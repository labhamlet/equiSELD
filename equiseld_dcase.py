from __future__ import annotations

import os

import numpy as np
import torch
import torch.nn as nn

from equiseld import (
    EquivISAB, EquivPMA, EquivSeldSetTransformer, FoaFeatures, SeldConfig,
    SVLayerNorm,
)

NB_FEAT_CH = FoaFeatures.N_SCALARS + 3 * FoaFeatures.N_VECTORS   # 5 + 6 = 11


# ---------------------------------------------------------------------------
# Config mapping
# ---------------------------------------------------------------------------

def make_seld_config(params: dict) -> SeldConfig:
    """SeldConfig from the baseline params dict (equiseld_* keys optional)."""
    hop = int(params['fs'] * params['hop_len_s'])                 # 480
    time_pool = int(params['label_hop_len_s'] // params['hop_len_s'])  # 5
    return SeldConfig(
        sample_rate=params['fs'],
        n_fft=params.get('equiseld_n_fft', 1024),
        hop=hop,
        n_mels=params['nb_mel_bins'],
        fmin=params.get('equiseld_fmin', 20.0),
        f_vec_max=params.get('equiseld_f_vec_max', 9000.0),
        time_pool=time_pool,
        num_classes=params['unique_classes'],
        num_tracks=3 if params['multi_accdoa'] else 1,
        d_s=params.get('equiseld_d_s', 128),
        d_v=params.get('equiseld_d_v', 32),
        heads=params.get('equiseld_heads', 4),
        freq_blocks=params.get('equiseld_freq_blocks', 2),
        isab_inducing=params.get('equiseld_isab_inducing', 16),
        temporal_blocks=params.get('equiseld_temporal_blocks', 4),
        dropout=params['dropout_rate'],
        equivariant=params.get('equiseld_equivariant', True),
        so3_only=params.get('equiseld_so3_only', False),
    )


# ---------------------------------------------------------------------------
# Feature extraction (replaces extract_all_feature + preprocess_features)
# ---------------------------------------------------------------------------

def features_from_wave(feat: FoaFeatures, wave: torch.Tensor) -> np.ndarray:
    """(4, N) float32 FOA wave -> [T_feat, 11*n_mels] float32 (untrimmed).

    Flattening is channel-major per frame (block c = cols [c*M:(c+1)*M]),
    matching DataGenerator's reshape to (T, nb_ch, mel)."""
    with torch.no_grad():
        s, v = feat(wave.unsqueeze(0))               # (1,T,M,5), (1,T,M,2,3)
    s, v = s[0], v[0]
    T, M = s.shape[0], s.shape[1]
    s_flat = s.permute(0, 2, 1).reshape(T, FoaFeatures.N_SCALARS * M)
    v_flat = v.permute(0, 2, 3, 1).reshape(T, 3 * FoaFeatures.N_VECTORS * M)
    out = torch.cat([s_flat, v_flat], dim=1).cpu().numpy().astype(np.float32)
    return out


def silence_pad_row(params: dict) -> np.ndarray:
    cfg = make_seld_config(params)
    feat = FoaFeatures(cfg.sample_rate, cfg.n_fft, cfg.hop, cfg.n_mels,
                       cfg.f_vec_max, fmin=cfg.fmin).eval()
    wave = torch.zeros(4, cfg.hop * 10)
    rows = features_from_wave(feat, wave)
    return rows[rows.shape[0] // 2].copy()


def extract_all_equiseld_features(params: dict, is_eval: bool = False) -> None:
    """Walk <dataset_dir>/<dataset>_{dev|eval} and write one feature .npy per
    recording into get_normalized_feat_dir() (same naming as the baseline, so
    label .npy names produced by extract_all_labels match)."""
    import cls_feature_class

    assert params['dataset'] == 'foa', "equiseld consumes FOA only"
    fc = cls_feature_class.FeatureClass(params, is_eval=is_eval)
    feat_dir = fc.get_normalized_feat_dir()
    cls_feature_class.create_folder(feat_dir)

    cfg = make_seld_config(params)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    feat = FoaFeatures(cfg.sample_rate, cfg.n_fft, cfg.hop, cfg.n_mels,
                       cfg.f_vec_max, fmin=cfg.fmin).to(device).eval()

    aud_dir = os.path.join(params['dataset_dir'], '{}_{}'.format(
        params['dataset'], 'eval' if is_eval else 'dev'))
    label_hop = int(params['fs'] * params['label_hop_len_s'])     # 2400
    tp = cfg.time_pool

    wavs = []
    for dirpath, _dirs, files in os.walk(aud_dir):
        wavs += [os.path.join(dirpath, f) for f in files if f.endswith('.wav')]
    print('equiseld feature extraction: {} files from {}\n\t-> {}'.format(
        len(wavs), aud_dir, feat_dir))

    for cnt, wav_path in enumerate(sorted(wavs)):
        audio, _fs = fc._load_audio(wav_path)         # (N, 4), baseline scaling
        nb_label_fr = int(audio.shape[0] / float(label_hop))      # baseline T
        wave = torch.from_numpy(audio.T.copy()).float().to(device)
        out = features_from_wave(feat, wave)
        T_feat = tp * nb_label_fr                     # exact 5:1 alignment
        assert out.shape[0] >= T_feat, (wav_path, out.shape, T_feat)
        out = out[:T_feat]
        name = os.path.splitext(os.path.basename(wav_path))[0]
        np.save(os.path.join(feat_dir, '{}.npy'.format(name)), out)
        print('{}: {}, {}'.format(cnt, name, out.shape))


# ---------------------------------------------------------------------------
# seldnet_model-API wrapper
# ---------------------------------------------------------------------------

class EquiSeldModel(nn.Module):
    def __init__(self, in_feat_shape, out_shape, params: dict):
        super().__init__()
        cfg = make_seld_config(params)
        assert in_feat_shape[1] == NB_FEAT_CH, \
            'expected {} feature channels, got {}'.format(NB_FEAT_CH,
                                                          in_feat_shape[1])
        assert in_feat_shape[-1] == cfg.n_mels
        assert out_shape[-1] == cfg.num_tracks * 3 * cfg.num_classes, \
            (out_shape, cfg.num_tracks, cfg.num_classes)
        self.cfg = cfg
        self.net = EquivSeldSetTransformer(cfg)
        self.last_sed_logits: torch.Tensor | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 11, T_feat, mel) -> (B, T_feat//time_pool, K*3*C) in the
        baseline layout: track-major, then axis (x,y,z), then class."""
        B, C, T, M = x.shape
        ns = FoaFeatures.N_SCALARS
        s = x[:, :ns].permute(0, 2, 3, 1)                          # (B,T,M,5)
        v = (x[:, ns:]
             .reshape(B, FoaFeatures.N_VECTORS, 3, T, M)
             .permute(0, 3, 4, 1, 2))                              # (B,T,M,2,3)
        out = self.net.forward_features(s, v)
        self.last_sed_logits = out['sed_logits']                   # (B,T',K,C)
        acc = out['accdoa']                                        # (B,T',K,C,3)
        B2, T2, K, C2, _ = acc.shape
        return acc.permute(0, 1, 2, 4, 3).reshape(B2, T2, K * 3 * C2)


# ---------------------------------------------------------------------------
# Optimizer recipe (equiseld.py docstring; validated by the overfit test)
# ---------------------------------------------------------------------------

def build_optimizer(model: nn.Module, lr: float, weight_decay: float):
    """AdamW with weight decay excluded for biases, LayerNorm params,
    SVLayerNorm gains, band_emb, and ISAB inducing points / PMA seeds."""
    no_decay: set[str] = set()
    for mn, m in model.named_modules():
        prefix = mn + '.' if mn else ''
        if isinstance(m, nn.LayerNorm):
            for pn, _ in m.named_parameters(recurse=False):
                no_decay.add(prefix + pn)
        elif isinstance(m, SVLayerNorm):
            no_decay.add(prefix + 'gv')
        elif isinstance(m, EquivISAB):
            no_decay.add(prefix + 'I')
        elif isinstance(m, EquivPMA):
            no_decay.add(prefix + 'S')
    for pn, _ in model.named_parameters():
        if pn.endswith('.bias') or pn.split('.')[-1] == 'band_emb':
            no_decay.add(pn)

    decay, nodecay = [], []
    for pn, p in model.named_parameters():
        if p.requires_grad:
            (nodecay if pn in no_decay else decay).append(p)
    return torch.optim.AdamW(
        [{'params': decay, 'weight_decay': weight_decay},
         {'params': nodecay, 'weight_decay': 0.0}],
        lr=lr, betas=(0.9, 0.95))


def lr_factor(step: int, warmup: int, total: int, floor: float = 0.01) -> float:
    import math
    if step < warmup:
        return (step + 1) / max(1, warmup)
    prog = min(1.0, (step - warmup) / max(1, total - warmup))
    return floor + (1.0 - floor) * 0.5 * (1.0 + math.cos(math.pi * prog))


def make_scheduler(optimizer, warmup: int, total_steps: int,
                   floor: float = 0.01):
    """floor is the cosine's final lr fraction; floor=1.0 gives a constant
    lr after warmup."""
    return torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda s: lr_factor(s, warmup, total_steps, floor))
