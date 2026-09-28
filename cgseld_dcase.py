from __future__ import annotations

import os

import numpy as np
import torch
import torch.nn as nn

from cgseld import CGSeldAccdoa, default_cgnet_params, wxyz_stft_to_sh_input

NB_FEAT_CH = 8            # 4 complex channels x (Re, Im)
NB_FREQ_BINS = 1024       # nfft 2048, DC dropped — hardcoded by the F-chain
NFFT = 2048


def stft_features_for_audio(audio: np.ndarray, fs: int,
                            scale: np.ndarray | None) -> np.ndarray:
    """(N, 4) float FOA audio -> [T_feat, 8*1024] float32 (untrimmed).

    Phase-aligned complex STFT, per-bin shared scaling."""
    import librosa
    hop = int(fs * 0.02)
    win = int(fs * 0.04)
    spectra = []
    for ch in range(4):
        s = librosa.stft(np.asfortranarray(audio[:, ch]), n_fft=NFFT,
                         hop_length=hop, win_length=win, window='hann')
        spectra.append(s[1:NB_FREQ_BINS + 1].T)        # drop DC -> [T, F]
    S = np.stack(spectra, axis=1)                      # [T, 4, F] complex
    # W-phase alignment by the conjugate unit phase (paper Eq. (22));
    # bounded and zero-safe, unlike dividing by W/(|W|+eps), which is
    # 0/0 on digitally silent bins
    S = S * (np.conj(S[:, 0:1]) / (np.abs(S[:, 0:1]) + 1e-12))
    if scale is not None:
        S = S * scale[None, None, :]
    out = np.empty((S.shape[0], NB_FEAT_CH, NB_FREQ_BINS), dtype=np.float32)
    out[:, 0::2] = S.real.astype(np.float32)
    out[:, 1::2] = S.imag.astype(np.float32)
    return out.reshape(S.shape[0], NB_FEAT_CH * NB_FREQ_BINS)


def fit_bin_scale(fc, wav_paths, fs: int, n_files: int = 60) -> np.ndarray:
    """One real scale per frequency bin: 1/sqrt(mean |W|^2), estimated on a
    file sample. Shared across channels/Re-Im — rotation-safe."""
    import librosa
    hop, win = int(fs * 0.02), int(fs * 0.04)
    acc, cnt = np.zeros(NB_FREQ_BINS), 0
    for p in wav_paths[:n_files]:
        audio, _fs = fc._load_audio(p)
        s = librosa.stft(np.asfortranarray(audio[:, 0]), n_fft=NFFT,
                         hop_length=hop, win_length=win, window='hann')
        acc += (np.abs(s[1:NB_FREQ_BINS + 1]) ** 2).mean(axis=1)
        cnt += 1
    scale = 1.0 / np.sqrt(acc / cnt + 1e-12)
    scale = np.minimum(scale, 100.0 * np.median(scale))
    return scale.astype(np.float64)


def extract_all_cgseld_features(params: dict) -> None:
    import cls_feature_class
    assert params['dataset'] == 'foa'
    fc = cls_feature_class.FeatureClass(params)
    feat_dir = fc.get_normalized_feat_dir()
    cls_feature_class.create_folder(feat_dir)
    fs = params['fs']
    label_hop = int(fs * params['label_hop_len_s'])

    aud_dir = os.path.join(params['dataset_dir'],
                           '{}_dev'.format(params['dataset']))
    wavs = []
    for dirpath, _d, files in os.walk(aud_dir):
        wavs += [os.path.join(dirpath, f) for f in files
                 if f.endswith('.wav')]
    wavs = sorted(wavs)
    folds = params.get('extract_folds')
    if folds:
        wavs = [w for w in wavs if int(os.path.basename(w)[4]) in folds]

    train_folds = tuple(params.get('train_splits') or (1, 2, 3))
    train_wavs = [w for w in wavs
                  if int(os.path.basename(w)[4]) in train_folds]
    mu = sd = None
    if params.get('cgseld_std_scaler'):
        scale = None
        stat_path = os.path.join(params['feat_label_dir'],
                                 'cgseld_std_wts.npz')
        if os.path.exists(stat_path):
            d = np.load(stat_path)
            mu, sd = d['mu'], d['sd']
        else:
            acc = acc2 = None
            n = 0
            for w in train_wavs:
                audio, _fs = fc._load_audio(w)
                feat = stft_features_for_audio(audio, fs, None).astype(
                    np.float64)
                acc = feat.sum(0) if acc is None else acc + feat.sum(0)
                acc2 = ((feat ** 2).sum(0) if acc2 is None
                        else acc2 + (feat ** 2).sum(0))
                n += feat.shape[0]
            mu = acc / n
            sd = np.sqrt(np.maximum(acc2 / n - mu ** 2, 0.0)) + 1e-8
            np.savez(stat_path, mu=mu, sd=sd)
    else:
        scale_path = os.path.join(params['feat_label_dir'], 'cgseld_wts.npy')
        if os.path.exists(scale_path):
            scale = np.load(scale_path)
        else:
            scale = fit_bin_scale(fc, train_wavs, fs)
            np.save(scale_path, scale)
    print('cgseld feature extraction: {} files from {}\n\t-> {}'.format(
        len(wavs), aud_dir, feat_dir), flush=True)

    for cnt, wav_path in enumerate(wavs):
        audio, _fs = fc._load_audio(wav_path)
        nb_label_fr = int(audio.shape[0] / float(label_hop))
        feat = stft_features_for_audio(audio, fs, scale)
        if mu is not None:
            feat = ((feat - mu) / sd).astype(np.float32)
        assert np.isfinite(feat).all(), \
            'non-finite features from {}'.format(wav_path)
        T_feat = 5 * nb_label_fr
        assert feat.shape[0] >= T_feat, (wav_path, feat.shape, T_feat)
        name = os.path.splitext(os.path.basename(wav_path))[0]
        np.save(os.path.join(feat_dir, '{}.npy'.format(name)),
                feat[:T_feat])
        print('{}: {}, {}'.format(cnt, name, (T_feat,
                                              NB_FEAT_CH * NB_FREQ_BINS)),
              flush=True)


class CGSeldModel(nn.Module):
    """seldnet-API wrapper around CGSeldAccdoa."""

    def __init__(self, in_feat_shape, out_shape, params: dict):
        super().__init__()
        assert in_feat_shape[1] == NB_FEAT_CH, in_feat_shape
        assert in_feat_shape[-1] == NB_FREQ_BINS, in_feat_shape
        K = 3 if params['multi_accdoa'] else 1
        C = params['unique_classes']
        assert out_shape[-1] == K * 3 * C, (out_shape, K, C)
        self.net = CGSeldAccdoa(default_cgnet_params(nb_class=C,
                                                     num_tracks=K))
        self.last_sed_logits = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, 8, T_feat, 1024] -> complex STFT [B, 4, T, F, 2] -> SH input
        stft = torch.stack([x[:, 0::2], x[:, 1::2]], dim=-1)
        out = self.net(wxyz_stft_to_sh_input(stft), update=self.training)
        self.last_sed_logits = self.net.last_sed_logits
        return out


def build_optimizer(model: nn.Module, lr: float, weight_decay: float):
    """AdamW; no decay for biases (incl. GRU bias_ih/hh) and BatchNorm."""
    bn_prefixes = set()
    for mn, m in model.named_modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
            bn_prefixes.add(mn + '.' if mn else '')
    decay, nodecay = [], []
    for pn, p in model.named_parameters():
        if not p.requires_grad:
            continue
        leaf = pn.split('.')[-1]
        is_bn = any(pn.startswith(pref) for pref in bn_prefixes)
        (nodecay if ('bias' in leaf or is_bn) else decay).append(p)
    return torch.optim.AdamW(
        [{'params': decay, 'weight_decay': weight_decay},
         {'params': nodecay, 'weight_decay': 0.0}],
        lr=lr, betas=(0.9, 0.95))


class OriginalHeadLoss(nn.Module):
    """Sato et al.'s training objective, for the single-DOA head.
    """

    def __init__(self, model=None, doa_weight: float = 1.0,
                 eps: float = 1e-7):
        super().__init__()
        self.doa_weight = doa_weight
        self.eps = eps

    def forward(self, output: torch.Tensor,
                target: torch.Tensor) -> torch.Tensor:
        B, T, C3 = target.shape
        C = C3 // 3
        tgt = target.reshape(B, T, 3, C)
        act = (tgt.norm(dim=2) > 0.5).float()              # [B, T, C]

        # activity is read from the output
        pred = output.reshape(B, T, 3, C)
        prob = pred.norm(dim=2).clamp(self.eps, 1.0 - self.eps)
        l_sed = nn.functional.binary_cross_entropy(prob, act)
        pu = pred / (pred.norm(dim=2, keepdim=True) + self.eps)
        tu = tgt / (tgt.norm(dim=2, keepdim=True) + self.eps)
        cos = (pu * tu).sum(dim=2).clamp(-1.0 + self.eps, 1.0 - self.eps)
        ang = torch.acos(cos)                              # [B, T, C]
        l_doa = (ang * act).sum() / act.sum().clamp(min=1.0)

        return l_sed + self.doa_weight * l_doa
