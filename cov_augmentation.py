from __future__ import annotations

import os

import joblib
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

import cls_feature_class
from make_rotated_copies import haar_o3

NB_COV_CH = 10


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------

def cov_features_for_spectra(fc, spect: np.ndarray) -> np.ndarray:
    """spect: [T, bins, 4] complex linear spectra (baseline _spectrogram
    output, ACN [W, Y, Z, X]) -> [T, 10 * nb_mel] float32."""
    W = spect[:, :, 0]
    d = spect[:, :, [3, 1, 2]]                        # (x, y, z) dipoles
    mel = fc._mel_wts                                 # [bins, nb_mel]
    nb_mel = mel.shape[1]

    P_w = np.dot(np.abs(W) ** 2, mel)                 # [T, mel]
    idx = [(0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2)]
    M = np.stack([np.dot(np.real(d[:, :, i] * np.conj(d[:, :, j])), mel)
                  for i, j in idx], axis=1)           # [T, 6, mel]

    E = fc._eps + (np.abs(W) ** 2
                   + (np.abs(spect[:, :, 1:]) ** 2).sum(-1) / 3.0)
    I = np.real(np.conj(W)[:, :, None] * d) / E[:, :, None]
    IV = np.stack([np.dot(I[:, :, k], mel) for k in range(3)], axis=1)

    out = np.concatenate([P_w[:, None], M, IV], axis=1)   # [T, 10, mel]
    return out.reshape(out.shape[0], NB_COV_CH * nb_mel).astype(np.float32)


def extract_all_cov_features(params: dict) -> None:
    """Writes [T_feat, 10*nb_mel] per recording under
    params['cov_feat_label_dir']"""
    p_cov = dict(params)
    p_cov['feat_label_dir'] = params['cov_feat_label_dir']
    fc = cls_feature_class.FeatureClass(p_cov)
    feat_dir = fc.get_normalized_feat_dir()
    cls_feature_class.create_folder(feat_dir)
    aud_dir = os.path.join(params['dataset_dir'],
                           '{}_dev'.format(params['dataset']))
    label_hop = int(params['fs'] * params['label_hop_len_s'])
    wavs = []
    for dirpath, _d, files in os.walk(aud_dir):
        wavs += [os.path.join(dirpath, f) for f in files
                 if f.endswith('.wav')]
    print('cov feature extraction: {} files -> {}'.format(
        len(wavs), feat_dir), flush=True)
    for cnt, wav_path in enumerate(sorted(wavs)):
        audio, _fs = fc._load_audio(wav_path)
        nb_feat = int(len(audio) / float(fc._hop_len))
        spect = fc._spectrogram(audio, nb_feat)
        feat = cov_features_for_spectra(fc, spect)
        T_feat = 5 * int(audio.shape[0] / float(label_hop))
        name = os.path.splitext(os.path.basename(wav_path))[0]
        np.save(os.path.join(feat_dir, name + '.npy'), feat[:T_feat])
        if cnt % 100 == 0:
            print('{}: {}, {}'.format(cnt, name, feat[:T_feat].shape),
                  flush=True)


# ---------------------------------------------------------------------------
# Assembly: stored cov tensor + rotation R -> baseline 7-ch z-scored feature
# ---------------------------------------------------------------------------

def _power_to_db(x: np.ndarray, top_db: float = 80.0) -> np.ndarray:
    """librosa.power_to_db(ref=1.0, amin=1e-10) semantics, per channel."""
    log_spec = 10.0 * np.log10(np.maximum(x, 1e-10))
    return np.maximum(log_spec, log_spec.max() - top_db)


def assemble_rotated(cov: np.ndarray, R: np.ndarray, nb_mel: int,
                     sc_mean: np.ndarray, sc_scale: np.ndarray) -> np.ndarray:
    """cov: [T, 10*nb_mel] -> baseline z-scored features [7, T, nb_mel] of
    the R-rotated scene. Baseline channel order [W, Y, Z, X | IVy, IVz, IVx].
    """
    T = cov.shape[0]
    c = cov.reshape(T, NB_COV_CH, nb_mel).astype(np.float64)
    P_w, Mv, IV = c[:, 0], c[:, 1:7], c[:, 7:10]

    # symmetric matrix rotation, diagonal only: d_i = sum_jk R_ij R_ik M_jk
    xx, yy, zz, xy, xz, yz = (Mv[:, k] for k in range(6))
    diag = []
    for i in range(3):
        r = R[i]
        diag.append(r[0] * r[0] * xx + r[1] * r[1] * yy + r[2] * r[2] * zz
                    + 2 * (r[0] * r[1] * xy + r[0] * r[2] * xz
                           + r[1] * r[2] * yz))
    dx, dy, dz = diag                                   # rotated |X|,|Y|,|Z|
    iv = np.einsum('ij,tjm->tim', R, IV)                # rotated IV (x,y,z)

    feat = np.empty((7, T, nb_mel))
    feat[0] = _power_to_db(P_w)
    feat[1] = _power_to_db(np.maximum(dy, 0.0))         # mel Y
    feat[2] = _power_to_db(np.maximum(dz, 0.0))         # mel Z
    feat[3] = _power_to_db(np.maximum(dx, 0.0))         # mel X
    feat[4], feat[5], feat[6] = iv[:, 1], iv[:, 2], iv[:, 0]  # IVy, IVz, IVx

    m = sc_mean.reshape(7, 1, nb_mel)
    s = sc_scale.reshape(7, 1, nb_mel)
    return ((feat - m) / s).astype(np.float32)


# ---------------------------------------------------------------------------
# Train-only generator (baseline API), mirroring WaveformDataGenerator
# ---------------------------------------------------------------------------

class _CovChunkDataset(Dataset):
    def __init__(self, params: dict, split):
        self._params = params
        base_fc = cls_feature_class.FeatureClass(params)   # baseline dirs
        scaler = joblib.load(base_fc.get_normalized_wts_file())
        self._mean = np.asarray(scaler.mean_)
        self._scale = np.asarray(scaler.scale_)
        self._nb_mel = params['nb_mel_bins']
        self._label_seq = params['label_sequence_length']
        self._ratio = int(params['label_hop_len_s'] // params['hop_len_s'])
        self._feat_seq = self._label_seq * self._ratio
        p_cov = dict(params)
        p_cov['feat_label_dir'] = params['cov_feat_label_dir']
        feat_dir = cls_feature_class.FeatureClass(p_cov) \
            .get_normalized_feat_dir()
        lab_dir = base_fc.get_label_dir()
        self._files, weights, n_grid = [], [], 0
        for fname in sorted(os.listdir(feat_dir)):
            if not fname.endswith('.npy') or int(fname[4]) not in split:
                continue
            T = np.load(os.path.join(lab_dir, fname), mmap_mode='r').shape[0]
            if T < self._label_seq:
                continue
            self._files.append((os.path.join(feat_dir, fname),
                                os.path.join(lab_dir, fname), T))
            weights.append(T - self._label_seq + 1)
            n_grid += T // self._label_seq
        assert self._files, 'no cov training files found'
        w = np.asarray(weights, dtype=np.float64)
        self._cum = np.cumsum(w / w.sum())
        self._epoch_chunks = int(params.get('cov_epoch_chunks') or n_grid)
        self._labels_cache: dict[int, np.ndarray] = {}

    def __len__(self):
        return self._epoch_chunks

    def _rng(self):
        info = torch.utils.data.get_worker_info()
        if not hasattr(self, '_worker_rng'):
            seed = (torch.initial_seed()
                    + (info.id if info is not None else 0)) % (2 ** 63)
            self._worker_rng = np.random.default_rng(seed)
        return self._worker_rng

    def __getitem__(self, _idx):
        rng = self._rng()
        fi = min(int(np.searchsorted(self._cum, rng.random(), 'right')),
                 len(self._files) - 1)
        feat_path, lab_path, T = self._files[fi]
        l0 = int(rng.integers(0, T - self._label_seq + 1))
        R = haar_o3(rng)
        cov = np.load(feat_path, mmap_mode='r')[
            l0 * self._ratio:(l0 + self._label_seq) * self._ratio]
        feat = assemble_rotated(np.asarray(cov), R, self._nb_mel,
                                self._mean, self._scale)
        if fi not in self._labels_cache:
            self._labels_cache[fi] = np.load(lab_path)
        lab = self._labels_cache[fi][l0:l0 + self._label_seq].copy()
        lab[:, :, 1:4, :] = np.einsum('ij,tkjc->tkic', R, lab[:, :, 1:4, :])
        return (torch.from_numpy(feat), torch.from_numpy(lab).float())


class CovRotDataGenerator(object):
    """Baseline-API train generator: continuous O(3) augmentation via
    rotation-complete covariance features."""

    def __init__(self, params, split=1, shuffle=True, per_file=False,
                 is_eval=False):
        assert not per_file and not is_eval and shuffle
        assert params.get('model', 'seldnet') == 'seldnet', \
            'rotation augmentation is a no-op for the equivariant models'
        self._params = params
        self._batch_size = params['batch_size']
        self._workers = int(params.get('cov_num_workers', 8))
        self._ds = _CovChunkDataset(params, list(np.atleast_1d(split)))
        self._nb_total_batches = len(self._ds) // self._batch_size
        print('\tCovRotDataGenerator (O(3) covariance-feature aug): '
              '{} recordings, {} chunks/epoch -> {} batches/epoch\n'.format(
                  len(self._ds._files), len(self._ds),
                  self._nb_total_batches), flush=True)

    def get_data_sizes(self):
        p = self._params
        feat = (self._batch_size, 7, p['feature_sequence_length'],
                p['nb_mel_bins'])
        lab = (self._batch_size, p['label_sequence_length'],
               p['unique_classes'] * 3 * (3 if p['multi_accdoa'] else 1))
        return feat, lab

    def get_total_batches_in_data(self):
        return self._nb_total_batches

    def generate(self):
        loader = DataLoader(self._ds, batch_size=self._batch_size,
                            num_workers=self._workers, drop_last=True,
                            persistent_workers=False)
        for feat, lab in loader:
            yield feat.numpy(), lab.numpy()
