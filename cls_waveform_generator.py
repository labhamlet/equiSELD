
from __future__ import annotations

import os

import joblib
import numpy as np
import soundfile as sf
import torch
from torch.utils.data import DataLoader, Dataset

import cls_feature_class
from make_rotated_copies import haar_o3, rotate_foa_audio


EQ_CTX = 2      # context frames kept at each chunk edge (equiseld)


class _WaveChunkDataset(Dataset):
    def __init__(self, params: dict, split):
        self._params = params
        self._fc = cls_feature_class.FeatureClass(params)
        self._equiseld = params.get('model', 'seldnet') == 'equiseld'
        self._feat_mod = None
        if not self._equiseld:
            scaler = joblib.load(self._fc.get_normalized_wts_file())
            self._mean = np.asarray(scaler.mean_)
            self._scale = np.asarray(scaler.scale_)

        self._label_seq = params['label_sequence_length']          # 50
        self._ratio = int(params['label_hop_len_s'] // params['hop_len_s'])
        self._feat_seq = self._label_seq * self._ratio             # 250
        self._hop = self._fc._hop_len                              # 640
        self._nfft = self._fc._nfft                                # 2048
        self._pad = self._nfft // 2

        label_dir = self._fc.get_label_dir()
        aud_dir = os.path.join(params['dataset_dir'],
                               '{}_dev'.format(params['dataset']))
        wav_by_stem = {}
        for dirpath, _d, files in os.walk(aud_dir):
            for f in files:
                if f.endswith('.wav'):
                    wav_by_stem[os.path.splitext(f)[0]] = \
                        os.path.join(dirpath, f)

        self._files = []          # (wav_path, label_path, T_label, n_samples)
        weights = []
        n_grid_chunks = 0         # baseline's non-overlapping chunk count
        for fname in sorted(os.listdir(label_dir)):
            if not fname.endswith('.npy') or int(fname[4]) not in split:
                continue
            stem = os.path.splitext(fname)[0]
            if stem not in wav_by_stem or '_rot' in stem:
                continue
            wav = wav_by_stem[stem]
            info = sf.info(wav)
            T = np.load(os.path.join(label_dir, fname),
                        mmap_mode='r').shape[0]
            if T < self._label_seq:
                continue
            self._files.append((wav, os.path.join(label_dir, fname),
                                T, info.frames))
            n_grid = T // self._label_seq
            weights.append(n_grid if self._equiseld
                           else T - self._label_seq + 1)
            n_grid_chunks += T // self._label_seq
        assert self._files, 'no training recordings found'
        w = np.asarray(weights, dtype=np.float64)
        self._cum = np.cumsum(w / w.sum())
        self._n_positions = int(w.sum())
        self._epoch_chunks = int(self._params.get('wave_epoch_chunks')
                                 or n_grid_chunks)
        self._labels_cache: dict[int, np.ndarray] = {}

    # ------------------------------------------------------------------
    def n_positions(self) -> int:
        return self._n_positions

    def __len__(self) -> int:
        """Chunks per epoch — defaults to the baseline's non-overlapping
        chunk count, so steps/epoch match the unaugmented runs."""
        return self._epoch_chunks

    def _rng(self) -> np.random.Generator:
        info = torch.utils.data.get_worker_info()
        if not hasattr(self, '_worker_rng'):
            seed = (torch.initial_seed() +
                    (info.id if info is not None else 0)) % (2 ** 63)
            self._worker_rng = np.random.default_rng(seed)
        return self._worker_rng

    def _read_chunk(self, wav_path: str, n_samples: int, f0: int,
                    n_frames: int, pad: int = None) -> np.ndarray:
        pad = self._pad if pad is None else pad
        lo = f0 * self._hop - pad
        hi = (f0 + n_frames - 1) * self._hop + pad + 1
        a, b = max(lo, 0), min(hi, n_samples)
        audio, _sr = sf.read(wav_path, start=a, frames=b - a,
                             dtype='int16', always_2d=True)
        audio = audio[:, :4] / 32768.0 + 1e-8      # baseline _load_audio
        return np.pad(audio, ((a - lo, hi - b), (0, 0)))

    def _extract(self, audio: np.ndarray) -> np.ndarray:
        """context buffer (N, 4) -> normalized features [7, feat_seq, mel]"""
        if self._equiseld:
            return self._extract_equiseld(audio)
        import librosa
        fcls = self._fc
        spectra = []
        for ch in range(4):
            s = librosa.core.stft(np.asfortranarray(audio[:, ch]),
                                  n_fft=self._nfft, hop_length=self._hop,
                                  win_length=fcls._win_len, window='hann',
                                  center=False)
            spectra.append(s[:, :self._feat_seq])
        spect = np.array(spectra).T                # [T, bins, 4]
        feat = np.concatenate((fcls._get_mel_spectrogram(spect),
                               fcls._get_foa_intensity_vectors(spect)),
                              axis=-1)             # [T, 7*mel]
        feat = (feat - self._mean) / self._scale
        M = fcls._nb_mel_bins
        return feat.reshape(-1, 7, M).transpose(1, 0, 2)

    def _extract_equiseld(self, audio: np.ndarray) -> np.ndarray:
        import torch as _torch
        import equiseld_dcase
        if self._feat_mod is None:
            cfg = equiseld_dcase.make_seld_config(self._params)
            self._feat_mod = equiseld_dcase.FoaFeatures(
                cfg.sample_rate, cfg.n_fft, cfg.hop, cfg.n_mels,
                cfg.f_vec_max, fmin=cfg.fmin).eval()
        wave = _torch.from_numpy(np.ascontiguousarray(audio.T)).float()
        feat = equiseld_dcase.features_from_wave(self._feat_mod, wave)
        feat = feat[EQ_CTX:EQ_CTX + self._feat_seq]     # [T, 11*mel]
        M = self._params['nb_mel_bins']
        return feat.reshape(-1, equiseld_dcase.NB_FEAT_CH, M).transpose(1, 0, 2)

    def __getitem__(self, _idx):
        rng = self._rng()
        fi = int(np.searchsorted(self._cum, rng.random(), side='right'))
        fi = min(fi, len(self._files) - 1)
        wav, lab_path, T, n_samples = self._files[fi]
        if self._equiseld:
            # the baseline's own grid, so the augmented twin and
            # EquiSELD see exactly the same crops and rotation
            # is the only difference between them
            l0 = int(rng.integers(0, T // self._label_seq)) * self._label_seq
        else:
            l0 = int(rng.integers(0, T - self._label_seq + 1))
        R = haar_o3(rng)

        if self._equiseld:
            audio = self._read_chunk(wav, n_samples,
                                     l0 * self._ratio - EQ_CTX,
                                     self._feat_seq + 2 * EQ_CTX, pad=0)
        else:
            audio = self._read_chunk(wav, n_samples, l0 * self._ratio,
                                     self._feat_seq)
        feat = self._extract(rotate_foa_audio(audio, R))

        if fi not in self._labels_cache:
            self._labels_cache[fi] = np.load(lab_path)
        lab = self._labels_cache[fi][l0:l0 + self._label_seq].copy()
        lab[:, :, 1:4, :] = np.einsum('ij,tkjc->tkic', R, lab[:, :, 1:4, :])
        return (torch.from_numpy(np.ascontiguousarray(feat)).float(),
                torch.from_numpy(lab).float())


class WaveformDataGenerator(object):
    """Baseline-API train generator: on-the-fly O(3)-augmented waveforms."""

    def __init__(self, params, split=1, shuffle=True, per_file=False,
                 is_eval=False):
        assert not per_file and not is_eval and shuffle, \
            'WaveformDataGenerator is train-only'
        assert params['dataset'] == 'foa' and not params['use_salsalite']
        assert (params.get('model', 'seldnet') != 'equiseld'
                or params.get('equiseld_equivariant', True) is False), \
            'O(3) augmentation is a no-op for the equivariant model; it is ' \
            'meaningful only for the non-equivariant twin'
        self._params = params
        self._batch_size = params['batch_size']
        self._workers = int(params.get('wave_num_workers', 8))
        self._ds = _WaveChunkDataset(params, list(np.atleast_1d(split)))
        self._nb_classes = params['unique_classes']
        self._nb_total_batches = len(self._ds) // self._batch_size
        print('\tWaveformDataGenerator (O(3) on-the-fly): {} recordings, '
              '{} chunk positions, {} chunks/epoch -> {} batches/epoch, '
              '{} workers\n'.format(
                  len(self._ds._files), self._ds.n_positions(),
                  len(self._ds), self._nb_total_batches, self._workers),
              flush=True)

    def get_data_sizes(self):
        p = self._params
        nb_ch = 11 if p.get('model') == 'equiseld' else 7
        feat_shape = (self._batch_size, nb_ch, p['feature_sequence_length'],
                      p['nb_mel_bins'])
        if p['multi_accdoa']:
            label_shape = (self._batch_size, p['label_sequence_length'],
                           self._nb_classes * 3 * 3)
        else:
            label_shape = (self._batch_size, p['label_sequence_length'],
                           self._nb_classes * 3)
        return feat_shape, label_shape

    def get_total_batches_in_data(self):
        return self._nb_total_batches

    def generate(self):
        loader = DataLoader(self._ds, batch_size=self._batch_size,
                            num_workers=self._workers, drop_last=True,
                            persistent_workers=False)
        for feat, lab in loader:
            yield feat.numpy(), lab.numpy()
