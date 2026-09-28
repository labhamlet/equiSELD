from __future__ import annotations

import joblib
import numpy as np

# (swap, sx, sy, sz); index 0 is the identity
TRANSFORMS = [(swap, sx, sy, sz)
              for swap in (0, 1) for sx in (1, -1)
              for sy in (1, -1) for sz in (1, -1)]

_MEL_W, _MEL_Y, _MEL_Z, _MEL_X, _IV_Y, _IV_Z, _IV_X = range(7)


def load_scaler(wts_file: str, nb_mel: int):
    """(mean, scale) as [7, nb_mel] float64 from the saved StandardScaler."""
    scaler = joblib.load(wts_file)
    mean = np.asarray(scaler.mean_, dtype=np.float64).reshape(-1, nb_mel)
    scale = np.asarray(scaler.scale_, dtype=np.float64).reshape(-1, nb_mel)
    assert mean.shape[0] == 7, \
        'ACS augmentation supports FOA mel+IV features only (7 ch), got %d' \
        % mean.shape[0]
    return mean, scale


def transform_xyz(x, y, z, t):
    """Apply transform t to cartesian components (arrays or scalars)."""
    swap, sx, sy, sz = t
    if swap:
        x, y = y, x
    return sx * x, sy * y, sz * z


def transform_features(fd: np.ndarray, t) -> np.ndarray:
    """fd: DENORMALIZED features [7, T, mel] -> transformed copy."""
    swap, sx, sy, sz = t
    out = fd.copy()
    if swap:
        out[_MEL_Y] = fd[_MEL_X]
        out[_MEL_X] = fd[_MEL_Y]
    out[_IV_X] = sx * (fd[_IV_Y] if swap else fd[_IV_X])
    out[_IV_Y] = sy * (fd[_IV_X] if swap else fd[_IV_Y])
    out[_IV_Z] = sz * fd[_IV_Z]
    return out


def transform_labels(lab: np.ndarray, t, nb_classes: int) -> np.ndarray:
    """lab: [T, 6, 4, C] ADPIT ([act,x,y,z] on dim 2) or [T, 3C] masked
    single-ACCDOA (x|y|z blocks) -> transformed copy."""
    out = lab.copy()
    if lab.ndim == 4:                                     # ADPIT [T, 6, 4, C]
        x, y, z = lab[:, :, 1], lab[:, :, 2], lab[:, :, 3]
        out[:, :, 1], out[:, :, 2], out[:, :, 3] = transform_xyz(x, y, z, t)
    else:                                                 # [T, 3C]
        C = nb_classes
        x, y, z = lab[:, :C], lab[:, C:2 * C], lab[:, 2 * C:3 * C]
        out[:, :C], out[:, C:2 * C], out[:, 2 * C:3 * C] = \
            transform_xyz(x, y, z, t)
    return out


def augment_batch(feat: np.ndarray, label: np.ndarray,
                  mean: np.ndarray, scale: np.ndarray,
                  nb_classes: int, rng=None) -> tuple[np.ndarray, np.ndarray]:
    assert feat.shape[1] == 7, 'expected 7-channel FOA mel+IV features'
    rng = rng or np.random
    m, s = mean[:, None, :], scale[:, None, :]
    for b in range(feat.shape[0]):
        t = TRANSFORMS[rng.randint(len(TRANSFORMS))]
        if t == (0, 1, 1, 1):
            continue
        fd = feat[b] * s + m                              # denormalize
        feat[b] = (transform_features(fd, t) - m) / s     # renormalize
        label[b] = transform_labels(label[b], t, nb_classes)
    return feat, label
