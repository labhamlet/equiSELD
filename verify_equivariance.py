"""Measure a trained checkpoint's equivariance error on real features.

For each group element Q we compare f(Q.x) against Q.f(x), where Q acts
on the two intensity-vector channels of the input features and on the
xyz axis of the ACCDOA output.  Reported as the max absolute deviation
alongside the output scale, so the number is readable as a fraction.

Proper rotations and reflections are reported separately: the O(3)
model should be at float32 round-off for both, the SO(3) variant only
for the proper half.

Usage: python verify_equivariance.py 93:so393 8:equi1 ...
"""

from __future__ import annotations

import os
import sys

import numpy as np

if not hasattr(np, 'float'):
    np.float = float
    np.int = int

import torch

import cls_feature_class
from evaluate_rotated import build_model, load_checkpoint, quiet_params
from make_rotated_copies import haar_o3

N_SCALARS = 5          # feature channels 0..4 are invariant scalars


def rotate_features(x: torch.Tensor, R: torch.Tensor) -> torch.Tensor:
    """x: (B, 11, T, M). Channels 5..10 are va(xyz), vr(xyz)."""
    out = x.clone()
    v = x[:, N_SCALARS:].reshape(x.shape[0], 2, 3, *x.shape[2:])
    v = torch.einsum('de,bnexy->bndxy', R, v)
    out[:, N_SCALARS:] = v.reshape(x.shape[0], 6, *x.shape[2:])
    return out


def main():
    pairs = sys.argv[1:]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    rng = np.random.default_rng(0)

    # a proper rotation and a reflection
    Rp = haar_o3(rng)
    Rp = Rp * np.sign(np.linalg.det(Rp))
    elems = [('rotation  (det +1)', Rp), ('reflection (det -1)', -Rp)]

    for pair in pairs:
        task_id, job_id = pair.split(':')
        params = quiet_params(task_id)
        tag = 'multiaccdoa' if params['multi_accdoa'] else 'accdoa'
        unique = '{}_{}_dev_split0_{}_foa'.format(task_id, job_id, tag)
        model = build_model(params, device).to(device)
        load_checkpoint(model, os.path.join(params['model_dir'],
                                            unique + '_model.h5'), device)
        model.eval()

        # one real test-fold feature file
        fc = cls_feature_class.FeatureClass(params)
        fd = fc.get_normalized_feat_dir()
        test_fold = 6 if '2021' in params['dataset_dir'] else 4
        name = sorted(f for f in os.listdir(fd)
                      if f.endswith('.npy') and int(f[4]) == test_fold)[0]
        feat = np.load(os.path.join(fd, name))
        M = params['nb_mel_bins']
        T = params['feature_sequence_length']
        n_seq = max(1, feat.shape[0] // T)
        x = torch.from_numpy(feat[:n_seq * T]).float()
        x = x.reshape(1, -1, 11, M).permute(0, 2, 1, 3).contiguous().to(device)

        K = 3 if params['multi_accdoa'] else 1
        C = params['unique_classes']
        print('##### {} ({}) #####'.format(unique, name), flush=True)
        with torch.no_grad():
            y = model(x).reshape(1, -1, K, 3, C)
            for label, Q in elems:
                Qt = torch.tensor(Q, dtype=torch.float32, device=device)
                yq = model(rotate_features(x, Qt)).reshape(1, -1, K, 3, C)
                ref = torch.einsum('de,btkec->btkdc', Qt, y)
                err = (yq - ref).abs().max().item()
                scale = y.abs().max().item()
                # the metric only sees predictions above the detection
                # threshold; inactive slots have near-zero magnitude and
                # an unconstrained direction, so their deviation is large
                # but invisible downstream
                mag = ref.norm(dim=3)                          # (1,T,K,C)
                ang = torch.rad2deg(torch.acos((
                    torch.nn.functional.normalize(yq, dim=3) *
                    torch.nn.functional.normalize(ref, dim=3)
                ).sum(3).clamp(-1, 1)))                        # (1,T,K,C)
                flat_m, flat_a = mag.flatten(), ang.flatten()
                k = min(100, flat_m.numel())
                idx = flat_m.topk(k).indices
                print('  {}: all slots {:.3e} ({:.1%} of {:.3f}) | '
                      'top-{} most confident: max angle {:6.2f} deg, '
                      'median {:6.2f} deg (magnitudes {:.3f}-{:.3f})'.format(
                          label, err, err / max(scale, 1e-12), scale, k,
                          flat_a[idx].max().item(),
                          flat_a[idx].median().item(),
                          flat_m[idx].min().item(),
                          flat_m[idx].max().item()), flush=True)


if __name__ == '__main__':
    main()
