
from __future__ import annotations

import math
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F


def _f(n: int) -> int:
    return math.factorial(n)


def clebsch_gordan(l1: int, l2: int, l3: int, m1: int, m2: int, m3: int) -> float:
    if m3 != m1 + m2 or not (abs(l1 - l2) <= l3 <= l1 + l2):
        return 0.0
    if abs(m1) > l1 or abs(m2) > l2 or abs(m3) > l3:
        return 0.0
    pref = (2 * l3 + 1) * _f(l3 + l1 - l2) * _f(l3 - l1 + l2) * _f(l1 + l2 - l3) \
        / _f(l1 + l2 + l3 + 1)
    pref *= _f(l3 + m3) * _f(l3 - m3) * _f(l1 - m1) * _f(l1 + m1) \
        * _f(l2 - m2) * _f(l2 + m2)
    s = 0.0
    for k in range(0, l1 + l2 - l3 + 1):
        d = [k, l1 + l2 - l3 - k, l1 - m1 - k, l2 + m2 - k,
             l3 - l2 + m1 + k, l3 - l1 - m2 + k]
        if min(d) < 0:
            continue
        s += (-1.0) ** k / (_f(d[0]) * _f(d[1]) * _f(d[2]) * _f(d[3])
                            * _f(d[4]) * _f(d[5]))
    return math.sqrt(pref) * s


def cg_matrix(l1: int, l2: int, l3: int) -> torch.Tensor:
    """[2l3+1, 2l1+1, 2l2+1] with C[m3, m1, m2]; m indexed from -l upward."""
    C = torch.zeros(2 * l3 + 1, 2 * l1 + 1, 2 * l2 + 1, dtype=torch.float64)
    for m1 in range(-l1, l1 + 1):
        for m2 in range(-l2, l2 + 1):
            m3 = m1 + m2
            if -l3 <= m3 <= l3:
                C[m3 + l3, m1 + l1, m2 + l2] = clebsch_gordan(
                    l1, l2, l3, m1, m2, m3)
    return C


# ---------------------------------------------------------------------------
# tau bookkeeping (channel axis = flattened (fragment j, degree l, order m))
# ---------------------------------------------------------------------------

def tau_decompose(taus: List[int]):
    mlos, mhis, l_begin, l_end = [], [], [], []
    for l, t in enumerate(taus):
        l_begin.append(0 if not mhis else mhis[-1])
        for _ in range(t):
            mlos.append(0 if not mhis else mhis[-1])
            mhis.append(mlos[-1] + 2 * l + 1)
        l_end.append(0 if not mhis else mhis[-1])
    return mlos, mhis, l_begin, l_end


def _trim_zeros_back(taus: List[int]) -> List[int]:
    t = list(taus)
    while t and t[-1] == 0:
        t.pop()
    return t


# ---------------------------------------------------------------------------
# Layers
# ---------------------------------------------------------------------------

class ComplexConv2d(nn.Module):
    """Complex 2D conv on [B, CH, T, F, 2] via two real convolutions."""

    def __init__(self, cin, cout, kernel_size, bias, padding):
        super().__init__()
        self.cout = cout
        self.conv_re = nn.Conv2d(cin, cout, kernel_size, padding=padding,
                                 bias=bias)
        self.conv_im = nn.Conv2d(cin, cout, kernel_size, padding=padding,
                                 bias=bias)

    def forward(self, x):
        re = self.conv_re(x[..., 0]) - self.conv_im(x[..., 1])
        im = self.conv_im(x[..., 0]) + self.conv_re(x[..., 1])
        return torch.stack((re, im), dim=4)


class SphHarmDistributedComplexConv2d(nn.Module):

    def __init__(self, taus_in, taus_out, kernel_size, padding,
                 skip_l0=False):
        super().__init__()
        self.taus_in = _trim_zeros_back(taus_in)
        self.taus_out = _trim_zeros_back(taus_out)
        self.skip_l0 = skip_l0
        if skip_l0:
            assert self.taus_in[0] >= self.taus_out[0]
        self.cconvs = nn.ModuleList([
            # reference parity: complex bias on the invariant (l=0) conv
            # only; biases on l>0 blocks would break equivariance
            ComplexConv2d(tin, tout, kernel_size, bias=(l == 0),
                          padding=padding)
            if (tin and tout) else None
            for l, (tin, tout) in enumerate(zip(self.taus_in, self.taus_out))
        ])
        _, _, self.l_begin, self.l_end = tau_decompose(self.taus_in)

    def forward(self, x):
        B, _, T_in, F_in, _ = x.shape
        out = []
        for l, (tin, tout) in enumerate(zip(self.taus_in, self.taus_out)):
            if tin == 0 or tout == 0:
                continue
            xl = x[:, self.l_begin[l]:self.l_end[l]]
            if l == 0 and self.skip_l0:
                out.append(xl[:, :tout])
                continue
            m = 2 * l + 1
            xl = (xl.reshape(B, tin, m, T_in, F_in, 2)
                  .permute(0, 2, 1, 3, 4, 5)
                  .reshape(B * m, tin, T_in, F_in, 2).contiguous())
            yl = self.cconvs[l](xl)
            _, _, T_out, F_out, _ = yl.shape
            out.append(yl.reshape(B, m, tout, T_out, F_out, 2)
                       .permute(0, 2, 1, 3, 4, 5)
                       .reshape(B, tout * m, T_out, F_out, 2))
        return torch.cat(out, dim=1)


class SphericalSigmaBN(nn.Module):
    def __init__(self, taus, momentum, eps):
        super().__init__()
        self.mlos, self.mhis, _, _ = tau_decompose(_trim_zeros_back(taus))
        # float64: squaring near-underflow fp32 activations must not
        # collapse a fragment's second moment to exactly zero
        self.sigma2 = nn.Parameter(
            torch.zeros(len(self.mlos), dtype=torch.float64),
            requires_grad=False)
        frag_id = torch.empty(self.mhis[-1], dtype=torch.long)
        for i, (lo, hi) in enumerate(zip(self.mlos, self.mhis)):
            frag_id[lo:hi] = i
        self.register_buffer('frag_id', frag_id, persistent=False)
        self.register_buffer(
            'frag_count',
            torch.tensor([hi - lo for lo, hi in zip(self.mlos, self.mhis)],
                         dtype=torch.float64), persistent=False)
        self._momentum_p = nn.Parameter(
            torch.tensor(float(momentum), dtype=torch.float64),
            requires_grad=False)
        self._eps_p = nn.Parameter(
            torch.tensor(float(eps), dtype=torch.float64),
            requires_grad=False)
        self.momentum = momentum
        self.eps = eps

    def forward(self, x, update: bool):
        if update:
            with torch.no_grad():
                ms = x.double().abs().mean(dim=(0, 2, 3, 4))
                s2 = torch.zeros_like(self.sigma2).index_add_(
                    0, self.frag_id, ms) / self.frag_count
                new = torch.where(self.sigma2 == 0, s2,
                                  self.sigma2 * (1 - self.momentum)
                                  + s2 * self.momentum)
                # never poison stats with NaN/inf batches
                self.sigma2.copy_(
                    torch.where(torch.isfinite(s2), new, self.sigma2))
        if not update:
            n_bad = int((~torch.isfinite(self.sigma2)).sum())
            all_zero = bool((self.sigma2 == 0).all())
            if n_bad or all_zero:
                raise RuntimeError(
                    'SphericalSigmaBN statistics invalid at eval: {} '
                    'non-finite of {} fragments{}'.format(
                        n_bad, len(self.sigma2),
                        '; all zero (never updated)' if all_zero else ''))
            # isolated zero stats are tolerated: a fragment with no signal
            # yet divides its (~zero) activations by eps — harmless
        denom = (torch.sqrt(self.sigma2) + self.eps)[self.frag_id]
        return x / denom.view(1, -1, 1, 1, 1).to(x.dtype)


class GroupAvgPoolFreq(nn.Module):
    """Frequency-only average pooling (equivariant: no mixing of m)."""

    def __init__(self, stride):
        super().__init__()
        self.stride = stride

    def forward(self, x):
        B, CH, T, Fq, RI = x.shape
        x = x.reshape(B, CH, T, Fq // self.stride, self.stride, RI)
        return x.mean(dim=4)


class ScaleInvariantSHActivation(nn.Module):
    """Each l>0 fragment (for l in l_use) is multiplied by a complex scalar
    computed linearly from the l=0 block, then divided by
    (eps + sqrt(fragment norm)). l=0 passes through."""

    def __init__(self, taus, l_use_list, eps):
        super().__init__()
        self.taus = _trim_zeros_back(taus)
        _, _, self.l_begin, self.l_end = tau_decompose(self.taus)
        self.lin_re = nn.ModuleList(
            [nn.Linear(self.taus[0], t, bias=True) for t in self.taus])
        self.lin_im = nn.ModuleList(
            [nn.Linear(self.taus[0], t, bias=True) for t in self.taus])
        self.l_use = [1 if l in l_use_list else 0
                      for l in range(len(self.taus))]
        assert self.l_use[0] == 0
        self.eps = eps

    def forward(self, x):
        if not any(self.l_use):
            return x
        B, CH, T, Fq, _ = x.shape
        x = x.permute(0, 2, 3, 1, 4).reshape(B * T * Fq, CH, 2)
        x0 = x[:, self.l_begin[0]:self.l_end[0]]
        re0, im0 = x0[..., 0], x0[..., 1]
        out = []
        for l, tau in enumerate(self.taus):
            if l == 0:
                out.append(x0)
                continue
            if tau == 0:
                continue
            xl = x[:, self.l_begin[l]:self.l_end[l]]
            if self.l_use[l]:
                m = 2 * l + 1
                xl = xl.reshape(-1, tau, m, 2)
                a_re = (self.lin_re[l](re0) - self.lin_im[l](im0)).unsqueeze(-1)
                a_im = (self.lin_im[l](re0) + self.lin_re[l](im0)).unsqueeze(-1)
                y_re = xl[..., 0] * a_re - xl[..., 1] * a_im
                y_im = xl[..., 1] * a_re + xl[..., 0] * a_im
                xl = torch.stack([y_re, y_im], dim=-1)
                norm = torch.sqrt((xl * xl).sum(dim=(2, 3)))
                xl = xl / (self.eps + torch.sqrt(norm)[:, :, None, None])
                xl = xl.reshape(-1, tau * m, 2)
            out.append(xl)
        x = torch.cat(out, dim=1)
        return x.reshape(B, T, Fq, CH, 2).permute(0, 3, 1, 2, 4).contiguous()


class CGElementProduct(nn.Module):
    """Degree-wise CG self-products, pure torch (no CUDA extension)."""

    def __init__(self, tau_in, Lout_max, eps, scale_invariance=True):
        super().__init__()
        self.tau_in = _trim_zeros_back(tau_in)
        self.Lmax_in = len(self.tau_in) - 1
        self.Lout_max = min(self.Lmax_in * 2, Lout_max)
        self.eps = eps
        self.scale_invariance = scale_invariance
        _, _, self.l_begin, self.l_end = tau_decompose(self.tau_in)

        # kept pairs per (l_source, lout), in (t1-major, t2-minor) order
        self.pairs = {}
        tau_out = [0] * (self.Lout_max + 1)
        self.order = [[] for _ in range(self.Lout_max + 1)]  # build recipe
        for l, T in enumerate(self.tau_in):
            if T:
                tau_out[l] += T
                self.order[l].append(('pass', l))
            if l > 0:
                for lout in range(min(2 * l, self.Lout_max) + 1):
                    kept = [(t1, t2) for t1 in range(T) for t2 in range(T)
                            if t1 < t2 or (t1 == t2 and lout % 2 == 0)]
                    if kept and lout <= self.Lout_max:
                        self.pairs[(l, lout)] = kept
                        tau_out[lout] += len(kept)
                        self.order[lout].append(('prod', l))
        self.tau_out = _trim_zeros_back(tau_out)

        for (l, lout), kept in self.pairs.items():
            C = cg_matrix(l, l, lout)          # float64: full CG precision
            self.register_buffer(f'cg_{l}_{lout}', C, persistent=False)
            idx = torch.tensor(kept, dtype=torch.long)
            self.register_buffer(f'idx_{l}_{lout}', idx, persistent=False)

    def forward(self, x):
        B, CH, T, Fq, _ = x.shape
        x = x.permute(0, 2, 3, 1, 4).reshape(B * T * Fq, CH, 2)
        frags = {}
        for l, tau in enumerate(self.tau_in):
            if tau:
                frags[l] = x[:, self.l_begin[l]:self.l_end[l]] \
                    .reshape(-1, tau, 2 * l + 1, 2)

        prods = {}
        for (l, lout), _kept in self.pairs.items():
            u = frags[l]
            idx = getattr(self, f'idx_{l}_{lout}')
            C = getattr(self, f'cg_{l}_{lout}').to(u.dtype)
            a = u[:, idx[:, 0]]                                  # [N,P,m1,2]
            b = u[:, idx[:, 1]]
            # complex bilinear CG product: w_m3 = sum C[m3,m1,m2] a_m1 b_m2
            ar, ai, br, bi = a[..., 0], a[..., 1], b[..., 0], b[..., 1]
            w_rr = torch.einsum('kmn,zpm,zpn->zpk', C, ar, br)
            w_ii = torch.einsum('kmn,zpm,zpn->zpk', C, ai, bi)
            w_ri = torch.einsum('kmn,zpm,zpn->zpk', C, ar, bi)
            w_ir = torch.einsum('kmn,zpm,zpn->zpk', C, ai, br)
            w = torch.stack([w_rr - w_ii, w_ri + w_ir], dim=-1)  # [N,P,m3,2]
            if self.scale_invariance:
                norm = torch.sqrt((w * w).sum(dim=(2, 3)))
                w = w / (self.eps + torch.sqrt(norm)[:, :, None, None])
            prods[(l, lout)] = w

        out = []
        for lout in range(len(self.tau_out)):
            for kind, l in self.order[lout]:
                if kind == 'pass':
                    out.append(frags[l].reshape(-1, self.tau_in[l]
                                                * (2 * l + 1), 2))
                else:
                    w = prods[(l, lout)]
                    out.append(w.reshape(-1, w.shape[1] * (2 * lout + 1), 2))
        x = torch.cat(out, dim=1)
        CH_out = tau_decompose(self.tau_out)[1][-1]
        return x.reshape(B, T, Fq, CH_out, 2).permute(0, 3, 1, 2, 4) \
            .contiguous()


class FirstOrderSphericalToVector(nn.Module):
    def forward(self, x):
        # x: [..., 3(m=-1,0,+1), 2]
        rx = (x[..., 0, 0] - x[..., 2, 0]) / math.sqrt(2.0)
        ry = (-x[..., 0, 1] - x[..., 2, 1]) / math.sqrt(2.0)
        rz = x[..., 1, 0]
        return torch.stack([rx, ry, rz], dim=-1)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def default_cgnet_params(nb_class: int = 13, num_tracks: int = 3) -> dict:
    """The published default configuration (their sizes untouched), plus the
    multi-ACCDOA head fields."""
    half_time = 16
    return dict(
        nb_class=nb_class,
        num_tracks=num_tracks,
        half_dense_feature=128,
        half_fc_feature=128,
        nb_skip_bin=4,
        nb_layer=5,
        nb_skip_l0_features=[-1, 8, 8, 8, 0],
        Lmax=3,
        taus_cgins=[[2, 2], [16, 4, 2], [48, 8, 4, 2], [64, 12, 8, 4],
                    [64, 16, 8, 8]],
        sphharm_activation_enabled_ls=[[], [], [], [1, 2, 3], [1]],
        pooling_ns=[8, 8, 4, 1, 1],
        cnn_kernel_sizes=[(3, 3), (3, 3), (3, 3),
                          (half_time * 2 + 1, 4), (half_time * 2 + 1, 1)],
        cnn_skip_l0s=[False, False, False, False, True],
        cnn_paddings=[(1, 1), (1, 1), (1, 1),
                      (half_time, 0), (half_time, 0)],
        dropout=0.5,
        use_GRU=True,
        nb_gru_layer=2,
        dropout_gru=0.2,
        sphstdbatchnorm_momentum=0.01,
        sphstdbatchnorm_eps=1e-10,
        sphharmactivation_eps=1e-10,
        scale_equivariance=True,
        cgbilinear_eps=1e-10,
        time_pool=5,          # 20 ms frames -> 100 ms STARSS23 label rate
    )


class CGSeldAccdoa(nn.Module):
    """Sato et al. trunk (reimplementation) with a Multi-ACCDOA head.
    """

    def __init__(self, p: dict):
        super().__init__()
        self.p = p
        self.nb_class = p['nb_class']
        self.K = p['num_tracks']
        self.time_pool = p.get('time_pool', 1)
        n_out_vec = self.K * self.nb_class

        ladd = [0] * (p['Lmax'] + 1)
        ladd[0] = (p['half_dense_feature']
                   - sum(p['nb_skip_l0_features'][1:]) * p['nb_skip_bin'])
        ladd[1] = n_out_vec
        self.taus_cgin = p['taus_cgins'] + [ladd]
        self.nb_layer = p['nb_layer']
        self.nb_skip_bin = p['nb_skip_bin']
        self.nb_skip_l0_features = p['nb_skip_l0_features']

        self.cgs = nn.ModuleList([
            CGElementProduct(self.taus_cgin[i], p['Lmax'],
                             p['cgbilinear_eps'], p['scale_equivariance'])
            for i in range(self.nb_layer)])
        self.bn_preconv = nn.ModuleList([
            SphericalSigmaBN(self.cgs[i].tau_out,
                             p['sphstdbatchnorm_momentum'],
                             p['sphstdbatchnorm_eps'])
            for i in range(self.nb_layer)])
        self.acts = nn.ModuleList([
            ScaleInvariantSHActivation(self.cgs[i].tau_out,
                                       p['sphharm_activation_enabled_ls'][i],
                                       p['sphharmactivation_eps'])
            for i in range(self.nb_layer)])
        self.convs = nn.ModuleList([
            SphHarmDistributedComplexConv2d(
                self.cgs[i].tau_out, self.taus_cgin[i + 1],
                p['cnn_kernel_sizes'][i], p['cnn_paddings'][i],
                skip_l0=p['cnn_skip_l0s'][i])
            for i in range(self.nb_layer)])
        self.pools = nn.ModuleList([
            GroupAvgPoolFreq(p['pooling_ns'][i])
            for i in range(self.nb_layer)])

        dense = 2 * p['half_dense_feature']
        self.bn_sed = nn.BatchNorm1d(dense)
        self.dropout = nn.Dropout(p['dropout'])
        # constructed for reference construction parity; unused when the
        # GRU path is enabled (dead in the reference too)
        self.fc_sed1 = nn.Linear(dense, dense)
        self.rnn = nn.GRU(dense, p['half_fc_feature'],
                          num_layers=p['nb_gru_layer'], batch_first=True,
                          dropout=p['dropout_gru'], bidirectional=True)
        self.fc_sed2 = nn.Linear(2 * p['half_fc_feature'], n_out_vec)
        nn.init.constant_(self.fc_sed2.bias, -2.0)   # low-activity prior
        self.to_vec = FirstOrderSphericalToVector()
        self.last_sed_logits = None

    def forward(self, x, update: bool):
        B = x.shape[0]
        assert x.shape[1] == 8 and x.shape[4] == 2
        skips = []
        for i in range(self.nb_layer):
            x = self.cgs[i](x)
            x = self.bn_preconv[i](x, update)
            x = self.acts[i](x)
            x = self.convs[i](x)
            t0 = self.taus_cgin[i + 1][0]
            x = torch.cat([F.relu(x[:, :t0]), x[:, t0:]], dim=1)
            x = self.pools[i](x)
            if i < self.nb_layer - 1:
                Fq = x.shape[3]
                fs = [j * Fq // self.nb_skip_bin
                      for j in range(self.nb_skip_bin)]
                skips.append(x[:, :self.nb_skip_l0_features[i + 1], :, fs])

        T_out = x.shape[2]
        t0 = self.taus_cgin[-1][0]
        sed = [x[:, :t0].permute(0, 2, 1, 3, 4).reshape(B, T_out, -1)]
        for s in skips:
            s = s.permute(0, 2, 1, 3, 4)
            tb = (s.shape[1] - T_out) // 2
            sed.append(s[:, tb:tb + T_out].reshape(B, T_out, -1))
        sed = torch.cat(sed, dim=2)
        sed = self.bn_sed(sed.permute(0, 2, 1)).permute(0, 2, 1)
        sed, _ = self.rnn(sed)
        logits = self.fc_sed2(self.dropout(sed))       # [B, T, K*C]

        vec = x[:, t0:]                                # [B, K*C*3, T, 1, 2]
        vec = vec.reshape(B, self.K * self.nb_class, 3, T_out, 2) \
            .permute(0, 3, 1, 2, 4)                    # [B, T, K*C, 3(m), 2]
        vec = self.to_vec(vec)                         # [B, T, K*C, 3(xyz)]

        act = torch.sigmoid(logits)                    # [B, T, K*C]
        unit = vec / (vec.pow(2).sum(-1, keepdim=True) + 1e-12).sqrt()
        accdoa = act.unsqueeze(-1) * unit              # [B, T, K*C, 3]

        tp = self.time_pool
        if tp > 1:
            T2 = (T_out // tp) * tp
            accdoa = accdoa[:, :T2].reshape(B, T2 // tp, tp,
                                            self.K * self.nb_class, 3) \
                .mean(dim=2)
            logits = logits[:, :T2].reshape(B, T2 // tp, tp, -1).mean(dim=2)

        self.last_sed_logits = logits.reshape(
            B, -1, self.K, self.nb_class)
        B2, T2 = accdoa.shape[0], accdoa.shape[1]
        accdoa = accdoa.reshape(B2, T2, self.K, self.nb_class, 3) \
            .permute(0, 1, 2, 4, 3) \
            .reshape(B2, T2, self.K * 3 * self.nb_class)
        return accdoa


# ---------------------------------------------------------------------------
# FOA STFT -> SH input features (WXYZ2Sph + conjugate copies)
# ---------------------------------------------------------------------------

def wxyz_stft_to_sh_input(stft: torch.Tensor) -> torch.Tensor:
    """stft: [B, 4, T, F, 2] complex STFT of ACN channels [W, Y, Z, X]
    -> [B, 8, T, F, 2]: rows {W; l=1 SH of (X,Y,Z)} + conjugate copies.

    l=1 SH (m=-1,0,+1) from cartesian STFTs (X, Y, Z complex):
        a_{-1} = (X - iY)/sqrt(2),  a_0 = Z,  a_{+1} = -(X + iY)/sqrt(2)
    chosen to invert FirstOrderSphericalToVector exactly."""
    W = stft[:, 0]
    Y_, Z_, X_ = stft[:, 1], stft[:, 2], stft[:, 3]
    xr, xi = X_[..., 0], X_[..., 1]
    yr, yi = Y_[..., 0], Y_[..., 1]
    s2 = math.sqrt(2.0)
    a_m1 = torch.stack([(xr + yi) / s2, (xi - yr) / s2], dim=-1)
    a_0 = Z_
    a_p1 = torch.stack([(-xr + yi) / s2, (-xi - yr) / s2], dim=-1)

    def conj(t):
        return torch.stack([t[..., 0], -t[..., 1]], dim=-1)

    # tau=[2,2]: l=0 fragments {W, conj W}; l=1 fragments {a, conj-SH of a}.
    # The conjugate of an l=1 SH function has coefficients
    # b_m = (-1)^m conj(a_{-m}), which is again a valid l=1 triplet.
    b_m1 = torch.stack([-conj(a_p1)[..., 0], -conj(a_p1)[..., 1]], dim=-1)
    b_0 = conj(a_0)
    b_p1 = torch.stack([-conj(a_m1)[..., 0], -conj(a_m1)[..., 1]], dim=-1)

    return torch.stack([W, conj(W),
                        a_m1, a_0, a_p1,
                        b_m1, b_0, b_p1], dim=1)
