from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as Fnn

EPS = 1e-8

# (X, Y, Z) channel indices inside ACN-ordered FOA [W, Y, Z, X]
ACN_TO_XYZ = (3, 1, 2)


# ---------------------------------------------------------------------------
# Rotation utilities (also used by the equivariance unit tests)
# ---------------------------------------------------------------------------

def random_rotation(dtype: torch.dtype = torch.float64, device=None) -> torch.Tensor:
    """Uniform-ish random R in SO(3) via QR, det forced to +1."""
    A = torch.randn(3, 3, dtype=torch.float64, device=device)
    Q, R = torch.linalg.qr(A)
    Q = Q @ torch.diag(torch.sign(torch.diagonal(R)))
    if torch.linalg.det(Q) < 0:
        Q = Q.clone()
        Q[:, 0] = -Q[:, 0]
    return Q.to(dtype)


def rotate_foa(wave: torch.Tensor, R: torch.Tensor) -> torch.Tensor:
    """Rotate the acoustic scene of an ACN FOA signal.

    wave: (B, 4, N) in [W, Y, Z, X];  R: (3, 3) rotation acting as x' = R x.
    """
    out = wave.clone()
    v = torch.stack([wave[:, 3], wave[:, 1], wave[:, 2]], dim=1)  # (B, 3, N)
    v = torch.einsum("ij,bjn->bin", R.to(wave), v)
    out[:, 3] = v[:, 0]
    out[:, 1] = v[:, 1]
    out[:, 2] = v[:, 2]
    return out


def rotate_vec(vec: torch.Tensor, R: torch.Tensor) -> torch.Tensor:
    """Apply R to the last dim of (..., 3)."""
    return torch.einsum("ij,...j->...i", R.to(vec), vec)


# ---------------------------------------------------------------------------
# Front end: FOA -> per-frame mel-band scalar + vector features
# ---------------------------------------------------------------------------

def mel_filterbank(sr: int, n_fft: int, n_mels: int,
                   fmin: float = 20.0, fmax: float | None = None) -> torch.Tensor:
    """Triangular mel filterbank, (n_mels, n_fft//2+1)."""
    fmax = fmax if fmax is not None else sr / 2

    def h2m(f):
        return 2595.0 * math.log10(1.0 + f / 700.0)

    def m2h(m):
        return 700.0 * (10.0 ** (m / 2595.0) - 1.0)

    m_pts = torch.linspace(h2m(fmin), h2m(fmax), n_mels + 2, dtype=torch.float64)
    f_pts = torch.tensor([m2h(m.item()) for m in m_pts], dtype=torch.float64)
    freqs = torch.linspace(0, sr / 2, n_fft // 2 + 1, dtype=torch.float64)
    lo, ce, hi = f_pts[:-2, None], f_pts[1:-1, None], f_pts[2:, None]
    up = (freqs[None] - lo) / (ce - lo).clamp_min(1e-9)
    dn = (hi - freqs[None]) / (hi - ce).clamp_min(1e-9)
    fb = torch.clamp(torch.minimum(up, dn), min=0.0)
    return fb.to(torch.float32)


class FoaFeatures(nn.Module):
    """FOA (ACN) waveform -> mel-band invariant scalars + equivariant vectors.

    scalars (B, T, M, 5): [log omni energy, log l=1 energy,
                           ||v_a||, ||v_r||, cos(v_a, v_r)]
    vectors (B, T, M, 2, 3): energy-normalized active / reactive intensity
                             (zeroed above f_vec_max, where FOA encoding
                             deviates from the ideal transform law).
    """

    N_SCALARS = 5
    N_VECTORS = 2

    def __init__(self, sr: int, n_fft: int, hop: int, n_mels: int,
                 f_vec_max: float, fmin: float = 20.0):
        super().__init__()
        self.n_fft, self.hop = n_fft, hop
        fb = mel_filterbank(sr, n_fft, n_mels, fmin=fmin)
        freqs = torch.linspace(0, sr / 2, n_fft // 2 + 1)
        vec_mask = (freqs <= f_vec_max).float()
        self.register_buffer("mel", fb)                               # (M, Fb)
        self.register_buffer("mel_vec", fb * vec_mask[None, :])       # (M, Fb)
        self.register_buffer("window", torch.hann_window(n_fft))

    def forward(self, wave: torch.Tensor):
        B, C, N = wave.shape
        assert C == 4, "expected ACN FOA [W, Y, Z, X]"
        X = torch.stft(wave.reshape(B * C, N), self.n_fft, self.hop,
                       window=self.window, return_complex=True, center=True)
        Fb, T = X.shape[-2], X.shape[-1]
        X = X.reshape(B, C, Fb, T)
        W = X[:, 0]                                                    # (B,Fb,T)
        V = torch.stack([X[:, i] for i in ACN_TO_XYZ], dim=1)          # (B,3,Fb,T)

        Ew = W.real ** 2 + W.imag ** 2                                 # invariant
        Ev = (V.real ** 2 + V.imag ** 2).sum(dim=1)                    # invariant
        P = W.conj().unsqueeze(1) * V                                  # (B,3,Fb,T)
        va, vr = P.real, P.imag                                        # equivariant
        E = 0.5 * (Ew + Ev / 3.0)                                      # invariant

        def agg(x, fb):
            return torch.einsum("mf,...ft->...mt", fb, x)

        Ew_b = agg(Ew, self.mel)                                       # (B,M,T)
        Ev_b = agg(Ev, self.mel)
        E_bv = agg(E, self.mel_vec)
        va_b = agg(va, self.mel_vec) / (E_bv.unsqueeze(1) + EPS)       # (B,3,M,T)
        vr_b = agg(vr, self.mel_vec) / (E_bv.unsqueeze(1) + EPS)

        va_b = va_b.permute(0, 3, 2, 1)                                # (B,T,M,3)
        vr_b = vr_b.permute(0, 3, 2, 1)
        na = (va_b.pow(2).sum(-1) + EPS).sqrt()                        # (B,T,M)
        nr = (vr_b.pow(2).sum(-1) + EPS).sqrt()
        cos = (va_b * vr_b).sum(-1) / (na * nr + EPS)

        s = torch.stack([
            torch.log(Ew_b + EPS).permute(0, 2, 1),
            torch.log(Ev_b + EPS).permute(0, 2, 1),
            na, nr, cos,
        ], dim=-1)                                                     # (B,T,M,5)
        v = torch.stack([va_b, vr_b], dim=-2)                          # (B,T,M,2,3)
        return s, v


# ---------------------------------------------------------------------------
# Equivariant primitives
# ---------------------------------------------------------------------------

class VecLinear(nn.Module):
    def __init__(self, cin: int, cout: int, init_scale: float = 1.0,
                 equivariant: bool = True):
        super().__init__()
        w = torch.randn(cout, cin) / math.sqrt(cin) * init_scale
        self.weight = nn.Parameter(w)
        self.equivariant = equivariant
        if not equivariant:
            self.mix3 = nn.Parameter(
                torch.eye(3) + 0.05 * torch.randn(3, 3))
            self.vbias = nn.Parameter(torch.zeros(cout, 3))

    def forward(self, v: torch.Tensor) -> torch.Tensor:
        out = torch.einsum("oi,...id->...od", self.weight, v)
        if not self.equivariant:
            out = torch.einsum("...od,de->...oe", out, self.mix3) + self.vbias
        return out


class SVLayerNorm(nn.Module):
    """LayerNorm on scalars; RMS-of-norms normalization on vector channels
    (shared scale across the 3 components => equivariant)."""

    def __init__(self, d_s: int, d_v: int, equivariant: bool = True):
        super().__init__()
        self.ls = nn.LayerNorm(d_s)
        self.equivariant = equivariant
        if equivariant:
            self.gv = nn.Parameter(torch.ones(d_v))
        else:
            # per-component gain and offset: a direction-dependent affine
            self.gv = nn.Parameter(torch.ones(d_v, 3))
            self.bv = nn.Parameter(torch.zeros(d_v, 3))

    def forward(self, s: torch.Tensor, v: torch.Tensor):
        n2 = v.pow(2).sum(-1)                                  # (..., d_v)
        rms = (n2.mean(-1, keepdim=True) + EPS).sqrt()         # (..., 1)
        if self.equivariant:
            v = v / rms.unsqueeze(-1) * self.gv.unsqueeze(-1)
        else:
            v = v / rms.unsqueeze(-1) * self.gv + self.bv
        return self.ls(s), v


class SVFeedForward(nn.Module):
    def __init__(self, d_s: int, d_v: int, expand: int = 2, dropout: float = 0.0,
                 equivariant: bool = True):
        super().__init__()
        self.d_s, self.d_v = d_s, d_v
        self.equivariant = equivariant
        # non-equivariant twin: the scalar MLP reads the raw vector
        self.dvf = d_v if equivariant else 3 * d_v
        self.norm = SVLayerNorm(d_s, d_v, equivariant=equivariant)
        self.vin = VecLinear(d_v, d_v, equivariant=equivariant)
        self.fc1 = nn.Linear(d_s + self.dvf, expand * d_s)
        self.fc2 = nn.Linear(expand * d_s, d_s + self.dvf)
        nn.init.normal_(self.fc2.weight, std=0.02)   # small-init residual branch
        nn.init.zeros_(self.fc2.bias)
        self.vout = VecLinear(d_v, d_v, init_scale=0.1, equivariant=equivariant)
        self.drop = nn.Dropout(dropout)

    def forward(self, s: torch.Tensor, v: torch.Tensor):
        sn, vn = self.norm(s, v)
        vh = self.vin(vn)
        if self.equivariant:
            n = (vh.pow(2).sum(-1) + EPS).sqrt()
        else:
            n = vh.reshape(*vh.shape[:-2], self.dvf)
        h = Fnn.gelu(self.fc1(torch.cat([sn, n], dim=-1)))
        h = self.drop(h)
        ds, gate = self.fc2(h).split([self.d_s, self.dvf], dim=-1)
        if self.equivariant:
            dv = vh * torch.sigmoid(gate).unsqueeze(-1)
        else:
            dv = vh * torch.sigmoid(gate).reshape(*vh.shape[:-2], self.d_v, 3)
        return s + ds, v + self.vout(dv)


def _apply_rope(x: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
    """RoPE on invariant Q/K features. x: (B, h, N, dh), dh even; pos: (N,)."""
    dh = x.shape[-1]
    half = dh // 2
    inv = 1.0 / (10000.0 ** (torch.arange(half, device=x.device, dtype=x.dtype) / half))
    ang = pos.to(x.dtype)[:, None] * inv[None]                 # (N, half)
    cos, sin = ang.cos()[None, None], ang.sin()[None, None]    # (1,1,N,half)
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)


class EquivAttention(nn.Module):
    """Multi-head (cross-)attention with invariant logits and residuals."""


    def __init__(self, d_s: int, d_v: int, heads: int,
                 dropout: float = 0.0, rope: bool = False,
                 equivariant: bool = True):
        super().__init__()
        assert d_s % heads == 0 and d_v % heads == 0
        self.h, self.dh, self.dvh = heads, d_s // heads, d_v // heads
        self.rope = rope
        self.equivariant = equivariant
        dvf = d_v if equivariant else 3 * d_v
        self.normq = SVLayerNorm(d_s, d_v, equivariant=equivariant)
        self.normk = SVLayerNorm(d_s, d_v, equivariant=equivariant)
        self.injq = nn.Linear(dvf, d_s, bias=False)
        self.injk = nn.Linear(dvf, d_s, bias=False)
        self.q = nn.Linear(d_s, d_s)
        self.k = nn.Linear(d_s, d_s)
        self.vs = nn.Linear(d_s, d_s)
        self.vv = VecLinear(d_v, d_v, equivariant=equivariant)
        self.os = nn.Linear(d_s, d_s)
        nn.init.normal_(self.os.weight, std=0.02)    # small-init residual branch
        nn.init.zeros_(self.os.bias)
        # NOTE: must not be zero-init — for PMA/ISAB seed queries (zero vector
        # part) this projection is the only path vectors take through the block
        self.ov = VecLinear(d_v, d_v, init_scale=0.1, equivariant=equivariant)
        self.drop = nn.Dropout(dropout)

    def forward(self, sq, vq, sk=None, vk=None, pos_q=None, pos_k=None):
        if sk is None:
            sk, vk = sq, vq
            pos_k = pos_q
        B, Nq, _ = sq.shape
        Nk = sk.shape[1]

        snq, vnq = self.normq(sq, vq)
        snk, vnk = self.normk(sk, vk)
        if self.equivariant:
            fq = (vnq.pow(2).sum(-1) + EPS).sqrt()
            fk = (vnk.pow(2).sum(-1) + EPS).sqrt()
        else:
            fq = vnq.reshape(B, Nq, -1)
            fk = vnk.reshape(B, Nk, -1)
        snq = snq + self.injq(fq)
        snk = snk + self.injk(fk)

        q = self.q(snq).view(B, Nq, self.h, self.dh).transpose(1, 2)   # (B,h,Nq,dh)
        k = self.k(snk).view(B, Nk, self.h, self.dh).transpose(1, 2)
        if self.rope and pos_q is not None:
            q = _apply_rope(q, pos_q)
            k = _apply_rope(k, pos_k)
        att = torch.softmax(q @ k.transpose(-2, -1) * self.dh ** -0.5, dim=-1)
        att = self.drop(att)

        vs = self.vs(snk).view(B, Nk, self.h, self.dh).transpose(1, 2)
        out_s = (att @ vs).transpose(1, 2).reshape(B, Nq, -1)

        vv = self.vv(vnk).view(B, Nk, self.h, self.dvh, 3).permute(0, 2, 1, 3, 4)
        out_v = torch.einsum("bhqk,bhkcd->bhqcd", att, vv)
        out_v = out_v.permute(0, 2, 1, 3, 4).reshape(B, Nq, -1, 3)

        return sq + self.os(out_s), vq + self.ov(out_v)


class EquivMAB(nn.Module):
    """Set-Transformer MAB: X attends to Y, then feed-forward (pre-norm)."""

    def __init__(self, d_s, d_v, heads, dropout=0.0, rope=False, expand=2,
                 equivariant=True):
        super().__init__()
        self.att = EquivAttention(d_s, d_v, heads, dropout=dropout, rope=rope,
                                  equivariant=equivariant)
        self.ff = SVFeedForward(d_s, d_v, expand=expand, dropout=dropout,
                                equivariant=equivariant)

    def forward(self, sx, vx, sy=None, vy=None, pos_q=None, pos_k=None):
        s, v = self.att(sx, vx, sy, vy, pos_q=pos_q, pos_k=pos_k)
        return self.ff(s, v)


class EquivISAB(nn.Module):
    """Induced set attention block. Inducing points are learnable scalars with
    a ZERO vector part (the only rotation-invariant constant vector)."""

    def __init__(self, d_s, d_v, heads, m, dropout=0.0, equivariant=True):
        super().__init__()
        self.d_v = d_v
        self.equivariant = equivariant
        self.I = nn.Parameter(torch.empty(m, d_s))
        nn.init.xavier_uniform_(self.I)
        if not equivariant:
            # a non-zero constant vector picks a preferred direction
            self.Iv = nn.Parameter(0.02 * torch.randn(m, d_v, 3))
        self.mab0 = EquivMAB(d_s, d_v, heads, dropout=dropout,
                             equivariant=equivariant)
        self.mab1 = EquivMAB(d_s, d_v, heads, dropout=dropout,
                             equivariant=equivariant)

    def forward(self, s, v):
        B = s.shape[0]
        sI = self.I.unsqueeze(0).expand(B, -1, -1)
        if self.equivariant:
            vI = torch.zeros(B, self.I.shape[0], self.d_v, 3, dtype=v.dtype, device=v.device)
        else:
            vI = self.Iv.unsqueeze(0).expand(B, -1, -1, -1).to(v.dtype)
        sh, vh = self.mab0(sI, vI, s, v)
        return self.mab1(s, v, sh, vh)


class EquivPMA(nn.Module):
    """Pooling by multi-head attention with k scalar-only learnable seeds."""

    def __init__(self, d_s, d_v, heads, k, dropout=0.0, equivariant=True):
        super().__init__()
        self.d_v = d_v
        self.equivariant = equivariant
        self.S = nn.Parameter(torch.empty(k, d_s))
        nn.init.xavier_uniform_(self.S)
        if not equivariant:
            self.Sv = nn.Parameter(0.02 * torch.randn(k, d_v, 3))
        self.mab = EquivMAB(d_s, d_v, heads, dropout=dropout,
                            equivariant=equivariant)

    def forward(self, s, v):
        B = s.shape[0]
        sS = self.S.unsqueeze(0).expand(B, -1, -1)
        if self.equivariant:
            vS = torch.zeros(B, self.S.shape[0], self.d_v, 3, dtype=v.dtype, device=v.device)
        else:
            vS = self.Sv.unsqueeze(0).expand(B, -1, -1, -1).to(v.dtype)
        return self.mab(sS, vS, s, v)


class EquivConvStem(nn.Module):
    """Per-band temporal context + pooling to label rate."""

    def __init__(self, d_s, d_v, pool: int, dropout: float = 0.0,
                 equivariant: bool = True):
        super().__init__()
        self.pool = pool
        self.equivariant = equivariant
        self.cs = nn.Conv1d(d_s, d_s, kernel_size=3, padding=1)
        self.cv = nn.Conv1d(d_v, d_v, kernel_size=3, padding=1,
                            bias=not equivariant)
        if not equivariant:
            self.cmix3 = nn.Parameter(torch.eye(3) + 0.05 * torch.randn(3, 3))
        self.ff = SVFeedForward(d_s, d_v, dropout=dropout,
                                equivariant=equivariant)

    def forward(self, s, v):
        B, T, M, ds = s.shape
        dv = v.shape[-2]
        s2 = self.cs(s.permute(0, 2, 3, 1).reshape(B * M, ds, T))
        s = s + s2.reshape(B, M, ds, T).permute(0, 3, 1, 2)
        v2 = self.cv(v.permute(0, 2, 4, 3, 1).reshape(B * M * 3, dv, T))
        v2 = v2.reshape(B, M, 3, dv, T).permute(0, 4, 1, 3, 2)
        if not self.equivariant:
            v2 = torch.einsum("...od,de->...oe", v2, self.cmix3)
        v = v + v2

        T2 = (T // self.pool) * self.pool
        s = s[:, :T2].reshape(B, T2 // self.pool, self.pool, M, ds).mean(dim=2)
        v = v[:, :T2].reshape(B, T2 // self.pool, self.pool, M, dv, 3).mean(dim=2)
        return self.ff(s, v)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

@dataclass
class SeldConfig:
    # front end
    sample_rate: int = 24000
    n_fft: int = 1024
    hop: int = 480            # 20 ms
    n_mels: int = 64
    fmin: float = 20.0
    f_vec_max: float = 9000.0  # FOA encoding trustworthy up to ~9 kHz
    time_pool: int = 5         # 20 ms features -> 100 ms label frames
    # task
    num_classes: int = 13
    num_tracks: int = 3        # 3 => Multi-ACCDOA, 1 => single ACCDOA
    # network
    d_s: int = 128             # invariant (scalar) width
    d_v: int = 32              # equivariant (vector) channels
    heads: int = 4
    freq_blocks: int = 2
    freq_block_type: str = "isab"   # "isab" | "sab"
    isab_inducing: int = 16
    temporal_blocks: int = 4
    track_mix: bool = True
    dropout: float = 0.05
    so3_only: bool = False
    equivariant: bool = True


class EquivSeldSetTransformer(nn.Module):
    """FOA waveform -> per-frame invariant SED logits + equivariant DOA vectors.

    Pipeline: intensity-vector front end -> equivariant conv stem (pool to
    label rate) -> Set Transformer over mel bands per frame (ISAB stack + PMA
    with num_tracks seeds) -> temporal SAB stack with RoPE per track token
    (interleaved with cheap cross-track SABs) -> decoupled heads.
    """

    def __init__(self, cfg: SeldConfig):
        super().__init__()
        self.cfg = cfg
        d_s, d_v, h, p = cfg.d_s, cfg.d_v, cfg.heads, cfg.dropout

        self.feat = FoaFeatures(cfg.sample_rate, cfg.n_fft, cfg.hop,
                                cfg.n_mels, cfg.f_vec_max, fmin=cfg.fmin)
        self.lift_s = nn.Linear(FoaFeatures.N_SCALARS, d_s)
        self.band_emb = nn.Parameter(torch.empty(cfg.n_mels, d_s))
        nn.init.normal_(self.band_emb, std=0.02)
        eq = cfg.equivariant
        n_vec_in = FoaFeatures.N_VECTORS + (1 if cfg.so3_only else 0)
        self.lift_v = VecLinear(n_vec_in, d_v, equivariant=eq)
        self.stem = EquivConvStem(d_s, d_v, pool=cfg.time_pool, dropout=p,
                                  equivariant=eq)

        if cfg.freq_block_type == "isab":
            self.freq = nn.ModuleList(
                [EquivISAB(d_s, d_v, h, cfg.isab_inducing, dropout=p,
                           equivariant=eq)
                 for _ in range(cfg.freq_blocks)])
        else:
            self.freq = nn.ModuleList(
                [EquivMAB(d_s, d_v, h, dropout=p, equivariant=eq)
                 for _ in range(cfg.freq_blocks)])
        self.pma = EquivPMA(d_s, d_v, h, k=cfg.num_tracks, dropout=p,
                            equivariant=eq)

        self.temporal = nn.ModuleList(
            [EquivMAB(d_s, d_v, h, dropout=p, rope=True, equivariant=eq)
             for _ in range(cfg.temporal_blocks)])
        self.mix = nn.ModuleList(
            [EquivMAB(d_s, d_v, h, dropout=p, equivariant=eq)
             for _ in range(cfg.temporal_blocks)]) \
            if (cfg.track_mix and cfg.num_tracks > 1) else None

        self.head_norm = SVLayerNorm(d_s, d_v, equivariant=eq)
        self.head_s = nn.Linear(d_s, cfg.num_classes)
        nn.init.constant_(self.head_s.bias, -2.0)   # low-activity prior
        self.head_v = VecLinear(d_v, cfg.num_classes, equivariant=eq)

    def forward(self, wave: torch.Tensor) -> dict:
        s, v = self.feat(wave)                       # (B,T,M,5), (B,T,M,2,3)
        return self.forward_features(s, v)

    def forward_features(self, s: torch.Tensor, v: torch.Tensor) -> dict:
        cfg = self.cfg
        s = self.lift_s(s) + self.band_emb           # (B,T,M,d_s)
        if cfg.so3_only:
            # axial channel: breaks reflection equivariance by design
            vx = torch.cross(v[..., 0, :], v[..., 1, :], dim=-1)
            v = torch.cat([v, vx.unsqueeze(-2)], dim=-2)
        v = self.lift_v(v)                           # (B,T,M,d_v,3)
        s, v = self.stem(s, v)                       # (B,T',M,...)

        B, T, M, d_s = s.shape
        d_v = v.shape[-2]
        K = cfg.num_tracks

        # --- Set Transformer over mel bands, per frame -----------------
        s = s.reshape(B * T, M, d_s)
        v = v.reshape(B * T, M, d_v, 3)
        for blk in self.freq:
            s, v = blk(s, v)
        s, v = self.pma(s, v)                        # (B*T, K, ...)

        # --- temporal attention per track token -------------------------
        s = s.reshape(B, T, K, d_s).transpose(1, 2).reshape(B * K, T, d_s)
        v = v.reshape(B, T, K, d_v, 3).transpose(1, 2).reshape(B * K, T, d_v, 3)
        pos = torch.arange(T, device=s.device, dtype=s.dtype)
        for i, blk in enumerate(self.temporal):
            s, v = blk(s, v, pos_q=pos)
            if self.mix is not None:
                s = s.reshape(B, K, T, d_s).transpose(1, 2).reshape(B * T, K, d_s)
                v = v.reshape(B, K, T, d_v, 3).transpose(1, 2).reshape(B * T, K, d_v, 3)
                s, v = self.mix[i](s, v)
                s = s.reshape(B, T, K, d_s).transpose(1, 2).reshape(B * K, T, d_s)
                v = v.reshape(B, T, K, d_v, 3).transpose(1, 2).reshape(B * K, T, d_v, 3)

        s = s.reshape(B, K, T, d_s).transpose(1, 2)  # (B,T,K,d_s)
        v = v.reshape(B, K, T, d_v, 3).transpose(1, 2)

        # --- decoupled heads --------------------------------------------
        s, v = self.head_norm(s, v)
        logits = self.head_s(s)                      # (B,T,K,C)   invariant
        vec = self.head_v(v)                         # (B,T,K,C,3) equivariant
        act = torch.sigmoid(logits)
        unit = vec / (vec.pow(2).sum(-1, keepdim=True) + EPS).sqrt()
        accdoa = act.unsqueeze(-1) * unit            # magnitude=activity
        return {"sed_logits": logits, "doa": vec, "accdoa": accdoa}


# ---------------------------------------------------------------------------
# Losses
# ---------------------------------------------------------------------------

class MultiAccdoaAdpitLoss(nn.Module):
    """ADPIT loss for Multi-ACCDOA, DCASE-baseline compatible.

    pred:   (B, T, 3, C, 3)  model 'accdoa' output (tracks, classes, xyz)
    target: (B, T, 6, 4, C)  baseline loader format:
            dim2 = [A0, B0, B1, C0, C1, C2] (1/2/3 same-class sources)
            dim3 = [activity, x, y, z]
    13 permutation hypotheses = 1 (A) + 6 (B) + 6 (C); inactive hypotheses
    are padded with the active group's first ordering (auxiliary duplication)
    so the min never selects a degenerate all-zero target.
    """

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        B, T, K, C, _ = pred.shape
        assert K == 3, "Multi-ACCDOA ADPIT expects 3 tracks"
        out = pred.permute(0, 1, 2, 4, 3).reshape(B, T, 9, C)  # trk-major, xyz inner

        def tgt(i):  # activity-scaled Cartesian target of dummy track i
            return target[:, :, i, 0:1, :] * target[:, :, i, 1:, :]   # (B,T,3,C)

        A0, B0, B1, C0, C1, C2 = (tgt(i) for i in range(6))

        def cat3(a, b, c):
            return torch.cat([a, b, c], dim=2)                        # (B,T,9,C)

        tA = cat3(A0, A0, A0)
        perms_B = [cat3(B0, B0, B1), cat3(B0, B1, B0), cat3(B0, B1, B1),
                   cat3(B1, B0, B0), cat3(B1, B0, B1), cat3(B1, B1, B0)]
        perms_C = [cat3(C0, C1, C2), cat3(C0, C2, C1), cat3(C1, C0, C2),
                   cat3(C1, C2, C0), cat3(C2, C0, C1), cat3(C2, C1, C0)]

        padA = perms_B[0] + perms_C[0]
        padB = tA + perms_C[0]
        padC = tA + perms_B[0]

        def mse(t):  # class-wise, frame-wise mean over the 9 track*xyz slots
            return (out - t).pow(2).mean(dim=2)                       # (B,T,C)

        losses = [mse(tA + padA)]
        losses += [mse(t + padB) for t in perms_B]
        losses += [mse(t + padC) for t in perms_C]
        return torch.stack(losses, dim=0).min(dim=0).values.mean()


class SingleAccdoaLoss(nn.Module):
    """Plain ACCDOA MSE. pred: (B,T,1,C,3) or (B,T,C,3); target: (B,T,C,3)
    with target = activity * unit_direction."""

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if pred.dim() == 5:
            assert pred.shape[2] == 1, "single-ACCDOA expects num_tracks == 1"
            pred = pred[:, :, 0]
        return Fnn.mse_loss(pred, target)


def sed_bce_aux(sed_logits: torch.Tensor, adpit_target: torch.Tensor) -> torch.Tensor:
    """Assignment-free detection warm-up: BCE between max-over-tracks logits
    and any-source class activity. Use with a small weight (e.g. 0.2) for the
    first few epochs, then anneal to 0."""
    act_any = adpit_target[:, :, :, 0, :].amax(dim=2)          # (B,T,C) in {0,1}
    return Fnn.binary_cross_entropy_with_logits(sed_logits.amax(dim=2), act_any)


# ---------------------------------------------------------------------------
# Inference decoding
# ---------------------------------------------------------------------------

@torch.no_grad()
def decode_multi_accdoa(accdoa: torch.Tensor, thresh: float = 0.5,
                        merge_deg: float = 15.0):
    """accdoa: (T, K, C, 3) -> list over frames of (class_idx, unit_xyz tensor).

    Per class: tracks with |vec| > thresh are active; near-duplicate tracks
    (angular distance < merge_deg) are unified by averaging."""
    T, K, C, _ = accdoa.shape
    cos_thr = math.cos(math.radians(merge_deg))
    norms = accdoa.norm(dim=-1)                                # (T,K,C)
    events = [[] for _ in range(T)]
    for t in range(T):
        for c in range(C):
            dirs = [accdoa[t, k, c] / norms[t, k, c].clamp_min(EPS)
                    for k in range(K) if norms[t, k, c] > thresh]
            groups: list[list[torch.Tensor]] = []
            for d in dirs:
                for g in groups:
                    gm = torch.stack(g).mean(0)
                    gm = gm / gm.norm().clamp_min(EPS)
                    if float((d * gm).sum()) > cos_thr:
                        g.append(d)
                        break
                else:
                    groups.append([d])
            for g in groups:
                gm = torch.stack(g).mean(0)
                events[t].append((c, gm / gm.norm().clamp_min(EPS)))
    return events


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    cfg = SeldConfig()
    model = EquivSeldSetTransformer(cfg)
    n_params = sum(p.numel() for p in model.parameters())
    wave = torch.randn(2, 4, cfg.sample_rate * 5)              # 5 s batch
    out = model(wave)
    print(f"params: {n_params / 1e6:.2f} M")
    for k_, v_ in out.items():
        print(f"{k_}: {tuple(v_.shape)}")