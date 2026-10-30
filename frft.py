"""Discrete fractional Fourier transform (FrFT) in PyTorch, with two methods.

Convention: dimensionless units, N samples with spacing 1/sqrt(N), kernel
exp(-i 2 pi u x) at order a = 1 (same sign as torch.fft.fft, ortho norm).
Order a rotates by alpha = a*pi/2: a=0 identity, a=1 FFT, a=2 parity, a=-1 inverse.

method="eig"   Hermite-like DFT eigenvectors (Candan et al., 2000).
               Exactly unitary and index-additive; O(N^2) per row after an
               O(N^3) eigendecomposition cached per N. `a` may be a tensor
               (differentiable w.r.t. the order).
method="chirp" Sampled chirp multiplication -> chirp convolution (FFT) ->
               chirp multiplication, with 2x band-limited interpolation.
               O(N log N), approximates the continuous FrFT (only approximately
               unitary). `a` is treated as a plain float. Gradients w.r.t. the
               input work; gradients w.r.t. `a` do not.
"""
import cmath
import math
from functools import lru_cache

import torch

METHODS = ("eig", "chirp")


# ----------------------------------------------------------------------------
# Method "eig": Hermite-like eigenvectors of the DFT
# ----------------------------------------------------------------------------
@lru_cache(maxsize=8)
def _hermite_like_eigvecs(N: int, device: torch.device = torch.device("cpu")):
    """Orthonormal Hermite-like eigenvectors of the DFT and their orders k."""
    n = torch.arange(N, dtype=torch.float64)
    I = torch.eye(N, dtype=torch.float64)
    S = torch.diag(2 * torch.cos(2 * math.pi * n / N)) + torch.roll(I, 1, 0) + torch.roll(I, -1, 0)

    r = 1 / math.sqrt(2)
    n_even, n_odd = N // 2 + 1, (N - 1) // 2
    E = torch.zeros(N, n_even, dtype=torch.float64)
    O = torch.zeros(N, n_odd, dtype=torch.float64)
    E[0, 0] = 1.0
    j = torch.arange(1, (N - 1) // 2 + 1)
    E[j, j] = r
    E[N - j, j] = r
    O[j, j - 1] = r
    O[N - j, j - 1] = -r
    if N % 2 == 0:
        E[N // 2, N // 2] = 1.0

    _, Ue = torch.linalg.eigh(E.T @ S @ E)
    _, Uo = torch.linalg.eigh(O.T @ S @ O)
    Ve = E @ Ue.flip(1)  # descending eigenvalue -> increasing Hermite order
    Vo = O @ Uo.flip(1)

    k_even = torch.arange(0, 2 * Ve.shape[1], 2, dtype=torch.float64)
    k_odd = torch.arange(1, 2 * Vo.shape[1] + 1, 2, dtype=torch.float64)
    return torch.cat([Ve, Vo], dim=1).to(device), torch.cat([k_even, k_odd]).to(device)


def frft_matrix(N, a, device="cpu"):
    """N x N FrFT matrix of order `a` (method "eig"), origin at index 0."""
    V, k = _hermite_like_eigvecs(N, torch.device(device))
    a = torch.as_tensor(a, dtype=torch.float64, device=V.device)
    phase = torch.exp(-0.5j * math.pi * a * k)
    return (V * phase) @ V.T


def _frft_eig(x, a, dim, centered):
    N = x.shape[dim]
    if centered:
        x = torch.fft.ifftshift(x, dim=dim)
    F = frft_matrix(N, a, x.device).to(x.dtype)
    y = torch.movedim(x, dim, -1) @ F.T  # F is symmetric
    y = torch.movedim(y, -1, dim)
    if centered:
        y = torch.fft.fftshift(y, dim=dim)
    return y


# ----------------------------------------------------------------------------
# Method "chirp": chirp multiplication / convolution / multiplication
# ----------------------------------------------------------------------------
def _integer_ft(x, m, dim):
    """F^m for integer m on a centered array (origin at index N//2)."""
    m %= 4
    if m == 0:
        return x
    N = x.shape[dim]
    xs = torch.fft.ifftshift(x, dim=dim)
    if m == 1:
        xs = torch.fft.fft(xs, dim=dim, norm="ortho")
    elif m == 3:
        xs = torch.fft.ifft(xs, dim=dim, norm="ortho")
    else:  # m == 2: parity, y[n] = x[-n]
        idx = (-torch.arange(N, device=x.device)) % N
        xs = xs.index_select(dim, idx)
    return torch.fft.fftshift(xs, dim=dim)


def _upsample2(x):
    """Band-limited 2x interpolation along the last dim (y[2n] == x[n])."""
    N = x.shape[-1]
    X = torch.fft.fft(x, dim=-1)
    h = (N + 1) // 2
    if N % 2 == 0:
        nyq = 0.5 * X[..., h:h + 1]
        zeros = torch.zeros(*x.shape[:-1], N - 1, dtype=x.dtype, device=x.device)
        Y = torch.cat([X[..., :h], nyq, zeros, nyq, X[..., h + 1:]], dim=-1)
    else:
        zeros = torch.zeros(*x.shape[:-1], N, dtype=x.dtype, device=x.device)
        Y = torch.cat([X[..., :h], zeros, X[..., h:]], dim=-1)
    return 2 * torch.fft.ifft(Y, dim=-1)


@lru_cache(maxsize=16)
def _chirp_kernels(N, a, device, dtype):
    """Chirp factor and FFT of the (scaled) convolution chirp, for a in [0.5, 1.5)."""
    M = 2 * N
    alpha = a * math.pi / 2
    dx = 1 / (2 * math.sqrt(N))  # spacing after 2x interpolation
    c = N // 2
    f64 = torch.float64
    xm = (torch.arange(M, dtype=f64, device=device) - 2 * c) * dx
    pre = torch.exp(-1j * math.pi * math.tan(alpha / 2) * xm ** 2)
    t = torch.arange(-(M - 1), M, dtype=f64, device=device) * dx
    h = torch.exp(1j * math.pi / math.sin(alpha) * t ** 2)
    A = cmath.exp(-1j * (math.pi / 4 - alpha / 2)) / math.sqrt(math.sin(alpha))
    Hk = torch.fft.fft(h, n=3 * M) * (A * dx)
    return pre.to(dtype), Hk.to(dtype)


def _chirp_last(x, a):
    """FrFT of order a in [0.5, 1.5) along the last dim of a centered array."""
    N = x.shape[-1]
    M = 2 * N
    pre, Hk = _chirp_kernels(N, a, x.device, x.dtype)
    g = _upsample2(x) * pre
    conv = torch.fft.ifft(torch.fft.fft(g, n=3 * M, dim=-1) * Hk, dim=-1)[..., M - 1:2 * M - 1]
    return (conv * pre)[..., ::2].contiguous()


def _frft_chirp(x, a, dim, centered):
    a = float(a)
    if not centered:
        x = torch.fft.fftshift(x, dim=dim)  # move origin to N//2
    r = round(a)
    if abs(a - r) < 1e-9:
        y = _integer_ft(x, r, dim)
    else:
        m = round(a - 1)  # F^a = F^(a-m) F^m with a-m in [0.5, 1.5)
        x = torch.movedim(_integer_ft(x, m, dim), dim, -1)
        y = torch.movedim(_chirp_last(x, a - m), -1, dim)
    if not centered:
        y = torch.fft.ifftshift(y, dim=dim)
    return y


# ----------------------------------------------------------------------------
# Public API
# ----------------------------------------------------------------------------
def frft(x, a, dim=-1, centered=True, method="eig"):
    """FrFT of order `a` along `dim`.

    centered=True  -> origin at the array centre (index N//2), e.g. images/holograms.
    centered=False -> origin at index 0 (same layout as torch.fft.fft input).
    method         -> "eig" or "chirp" (see module docstring).
    """
    if method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}, got {method!r}")
    if not x.is_complex():
        x = torch.complex(x, torch.zeros_like(x))
    if method == "eig":
        return _frft_eig(x, a, dim, centered)
    return _frft_chirp(x, a, dim, centered)


def frft2(x, a_x, a_y=None, centered=True, method="eig"):
    """Separable 2D FrFT: order a_x along the last dim, a_y along the one before."""
    a_y = a_x if a_y is None else a_y
    y = frft(x, a_x, dim=-1, centered=centered, method=method)
    return frft(y, a_y, dim=-2, centered=centered, method=method)
