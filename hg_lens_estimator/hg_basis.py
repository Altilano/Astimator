"""
Fast Hermite-Gaussian basis on a pixel grid.

Why this is fast:
  * Separability: HG_nm(x, y) = psi_n(x) * psi_m(y). Only 2*(N+1) one-dimensional
    profiles are computed (stable three-term recurrence), never the 2D modes
    one by one.
  * Triangular truncation: only modes with n + m <= N are built, i.e.
    (N+1)(N+2)/2 columns instead of (N+1)^2.
  * All 2D modes are assembled in one batched gather + broadcast product,
    which runs in parallel on the chosen device (CPU or GPU).

Conventions (check once against your beam.py):
  * mode (n, m): order n along x (columns of the image), order m along y (rows).
  * pixel rows of `basis` follow C-order ravel of an (H, W) image: index = y*W + x.
  * field profile psi_n(x) ~ H_n(sqrt(2) x / waist) * exp(-x^2 / waist^2).
  * pixel grid is centred: x = (col - cx) * pixel_size, y = (row - cy) * pixel_size,
    with (cy, cx) = ((H-1)/2, (W-1)/2) unless you pass `center`.
"""
import math
from dataclasses import dataclass

import numpy as np
import torch

from .torch_engine import resolve_device


@dataclass
class HGBasis:
    basis: torch.Tensor   # (n_pix, n_modes) complex128, one column per mode
    ni: torch.Tensor      # (n_modes,) long, order n along x
    mi: torch.Tensor      # (n_modes,) long, order m along y
    N: int
    shape: tuple          # (H, W)


def _hermite_functions_1d(coord, waist, N):
    """(N+1, len(coord)) real tensor: psi_0 .. psi_N via the stable recurrence
    psi_{n+1} = sqrt(2/(n+1)) u psi_n - sqrt(n/(n+1)) psi_{n-1},  u = sqrt(2) x / waist.
    (Overall scale is fixed afterwards by the normalisation step.)"""
    u = math.sqrt(2.0) * coord / waist
    psi = torch.empty(N + 1, coord.numel(), dtype=coord.dtype, device=coord.device)
    psi[0] = torch.exp(-u ** 2 / 2)
    if N >= 1:
        psi[1] = math.sqrt(2.0) * u * psi[0]
    for n in range(1, N):
        psi[n + 1] = math.sqrt(2.0 / (n + 1)) * u * psi[n] - math.sqrt(n / (n + 1)) * psi[n - 1]
    return psi


def make_hg_basis(N, shape, pixel_size, waist, norm="max1", device="auto", center=None):
    """
    N:          maximum HG order (modes with n + m <= N are built)
    shape:      (H, W) of ONE image
    pixel_size: camera pixel pitch (same length unit as waist)
    waist:      beam waist
    norm:       "max1" -> every 2D mode has max |value| = 1 on the grid
                "l2"   -> every 2D mode has sum |value|^2 = 1 on the grid
    center:     (cy, cx) beam centre in pixels; default = image centre
    """
    if norm not in ("max1", "l2"):
        raise ValueError("norm must be 'max1' or 'l2'")
    dev = resolve_device(device)
    H, W = shape
    cy, cx = center if center is not None else ((H - 1) / 2, (W - 1) / 2)

    x = (torch.arange(W, dtype=torch.float64, device=dev) - cx) * pixel_size
    y = (torch.arange(H, dtype=torch.float64, device=dev) - cy) * pixel_size
    psi_x = _hermite_functions_1d(x, waist, N)    # (N+1, W)
    psi_y = _hermite_functions_1d(y, waist, N)    # (N+1, H)

    # Normalising the 1D profiles normalises the 2D modes too (separable product).
    if norm == "max1":
        psi_x = psi_x / psi_x.abs().amax(dim=1, keepdim=True)
        psi_y = psi_y / psi_y.abs().amax(dim=1, keepdim=True)
    else:
        psi_x = psi_x / psi_x.pow(2).sum(dim=1, keepdim=True).sqrt()
        psi_y = psi_y / psi_y.pow(2).sum(dim=1, keepdim=True).sqrt()

    pairs = [(n, m) for n in range(N + 1) for m in range(N + 1 - n)]   # n + m <= N
    ni = torch.tensor([p[0] for p in pairs], dtype=torch.long, device=dev)
    mi = torch.tensor([p[1] for p in pairs], dtype=torch.long, device=dev)

    modes = psi_y[mi][:, :, None] * psi_x[ni][:, None, :]              # (n_modes, H, W)
    basis = modes.reshape(len(pairs), H * W).T.contiguous().to(torch.complex128)
    return HGBasis(basis=basis, ni=ni, mi=mi, N=N, shape=(H, W))


def coeffs_to_matrix(coef, ni, mi, N):
    """Flat coefficient vector(s) (..., n_modes) -> complex array (..., N+1, N+1)
    indexed [n, m] (n along x, m along y). Handy for displaying/plotting."""
    coef = np.asarray(coef)
    ni = torch.as_tensor(ni).cpu().numpy()
    mi = torch.as_tensor(mi).cpu().numpy()
    out = np.zeros(coef.shape[:-1] + (N + 1, N + 1), dtype=complex)
    out[..., ni, mi] = coef
    return out
