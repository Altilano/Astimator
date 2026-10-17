"""
Turn a basis into the per-image transformed bases the Estimator needs.

The tilted-lens transformation multiplies HG mode (n, m) by
    exp(i * (phi_x * n + phi_y * m)),
one (phi_x, phi_y) pair per image -- the values you used to call ax, ay.

  * transform_hg_basis: your basis IS the HG basis -> direct per-mode phase.
  * transform_basis:    your basis is anything else (same pixel grid) -> project
                        it onto the HG basis, apply the phase there (where it is
                        diagonal), recombine.

Both return a tensor of shape (K, n_pix, n_modes): index k is the basis as seen
by image k. Pass it straight to Estimator.
"""
import torch


def phase_factors(ns, ms, phis):
    """phis: K pairs (phi_x, phi_y) -> (K, n_modes) complex128 phases."""
    phis = torch.as_tensor(phis, dtype=torch.float64, device=ns.device).reshape(-1, 2)
    angle = phis[:, 0:1] * ns[None, :] + phis[:, 1:2] * ms[None, :]
    return torch.polar(torch.ones_like(angle), angle)


def transform_hg_basis(hg, phis):
    """hg: HGBasis from make_hg_basis. Returns (K, n_pix, n_modes)."""
    phase = phase_factors(hg.ns, hg.ms, phis)                  # (K, n_modes)
    return hg.basis[None, :, :] * phase[:, None, :]


def transform_basis(basis, hg, phis, rtol=None):
    """
    basis: (n_pix, M) array/tensor, any basis on the same pixel grid as `hg`.
    hg:    HGBasis of high enough order N to represent `basis` well
           (it is only the computational intermediate).
    Returns (K, n_pix, M).
    """
    basis = torch.as_tensor(basis).to(device=hg.basis.device, dtype=torch.complex128)
    coeffs = torch.linalg.pinv(hg.basis, rtol=rtol) @ basis            # (n_hg, M), computed once
    phase = phase_factors(hg.ns, hg.ms, phis)                          # (K, n_hg)
    return hg.basis @ (phase[:, :, None] * coeffs[None, :, :])         # (K, n_pix, M)
