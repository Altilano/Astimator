"""
The tilted-lens transformation is only diagonal (a simple per-mode phase) in
the Hermite-Gaussian representation. If your basis isn't HG, you can't just
multiply it by exp(i*(ax*n + ay*m)) directly -- you first have to express it
in the HG representation, apply the phase there, then convert back.

That's exactly what `tilted_lens_transform` does:

    1. decompose_into_hg   : project the input basis onto an HG basis
                             (least-squares -- works even if the HG basis
                             given isn't perfectly orthonormal, or the input
                             basis isn't exactly representable at finite N)
    2. apply_hg_diagonal_phase : multiply each HG coefficient by
                             exp(i*(ax*n + ay*m)) -- the actual physics
    3. recombine_from_hg   : reconstruct the transformed field on the pixel
                             grid from the phase-shifted HG coefficients

This makes the lens-tilt model usable with any basis you like (measured
modes, Laguerre-Gaussian, anything defined on the same pixel grid) as long
as you can supply an HG basis of high enough order to represent it -- the HG
basis here is a computational intermediate, not a constraint on your data.
"""
import numpy as np


def decompose_into_hg(basis, hg_basis, rcond=None):
    """
    basis:    (n_pix, n_modes) complex/real, arbitrary field patterns on a pixel grid
    hg_basis: (n_pix, n_hg_modes) complex, HG basis on the SAME pixel grid
    returns:  (n_hg_modes, n_modes) decomposition coefficients

    Uses a least-squares fit rather than assuming exact orthonormality of
    hg_basis, so mild normalization differences don't bias the result.
    """
    coeffs, *_ = np.linalg.lstsq(hg_basis, basis, rcond=rcond)
    return coeffs


def apply_hg_diagonal_phase(coeffs, ni, mi, ax, ay):
    """
    coeffs:  (n_hg_modes, n_modes)
    ni, mi:  (n_hg_modes,) HG order (n, m) of each row of `coeffs`
    ax, ay:  scalars for this measurement (e.g. from calculate_ax_ay, or
             read back from wherever you stored them -- an .h5 file, say)
    """
    phase = np.exp(1j * (ax * ni + ay * mi))
    return coeffs * phase[:, None]


def recombine_from_hg(coeffs, hg_basis):
    """coeffs: (n_hg_modes, n_modes); hg_basis: (n_pix, n_hg_modes) -> (n_pix, n_modes)"""
    return hg_basis @ coeffs


def tilted_lens_transform(basis, hg_basis, ni, mi, ax, ay, rcond=None):
    """
    Full pipeline: decompose `basis` into `hg_basis`, apply the tilted-lens
    phase per (n, m), recombine into pixel space.

    basis:    (n_pix, n_modes)     -- the basis you actually care about
    hg_basis: (n_pix, n_hg_modes)  -- an HG basis on the same pixel grid,
                                       high enough order to represent `basis`
                                       with negligible truncation error
    ni, mi:   (n_hg_modes,)         -- HG index of each hg_basis column
    ax, ay:   scalars for this measurement

    Sanity check: if `basis` IS `hg_basis` (same object/columns, same order),
    this reduces to a plain per-column phase multiply, matching the direct
    HG phase-shift approach used when you know your basis already is HG.
    """
    coeffs = decompose_into_hg(basis, hg_basis, rcond=rcond)
    coeffs = apply_hg_diagonal_phase(coeffs, ni, mi, ax, ay)
    return recombine_from_hg(coeffs, hg_basis)
