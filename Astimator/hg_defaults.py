"""
Everything HG- and lens-tilt-specific lives here, as optional implementations
of the generic BasisProvider interface (basis_provider.py). Two flavors:

- hg_transform_provider(...): fast path for when your basis already IS the
  HG basis -- the lens-tilt transform is then just a per-mode phase multiply.

- generic_lens_transform_provider(...): for when your basis is something
  else (measured modes, Laguerre-Gaussian, anything on the same pixel grid).
  It decomposes your basis into an HG basis, applies the lens-tilt phase in
  that representation (where it's actually diagonal), and recombines back --
  see lens_transform.py for the underlying math.

If you switch to a different basis or different transform physics entirely,
you don't touch this file -- write your own base_basis_fn/transform_fn (or
just build a PrecomputedBasisProvider) and the rest of the pipeline is
unaffected.
"""
import numpy as np

from .basis_provider import TransformBasisProvider
from .lens_transform import tilted_lens_transform


def build_space(nix, im_size, waist):
    """Thin wrapper around beam.Space so callers don't need `from beam import *`."""
    from beam import Space  # your existing beam.py
    b1 = Space(nix, im_size)
    b1.waist = waist
    return b1


def build_hg_grids(b1, N):
    """HG index grids (ni, mi) for max order N -- only depend on N, not the image."""
    ni, mi, _ = b1.hg_projector(N)
    return ni, mi


def build_basis_matrix(b1, N, waist):
    """Flattened HG basis matrix (n_pix, (N+1)^2) for a given waist."""
    b1.waist = waist
    basis = b1.hg_basis(N, norm="max1").transpose(0, 1, 2, 3)
    B = np.array([basis[i, j].ravel() for i in range(N + 1) for j in range(N + 1)]).T
    return B


def hg_valid_indices(N):
    """Triangular n+m<=N mode-truncation mask, flattened to indices into the (N+1)^2 basis columns."""
    ni, mi = np.meshgrid(np.arange(N + 1), np.arange(N + 1), indexing="ij")
    valid = (ni + mi) <= N
    n_valid = int(np.count_nonzero(valid))
    valid_indices = np.where(valid.ravel())[0]
    return valid, n_valid, valid_indices


def reshape_hg_coefficients(coef_flat, N):
    """Flat (N+1)^2 coefficient vector -> (N+1, N+1) matrix indexed by [n, m], for display."""
    return coef_flat.reshape(N + 1, N + 1)


# ---------------------------------------------------------------------------
# Fast path: your basis already IS the HG basis, so the lens transform is a
# direct per-mode phase multiply (no decomposition needed).
# ---------------------------------------------------------------------------

def hg_lens_transform_fn(ni, mi, lamb, z0, f):
    """transform_fn(base_basis, transform_spec, waist) using calculate_ax_ay
    directly on an HG basis (base_basis's columns must be HG modes ordered
    to match ni.ravel(), mi.ravel())."""
    from utils import calculate_ax_ay  # your existing optics function

    def transform_fn(base_basis, transform_spec, waist):
        ax, ay = calculate_ax_ay(
            waist=waist, lamb=lamb, z0=z0, f=f,
            theta=transform_spec.theta, z=transform_spec.z,
            flip_lens=transform_spec.flip_lens,
        )
        phase = np.exp(1j * (ax * ni + ay * mi)).ravel()
        return base_basis * phase[None, :]

    return transform_fn


def hg_transform_provider(nix, im_size, N, lamb=800e-9, z0=20e-2, f=30e-2, waist=2e-4):
    """
    Reproduces the original notebooks' default behavior: builds an HG basis
    via beam.Space, and transforms it with the lens-tilt phase model for
    each TransformSpec. Returns a TransformBasisProvider.
    """
    b1 = build_space(nix, im_size, waist)
    ni, mi = build_hg_grids(b1, N)

    def base_basis_fn(w):
        return build_basis_matrix(b1, N, w if w is not None else waist)

    transform_fn = hg_lens_transform_fn(ni, mi, lamb, z0, f)
    return TransformBasisProvider(base_basis_fn, transform_fn)


# ---------------------------------------------------------------------------
# Generic path: your basis is NOT (necessarily) HG. Decompose into an HG
# basis, apply the lens-tilt phase there (where it's diagonal), recombine.
# ---------------------------------------------------------------------------

def make_hg_basis_builder(nix, im_size, N):
    """
    Returns hg_basis_builder(waist) -> (hg_basis, ni, mi), a reusable HG
    representation (built via beam.Space) used purely as the computational
    intermediate for the generic lens transform below -- it does not need
    to have anything to do with the basis you're actually fitting.
    """
    b1 = build_space(nix, im_size, 2e-4)  # placeholder waist, reset per call
    ni, mi = build_hg_grids(b1, N)

    def hg_basis_builder(waist):
        hg_basis = build_basis_matrix(b1, N, waist)
        return hg_basis, ni.ravel(), mi.ravel()

    return hg_basis_builder


def generic_lens_transform_fn(hg_basis_builder, lamb, z0, f, rcond=None):
    """
    transform_fn(base_basis, transform_spec, waist) for ANY base_basis
    (need not be HG). Computes ax, ay via calculate_ax_ay for this
    transform, then applies lens_transform.tilted_lens_transform using the
    HG basis from hg_basis_builder as the intermediate representation.
    """
    from utils import calculate_ax_ay  # your existing optics function

    def transform_fn(base_basis, transform_spec, waist):
        hg_basis, ni, mi = hg_basis_builder(waist)
        ax, ay = calculate_ax_ay(
            waist=waist, lamb=lamb, z0=z0, f=f,
            theta=transform_spec.theta, z=transform_spec.z,
            flip_lens=transform_spec.flip_lens,
        )
        return tilted_lens_transform(base_basis, hg_basis, ni, mi, ax, ay, rcond=rcond)

    return transform_fn


def h5_lens_transform_fn(h5_dataset, hg_basis_builder, rcond=None):
    """
    Same idea as generic_lens_transform_fn, but reads ax, ay back from an
    H5Dataset (h5_dataset.py) instead of recomputing them with
    calculate_ax_ay -- use this if you trust the ax/ay values already baked
    into your .h5 file (e.g. from a separate calibration step).
    """
    def transform_fn(base_basis, transform_spec, waist):
        hg_basis, ni, mi = hg_basis_builder(waist)
        ax, ay = h5_dataset.get_ax_ay(transform_spec.name)
        return tilted_lens_transform(base_basis, hg_basis, ni, mi, ax, ay, rcond=rcond)

    return transform_fn


def generic_transform_provider(base_basis, nix, im_size, N, lamb=800e-9, z0=20e-2, f=30e-2, rcond=None):
    """
    Convenience one-liner for the generic (non-HG) case: give it your basis
    (or a callable(waist) -> basis, if it depends on waist) and it builds a
    TransformBasisProvider that decomposes-transforms-recombines through an
    internal HG representation for each TransformSpec.
    """
    hg_basis_builder = make_hg_basis_builder(nix, im_size, N)
    transform_fn = generic_lens_transform_fn(hg_basis_builder, lamb, z0, f, rcond=rcond)

    def base_basis_fn(waist):
        return base_basis(waist) if callable(base_basis) else base_basis

    return TransformBasisProvider(base_basis_fn, transform_fn)
