import numpy as np


def build_space(nix, im_size, waist):
    """Thin wrapper around beam.Space so pipeline.py doesn't need `from beam import *`."""
    from structured_optics import Beam  # your existing beam.py
    b1 = Beam(nix, im_size)
    b1.waist = waist
    return b1


def build_hg_grids(b1, N):
    """
    Returns ni, mi (HG index grids) using the (n=0, m=0) projector, exactly as in
    both original notebooks -- these only depend on N, not on the image itself.
    """
    ni, mi, _ = b1.hg_projector(N)
    return ni, mi


def build_basis_matrix(b1, N, waist):
    """
    Build the flattened HG basis matrix B (n_pix, (N+1)^2) for a given waist,
    i.e. set_Bmat[k] from the original notebooks.
    """
    b1.waist = waist
    basis = b1.hg_basis(N, norm="max1")
    B = np.array([basis[i, j].ravel() for i in range(N + 1) for j in range(N + 1)]).T
    return B


def build_stacked_basis(B, phases_by_name, transform_names):
    """
    Phase-shift B for each active transform and stack them, i.e.
    the B_stacked from the original notebooks, generalized to any
    number/order of transforms.

    B:               (n_pix, (N+1)^2) complex, unshifted basis
    phases_by_name:  dict name -> raveled phase vector (from optics.compute_transform_phases)
    transform_names: ordered list of names to actually include (len == n_transforms)
    """
    blocks = [B * phases_by_name[name][None, :] for name in transform_names]
    return np.vstack(blocks)


def valid_mode_mask(ni, mi, N):
    valid = ni + mi <= N
    n_valid = int(np.count_nonzero(valid))
    valid_indices = np.where(valid.ravel())[0]
    return valid, n_valid, valid_indices
