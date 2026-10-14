"""
Real-parameter-vector <-> flat complex-coefficient-vector conversion.
Basis-agnostic: works for any number of modes and any subset of "valid"
mode indices, with no assumption of a square (N+1)x(N+1) layout.

If you're using the HG basis and want the familiar [n, m] matrix view for
display, see hg_defaults.reshape_hg_coefficients.
"""
import numpy as np


def vector_to_coef_np(v, n_modes, valid_indices, n_valid):
    """Inverse of coef_to_vector: real vector -> flat complex coefficient vector."""
    coef = np.zeros(n_modes, dtype=complex)
    real_part = v[:n_valid]
    imag_part = np.zeros(n_valid)
    imag_part[1:] = v[n_valid:]
    coef[valid_indices] = real_part + 1j * imag_part
    return coef


def coef_to_vector_np(coef, valid_indices):
    """Flat complex coefficient vector -> real parameter vector (global phase fixed)."""
    valid_elements = coef[valid_indices]
    return np.concatenate([valid_elements.real, valid_elements[1:].imag])
