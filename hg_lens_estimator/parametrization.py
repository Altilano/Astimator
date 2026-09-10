import numpy as np


def vector_to_matrix_np(v, N, valid, n_valid):
    """Inverse of matrix_to_vector: real vector -> complex (N+1, N+1) coefficient matrix."""
    A = np.zeros((N + 1, N + 1), dtype=complex)
    real_part = v[:n_valid]
    imag_part = np.zeros(n_valid)
    imag_part[1:] = v[n_valid:]
    A[valid] = real_part + 1j * imag_part
    return A


def matrix_to_vector(A, valid):
    """Complex coefficient matrix -> real parameter vector (global phase fixed)."""
    valid_elements = A[valid]
    return np.concatenate([valid_elements.real, valid_elements[1:].imag])
