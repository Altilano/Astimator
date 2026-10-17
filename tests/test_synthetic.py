"""Synthetic tests: generate noiseless pictures from known coefficients, then
check that Estimator recovers them. Run with:  pytest tests/ -s"""
import numpy as np
from numpy.polynomial.hermite import hermval
from math import factorial, sqrt

from hg_lens_estimator import (
    EstimationConfig, Estimator,
    make_hg_basis, transform_hg_basis, transform_basis, coeffs_to_matrix,
)

N, RES, W_BEAM = 3, 64, 1.0
PIXEL = 6.0 / (RES - 1)            # np.linspace(-3, 3, RES) pitch
PHIX = [0.0, 0.9, 2.1, 3.0]
PHIY = [0.0, 1.3, 0.4, 2.2]
PHIS = list(zip(PHIX, PHIY))


def hg1d(n, x, w):
    c = np.zeros(n + 1); c[n] = 1
    return hermval(sqrt(2) * x / w, c) * np.exp(-x**2 / w**2) / sqrt(2**n * factorial(n))


def reference_basis(N, res=RES, w=W_BEAM):
    """Independent basis with modes n+m<=N as columns (res*res, n_modes); returns also ni, mi.
    Same column order as make_hg_basis: n outer loop, m inner loop; n along x, m along y."""
    x = np.linspace(-3, 3, res)
    X, Y = np.meshgrid(x, x)
    cols, ni, mi = [], [], []
    for n in range(N + 1):
        for m in range(N + 1 - n):
            b = hg1d(n, X, w) * hg1d(m, Y, w)
            cols.append((b / np.sqrt((np.abs(b) ** 2).sum())).ravel())
            ni.append(n); mi.append(m)
    return np.array(cols).T.astype(complex), np.array(ni), np.array(mi)


def make_truth(ni, mi, seed=0):
    rng = np.random.default_rng(seed)
    c = rng.standard_normal(len(ni)) + 1j * rng.standard_normal(len(ni))
    return c / np.linalg.norm(c)


def make_images(transformed, c):
    """transformed: (K, n_pix, n_modes). Returns the K pictures v-stacked -> (K*RES, RES)."""
    imgs = []
    for b in transformed:
        I = np.abs(b @ c) ** 2
        imgs.append((I / I.max()).reshape(RES, RES))
    return np.vstack(imgs)


def align(est, truth):
    """Remove the unobservable global phase of the estimate.
    vdot(est, truth) = sum(conj(est) * truth) has phase -delta when est = truth * e^{i delta}."""
    return est * np.exp(1j * np.angle(np.vdot(est, truth)))


def check_recovery(basis, images, truth):
    np.random.seed(0)
    est = Estimator(basis, EstimationConfig(device="cpu", n_restarts=5, maxiter=1000))
    res = est.estimate(images)
    err = np.linalg.norm(align(res["best"], truth) - truth)
    print("loss", res["results"].min(), "err", err)
    assert err < 1e-3
    return res


def test_make_hg_basis_matches_reference():
    hg = make_hg_basis(N, (RES, RES), PIXEL, W_BEAM, norm="l2", device="cpu")
    B, ni, mi = reference_basis(N)
    assert np.array_equal(hg.ni.numpy(), ni) and np.array_equal(hg.mi.numpy(), mi)
    assert np.abs(hg.basis.numpy() - B).max() < 1e-10


def test_recovers_coefs_with_helpers():
    hg = make_hg_basis(N, (RES, RES), PIXEL, W_BEAM, norm="l2", device="cpu")
    ni, mi = hg.ni.numpy(), hg.mi.numpy()
    C = make_truth(ni, mi)
    basis = transform_hg_basis(hg, PHIS)                      # (4, n_pix, n_modes)
    res = check_recovery(basis, make_images(basis.numpy(), C), C)
    # matrix view [n, m], like the old (N+1, N+1) layout
    Cm = coeffs_to_matrix(align(res["best"], C), ni, mi, N)
    assert Cm.shape == (N + 1, N + 1)


def test_recovers_coefs_with_custom_basis_list():
    """Your own basis, with the per-picture transformed bases built by hand."""
    B, ni, mi = reference_basis(N)
    C = make_truth(ni, mi, seed=1)
    basis_list = [B * np.exp(1j * (px * ni + py * mi))[None, :] for px, py in PHIS]
    check_recovery(basis_list, make_images(np.array(basis_list), C), C)


def test_transform_basis_generic_matches_manual():
    """Generic path (project onto HG, phase, recombine) == manual phases for an HG-spanned basis."""
    hg = make_hg_basis(N, (RES, RES), PIXEL, W_BEAM, norm="l2", device="cpu")
    B, ni, mi = reference_basis(N)
    manual = np.array([B * np.exp(1j * (px * ni + py * mi))[None, :] for px, py in PHIS])
    generic = transform_basis(B, hg, PHIS).numpy()
    assert np.abs(generic - manual).max() < 1e-8
