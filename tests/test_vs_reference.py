"""
Equivalence tests: Astimator package  vs  the original script `estimate_pytorch.py`.

HOW TO RUN
----------
1. Copy `estimate_pytorch.py` to the repository root (temporary; or leave it in
   PATH_FOLDER, it is searched there too).
2. Set PATH_FOLDER below (or the environment variable PATH_FOLDER). It is the
   folder that contains
       data_cheat/<FOLDER>/random_modes_N*_f0.300_z*_theta*.h5
       coefs_modes/random_modes_N*.h5
   and also structured_optics.py, utils_lens.py, utils_img.py (the original
   script's own imports).
3.  pytest tests/test_vs_reference.py -v -s

WHAT IS TESTED (each level isolates one possible source of mismatch)
--------------------------------------------------------------------
 1. test_hg_basis_matches_reference        make_hg_basis      vs Beam/hg_basis (norm1=True)
 2. test_transformed_basis_matches_...     transform_hg_basis vs B * exp(i(phix*ni+phiy*mi))
                                           (two different waists, like w1/w2 in the script)
 3. test_loss_and_gradient_match_reference TorchEngine loss/grad vs negative_log_likelihood_torch
 4. test_same_start_same_result            Estimator vs the script's minimize(), SAME x0
 5. test_random_restarts_statistically_..  both with random restarts: best loss + fidelity to truth

The reference functions (loss, gradient, vector_to_matrix_np, ...) are NOT
re-typed here: they are extracted from the script with `ast`, so the test runs
the script's own code. Only the per-N setup block of the script (which lives
inside a `for N` loop and cannot be imported) is copied, in `build_reference`.

Mode ordering: the script uses a full (N+1)x(N+1) grid [i, j] and keeps the
entries with ni+mi<=N; the package uses a flat list of modes. The tests map
modes by their (n, m) values (taken from the script's own `hg_proj`), so
a different ordering is NOT a failure, but a different (n, m) meaning is.
"""
import ast
import os
import sys
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from scipy.optimize import minimize

from Astimator import EstimationConfig, Estimator, make_hg_basis, transform_hg_basis
from Astimator.torch_engine import TorchEngine

# =============================================================================
# CONFIG  -- edit here
# =============================================================================
# Folder with data_cheat/, coefs_modes/ and the reference modules
# (structured_optics, utils_lens, utils_img).  Env var PATH_FOLDER overrides it.
PATH_FOLDER = os.environ.get("PATH_FOLDER", r"C:/Python/tilted_lens")

# Where the original script is. Default: repo root, then PATH_FOLDER.
REFERENCE_SCRIPT = os.environ.get("REFERENCE_SCRIPT", "")

# Same constants as in estimate_pytorch.py
FOLDER = "biconvex_stretched_gaussref_equalsize"
PIXEL_SIZE = 2.4e-6
RES = 230
Z1, THETA1 = 30e-2, 0
Z2, THETA2 = 22e-2, 20
LAMB, F_LENS, N_GLASS = 633e-9, 30e-2, 1.5151
WAISTX = WAISTY = 700e-6
Z0 = 118e-2
SIGMA = 1.0

# Test size (keep small: the script's N goes to 10, equivalence shows at small N)
N_VALUES = [1, 2, 3]
SAMPLES = [0, 1]            # which images of each data file are fitted
NUM_REP = 5                 # random restarts in the statistical test (script uses 5)
TEST_WAISTS = (1.2e-4, 1.5e-4)   # waists for the data-free tests (differ on purpose)

# Conventions that make the package reproduce the script (found with the first run of
# these tests; test_diagnose_reference_grid re-checks them against your Beam class):
NORM = "hermite"                         # script's hg_basis(norm1=True) = H_n/sqrt(2^n n!)
GRID_PITCH = RES * PIXEL_SIZE / (RES - 1)   # script's Beam grid: linspace(-nix, nix, res)
SWAP_AXES = True    # script's `ni` (the index multiplying phix) is the order along image ROWS (y)

# Tolerances
BASIS_ATOL = 1e-8           # |s*basis_pkg - basis_script|, after removing one global scale s
SCALE_TOL = 1e-3            # allowed |s - 1|
LOSS_RTOL = 1e-9            # loss at the same parameters
GRAD_RTOL = 1e-8            # gradient, relative to max|grad|
OPT_LOSS_RTOL = 0.05        # final losses of two optimisations
OPT_LOSS_ATOL = 1e-6
FID_MIN = 0.999             # |<a,b>| of two unit-norm coefficient vectors

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =============================================================================
# Locating / loading the reference
# =============================================================================
def _find_reference_script():
    candidates = []
    if REFERENCE_SCRIPT:
        candidates.append(Path(REFERENCE_SCRIPT))
    candidates += [Path(__file__).resolve().parents[1] / "estimate_pytorch.py",
                   Path(PATH_FOLDER) / "estimate_pytorch.py"]
    for c in candidates:
        if c.is_file():
            return c
    pytest.skip("estimate_pytorch.py not found. Looked in: " + ", ".join(map(str, candidates)))


@pytest.fixture(scope="session")
def ref():
    """The reference's own helper modules (structured_optics, utils_lens, utils_img)."""
    if not Path(PATH_FOLDER).is_dir():
        pytest.skip(f"PATH_FOLDER does not exist: {PATH_FOLDER}")
    if PATH_FOLDER not in sys.path:
        sys.path.append(str(PATH_FOLDER))      # append: never shadow the repo's modules
    try:
        import structured_optics as so
        import utils_lens as ul
        import utils_img as ui
    except ImportError as e:
        pytest.skip(f"cannot import reference modules from {PATH_FOLDER}: {e}")
    return SimpleNamespace(Beam=so.Beam, hg_proj=so.hg_proj, hg_basis=so.hg_basis,
                           tilted_lens_focal_lengths=ul.tilted_lens_focal_lengths,
                           calculate_phix_phiy=ul.calculate_phix_phiy,
                           fit_beam_widths=ui.fit_beam_widths)


def _load_reference_functions():
    """Execute ONLY the top-level `def`s of the script (loss, grad, vector_to_matrix_np...).
    The `for N` loop is not executed. Globals the functions expect (n_valid, total,
    valid_indices_t, valid, A_np) are injected by build_reference."""
    path = _find_reference_script()
    tree = ast.parse(path.read_text(encoding="utf-8"))
    defs = [n for n in tree.body if isinstance(n, ast.FunctionDef)]
    ns = {"np": np, "torch": torch, "device": DEVICE, "dtype": torch.float64}
    exec(compile(ast.Module(body=defs, type_ignores=[]), str(path), "exec"), ns)
    return ns


def _reference_columns(ref, N, waist):
    """Copy of the script's `build_Bmat` for ONE waist, restricted to valid modes.
    Returns B_valid (n_pix, n_valid), valid_indices, ni_flat, mi_flat, ni, mi."""
    beam = ref.Beam(nix=RES * PIXEL_SIZE / 2, niy=RES * PIXEL_SIZE / 2,
                    Dx=RES, Dy=RES, lamb=633e-9, waist=waist)
    beam.hg(0, 0)
    ni, mi, _overlap = ref.hg_proj(beam, N)
    beam.waist = waist
    basis = ref.hg_basis(beam, N, norm1=True)
    Bmat = np.array([basis[i, j].ravel() for i in range(N + 1) for j in range(N + 1)]).T
    valid = ni + mi <= N
    valid_indices = np.where(valid.ravel())[0]
    return (Bmat[:, valid_indices], valid_indices,
            ni.ravel()[valid_indices], mi.ravel()[valid_indices], ni, mi, Bmat, valid)


_REF_CACHE = {}


def build_reference(ref, N, w1, w2):
    """Copy of the per-N setup block of estimate_pytorch.py (plus_N = 0)."""
    key = (N, w1, w2)
    if key in _REF_CACHE:
        return _REF_CACHE[key]

    fx1, fy1 = ref.tilted_lens_focal_lengths('biconvex', THETA1, f0=F_LENS, n=N_GLASS)
    phix1, phiy1 = ref.calculate_phix_phiy(fx1, fy1, WAISTX, WAISTY, LAMB, Z0, Z1)
    fx2, fy2 = ref.tilted_lens_focal_lengths('biconvex', THETA2, f0=F_LENS, n=N_GLASS)
    phix2, phiy2 = ref.calculate_phix_phiy(fx2, fy2, WAISTX, WAISTY, LAMB, Z0, Z2)

    B1v, valid_indices, ni_flat, mi_flat, ni, mi, Bmat1, valid = _reference_columns(ref, N, w1)
    B2v, *_rest, Bmat2, _ = _reference_columns(ref, N, w2)

    n_valid = int(np.count_nonzero(valid))
    forw = np.exp(1j * (phix1 * ni + phiy1 * mi)).ravel()
    tilt = np.exp(1j * (phix2 * ni + phiy2 * mi)).ravel()
    B_stacked_full = np.vstack([Bmat1 * forw[None, :], Bmat2 * tilt[None, :]])   # (2*n_pix, (N+1)^2)
    B_valid = B_stacked_full[:, valid_indices].reshape(2, RES * RES, n_valid)    # (2, n_pix, n_valid)

    funcs = _load_reference_functions()
    funcs.update(n_valid=n_valid, total=(N + 1) ** 2, valid=valid,
                 valid_indices_t=torch.tensor(valid_indices, dtype=torch.long, device=DEVICE),
                 A_np=np.zeros((N + 1, N + 1), dtype=complex))

    out = SimpleNamespace(N=N, w1=w1, w2=w2, funcs=funcs, n_valid=n_valid,
                          valid_indices=valid_indices, ni_flat=ni_flat, mi_flat=mi_flat,
                          B_stacked_full=B_stacked_full, B_valid=B_valid,
                          phis=[(float(phix1), float(phiy1)), (float(phix2), float(phiy2))])
    _REF_CACHE[key] = out
    return out


def repo_phis(setup):
    """The script's (phix, phiy) pairs in the PACKAGE's convention (n along x/columns, m along y/rows)."""
    return [(py, px) if SWAP_AXES else (px, py) for px, py in setup.phis]


def build_repo(setup):
    """The package's way: one HG basis per waist, phases from the same phis."""
    hg1 = make_hg_basis(setup.N, (RES, RES), GRID_PITCH, setup.w1, norm=NORM, device="auto")
    hg2 = make_hg_basis(setup.N, (RES, RES), GRID_PITCH, setup.w2, norm=NORM, device="auto")
    phis = repo_phis(setup)
    b1 = transform_hg_basis(hg1, [phis[0]])[0]
    b2 = transform_hg_basis(hg2, [phis[1]])[0]
    return hg1, torch.stack([b1, b2])                    # (2, n_pix, n_modes)


# =============================================================================
# Mode bookkeeping  (script: matrix [i, j] / valid order;  package: flat (n, m) list)
# =============================================================================
def mode_permutation(setup, hg, swap=None):
    """perm[p] = index (in the script's valid-mode order) of the package's mode p.
    The script's label (ni, mi) is matched to the package's (n, m) -- or to (m, n) when
    SWAP_AXES, because the script's `ni` runs along image rows and the package's `n` along columns."""
    swap = SWAP_AXES if swap is None else swap
    ns, ms = hg.ns.cpu().numpy(), hg.ms.cpu().numpy()
    lookup = {(int(n), int(m)): q for q, (n, m) in enumerate(zip(setup.ni_flat, setup.mi_flat))}
    assert len(ns) == setup.n_valid, (
        f"number of modes differs: package {len(ns)} vs script {setup.n_valid}")
    keys = [(int(m), int(n)) if swap else (int(n), int(m)) for n, m in zip(ns, ms)]
    try:
        perm = np.array([lookup[k] for k in keys])
    except KeyError as e:
        raise AssertionError(f"mode {e.args[0]} exists in the package but not in the script")
    assert perm[0] == 0, "the first mode must be (0, 0) in both: it carries the fixed global phase"
    return perm


def ref_vec_to_repo(v, perm, n):
    """Parameter vector [real(n) | imag(n-1)] script order -> package order."""
    imag = np.concatenate([[0.0], v[n:]])[perm]
    return np.concatenate([v[:n][perm], imag[1:]])


def repo_vec_to_ref(v, perm, n):
    real, imag = np.empty(n), np.empty(n)
    real[perm] = v[:n]
    imag[perm] = np.concatenate([[0.0], v[n:]])
    return np.concatenate([real, imag[1:]])


def ref_matrix_to_repo_flat(A, setup, perm):
    return np.asarray(A).ravel()[setup.valid_indices][perm]


def fidelity(a, b):
    a, b = np.asarray(a).ravel(), np.asarray(b).ravel()
    return float(abs(np.vdot(a, b)) / (np.linalg.norm(a) * np.linalg.norm(b)))


# =============================================================================
# Data
# =============================================================================
@lru_cache(maxsize=None)
def load_dataset(N):
    h5py = pytest.importorskip("h5py")
    root = Path(PATH_FOLDER)
    p1 = root / "data_cheat" / FOLDER / f"random_modes_N{N}_f0.300_z{Z1:.3f}_theta{THETA1:.3f}.h5"
    p2 = root / "data_cheat" / FOLDER / f"random_modes_N{N}_f0.300_z{Z2:.3f}_theta{THETA2:.3f}.h5"
    pc = root / "coefs_modes" / f"random_modes_N{N}.h5"
    for p in (p1, p2, pc):
        if not p.is_file():
            pytest.skip(f"data file not found: {p}  (check PATH_FOLDER)")

    def read(path, key):
        with h5py.File(path, "r") as fh:
            return np.asarray(fh[key])

    imgs1, imgs2 = read(p1, "images"), read(p2, "images")
    imgs1 = imgs1 / imgs1.max(axis=(1, 2), keepdims=True)       # same as the script
    imgs2 = imgs2 / imgs2.max(axis=(1, 2), keepdims=True)
    return SimpleNamespace(combined=np.concatenate([imgs1, imgs2], axis=1),   # (n, 2*RES, RES)
                           coefs=read(pc, "coefs"), imgs1=imgs1, imgs2=imgs2)


def fitted_waists(ref, data):
    """Per-image waists exactly as in the script: sqrt(wx*wy) of image 0 of each file."""
    wx1, wy1 = ref.fit_beam_widths(data.imgs1[0], pixel_size=PIXEL_SIZE)
    wx2, wy2 = ref.fit_beam_widths(data.imgs2[0], pixel_size=PIXEL_SIZE)
    return float(np.sqrt(wx1 * wy1)), float(np.sqrt(wx2 * wy2))


def run_reference_fit(setup, img_ravel, x0):
    """The script's inner call: scipy L-BFGS-B with loss_np / grad_np."""
    f = setup.funcs
    B_t = f["_to_complex_tensor"](setup.B_stacked_full)
    data_t = torch.tensor(img_ravel, dtype=torch.float64, device=DEVICE)
    res = minimize(f["loss_np"], x0, args=(B_t, data_t, SIGMA), method="L-BFGS-B",
                   jac=f["grad_np"], tol=1e-12, options={"maxiter": 5000})
    A = f["vector_to_matrix_np"](res.x, setup.N)
    return float(res.fun), A / np.sqrt(np.sum(np.abs(A) ** 2))


def basis_for_estimator(source, setup, hg, perm):
    if source == "repo":
        return build_repo(setup)[1]
    return torch.as_tensor(setup.B_valid[:, :, perm])       # script's own basis, package order


def close_enough(l_a, l_b):
    return l_a <= l_b * (1 + OPT_LOSS_RTOL) + OPT_LOSS_ATOL


# =============================================================================
# 1. HG basis
# =============================================================================
def _fit_global_scale(B_new, B_ref):
    """Best real s with s*B_new ~ B_ref (least squares); returns (s, max|s*B_new - B_ref|).
    The script divides its whole basis by the sampled peak of the (0,0) mode, a constant
    ~ exp(pitch^2/(2 waist^2)) = 1 + 2e-4 (even `res`: no pixel sits exactly at the centre).
    A common factor on the basis cancels in the loss (I / I.max) and in the normalised
    coefficients, so it is compared separately instead of counted as a mismatch."""
    s = np.real(np.vdot(B_new, B_ref)) / np.real(np.vdot(B_new, B_new))
    return float(s), float(np.abs(s * B_new - B_ref).max())


def _convention_table(B_valid, ni_flat, mi_flat, N, waist):
    """max|diff| between the script's basis and the package's for every combination of the
    usual suspects (axis swap, normalisation, grid pitch). The right one shows ~1e-12."""
    setup = SimpleNamespace(ni_flat=ni_flat, mi_flat=mi_flat, n_valid=B_valid.shape[1])
    rows = []
    for swap in (False, True):
        for norm in ("hermite", "max1"):
            for label, pitch in (("pixel_size", PIXEL_SIZE), ("res*pixel/(res-1)", PIXEL_SIZE * RES / (RES - 1))):
                hg = make_hg_basis(N, (RES, RES), pitch, waist, norm=norm, device="cpu")
                d = _fit_global_scale(hg.basis.numpy(), B_valid[:, mode_permutation(setup, hg, swap=swap)])[1]
                rows.append(f"swap_axes={swap!s:5}  norm={norm:8}  pitch={label:18} max|diff|={d:.3e}")
    return "\n  ".join(rows)


@pytest.mark.parametrize("waist", TEST_WAISTS)
@pytest.mark.parametrize("N", N_VALUES)
def test_hg_basis_matches_reference(ref, N, waist):
    B_valid, _vi, ni_flat, mi_flat, *_ = _reference_columns(ref, N, waist)
    hg = make_hg_basis(N, (RES, RES), GRID_PITCH, waist, norm=NORM, device="cpu")

    setup = SimpleNamespace(ni_flat=ni_flat, mi_flat=mi_flat, n_valid=B_valid.shape[1])
    perm = mode_permutation(setup, hg)
    B_ref = B_valid[:, perm]
    B_new = hg.basis.numpy()
    assert B_new.shape == B_ref.shape, f"shape: package {B_new.shape} vs script {B_ref.shape}"

    scale, diff = _fit_global_scale(B_new, B_ref)
    expected = np.exp(GRID_PITCH ** 2 / (2 * waist ** 2)) - 1
    print(f"\nN={N} waist={waist:g}: global scale s-1 = {scale - 1:.3e} (sampled-peak estimate {expected:.3e}), "
          f"residual after scaling {diff:.2e}")
    assert abs(scale - 1) < SCALE_TOL, f"global scale between the bases is {scale:.6f}, expected ~1"
    assert diff < BASIS_ATOL, (
        f"basis differs by {diff:.3e} (after a global scale) with NORM={NORM!r}, GRID_PITCH={GRID_PITCH:.6e}, "
        f"SWAP_AXES={SWAP_AXES}.\nAll combinations (the matching one should be ~1e-12):\n  "
        + _convention_table(B_valid, ni_flat, mi_flat, N, waist))


def _fit_gaussian_1d(profile, waist):
    """log|profile| = -(pitch/waist)^2 (k - k0)^2 + c  ->  (pitch, k0)."""
    k = np.arange(len(profile))
    ok = profile > 1e-8 * profile.max()
    a2, a1, _ = np.polyfit(k[ok], np.log(profile[ok]), 2)
    return waist * np.sqrt(-a2), -a1 / (2 * a2)


def test_diagnose_reference_grid(ref):
    """Measures the script's Beam grid (pitch, centre) and the axis its `ni` runs along,
    straight from its modes, and checks them against NORM / GRID_PITCH / SWAP_AXES."""
    N, waist = 1, TEST_WAISTS[0]
    B_valid, _vi, ni_flat, mi_flat, *_ = _reference_columns(ref, N, waist)
    col = {(int(a), int(b)): q for q, (a, b) in enumerate(zip(ni_flat, mi_flat))}

    G = np.abs(B_valid[:, col[(0, 0)]]).reshape(RES, RES)
    pitch_x, c_x = _fit_gaussian_1d(G[RES // 2, :], waist)     # along columns
    pitch_y, c_y = _fit_gaussian_1d(G[:, RES // 2], waist)     # along rows
    G1 = B_valid[:, col[(1, 0)]].reshape(RES, RES)
    odd_along_columns = np.abs(G1.sum(axis=1)).max() < np.abs(G1.sum(axis=0)).max()
    ni_axis = "COLUMNS (x)" if odd_along_columns else "ROWS (y)"
    print(f"\nscript grid: pitch along x = {pitch_x:.6e} ({pitch_x / PIXEL_SIZE:.6f} x pixel_size), "
          f"along y = {pitch_y:.6e} ({pitch_y / PIXEL_SIZE:.6f} x pixel_size)")
    print(f"             centre (pixel index): x = {c_x:.4f}, y = {c_y:.4f}   (package assumes {(RES - 1) / 2})")
    print(f"             res/(res-1) = {RES / (RES - 1):.6f}")
    print(f"script mode ni (index multiplying phix) runs along image {ni_axis}")

    assert pitch_x == pytest.approx(GRID_PITCH, rel=1e-4) and pitch_y == pytest.approx(GRID_PITCH, rel=1e-4), (
        "script grid pitch != GRID_PITCH; use the fitted values printed above")
    assert c_x == pytest.approx((RES - 1) / 2, abs=1e-3) and c_y == pytest.approx((RES - 1) / 2, abs=1e-3), (
        "script grid centre differs from (RES-1)/2; pass center=(cy, cx) to make_hg_basis")
    assert odd_along_columns == (not SWAP_AXES), f"ni runs along {ni_axis}: fix SWAP_AXES"


# =============================================================================
# 2. Transformed basis (phases + two waists)
# =============================================================================
@pytest.mark.parametrize("N", N_VALUES)
def test_transformed_basis_matches_reference(ref, N):
    setup = build_reference(ref, N, *TEST_WAISTS)
    hg, basis = build_repo(setup)
    perm = mode_permutation(setup, hg)
    B_ref = setup.B_valid[:, :, perm]
    assert basis.shape == B_ref.shape
    B_new = basis.cpu().numpy()
    for k in range(B_new.shape[0]):             # one global scale per image (each has its own waist)
        scale, diff = _fit_global_scale(B_new[k], B_ref[k])
        assert abs(scale - 1) < SCALE_TOL, f"image {k}: global scale {scale:.6f}, expected ~1"
        assert diff < BASIS_ATOL, (
            f"image {k}: transformed basis differs by {diff:.3e}; phis (script) {setup.phis}, "
            f"used in package order {repo_phis(setup)}. If test_hg_basis_matches_reference passes, the "
            f"phase convention or the image order is what differs.")


# =============================================================================
# 3. Loss and gradient at identical parameters
# =============================================================================
@pytest.mark.parametrize("sigma", [1.0, 0.5])
@pytest.mark.parametrize("N", N_VALUES)
def test_loss_and_gradient_match_reference(ref, N, sigma):
    setup = build_reference(ref, N, *TEST_WAISTS)
    hg, _ = build_repo(setup)
    perm = mode_permutation(setup, hg)
    n = setup.n_valid
    f = setup.funcs

    rng = np.random.default_rng(0)
    data = rng.random((2, RES * RES))
    data /= data.max(axis=1, keepdims=True)
    v_ref = rng.standard_normal(2 * n - 1)

    # script
    B_t = f["_to_complex_tensor"](setup.B_stacked_full)
    loss_ref, grad_ref = f["loss_and_grad_np"](
        v_ref, B_t, torch.tensor(data.ravel(), dtype=torch.float64, device=DEVICE), sigma)

    # package, fed with the SCRIPT's basis -> isolates the loss function itself
    engine = TorchEngine("auto")
    lg = engine.make_loss_and_grad(n)
    basis = engine.to_basis_tensor(setup.B_valid[:, :, perm])
    loss_new, grad_new = lg(ref_vec_to_repo(v_ref, perm, n), basis, engine.to_data_tensor(data), sigma)
    grad_new = repo_vec_to_ref(grad_new, perm, n)

    assert loss_new == pytest.approx(loss_ref, rel=LOSS_RTOL), f"loss: {loss_new} vs {loss_ref}"
    gmax = np.abs(grad_ref).max()
    assert np.abs(grad_new - grad_ref).max() <= GRAD_RTOL * gmax, (
        f"gradient differs: max|diff|={np.abs(grad_new - grad_ref).max():.3e} (max|grad|={gmax:.3e})")


# =============================================================================
# 4. Same starting point -> same optimum
# =============================================================================
@pytest.mark.parametrize("basis_source", ["repo", "reference"])
@pytest.mark.parametrize("N", N_VALUES)
def test_same_start_same_result(ref, N, basis_source, monkeypatch):
    """basis_source='reference' feeds the package the script's own basis: it isolates
    Estimator/optimiser from the basis. 'repo' is the real end-to-end path."""
    data = load_dataset(N)
    setup = build_reference(ref, N, *fitted_waists(ref, data))
    hg, _ = build_repo(setup)
    perm = mode_permutation(setup, hg)
    basis = basis_for_estimator(basis_source, setup, hg, perm)
    n = setup.n_valid

    rng = np.random.RandomState(0)
    print()
    for r in [s for s in SAMPLES if s < len(data.combined)]:
        x0 = 0.01 * rng.randn(2 * n - 1)
        x0 /= np.linalg.norm(x0)                       # the script normalises its initial guess
        loss_ref, A_ref = run_reference_fit(setup, data.combined[r].ravel(), x0)

        x0_repo = ref_vec_to_repo(x0, perm, n)
        # Estimator draws x0 = 0.01 * np.random.randn(...): hand it exactly x0_repo
        monkeypatch.setattr(np.random, "randn", lambda *a, _x=x0_repo: _x / 0.01)
        out = Estimator(basis, EstimationConfig(device="auto", n_restarts=1, sigma=SIGMA)).estimate(
            data.combined[r])
        loss_new = float(out["results"][0])

        fid = fidelity(out["best"], ref_matrix_to_repo_flat(A_ref, setup, perm))
        print(f"N={N} sample={r} [{basis_source}]  loss script={loss_ref:.3e} package={loss_new:.3e}  fidelity={fid:.8f}")
        assert close_enough(loss_new, loss_ref) and close_enough(loss_ref, loss_new), (
            f"final losses differ: script {loss_ref:.4e} vs package {loss_new:.4e}")
        assert fid > FID_MIN, f"estimated coefficients differ (fidelity {fid:.6f})"


# =============================================================================
# 5. Random restarts: best loss and recovery of the true coefficients
# =============================================================================
@pytest.mark.parametrize("basis_source", ["repo", "reference"])
@pytest.mark.parametrize("N", N_VALUES)
def test_random_restarts_statistically_equivalent(ref, N, basis_source):
    data = load_dataset(N)
    setup = build_reference(ref, N, *fitted_waists(ref, data))
    hg, _ = build_repo(setup)
    perm = mode_permutation(setup, hg)
    basis = basis_for_estimator(basis_source, setup, hg, perm)
    n = setup.n_valid

    np.random.seed(123)
    est = Estimator(basis, EstimationConfig(device="auto", n_restarts=NUM_REP, sigma=SIGMA))
    rng = np.random.RandomState(321)
    print()
    for r in [s for s in SAMPLES if s < len(data.combined)]:
        # script: NUM_REP restarts, keep the lowest loss
        best_ref = (np.inf, None)
        for _ in range(NUM_REP):
            x0 = 0.01 * rng.randn(2 * n - 1)
            x0 /= np.linalg.norm(x0)
            l, A = run_reference_fit(setup, data.combined[r].ravel(), x0)
            if l < best_ref[0]:
                best_ref = (l, A)
        loss_ref, A_ref = best_ref
        out = est.estimate(data.combined[r])
        loss_new = float(out["results"].min())

        msg = f"N={N} sample={r} [{basis_source}]  best loss script={loss_ref:.3e} package={loss_new:.3e}"
        c_ref = ref_matrix_to_repo_flat(A_ref, setup, perm)
        truth_r = np.asarray(data.coefs[r])
        if truth_r.size == (N + 1) ** 2:
            truth = ref_matrix_to_repo_flat(truth_r.reshape(N + 1, N + 1), setup, perm)
            f_ref, f_new = fidelity(c_ref, truth), fidelity(out["best"], truth)
            msg += f"  fidelity to truth: script={f_ref:.6f} package={f_new:.6f}"
            print(msg)
            assert f_new >= f_ref - 0.01, "package recovers the true coefficients worse than the script"
        else:
            print(msg + f"  (truth skipped: coefs[r] has shape {truth_r.shape}, expected ({N+1},{N+1}))")
        assert close_enough(loss_new, loss_ref), (
            f"package best loss {loss_new:.4e} is worse than script {loss_ref:.4e}")