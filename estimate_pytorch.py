from structured_optics import *
import numpy as np
from scipy.optimize import minimize
import tqdm
import h5py
from utils_lens import *
from utils_img import *
import os
from pathlib import Path
import torch


# ── Device setup ─────────────────────────────────────────────────────────────
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dtype = torch.float64
print(f"Using device: {device}")


def _to_complex_tensor(arr):
    return torch.tensor(arr, dtype=torch.complex128, device=device)


def vector_to_coef_torch(v: torch.Tensor) -> torch.Tensor:
    """
    v layout:  [ real_part (n_valid) | imag_part (n_valid-1) ]
    The first coefficient is forced to be real (fixes global phase).
    Uses the globals n_valid, total, valid_indices_t set for the current N.
    """
    real_part = v[:n_valid]
    imag_part = torch.cat([v.new_zeros(1), v[n_valid:]])

    coef_real = torch.zeros(total, dtype=dtype, device=device)
    coef_imag = torch.zeros(total, dtype=dtype, device=device)
    coef_real = coef_real.scatter(0, valid_indices_t, real_part)
    coef_imag = coef_imag.scatter(0, valid_indices_t, imag_part)

    return torch.complex(coef_real, coef_imag)


# ── Loss function (pure PyTorch) ─────────────────────────────────────────────
def negative_log_likelihood_torch(params, B_stacked, data, sigma):
    coef = vector_to_coef_torch(params)
    E_all = B_stacked @ coef
    n_pix = E_all.shape[0] // 2

    def intensity_normalized(E_seg):
        I = E_seg.abs() ** 2
        return I / I.max()

    I0 = intensity_normalized(E_all[:n_pix])
    I1 = intensity_normalized(E_all[n_pix:2 * n_pix])

    predicted = torch.cat([I0, I1])
    residual = data - predicted
    return 0.5 * ((residual / sigma) ** 2).sum()


def loss_and_grad_np(params_np, B_stacked, data, sigma):
    params_t = torch.tensor(params_np, dtype=dtype, device=device,
                            requires_grad=True)
    loss = negative_log_likelihood_torch(params_t, B_stacked, data, sigma)
    loss.backward()
    grad = params_t.grad.detach().cpu().numpy().astype(np.float64)
    return float(loss.detach().cpu()), grad


def loss_np(params, B_stacked, data, sigma):
    return loss_and_grad_np(params, B_stacked, data, sigma)[0]


def grad_np(params, B_stacked, data, sigma):
    return loss_and_grad_np(params, B_stacked, data, sigma)[1]


def vector_to_matrix_np(v, valid, n_valid):
    A = np.zeros(valid.shape, dtype=complex)
    imag_part = np.zeros(n_valid)
    imag_part[1:] = v[n_valid:]
    A[valid] = v[:n_valid] + 1j * imag_part
    return A


# ── Settings ─────────────────────────────────────────────────────────────────
plus_N = 0
pixel_size = 2.4e-6
res = 232
folder = "biconvex_stretched_gaussref_equalsize"

lamb = 633e-9
f = 30e-2
n = 1.5151
waistx = waisty = 700e-6
z0 = 118e-2

z1, theta1 = 30e-2, 0
z2, theta2 = 22e-2, 20

sigma = 1.0
num_rep = 5

# New folder so the old (buggy-phase) results are not overwritten
output_path = f"results/waist_difbasis/{folder}_phasefix"


def load_normalized(N, z, theta):
    with h5py.File(f"data_treated/{folder}/random_modes_N{N}_f0.300_"
                   f"z{z:.3f}_theta{theta:.3f}.h5", 'r') as file:
        imgs = np.asarray(file['images'])
    return imgs / imgs.max(axis=(1, 2), keepdims=True)


for N in range(1, 11):

    Nt = N + plus_N

    imgs1 = load_normalized(N, z1, theta1)
    imgs2 = load_normalized(N, z2, theta2)
    combined = np.concatenate([imgs1, imgs2], axis=1)

    print(f"Loaded {combined.shape[0]} images for N={N}.")

    used_data = dict(zs=[z1, z2], thetas=[theta1, theta2])

    # ── Per-image waists ─────────────────────────────────────────────────────
    wx1, wy1 = fit_beam_widths(imgs1[0], pixel_size=pixel_size)
    wx2, wy2 = fit_beam_widths(imgs2[0], pixel_size=pixel_size)
    w1 = np.sqrt(wx1 * wy1)
    w2 = np.sqrt(wx2 * wy2)
    print("Estimated beam waists:")
    print(f"Image 1: wx={wx1}, wy={wy1} -> w1={w1}")
    print(f"Image 2: wx={wx2}, wy={wy2} -> w2={w2}")

    fx1, fy1 = tilted_lens_focal_lengths('biconvex', theta1, f0=f, n=n)
    phix1, phiy1 = calculate_phix_phiy(fx1, fy1, waistx, waisty, lamb, z0, z1)

    fx2, fy2 = tilted_lens_focal_lengths('biconvex', theta2, f0=f, n=n)
    phix2, phiy2 = calculate_phix_phiy(fx2, fy2, waistx, waisty, lamb, z0, z2)

    print("Calculated phi_x and phi_y.")

    # ── Beams ────────────────────────────────────────────────────────────────
    beam_kwargs = dict(nix=res * pixel_size / 2, niy=res * pixel_size / 2,
                       Dx=res, Dy=res, lamb=lamb)
    beam1 = Beam(**beam_kwargs, waist=w1)
    beam2 = Beam(**beam_kwargs, waist=w2)
    beam1.hg(0, 0)
    beam2.hg(0, 0)

    # ── Index grids: "ij" convention (replaces hg_proj) ──────────────────────
    # Column order of the propagation matrix is index = i*(Nt+1) + j,
    # so ni = i (row index) and mi = j (column index).
    ni, mi = np.meshgrid(np.arange(Nt + 1), np.arange(Nt + 1), indexing='ij')

    scales = np.array([1.0])
    waists1 = w1 * scales
    waists2 = w2 * scales

    def build_Bmat(beam, waist):
        beam.waist = waist
        basis = hg_basis(beam, Nt, norm1=True)
        return np.array([basis[i, j].ravel()
                         for i in range(Nt + 1)
                         for j in range(Nt + 1)]).T

    set_Bmat1 = [build_Bmat(beam1, w) for w in waists1]
    set_Bmat2 = [build_Bmat(beam2, w) for w in waists2]
    print('Built basis matrices for all waists.')

    # ── Valid-index bookkeeping ──────────────────────────────────────────────
    valid = (ni + mi) <= Nt
    n_valid = int(np.count_nonzero(valid))
    valid_indices = np.where(valid.ravel())[0]
    valid_indices_t = torch.tensor(valid_indices, dtype=torch.long,
                                   device=device)
    total = (Nt + 1) * (Nt + 1)

    # ── Phase vectors (px with ni, py with mi) ───────────────────────────────
    forw_ravel = np.exp(1j * (phix1 * ni + phiy1 * mi)).ravel()
    tilt_ravel = np.exp(1j * (phix2 * ni + phiy2 * mi)).ravel()

    set_Bmat_stacked_torch = []
    for k in range(len(scales)):
        Bf = set_Bmat1[k] * forw_ravel[None, :]
        Bt = set_Bmat2[k] * tilt_ravel[None, :]
        set_Bmat_stacked_torch.append(_to_complex_tensor(np.vstack([Bf, Bt])))

    # ── Output ───────────────────────────────────────────────────────────────
    file_name = (f"N_{N}+{plus_N}_used_z{used_data['zs']}"
                 f"_theta{used_data['thetas']}_cheat.h5")
    output_file = os.path.join(output_path, file_name)
    Path(output_path).mkdir(parents=True, exist_ok=True)

    print(f"Starting optimization for N={N} with {len(waists1)} waist(s).")

    with h5py.File(output_file, "w") as fh:

        results_ds = fh.create_dataset(
            "results", shape=(len(combined), num_rep), dtype=np.float64)
        coeffs_ds = fh.create_dataset(
            "coeffs", shape=(len(combined), num_rep, Nt + 1, Nt + 1),
            dtype=np.complex128)
        waists_ds = fh.create_dataset(
            "waists", shape=(len(combined), num_rep, 2), dtype=np.float64)

        for r in tqdm.tqdm(range(len(combined)), desc=f"N={N}"):
            img_ravel_t = torch.tensor(combined[r].ravel(), dtype=dtype,
                                       device=device)

            coeffs, results, waistest = [], [], []

            for k in range(num_rep):
                initial_guess = 0.01 * np.random.randn(2 * n_valid - 1)
                initial_guess /= np.linalg.norm(initial_guess)

                best_val, best_params, best_waist = np.inf, None, None

                for w in range(len(scales)):
                    result = minimize(
                        loss_np,
                        initial_guess,
                        args=(set_Bmat_stacked_torch[w], img_ravel_t, sigma),
                        method='L-BFGS-B',
                        jac=grad_np,
                        tol=1e-12,
                        options={'maxiter': 5000}
                    )
                    if result.fun < best_val:
                        best_val = result.fun
                        best_params = result.x
                        best_waist = (waists1[w], waists2[w])

                waistest.append(best_waist)
                results.append(best_val)
                coef_est = vector_to_matrix_np(best_params, valid, n_valid)
                coeffs.append(coef_est / np.sqrt(np.sum(np.abs(coef_est) ** 2)))

            results_ds[r] = np.array(results)
            coeffs_ds[r] = np.array(coeffs)
            waists_ds[r] = np.array(waistest)