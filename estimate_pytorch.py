from structured_optics import *
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import tqdm
import h5py
from utils_lens import *
from utils_img import *
import os
from pathlib import Path
import torch


# ── Device setup ─────────────────────────────────────────────────────────────
# PyTorch will automatically use CUDA if available (native Windows support, no WSL needed)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dtype  = torch.float64   # equivalent to jax_enable_x64
print(f"Using device: {device}")


def _to_complex_tensor(arr):
    """Convert a numpy complex128 array to a torch complex128 tensor on device."""
    return torch.tensor(arr, dtype=torch.complex128, device=device)

def vector_to_coef_torch(v: torch.Tensor) -> torch.Tensor:
    """
    Convert a real parameter vector to a complex coefficient vector of
    length (N+1)^2, placing values at valid_indices and zeros elsewhere.

    v layout:  [ real_part (n_valid) | imag_part (n_valid-1) ]
    The first coefficient is forced to be real (fixes global phase).
    """
    real_part = v[:n_valid]                                         # (n_valid,)
    imag_part = torch.cat([v.new_zeros(1), v[n_valid:]])            # (n_valid,)

    coef_real = torch.zeros(total, dtype=dtype, device=device)
    coef_imag = torch.zeros(total, dtype=dtype, device=device)
    coef_real = coef_real.scatter(0, valid_indices_t, real_part)
    coef_imag = coef_imag.scatter(0, valid_indices_t, imag_part)

    # Build complex tensor  (torch.complex requires float, not double, workaround below)
    return torch.view_as_complex(torch.stack([coef_real, coef_imag], dim=-1).to(torch.float64).contiguous())


# ── Loss function  (pure PyTorch) ─────────────────────────────────────────────
def negative_log_likelihood_torch(params: torch.Tensor,
                                B_stacked: torch.Tensor,
                                data: torch.Tensor,
                                sigma: float) -> torch.Tensor:
    coef  = vector_to_coef_torch(params)           # ((N+1)^2,) complex
    E_all = B_stacked @ coef                       # (2*n_pix,) complex
    n_pix = E_all.shape[0] // 2

    def intensity_normalized(E_seg):
        I = E_seg.abs() ** 2
        return I / I.max()

    I0 = intensity_normalized(E_all[:n_pix])
    I1 = intensity_normalized(E_all[n_pix:2 * n_pix])
    #I2 = intensity_normalized(E_all[2 * n_pix:])

    predicted = torch.cat([I0, I1]).real       # imaginary part is ~0
    residual  = data - predicted
    return 0.5 * ((residual / sigma) ** 2).sum()


# ── Scipy-compatible wrappers ─────────────────────────────────────────────────
def loss_and_grad_np(params_np: np.ndarray,
                    B_stacked: torch.Tensor,
                    data: torch.Tensor,
                    sigma: float):
    """
    Returns (loss_scalar, gradient_np) in one forward+backward pass.
    Using loss_and_grad together is more efficient than two separate calls.
    """
    params_t = torch.tensor(params_np, dtype=dtype, device=device, requires_grad=True)
    loss = negative_log_likelihood_torch(params_t, B_stacked, data, sigma)
    loss.backward()
    grad = params_t.grad.detach().cpu().numpy().astype(np.float64)
    return float(loss.detach().cpu()), grad


# Separate wrappers expected by scipy's minimize(jac=...) interface
def loss_np(params, B_stacked, data, sigma):
    val, _ = loss_and_grad_np(params, B_stacked, data, sigma)
    return val

def grad_np(params, B_stacked, data, sigma):
    _, g = loss_and_grad_np(params, B_stacked, data, sigma)
    return g



def vector_to_matrix_np(v, N):
    real_part        = v[:n_valid]
    imag_part        = np.zeros(n_valid)
    imag_part[1:]    = v[n_valid:]
    A_np[valid]      = real_part + 1j * imag_part
    return A_np.copy()

def matrix_to_vector(A):
    valid_elements = A[valid]
    return np.concatenate([valid_elements.real, valid_elements[1:].imag])













for N in range(1,11):

    plus_N = 0

    pixel_size = 2.4e-6
    res = 232
    folder = "biconvex_stretched_gaussref_equalsize"

    z1 = 30e-2
    theta1 = 0
    with h5py.File(f"data_cheat/{folder}/random_modes_N{N}_f0.300_z{z1:.3f}_theta{theta1:.3f}.h5", 'r') as file:
        imgs1 = file['images']
        imgs1 = np.asarray(imgs1)
        imgs1 = imgs1 / imgs1.max(axis=(1, 2), keepdims=True)
        #imgs1 = np.flip(imgs1, axis=2)

    z2 = 22e-2
    theta2 = 20
    with h5py.File(f"data_cheat/{folder}/random_modes_N{N}_f0.300_z{z2:.3f}_theta{theta2:.3f}.h5", 'r') as file:
        imgs2 = file['images']
        imgs2 = np.asarray(imgs2)
        imgs2 = imgs2 / imgs2.max(axis=(1, 2), keepdims=True)
        #imgs2 = np.flip(imgs2, axis=2)

    """z3 = 26e-2
    theta3 = 0
    with h5py.File(f"data_cheat/{folder}/random_modes_N{N}_f0.300_z{z3:.3f}_theta{theta3:.3f}.h5", 'r') as file:
        imgs3 = file['images']
        imgs3 = np.asarray(imgs3)
        #imgs3 = np.flip(imgs3, axis=2)"""

    combined = np.concatenate([imgs1, imgs2], axis=1)

    with h5py.File(f"coefs_modes/random_modes_N{N}.h5", 'r') as file_coefs:
        coefs = file_coefs['coefs']
        coefs = np.asarray(coefs)

    print(f"Loaded {combined.shape[0]} images, coefficient sets for N={N}.")

    used_data = dict(zs= [z1, z2], thetas=[theta1, theta2])

    wx1, wy1 = fit_beam_widths(imgs1[0], pixel_size=pixel_size)

    wx2, wy2 = fit_beam_widths(imgs2[0], pixel_size=pixel_size)

        # ── Per-image waists: average of the x and y fitted widths ───────────────────
    w1 = np.sqrt(wx1 * wy1)
    w2 = np.sqrt(wx2 * wy2)
    print("Estimated beam waists:")
    print(f"Image 1: wx={wx1}, wy={wy1} -> w1={w1}")
    print(f"Image 2: wx={wx2}, wy={wy2} -> w2={w2}")

    lamb = 633e-9
    f = 30e-2
    n = 1.5151
    waistx = waisty = 700e-6
    z0 = 118e-2

    fx1, fy1 = tilted_lens_focal_lengths('biconvex', theta1, f0 = f, n=n)
    phix1, phiy1 = calculate_phix_phiy(fx1, fy1, waistx, waisty, lamb, z0, z1)

    fx2, fy2 = tilted_lens_focal_lengths('biconvex', theta2, f0 = f, n=n)
    phix2, phiy2 = calculate_phix_phiy(fx2, fy2, waistx, waisty, lamb, z0, z2)

    print("Calculated phi_x and phi_y.")

    # ── One Beam object per image ────────────────────────────────────────────────
    beam_kwargs = dict(nix = res*pixel_size/2, niy = res*pixel_size/2,
                       Dx = res, Dy = res, lamb = 633e-9)
    beam1 = Beam(**beam_kwargs, waist=w1)
    beam2 = Beam(**beam_kwargs, waist=w2)
    beam1.hg(0, 0)
    beam2.hg(0, 0)

    # index grids depend only on N, so either beam works
    ni, mi, overlap = hg_proj(beam1, N+plus_N)

    # Waist scan: both waists are scaled together by the same factor
    scales  = np.array([1.0])     # same ±5% range as before
    waists1 = w1 * scales
    waists2 = w2 * scales

    def build_Bmat(beam, waist):
        beam.waist = waist
        basis = hg_basis(beam, N+plus_N, norm1=True)
        return np.array([basis[i, j].ravel()
                         for i in range(N + 1 + plus_N)
                         for j in range(N + 1 + plus_N)]).T

    set_Bmat1 = [build_Bmat(beam1, w) for w in waists1]
    set_Bmat2 = [build_Bmat(beam2, w) for w in waists2]

    print('Built basis matrices for all waists.')

    
    # ── Valid-index bookkeeping ───────────────────────────────────────────────────
    valid        = ni + mi <= N + plus_N
    n_valid      = int(np.count_nonzero(valid))
    n_imag       = n_valid - 1

    valid_flat    = valid.ravel()                       # shape: ((N+1)^2,)
    valid_indices = np.where(valid_flat)[0]
    valid_indices_t = torch.tensor(valid_indices, dtype=torch.long, device=device)

    # ── Phase vectors ─────────────────────────────────────────────────────────────
    forw_ravel = np.exp(1j * (phix1 * ni + phiy1 * mi)).ravel()
    tilt_ravel = np.exp(1j * (phix2 * ni + phiy2 * mi)).ravel()
    #gouy_ravel = np.exp(1j * (phix3 * ni + phiy3 * mi)).ravel()

    # ── Stacked phase-shifted B matrices  (moved to device) ──────────────────────

    set_Bmat_stacked_torch = []
    for k in range(len(scales)):
        Bf = set_Bmat1[k] * forw_ravel[None, :]    # image 1: basis built with w1*scale
        Bt = set_Bmat2[k] * tilt_ravel[None, :]    # image 2: basis built with w2*scale
        B_stacked = np.vstack([Bf, Bt])
        set_Bmat_stacked_torch.append(_to_complex_tensor(B_stacked))


        
    # ── vector_to_coef  (pure PyTorch, differentiable) ───────────────────────────
    total = (N + 1 + plus_N) * (N + 1 + plus_N)

    # ── Numpy helpers (post-processing) ──────────────────────────────────────────
    A_np = np.zeros((N + 1 + plus_N, N + 1 + plus_N), dtype=complex)


        
    # ── Optimization loop ─────────────────────────────────────────────────────────
    sigma          = 1.0
    output_path    = f"results/waist_difbasis/{folder}"
    file_name      = f"N_{N}+{plus_N}_used_z{used_data['zs']}_theta{used_data['thetas']}_cheat.h5"
    output_file    = os.path.join(output_path, file_name)


    folderi = Path(f"C:/Users/altil/Documents/Python/tilted_lens/{output_path}")
    folderi.mkdir(exist_ok=True)

    print("Created output folder:", folderi)

    print(f"Starting optimization for N={N} with {len(waists1)} waist(s).")
        
    num_rep = 5

    with h5py.File(output_file, "w") as f:

        results_ds = f.create_dataset(
            "results",
            shape=(len(combined), num_rep),
            dtype=np.float64
        )

        coeffs_ds = f.create_dataset(
            "coeffs",
            shape=(len(combined), num_rep, N + 1 + plus_N, N + 1 + plus_N),
            dtype=np.complex128
        )

        waists_ds = f.create_dataset(
            "waists",
            shape=(len(combined), num_rep, 2),     # [:, :, 0] = waist img1, [:, :, 1] = waist img2
            dtype=np.float64
        )

        for r in tqdm.tqdm(range(len(combined))):
            img_ravel = combined[r].ravel()

            img_ravel_t = torch.tensor(img_ravel, dtype=dtype, device=device)

            coeffs = []
            results = []
            waistest = []

            for k in range(num_rep):
                N_params      = n_valid
                initial_guess = 0.01 * np.random.randn(2 * N_params - 1)
                initial_guess /= np.linalg.norm(initial_guess)

                best_val    = np.inf
                best_params = None
                best_waist  = None
                best_index  = None


                for w in range(len(scales)):
                    B_stacked = set_Bmat_stacked_torch[w]
                    result = minimize(
                        loss_np,
                        initial_guess,
                        args=(B_stacked, img_ravel_t, sigma),
                        method='L-BFGS-B',
                        jac=grad_np,          # exact PyTorch gradient via autograd
                        tol=1e-12,
                        options={'maxiter': 5000}
                    )       # unchanged

                    if result.fun < best_val:
                        best_val    = result.fun
                        best_params = result.x
                        best_waist  = (waists1[w], waists2[w])
                        best_index  = w

                waistest.append(best_waist)
                results.append(best_val)
                coef_est = vector_to_matrix_np(best_params, N+plus_N)
                coef_est_norm = coef_est / np.sqrt(np.sum(np.abs(coef_est) ** 2))
                coeffs.append(coef_est_norm)


            results_ds[r] = np.array(results)
            coeffs_ds[r] = np.array(coeffs)
            waists_ds[r] = np.array(waistest)

