"""
Minimal driver: images + basis in, coefficients out.

  images : the K pictures of one sample v-stacked -> array/tensor (K*H, W)
           (or (n_samples, K*H, W) to fit many samples at once)
  basis  : (K, n_pix, n_modes) tensor, or a list of K (n_pix, n_modes) tensors;
           basis[k] is the basis already transformed for picture k
"""
import numpy as np
from hg_lens_estimator import (
    EstimationConfig, Estimator,
    make_hg_basis, transform_hg_basis, transform_basis, coeffs_to_matrix,
)

N = 12                                  # max HG order
H, W = 128, 128                         # size of ONE picture
pixel_size, waist = 6.45e-6, 2e-4

# One (phi_x, phi_y) per picture -- the values you used to call ax, ay,
# e.g. from your calculate_ax_ay(...) for each picture's lens position/tilt.
phis = [(0.0, 0.0), (0.42, -0.17), (-0.31, 0.25)]

# 1) fast HG basis, 2) transformed version for every picture
hg = make_hg_basis(N, (H, W), pixel_size, waist, device="auto")
basis = transform_hg_basis(hg, phis)                      # (3, H*W, n_modes)

# --- if your basis is NOT HG (any (H*W, M) array on the same pixel grid): ---
# basis = transform_basis(my_basis, hg, phis)             # (3, H*W, M)
# --- or if you already have the transformed bases: ---
# basis = [basis_pic1, basis_pic2, basis_pic3]            # each (H*W, M)

# images: your 3 pictures, each already background-subtracted / oriented the
# way the basis expects, v-stacked -> (3*H, W)
images = np.load("images.npy")

est = Estimator(basis, EstimationConfig(device="auto", n_restarts=10))
result = est.estimate(images)

print("best loss:", result["results"].min())
coef_matrix = coeffs_to_matrix(result["best"], hg.ni, hg.mi, N)   # [n, m] view, shape (N+1, N+1)
