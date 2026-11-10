"""
Minimal driver: images + basis in, coefficients out.

  images : the K pictures of one sample v-stacked -> array/tensor (K*H, W)
           (or (n_samples, K*H, W) to fit many samples at once)
  basis  : (K, n_pix, n_modes) tensor, or a list of K (n_pix, n_modes) tensors;
           basis[k] is the basis already transformed for picture k
"""
import numpy as np
from Astimator import (
    EstimationConfig, Estimator,
    make_hg_basis, transform_hg_basis, transform_basis, coeffs_to_matrix,
    tilted_lens_focal_lengths, calculate_phix_phiy,
)

N = 12                                  # max HG order
H, W = 128, 128                         # size of ONE picture
pixel_size, waist = 6.45e-6, 2e-4

# One (phi_x, phi_y) per picture -- the values you used to call ax, ay,
# e.g. computed for each picture's lens position/tilt:
#   fx, fy = tilted_lens_focal_lengths('biconvex', theta_deg, f0=0.3, n=1.5151)
#   phi = calculate_phix_phiy(fx, fy, waistx, waisty, lamb, z0, z)
# Physical phase by default (-pi/2). If the pictures are mirrored w.r.t. the
# basis (e.g. after a 4F system), pass flip_coordinates=True (+pi/2).
phis = [(0.0, 0.0), (0.42, -0.17), (-0.31, 0.25)]

# 1) fast HG basis, 2) transformed version for every picture
hgs = make_hg_basis(N, (H, W), pixel_size, waist, device="auto")
basis = transform_hg_basis(hgs, phis)                      # (3, H*W, n_modes)

# --- if your basis is NOT HG (any (H*W, M) array on the same pixel grid): ---
# basis = transform_basis(my_basis, hg, phis)             # (3, H*W, M)
# --- or if you already have the transformed bases: ---
# basis = [basis_pic1, basis_pic2, basis_pic3]            # each (H*W, M)

# images: your 3 pictures, each already background-subtracted / oriented the
# way the basis expects, v-stacked -> (3*H, W)
images = np.load("images.npy")
config = EstimationConfig(device="auto", n_restarts=10)
est = Estimator(basis, config)
result = est.estimate(images)

print("best loss:", result["results"].min())
coef_matrix = coeffs_to_matrix(result["best"], hgs.ns, hgs.ms, N)   # [n, m] view, shape (N+1, N+1)
