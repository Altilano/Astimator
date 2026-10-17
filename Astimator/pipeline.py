import numpy as np
import torch
from scipy.optimize import minimize

from .config import EstimationConfig
from .torch_engine import TorchEngine


class Estimator:
    """
    Fit complex mode coefficients so that, for every image k,
        |basis[k] @ coef|^2 / max  ~  images[k] / max

    basis:  list of K tensors/arrays (n_pix, n_modes), or one tensor
            (K, n_pix, n_modes). basis[k] is the basis *already transformed*
            for image k (see hg_basis.make_hg_basis + lens_transform).
    images: passed to estimate(); the K images of one sample v-stacked, i.e.
            shape (K*H, W) with H*W == n_pix. A 3D array (n_samples, K*H, W)
            fits every sample.
    """

    def __init__(self, basis, config=None):
        self.config = config or EstimationConfig()
        self.engine = TorchEngine(self.config.device)
        self.basis = self.engine.to_basis_tensor(basis)
        self.n_transforms, self.n_pix, self.n_modes = self.basis.shape
        if self.n_transforms < 2:
            raise ValueError("Need at least 2 images (basis.shape[0] >= 2).")
        self._loss_and_grad = self.engine.make_loss_and_grad(self.n_modes)

    def _split_images(self, image):
        """One v-stacked sample (K*H, W) -> (K, n_pix) tensor, each row normalised to max 1."""
        img = self.engine.to_data_tensor(image)
        if img.ndim != 2:
            raise ValueError(f"expected a 2D v-stacked image (K*H, W), got shape {tuple(img.shape)}")
        K = self.n_transforms
        if img.shape[0] % K != 0:
            raise ValueError(f"image has {img.shape[0]} rows, not divisible by n_transforms={K}")
        data = img.reshape(K, -1)
        if data.shape[1] != self.n_pix:
            raise ValueError(
                f"each of the {K} sub-images has {data.shape[1]} pixels but the basis has "
                f"n_pix={self.n_pix}. Check H*W of your images vs the basis shape."
            )
        return data / data.amax(dim=1, keepdim=True)

    def estimate(self, images):
        """2D input -> dict for that sample. 3D input -> list of dicts."""
        arr = torch.as_tensor(images)
        if arr.ndim == 3:
            return [self._estimate_one(a) for a in arr]
        return self._estimate_one(arr)

    def _estimate_one(self, image):
        cfg, M = self.config, self.n_modes
        data = self._split_images(image)

        costs, coeffs = [], []
        for _ in range(cfg.n_restarts):
            x0 = 0.01 * np.random.randn(2 * M - 1)
            res = minimize(
                self._loss_and_grad, x0,
                args=(self.basis, data, cfg.sigma),
                method="L-BFGS-B", jac=True, tol=cfg.tol,
                options={"maxiter": cfg.maxiter},
            )
            coef = res.x[:M] + 1j * np.concatenate([[0.0], res.x[M:]])
            coeffs.append(coef / np.linalg.norm(coef))
            costs.append(res.fun)

        costs, coeffs = np.array(costs), np.array(coeffs)
        return {
            "results": costs,                       # (n_restarts,) final loss of each restart
            "coeffs": coeffs,                       # (n_restarts, n_modes), unit norm
            "best": coeffs[np.argmin(costs)],       # (n_modes,) coefficients of the lowest-loss restart
        }
