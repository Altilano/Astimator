import numpy as np
from scipy.optimize import minimize

from .backends import get_backend
from .basis import build_space, build_hg_grids, build_basis_matrix, build_stacked_basis, valid_mode_mask
from .optics import compute_transform_phases
from .parametrization import vector_to_matrix_np
from .data import Dataset


class Estimator:
    """
    Usage:
        cfg = EstimationConfig(backend="torch", device="auto", n_transforms=3)
        est = Estimator(cfg)
        est.prepare(nix, im_size)          # build HG basis / stacked B matrices once
        result = est.estimate_image(image_index=0)
        est.run(range(180))                # loop + save to disk, like the notebooks did
    """

    def __init__(self, config):
        self.config = config
        self.backend = get_backend(config.backend)
        self.backend.setup_device(config.device)
        self.dataset = Dataset(config)

        self._B_stacked_device = None   # list over waist scan, backend tensors
        self.ni = self.mi = self.valid = None
        self.n_valid = self.valid_indices = None
        self.loss_np = self.grad_np = None

    def prepare(self, nix, im_size):
        cfg = self.config
        active = cfg.active_transforms()
        if len(active) != cfg.n_transforms:
            raise ValueError(
                f"n_transforms={cfg.n_transforms} but active_transforms() "
                f"returned {len(active)} entries. Check `transforms` in your config."
            )

        b1 = build_space(nix, im_size, cfg.waist)
        self.ni, self.mi = build_hg_grids(b1, cfg.N)
        self.valid, self.n_valid, self.valid_indices = valid_mode_mask(self.ni, self.mi, cfg.N)

        phases = compute_transform_phases(
            active, self.ni, self.mi,
            waist=cfg.waist, lamb=cfg.lamb, z0=cfg.z0, f=cfg.f,
        )
        transform_names = [t.name for t in active]

        B_stacked_np = []
        for w in cfg.waists():
            B = build_basis_matrix(b1, cfg.N, w)
            B_stacked_np.append(build_stacked_basis(B, phases, transform_names))

        self._B_stacked_device = self.backend.to_device_stack(B_stacked_np)
        self.loss_np, self.grad_np = self.backend.make_loss_grad(
            self.n_valid, self.valid_indices, cfg.N, cfg.n_transforms
        )
        self.b1 = b1
        return self

    def estimate_image(self, image_index):
        cfg = self.config
        sample = self.dataset.get_image(image_index, cfg.n_transforms)
        data_device = self.backend.to_device_data(sample.img_ravel)

        coeffs, results, waistest = [], [], []
        n_params = self.n_valid

        for _ in range(cfg.n_restarts):
            initial_guess = 0.01 * np.random.randn(2 * n_params - 1)

            best_val, best_params, best_waist = np.inf, None, None
            for w, B_stacked in zip(cfg.waists(), self._B_stacked_device):
                res = minimize(
                    self.loss_np,
                    initial_guess,
                    args=(B_stacked, data_device, cfg.sigma),
                    method="L-BFGS-B",
                    jac=self.grad_np,
                    tol=cfg.tol,
                    options={"maxiter": cfg.maxiter},
                )
                if res.fun < best_val:
                    best_val, best_params, best_waist = res.fun, res.x, w

            waistest.append(best_waist)
            results.append(best_val)
            coef_est = vector_to_matrix_np(best_params, cfg.N, self.valid, self.n_valid)
            coef_est_norm = coef_est / np.sqrt(np.sum(np.abs(coef_est) ** 2))
            coeffs.append(coef_est_norm)

        return {
            "results": np.array(results),
            "coeffs": np.array(coeffs),
            "waists": np.array(waistest),
        }

    def run(self, image_indices, save=True):
        all_out = {}
        for r in image_indices:
            out = self.estimate_image(r)
            all_out[r] = out
            if save:
                path = self.config.output_pattern.format(r=r + 1)
                np.savez(path, **out)
        return all_out
