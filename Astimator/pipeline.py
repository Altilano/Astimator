import numpy as np
from scipy.optimize import minimize

from .backends import get_backend
from .parametrization import vector_to_coef_np, coef_to_vector_np


class Estimator:
    """
    Usage:
        cfg = EstimationConfig(backend="torch", device="auto", n_transforms=3)

        # any BasisProvider: PrecomputedBasisProvider, TransformBasisProvider,
        # or the HG convenience one from hg_defaults.hg_transform_provider(...)
        basis_provider = ...

        # any DataProvider: NpzDataset, H5Dataset, ArrayDataProvider, CallableDataProvider
        data_provider = ...

        est = Estimator(cfg, basis_provider, data_provider)
        est.prepare()
        result = est.estimate_image(image_index=0)
        est.run(range(180))
    """

    def __init__(self, config, basis_provider, data_provider):
        self.config = config
        self.basis_provider = basis_provider
        self.data_provider = data_provider

        self.backend = get_backend(config.backend)
        self.backend.setup_device(config.device)

        self._B_stacked_device = None   # one entry per config.param_scan value
        self.n_modes = None
        self.valid_indices = None
        self.n_valid = None
        self.loss_np = self.grad_np = None

    def prepare(self):
        cfg = self.config
        active = cfg.active_transforms()
        if len(active) != cfg.n_transforms:
            raise ValueError(
                f"n_transforms={cfg.n_transforms} but active_transforms() "
                f"returned {len(active)} entries. Check `transforms` in your config."
            )

        B_stacked_np = [self.basis_provider.build_stacked_basis(p, active)
                        for p in cfg.param_scan]

        n_modes = B_stacked_np[0].shape[1]
        for B in B_stacked_np[1:]:
            if B.shape[1] != n_modes:
                raise ValueError("All basis matrices in param_scan must have the same n_modes.")
        self.n_modes = n_modes

        if cfg.valid_indices is not None:
            self.valid_indices = np.asarray(cfg.valid_indices)
        else:
            self.valid_indices = np.arange(n_modes)
        self.n_valid = len(self.valid_indices)

        self._B_stacked_device = self.backend.to_device_stack(B_stacked_np)
        self.loss_np, self.grad_np = self.backend.make_loss_grad(
            self.n_valid, self.valid_indices, self.n_modes, cfg.n_transforms
        )
        return self

    def estimate_image(self, image_index):
        cfg = self.config
        img_ravel = self.data_provider.get_image(image_index, cfg.n_transforms)
        data_device = self.backend.to_device_data(img_ravel)

        coeffs, results, params_used = [], [], []
        n_params = self.n_valid

        for _ in range(cfg.n_restarts):
            initial_guess = 0.01 * np.random.randn(2 * n_params - 1)

            best_val, best_params, best_scan_param = np.inf, None, None
            for p, B_stacked in zip(cfg.param_scan, self._B_stacked_device):
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
                    best_val, best_params, best_scan_param = res.fun, res.x, p

            params_used.append(best_scan_param)
            results.append(best_val)
            coef = vector_to_coef_np(best_params, self.n_modes, self.valid_indices, self.n_valid)
            coef_norm = coef / np.sqrt(np.sum(np.abs(coef) ** 2))
            coeffs.append(coef_norm)

        return {
            "results": np.array(results),
            "coeffs": np.array(coeffs),          # flat, shape (n_restarts, n_modes)
            "param_scan_used": np.array(params_used, dtype=object),
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
