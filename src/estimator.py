"""Maximum-likelihood estimation of Hermite-Gauss coefficients from images."""
from dataclasses import dataclass

import numpy as np
import torch
from scipy.optimize import minimize


@dataclass
class EstimationResult:
    coefs: np.ndarray      # (N+1, N+1) complex, unit-norm, C[0,0] real >= gauge fixed
    loss: float            # final negative log-likelihood
    success: bool          # scipy convergence flag
    n_iter: int
    message: str


class CoefficientEstimator:
    """Estimate HG coefficients c[n, m] (n + m <= N) from intensity images.

    Model: image k has field  E_k = sum_{n,m} c[n,m] * exp(i (phix_k n + phiy_k m)) * HG_nm,
    intensity I_k = |E_k|^2 / max(|E_k|^2), fitted to the (max-normalized) data with a
    Gaussian likelihood. The global phase is fixed by forcing c[0,0] to be real.

    Parameters
    ----------
    basis : provided by ``structured_optics.hg_basis``; indexable as basis[i][j] -> (H, W)
            array, for 0 <= i, j <= N (a (N+1, N+1, H, W) ndarray also works).
    N     : maximum mode order. Only modes with n + m <= N are fitted.
    sigma : noise standard deviation in the likelihood (only scales the loss).
    device: torch device; defaults to CUDA if available, else CPU.
    """

    def __init__(self, basis, N, *, sigma=1.0, device=None):
        self.N = int(N)
        self.sigma = float(sigma)
        self.device = torch.device(device) if device is not None else \
            torch.device("cuda" if torch.cuda.is_available() else "cpu")

        N = self.N
        cols = [np.asarray(basis[i][j]).ravel() for i in range(N + 1) for j in range(N + 1)]
        B = np.array(cols).T                                    # (n_pix, (N+1)^2)
        self.n_pix = B.shape[0]

        # index grids; flat index = n*(N+1) + m
        ni, mi = np.meshgrid(np.arange(N + 1), np.arange(N + 1), indexing="ij")
        valid = (ni + mi <= N).ravel()
        self._valid_idx = np.where(valid)[0]
        self.n_valid = len(self._valid_idx)
        self._n_vals = ni.ravel()[self._valid_idx].astype(float)
        self._m_vals = mi.ravel()[self._valid_idx].astype(float)

        # only keep columns of valid modes
        self._B = torch.tensor(B[:, self._valid_idx], dtype=torch.complex128, device=self.device)

    # ------------------------------------------------------------------ helpers
    def _params_to_coef(self, p):
        nv = self.n_valid
        real = p[:nv]
        imag = torch.cat([p.new_zeros(1), p[nv:]])      # first coef real -> fixes global phase
        return torch.complex(real, imag)

    def _loss(self, p, phase, data):
        c = self._params_to_coef(p)                      # (nv,)
        E = (c[None, :] * phase) @ self._B.T             # (n_img, n_pix)
        I = E.real ** 2 + E.imag ** 2
        I = I / I.max(dim=1, keepdim=True).values
        return 0.5 * (((data - I) / self.sigma) ** 2).sum()

    def _matrix_to_params(self, C):
        v = np.asarray(C, dtype=complex).ravel()[self._valid_idx]
        v = v * np.exp(-1j * np.angle(v[0]))             # rotate so first coef is real
        return np.concatenate([v.real, v[1:].imag])

    def _params_to_matrix(self, p):
        nv = self.n_valid
        v = p[:nv] + 1j * np.concatenate([[0.0], p[nv:]])
        C = np.zeros((self.N + 1) ** 2, dtype=complex)
        C[self._valid_idx] = v
        return C.reshape(self.N + 1, self.N + 1)

    # --------------------------------------------------------------------- API
    def estimate(self, images, phix, phiy, *, x0=None, rng=None,
                 normalize_images=True, maxiter=5000, tol=1e-12):
        """Estimate coefficients for one set of images.

        Parameters
        ----------
        images : array (N_IM, H, W); H*W must match the basis size.
        phix, phiy : sequences of length N_IM (one value per image).
        x0     : optional (N+1, N+1) complex initial guess. Random if omitted.
        rng    : ``np.random.Generator`` / seed for the random initial guess.
        normalize_images : divide each image by its max (as the model does).

        Returns
        -------
        EstimationResult (in RAM; saving is up to the caller).
        """
        images = np.asarray(images, dtype=float)
        if images.ndim == 2:
            images = images[None]
        if images.ndim != 3:
            raise ValueError("images must have shape (N_IM, H, W)")
        n_img = images.shape[0]
        if images[0].size != self.n_pix:
            raise ValueError(f"Image has {images[0].size} pixels but basis has {self.n_pix}.")

        phix = np.asarray(phix, dtype=float).reshape(-1)
        phiy = np.asarray(phiy, dtype=float).reshape(-1)
        if len(phix) != n_img or len(phiy) != n_img:
            raise ValueError("phix and phiy must each have one value per image.")

        data = images.reshape(n_img, -1)
        if normalize_images:
            data = data / data.max(axis=1, keepdims=True)

        phase_np = np.exp(1j * (phix[:, None] * self._n_vals[None, :]
                                + phiy[:, None] * self._m_vals[None, :]))
        phase = torch.tensor(phase_np, dtype=torch.complex128, device=self.device)
        data_t = torch.tensor(data, dtype=torch.float64, device=self.device)

        if x0 is None:
            rng = np.random.default_rng(rng)
            p0 = 0.01 * rng.standard_normal(2 * self.n_valid - 1)
            p0 /= np.linalg.norm(p0)
        else:
            p0 = self._matrix_to_params(x0)

        def fun(v):
            p = torch.tensor(v, dtype=torch.float64, device=self.device, requires_grad=True)
            loss = self._loss(p, phase, data_t)
            loss.backward()
            return float(loss.detach().cpu()), p.grad.detach().cpu().numpy()

        res = minimize(fun, p0, jac=True, method="L-BFGS-B", tol=tol,
                       options={"maxiter": maxiter})

        C = self._params_to_matrix(res.x)
        C /= np.sqrt(np.sum(np.abs(C) ** 2))
        return EstimationResult(coefs=C, loss=float(res.fun), success=bool(res.success),
                                n_iter=int(res.nit), message=str(res.message))
