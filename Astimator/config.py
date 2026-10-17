"""Optimisation settings. Everything about *what* you fit (images, basis) is
passed to Estimator directly, not through the config."""
from dataclasses import dataclass


@dataclass
class EstimationConfig:
    device: str = "auto"     # "auto" | "cpu" | "cuda"  (Cuda or AMD/ROCm torch builds use "cuda")
    n_restarts: int = 10     # random restarts of L-BFGS-B per image
    sigma: float = 1.0       # noise scale in the least-squares loss
    maxiter: int = 5000
    tol: float = 1e-12

    def __post_init__(self):
        if self.n_restarts < 1:
            raise ValueError(f"n_restarts must be >= 1, got {self.n_restarts}")
