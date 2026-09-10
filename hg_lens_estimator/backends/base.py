from abc import ABC, abstractmethod


class Backend(ABC):
    """
    Every backend just needs to:
      1. put numpy arrays on the chosen device,
      2. hand back scipy-compatible (loss, grad) functions for L-BFGS-B.

    Everything else (data loading, HG basis, optimisation loop, saving)
    is backend-agnostic and lives in pipeline.py.
    """

    name = "base"

    @abstractmethod
    def setup_device(self, device: str) -> None:
        ...

    @abstractmethod
    def to_device_stack(self, B_stacked_np_list):
        """list[np.ndarray complex] -> list[backend tensor] on device"""
        ...

    @abstractmethod
    def to_device_data(self, data_np):
        """np.ndarray float64 -> backend tensor on device"""
        ...

    @abstractmethod
    def make_loss_grad(self, n_valid: int, valid_indices, N: int, n_transforms: int):
        """
        Returns (loss_np, grad_np), both with signature
            f(params: np.ndarray, B_stacked, data, sigma: float) -> float | np.ndarray
        i.e. directly usable as fun= / jac= in scipy.optimize.minimize.
        """
        ...


def get_backend(name: str) -> Backend:
    if name == "jax":
        from .jax_backend import JaxBackend
        return JaxBackend()
    elif name == "torch":
        from .torch_backend import TorchBackend
        return TorchBackend()
    else:
        raise ValueError(f"Unknown backend '{name}'. Use 'jax' or 'torch'.")
