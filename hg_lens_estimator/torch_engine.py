"""PyTorch side of the fit: device handling + autodiff loss/gradient that
scipy's L-BFGS-B can call."""
import numpy as np
import torch


def resolve_device(device="auto"):
    """'auto' -> cuda if available (NVIDIA CUDA or AMD ROCm builds), else cpu."""
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


class TorchEngine:
    def __init__(self, device="auto"):
        self.device = resolve_device(device)
        print(f"Using device: {self.device}")

    def to_basis_tensor(self, basis):
        """list/tuple of (n_pix, n_modes) arrays or a (K, n_pix, n_modes)
        array/tensor -> complex128 tensor (K, n_pix, n_modes) on device."""
        if isinstance(basis, (list, tuple)):
            basis = torch.stack([torch.as_tensor(b).to(torch.complex128) for b in basis])
        else:
            basis = torch.as_tensor(basis).to(torch.complex128)
        if basis.ndim != 3:
            raise ValueError(
                f"basis must have shape (n_transforms, n_pix, n_modes), got {tuple(basis.shape)}"
            )
        return basis.to(self.device)

    def to_data_tensor(self, data):
        return torch.as_tensor(data).to(device=self.device, dtype=torch.float64)

    def make_loss_and_grad(self, n_modes):
        """
        Parameter vector v (real, length 2*n_modes - 1):
            coef = v[:n_modes] + 1j * [0, v[n_modes:]]
        i.e. the imaginary part of the first mode is fixed to 0 (global phase).

        Returns f(v_np, basis, data, sigma) -> (loss, grad_np), for
        scipy.optimize.minimize(..., jac=True).
        basis: (K, n_pix, n_modes) complex tensor; data: (K, n_pix) real tensor,
        each row already normalised to max 1.
        """
        device = self.device

        def loss_and_grad(v_np, basis, data, sigma):
            v = torch.tensor(v_np, dtype=torch.float64, device=device, requires_grad=True)
            imag = torch.cat([v.new_zeros(1), v[n_modes:]])
            coef = torch.complex(v[:n_modes], imag)
            E = basis @ coef                                   # (K, n_pix)
            I = E.real ** 2 + E.imag ** 2
            pred = I / I.amax(dim=1, keepdim=True)             # each image normalised to max 1
            loss = 0.5 * (((data - pred) / sigma) ** 2).sum()
            loss.backward()
            return float(loss.detach().cpu()), v.grad.detach().cpu().numpy().astype(np.float64)

        return loss_and_grad
