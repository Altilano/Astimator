import numpy as np
from .base import Backend


class TorchBackend(Backend):
    name = "torch"

    def __init__(self):
        import torch
        self._torch = torch
        self.device = None
        self.dtype = torch.float64

    def setup_device(self, device: str) -> None:
        torch = self._torch
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            # NOTE: on an AMD GPU with a ROCm build of torch installed,
            # torch.cuda.is_available() / device("cuda") is still the
            # correct call -- ROCm is exposed through the same 'cuda' API.
            self.device = torch.device(device)
        print(f"Using device: {self.device}")

    def to_device_stack(self, B_stacked_np_list):
        torch = self._torch
        return [torch.tensor(B, dtype=torch.complex128, device=self.device)
                for B in B_stacked_np_list]

    def to_device_data(self, data_np):
        torch = self._torch
        return torch.tensor(data_np, dtype=self.dtype, device=self.device)

    def make_loss_grad(self, n_valid, valid_indices, N, n_transforms):
        torch = self._torch
        total = (N + 1) * (N + 1)
        device = self.device
        dtype = self.dtype
        valid_idx_t = torch.tensor(valid_indices, dtype=torch.long, device=device)

        def vector_to_coef(v):
            real_part = v[:n_valid]
            imag_part = torch.cat([v.new_zeros(1), v[n_valid:]])
            coef_real = torch.zeros(total, dtype=dtype, device=device).scatter(0, valid_idx_t, real_part)
            coef_imag = torch.zeros(total, dtype=dtype, device=device).scatter(0, valid_idx_t, imag_part)
            return torch.view_as_complex(torch.stack([coef_real, coef_imag], dim=-1).contiguous())

        def nll(params, B_stacked, data, sigma):
            coef = vector_to_coef(params)
            E_all = B_stacked @ coef
            n_pix = E_all.shape[0] // n_transforms

            segs = []
            for k in range(n_transforms):
                seg = E_all[k * n_pix:(k + 1) * n_pix]
                I = seg.abs() ** 2
                segs.append(I / I.max())
            predicted = torch.cat(segs).real
            residual = data - predicted
            return 0.5 * ((residual / sigma) ** 2).sum()

        def loss_and_grad_np(params_np, B_stacked, data, sigma):
            params_t = torch.tensor(params_np, dtype=dtype, device=device, requires_grad=True)
            loss = nll(params_t, B_stacked, data, sigma)
            loss.backward()
            grad = params_t.grad.detach().cpu().numpy().astype(np.float64)
            return float(loss.detach().cpu()), grad

        def loss_np(params, B_stacked, data, sigma):
            val, _ = loss_and_grad_np(params, B_stacked, data, sigma)
            return val

        def grad_np(params, B_stacked, data, sigma):
            _, g = loss_and_grad_np(params, B_stacked, data, sigma)
            return g

        return loss_np, grad_np
