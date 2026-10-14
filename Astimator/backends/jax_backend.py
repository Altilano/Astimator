import numpy as np
from .base import Backend


class JaxBackend(Backend):
    name = "jax"

    def __init__(self):
        import jax
        jax.config.update("jax_enable_x64", True)
        self._jax = jax

    def setup_device(self, device: str) -> None:
        jax = self._jax
        print("JAX backend:", jax.default_backend())
        print("JAX devices:", jax.devices())

    def to_device_stack(self, B_stacked_np_list):
        import jax.numpy as jnp
        return [jnp.array(B) for B in B_stacked_np_list]

    def to_device_data(self, data_np):
        import jax.numpy as jnp
        return jnp.array(data_np)

    def make_loss_grad(self, n_valid, valid_indices, n_modes, n_transforms):
        import jax
        import jax.numpy as jnp

        total = n_modes
        valid_idx = jnp.array(valid_indices)

        def vector_to_coef(v):
            real_part = v[:n_valid]
            imag_part = jnp.concatenate([jnp.zeros(1), v[n_valid:]])
            coef_real = jnp.zeros(total).at[valid_idx].set(real_part)
            coef_imag = jnp.zeros(total).at[valid_idx].set(imag_part)
            return coef_real + 1j * coef_imag

        def nll(params, B_stacked, data, sigma):
            coef = vector_to_coef(params)
            E_all = B_stacked @ coef
            n_pix = E_all.shape[0] // n_transforms

            segs = []
            for k in range(n_transforms):
                seg = E_all[k * n_pix:(k + 1) * n_pix]
                I = jnp.abs(seg) ** 2
                segs.append(I / I.max())
            predicted = jnp.concatenate(segs)
            residual = data - predicted
            return 0.5 * jnp.sum((residual / sigma) ** 2)

        loss_jit = jax.jit(nll)
        grad_jit = jax.jit(jax.grad(nll))

        def loss_np(params, B_stacked, data, sigma):
            return float(loss_jit(params, B_stacked, data, sigma))

        def grad_np(params, B_stacked, data, sigma):
            return np.array(grad_jit(params, B_stacked, data, sigma), dtype=np.float64)

        return loss_np, grad_np
