# Astimator

The ASTIgmatic estiMATOR. Estimate complex mode coefficients from several intensity pictures of the same
beam, each taken after a different transformation. PyTorch only.

```
images (K*H, W)  ─┐
                  ├─> Estimator ──> coefficients
basis (K, P, M)  ─┘      (L-BFGS-B, torch autodiff, random restarts)
```

## Inputs

- **images**: the K pictures of one sample v-stacked, shape `(K*H, W)` (array or
  tensor). A 3D input `(n_samples, K*H, W)` fits every sample. Each picture is
  normalised to max 1 internally; background subtraction / flips / cropping are
  up to you, beforehand.
- **basis**: tensor `(K, n_pix, n_modes)` or a list of K tensors `(n_pix, n_modes)`.
  `basis[k]` is the basis already transformed for picture k, `n_pix = H*W`
  (C-order ravel of an `(H, W)` picture).

```python
est = Estimator(basis, EstimationConfig(device="auto", n_restarts=10))
result = est.estimate(images)     # dict: results, coeffs (n_restarts, M), best (M,)
```

## Helpers

- `make_hg_basis(N, (H, W), pixel_size, waist)` — fast HG basis. Separable 1D
  recurrence + one batched product on the device; only modes with n+m<=N are built.
  Returns `HGBasis(basis, ni, mi, N, shape)`.
- `transform_hg_basis(hg, phis)` — applies `exp(i(phi_x n + phi_y m))` per picture
  → `(K, n_pix, n_modes)`. `phis` is K pairs `(phi_x, phi_y)`.
- `transform_basis(my_basis, hg, phis)` — same for a non-HG basis: project onto
  `hg`, apply the phases, recombine. `hg`'s order must be high enough to
  represent `my_basis`.
- `coeffs_to_matrix(coef, ni, mi, N)` — flat coefficients → `[n, m]` matrix.

## Conventions to verify once against your beam.py

mode (n, m): n along x (columns), m along y (rows) · field ∝ H_n(√2x/w)e^(−x²/w²)/√(2ⁿn!) ·
grid centred at ((H−1)/2, (W−1)/2) (override with `center=`), pitch = `pixel_size` ·
`norm="hermite"` (default; the original script's scale), `"max1"` or `"l2"`.
If the coefficients look transposed or scaled,
this is where to look.

## Files

`pipeline.py` Estimator · `torch_engine.py` device + loss/grad · `hg_basis.py` ·
`lens_transform.py` · `config.py` · `examples/run_estimation.py`
