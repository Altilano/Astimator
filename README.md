# Astimator

The ASTIgmatic estiMATOR. Estimate complex mode coefficients from several intensity pictures of the same
beam, each taken after a different transformation. PyTorch based.

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
- `tilted_lens_focal_lengths(lens_type, theta_deg, f0, R, n)` — effective `(f_x, f_y)` of a tilted lens.
- `calculate_phix_phiy(fx, fy, waistx, waisty, lamb, z0, z, flip_lens=False, flip_coordinates=False)` —
  the `(phi_x, phi_y)` of a picture. Default is the physical phase (`... - pi/2`). With
  `flip_coordinates=True` it uses `+pi/2`, which is what matches pictures mirrored w.r.t. the basis,
  e.g. after a 4F system (the case of the original notebooks).
- `coeffs_to_matrix(coef, ni, mi, N)` — flat coefficients → `[n, m]` matrix.


## Files

`pipeline.py` Estimator · `torch_engine.py` device + loss/grad · `hg_basis.py` ·
`lens_transform.py` · `lens_phase.py` · `config.py` · `examples/run_estimation.py`

### Made with Claude, based on a series of Jupyter notebooks used in the scientific paper "High-Dimensional Structured Light Tomography with a Tilted Lens". The Jupyter notebooks were made by hand. Claude organized and fixed bugs.

## Citing

If you use Astimator in your work, please cite it. GitHub reads `CITATION.cff` (the "Cite this repository"
button); a BibTeX entry for LaTeX is in `CITATION.bib`.

### Made with Claude, based on a series of Jupyter notebooks used in the scientific paper "High-Dimensional Structured Light Tomography with a Tilted Lens". The Jupyter notebooks were made by hand. Claude organized and fixed bugs.