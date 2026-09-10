# hg_lens_estimator

Reorganized version of your two estimation notebooks
(`estimate_3p_paper2026.ipynb` [JAX] and `estimate_3p_pytorch_tiltelens.ipynb` [PyTorch]).

Both notebooks did the same thing: load an image made of 2 or 3 stacked
sub-images (each corresponding to a lens transformation: `forw`, `tilt`,
`gouy`), build a Hermite-Gaussian basis, and fit complex mode coefficients
by minimizing a negative-log-likelihood with L-BFGS-B, using autodiff for
the gradient. The only real differences between the two notebooks were:

1. which autodiff library computed the gradient (JAX vs PyTorch),
2. whether 2 or 3 sub-images were stacked into the loss.

This package pulls that shared logic into one place and turns those two
differences into config fields.

## Layout

```
hg_lens_estimator/
  config.py            EstimationConfig / TransformSpec  <- what you edit
  backends/
    base.py            Backend interface + get_backend()
    jax_backend.py      JAX implementation (autodiff, jit)
    torch_backend.py    PyTorch implementation (autodiff)
  optics.py            wraps your existing calculate_ax_ay (utils.py)
  basis.py             wraps your existing beam.Space / hg_basis (beam.py)
  parametrization.py   vector <-> coefficient-matrix (numpy, shared)
  data.py              loads the .npz dataset, replaces the old get_pic()
                       global-variable hack with a plain Dataset/ImageSample
  pipeline.py          Estimator: ties it all together, the optimisation loop
examples/
  run_estimation.py    the only file you edit day-to-day
```

`beam.py` and `utils.py` are still your existing files -- this package
imports `Space`, `calculate_ax_ay`, and `normalize` from them exactly as the
notebooks did, so nothing about your optics/beam code needs to change.

## What changes for each switch

All of it is one `EstimationConfig(...)` call in `examples/run_estimation.py`:

| You want to...                  | Change this field                          |
|----------------------------------|---------------------------------------------|
| JAX ↔ PyTorch                    | `backend="jax"` / `backend="torch"`         |
| NVIDIA ↔ AMD GPU                 | nothing in code -- install the matching ROCm build of torch or jax; keep `device="cuda"` (both libraries expose ROCm through the same `"cuda"` device string) or `device="auto"` |
| N-picture estimation (any N ≥ 2) | `n_transforms=<int>`                        |
| Max Hermite-Gaussian order       | `N=<int>`                                   |

`n_transforms` slices `active_transforms()` from the `transforms` list, so
with `n_transforms=2` only `forw` and `tilt` (the first two entries) are
loaded, phase-shifted, stacked, and included in the loss -- matching the
`_2IM` runs in your paper notebook. `n_transforms` can be any integer ≥ 2
(not just 2 or 3) as long as `transforms` has at least that many entries --
add more `TransformSpec`s (e.g. a second tilt angle, another distance) if you
want to fit against 4, 5, ... images at once. `EstimationConfig` validates
this at construction time and raises a clear `ValueError` if `n_transforms`
is too small or `transforms` is too short. Reorder the list if you want a
different subset used first.

`N` (max HG order) is a plain config field used everywhere: basis size,
valid-mode mask, coefficient parametrization, and the loss -- change it in
one place and everything else follows.

## Notes / things worth checking before you trust it fully

- I could not execute this against your actual `beam.py`/`utils.py`/data
  files (they weren't uploaded, and `jax`/`torch` aren't installed in this
  sandbox), so the file has not been run end-to-end. The math in
  `backends/jax_backend.py` and `backends/torch_backend.py` is copied
  essentially line-for-line from your two notebooks (same parametrization,
  same loss, same restart/L-BFGS-B loop) — only the "how many transforms"
  loop was generalized from hardcoded 2 or 3 to `n_transforms`. Please run
  it once on a known case and compare against your old results before
  relying on it.
- The JAX gouy phase (`ax_gouy`/`ay_gouy`) in your paper notebook was built
  but the loss only ever concatenated `[I0, I1]` (2 images) even though 3
  were computed. This refactor makes that explicit and controllable via
  `n_transforms` instead of implicit in which variables happened to be
  concatenated.
- The waist-scan feature (`waist_scan`, looping over multiple candidate
  waists) is preserved but optional; both your recent notebooks were
  effectively running it with a single-element list.
- AMD GPU support depends entirely on you having ROCm-enabled builds of
  `torch`/`jax` installed; JAX's ROCm support is less mature/complete than
  PyTorch's, so for AMD hardware `backend="torch"` is the safer bet.
