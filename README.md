# hg_lens_estimator

Reorganized version of your two estimation notebooks
(`estimate_3p_paper2026.ipynb` [JAX] and `estimate_3p_pytorch_tiltelens.ipynb` [PyTorch]).

The core pipeline (`Estimator`) doesn't know about Hermite-Gaussian modes,
lens tilts, or any particular file format -- it only needs:

1. a **basis**: a `(n_transforms * n_pix, n_modes)` complex matrix, and
2. **data**: a `(n_transforms * n_pix,)` real intensity vector,

for each image you want to fit. Everything HG-specific lives in
`hg_defaults.py` as one convenience implementation of two generic
interfaces (`BasisProvider`, `DataProvider`), not baked into the core.

## Data: build once, load with one path

`h5_dataset.py` replaces the old multi-piece `.npz` path
(`parent_folder + mode_folder + mode_file`) with a single-file `.h5`
pipeline:

```python
from hg_lens_estimator.h5_dataset import build_h5_dataset, H5Dataset

build_h5_dataset(
    picture_paths=[
        ["forw_0.tif", "forw_1.tif", ...],   # aligned with transforms[0]
        ["tilt_0.tif", "tilt_1.tif", ...],   # aligned with transforms[1]
        ["gouy_0.tif", "gouy_1.tif", ...],   # aligned with transforms[2]
    ],
    transforms=cfg.transforms,
    output_path="dataset.h5",                # <- one file, everything else uses just this
    waist=2e-4, lamb=800e-9, z0=20e-2, f=30e-2,
)

data_provider = H5Dataset("dataset.h5")      # one path in, ax/ay come along with it
ax, ay = data_provider.get_ax_ay("tilt")     # read back per-transform ax/ay any time
```

`build_h5_dataset` computes `ax`/`ay` for each transform (via your
`utils.calculate_ax_ay`) at build time and stores them as attributes
alongside the images, so downstream code never needs to recompute them --
though it still can, if you'd rather derive them fresh (see Option A/B in
`examples/run_estimation.py`).

Any number of transforms (>= 2) works: `picture_paths` just needs one list
per entry in `transforms`, all the same length (one path per sample index).
`loader=` lets you override how a single picture file is read (default:
`.npy` via `np.load`, otherwise PIL) if your images are in some other format.

`NpzDataset` is kept for backward compatibility with your existing `.npz`
files, unchanged.

## Basis: precomputed, or transform-on-demand -- for ANY basis

`basis_provider.py`:

- `PrecomputedBasisProvider({name: basis_matrix, ...})` -- you already have
  the fully-transformed basis for each measurement.
- `TransformBasisProvider(base_basis_fn, transform_fn)` -- you have one
  *untransformed* basis and a rule for how each `TransformSpec` modifies it.

The lens-tilt transform is genuinely a simple, diagonal phase multiply
**only when your basis is Hermite-Gaussian**. `hg_defaults.py` now offers
both cases:

```python
from hg_lens_estimator.hg_defaults import hg_transform_provider          # basis IS HG
from hg_lens_estimator.hg_defaults import generic_transform_provider     # basis is anything else
```

`generic_transform_provider` (backed by `lens_transform.py`) handles a
non-HG basis by:

1. **decomposing** it onto an internal HG basis (least-squares projection,
   so it doesn't need exact orthonormality),
2. **applying** the lens-tilt phase in that HG representation, where it
   actually is diagonal,
3. **recombining** back into pixel space.

```python
my_basis = ...  # np.ndarray (n_pix, n_modes), any basis you like, or a callable(waist)->that
basis_provider = generic_transform_provider(
    my_basis, nix, im_size, N=20,   # N here = HG order used only as the internal
    lamb=800e-9, z0=20e-2, f=30e-2, # representation; pick it large enough that N
)                                    # represents `my_basis` with negligible error
```

The lower-level building blocks (`decompose_into_hg`, `apply_hg_diagonal_phase`,
`recombine_from_hg`, and the combined `tilted_lens_transform`) are in
`lens_transform.py` if you want to compose them differently, or reuse just
the decomposition/recombination steps for something else.

There's also `h5_lens_transform_fn`, which does the same decompose-transform-
recombine but reads `ax`/`ay` back from an `H5Dataset` instead of
recomputing them -- useful if you trust values baked in from a separate
calibration.

## Layout

```
hg_lens_estimator/
  config.py            EstimationConfig / TransformSpec
  basis_provider.py    BasisProvider interface: Precomputed / Transform
  data.py              DataProvider interface: Npz (legacy) / Array / Callable
  h5_dataset.py        build_h5_dataset() + H5Dataset (the new, preferred data path)
  lens_transform.py    generic decompose/apply-phase/recombine math
  hg_defaults.py        HG-specific convenience: fast path + generic path
  parametrization.py   vector <-> flat coefficient vector (basis-agnostic)
  backends/
    base.py            Backend interface + get_backend()
    jax_backend.py       JAX autodiff loss+grad
    torch_backend.py     PyTorch autodiff loss+grad
  pipeline.py           Estimator: ties a BasisProvider + DataProvider + backend together
examples/
  run_estimation.py     shows the h5 build step + both HG and generic basis options
```

## What changes for each switch

| You want to...                          | Change this                                        |
|-------------------------------------------|------------------------------------------------------|
| JAX ↔ PyTorch                             | `EstimationConfig(backend=...)`                       |
| NVIDIA ↔ AMD GPU                          | nothing in code -- install the matching ROCm build; `device="cuda"`/`"auto"` |
| N-picture estimation (any N ≥ 2)          | `n_transforms=<int>` (+ enough `transforms` entries)  |
| Which/how many modes are fit              | `valid_indices=<list of int>`, `None` = all modes     |
| Scan over a basis parameter (e.g. waist)  | `param_scan=[v1, v2, ...]`                            |
| Where the data comes from                 | pick a `DataProvider`: `H5Dataset`, `NpzDataset`, `ArrayDataProvider`, `CallableDataProvider` |
| Where/how the basis is built              | pick a `BasisProvider`: `PrecomputedBasisProvider`, or `hg_defaults.hg_transform_provider` / `generic_transform_provider` |

## Notes / things worth checking before you trust it fully

- I don't have your actual `beam.py`/`utils.py`/image files, so
  `hg_defaults.py` and `h5_dataset.py`'s calls into `calculate_ax_ay` /
  `normalize` haven't been run against the real functions. What I *did*
  verify, with synthetic stand-ins:
  - the generic `Estimator` + provider plumbing end-to-end (recovers known
    ground-truth coefficients, >0.999 overlap, using a non-HG random basis),
  - `tilted_lens_transform` reduces to an exact per-mode phase multiply when
    the input basis IS the HG basis (max error ~2e-14), and correctly
    transforms a basis that's a genuine linear combination of HG modes
    (max error ~7e-14) -- i.e. the decompose/transform/recombine math is
    correct, not just plausible,
  - `build_h5_dataset` / `H5Dataset` round-trip correctly, including a real
    bug I caught and fixed: h5py's default group ordering is *alphabetical*,
    not insertion order, which would have silently reordered your
    forw/tilt/gouy transforms. Order is now stored explicitly as a
    `transform_order` attribute and read back from that.
- `coeffs` from `estimate_image()` is a flat vector of length `n_modes`. For
  the HG case, `hg_defaults.reshape_hg_coefficients(coef, N)` gets you the
  `[n, m]` matrix view back.
- In `generic_transform_provider`, pick the internal HG order `N` high
  enough to represent your actual basis well -- if your basis has structure
  the internal HG truncation can't capture, the decomposition step will
  lose it. There's no automatic check for this; a quick sanity test is to
  decompose-then-recombine *without* the phase step and confirm you get
  your original basis back to the precision you need.
