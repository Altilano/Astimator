"""
Example driver, now covering:
  1. Building an .h5 dataset from individual picture files (one list per transform)
  2. Loading it back with H5Dataset (single file path)
  3. Two basis choices: the HG fast path, or the generic decompose/transform/
     recombine path for a basis that ISN'T Hermite-Gaussian
"""
from hg_lens_estimator import EstimationConfig, TransformSpec, Estimator
from hg_lens_estimator.h5_dataset import build_h5_dataset, H5Dataset

# ── CONFIG: backend / device / how many images -------------------------------
cfg = EstimationConfig(
    backend="torch",        # <- "jax" or "torch"
    device="auto",          # <- "auto" | "cpu" | "cuda"
    n_transforms=3,          # <- any integer >= 2

    transforms=[
        TransformSpec("forw", theta=0.0,               z=32.0e-2),
        TransformSpec("tilt", theta=18 * 3.14159265/180, z=31.5e-2),
        TransformSpec("gouy", theta=0.0,               z=35.0e-2),
    ],

    N=12,
    n_restarts=10,
    output_pattern="results/results_{r}.npz",
)


# ── STEP 1 (one-time preprocessing): build the .h5 dataset --------------------
# Point this at your individual picture files, one list per transform, in the
# same order as cfg.transforms. ax/ay for each transform are computed and
# stored alongside the images -- you don't need to re-derive them later.
#
# build_h5_dataset(
#     picture_paths=[
#         ["data/forw_0.tif", "data/forw_1.tif", ...],   # matches transforms[0] = "forw"
#         ["data/tilt_0.tif", "data/tilt_1.tif", ...],   # matches transforms[1] = "tilt"
#         ["data/gouy_0.tif", "data/gouy_1.tif", ...],   # matches transforms[2] = "gouy"
#     ],
#     transforms=cfg.transforms,
#     output_path="dataset.h5",       # <- the one file path everything else uses
#     waist=2e-4, lamb=800e-9, z0=20e-2, f=30e-2,
# )

data_provider = H5Dataset("dataset.h5")   # just one path -- ax/ay come along with it
im_size = data_provider.im_size
cam_pixel_size = 6.45e-6
nix = im_size // 2 * cam_pixel_size


# ── STEP 2: choose a basis provider -------------------------------------------

# Option A: your basis already IS Hermite-Gaussian -> fast path (direct phase multiply)
from hg_lens_estimator.hg_defaults import hg_transform_provider, hg_valid_indices
_, _, valid_idx = hg_valid_indices(cfg.N)     # triangular n+m<=N truncation
cfg.valid_indices = valid_idx
basis_provider = hg_transform_provider(nix, im_size, N=cfg.N, lamb=800e-9, z0=20e-2, f=30e-2, waist=2e-4)

# Option B: your basis is something else entirely (uncomment to use instead).
# The lens-tilt transform only has a simple form in the HG representation, so
# this decomposes your basis into HG modes, applies the phase there, and
# recombines back -- your basis itself can be anything defined on the same
# pixel grid (measured modes, LG modes, a custom simulated response, ...).
#
# from hg_lens_estimator.hg_defaults import generic_transform_provider
# my_basis = ...   # np.ndarray (n_pix, n_modes), or a callable(waist) -> that
# basis_provider = generic_transform_provider(my_basis, nix, im_size, N=cfg.N,
#                                              lamb=800e-9, z0=20e-2, f=30e-2)
# cfg.valid_indices = None   # no HG-shaped mode truncation makes sense here

# Option C: use the ax/ay already stored in the .h5 file instead of recomputing them
# from hg_lens_estimator.hg_defaults import make_hg_basis_builder, h5_lens_transform_fn
# from hg_lens_estimator.basis_provider import TransformBasisProvider
# hg_basis_builder = make_hg_basis_builder(nix, im_size, N=cfg.N)
# transform_fn = h5_lens_transform_fn(data_provider, hg_basis_builder)
# basis_provider = TransformBasisProvider(lambda w: my_basis, transform_fn)


# ── run --------------------------------------------------------------------
est = Estimator(cfg, basis_provider, data_provider)
est.prepare()

result = est.estimate_image(image_index=0)
print("best cost:", result["results"].min())

# Or batch over many images, saving one .npz per image:
# est.run(range(180))
