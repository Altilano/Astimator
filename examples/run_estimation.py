"""
Example driver. Everything that used to be scattered/duplicated across your
two notebooks now boils down to editing the CONFIG block below.
"""
import sys
from pathlib import Path

# Calculate the parent directory and add it to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from hg_lens_estimator import EstimationConfig, TransformSpec, Estimator


# ── CONFIG: this is the only block you should need to touch ──────────────────
cfg = EstimationConfig(
    backend="torch",        # <- "jax" or "torch"
    device="auto",          # <- "auto" | "cpu" | "cuda"  (AMD GPU: install a
                             #    ROCm build of torch/jax, still say "cuda")
    n_transforms=3,          # <- any integer >= 2 (uses the first N entries of `transforms`)

    transforms=[
        TransformSpec("forw", theta=0.0,               z=32.0e-2),
        TransformSpec("tilt", theta=18 * 3.14159265/180, z=31.5e-2),
        TransformSpec("gouy", theta=0.0,               z=35.0e-2),
        # add more TransformSpec entries here if you want n_transforms > 3
    ],

    N=10,                    # <- max Hermite-Gaussian order, any positive int
    waist=2e-4,
    lamb=800e-9,
    z0=20e-2,
    f=30e-2,

    n_restarts=10,
    parent_folder="paper_tilted_lens/",
    mode_folder="mixed HGs/",
    mode_file="processed_images3.npz",
    output_pattern="paper_tilted_lens/results/results_{r}.npz",
)
# ───────────────────────────────────────────────────────────────────────────

# im_size / nix come from your camera setup, same as in the notebooks:
w_px = 70
pixel_size = 8e-6
cam_pixel_size = cfg.cam_pixel_size

# Peek at one image just to get im_size (mirrors the old get_pic(0) call)
probe = Estimator(cfg)  # backend + dataset are ready after this
sample0 = probe.dataset.get_image(0, cfg.n_transforms)
im_size = sample0.im_size
nix = im_size // 2 * cam_pixel_size

est = probe.prepare(nix, im_size)

# Single image:
result = est.estimate_image(image_index=0)
print("best cost:", result["results"].min())

# Or batch over many images, saving one .npz per image like the notebooks did:
# est.run(range(180))
