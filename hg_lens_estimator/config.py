"""
All the "few lines you change" live here.

- backend:      "jax" or "torch"
- device:       "auto" | "cpu" | "cuda"
                (For AMD GPUs: install a ROCm build of torch or jax, and pass
                device="cuda" anyway -- both libraries expose ROCm through the
                same 'cuda' device string. See README.)
- n_transforms: any integer >= 2 -> how many of `transforms` are actually used
                in the loss / fit. `transforms` must have at least this many
                entries; only the first `n_transforms` are used, so you can
                keep extra ones defined and just dial n_transforms down/up.
- N:            max Hermite-Gaussian order used everywhere (basis size,
                parametrization, valid-mode mask, loss).
"""
from dataclasses import dataclass, field
from typing import List, Literal, Optional


@dataclass
class TransformSpec:
    """One physical transformation (a lens position/tilt) that produced one image."""
    name: str
    theta: float = 0.0      # lens tilt angle, radians
    z: float = 0.0          # camera distance, metres
    flip_lens: bool = False


@dataclass
class EstimationConfig:
    # ---- switches you actually flip -------------------------------------
    backend: Literal["jax", "torch"] = "torch"
    device: str = "auto"                 # "auto" | "cpu" | "cuda"
    n_transforms: int = 3                # any integer >= 2

    transforms: List[TransformSpec] = field(default_factory=lambda: [
        TransformSpec("forw", theta=0.0,            z=32.0e-2),
        TransformSpec("tilt", theta=18 * 3.14159265 / 180, z=31.5e-2),
        TransformSpec("gouy", theta=0.0,            z=35.0e-2),
    ])

    # ---- physical / optimization parameters ------------------------------
    N: int = 12                 # max HG order (any positive int)
    waist: float = 2e-4
    lamb: float = 800e-9
    z0: float = 20e-2
    f: float = 30e-2
    waist_scan: Optional[List[float]] = None   # None -> just use `waist`

    n_restarts: int = 10
    sigma: float = 1.0
    maxiter: int = 5000
    tol: float = 1e-12

    # ---- data ------------------------------------------------------------
    parent_folder: str = "paper_tilted_lens/"
    mode_folder: str = "mixed HGs/"
    mode_file: str = "processed_images3.npz"
    cam_pixel_size: float = 6.45e-6

    # ---- output ------------------------------------------------------------
    output_pattern: str = "paper_tilted_lens/results/results_{r}.npz"

    def __post_init__(self):
        if self.n_transforms < 2:
            raise ValueError(f"n_transforms must be >= 2, got {self.n_transforms}")
        if len(self.transforms) < self.n_transforms:
            raise ValueError(
                f"n_transforms={self.n_transforms} but only "
                f"{len(self.transforms)} entries given in `transforms`. "
                "Add more TransformSpec entries (or lower n_transforms)."
            )
        if self.N < 1:
            raise ValueError(f"N must be >= 1, got {self.N}")

    def active_transforms(self) -> List[TransformSpec]:
        return self.transforms[: self.n_transforms]

    def waists(self) -> List[float]:
        return self.waist_scan if self.waist_scan else [self.waist]
