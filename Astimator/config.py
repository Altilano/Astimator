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
- N:            optional, informational/convenience field for basis
                implementations that care about a max mode order (e.g. the
                built-in HG helpers in hg_defaults.py). The core pipeline
                itself doesn't assume any particular basis size or shape --
                it just reads n_modes off whatever basis matrix your
                BasisProvider returns.

Note: this config no longer says anything about *what* basis or *what* data
you're using -- those are supplied separately as a BasisProvider and a
DataProvider (see basis_provider.py / data.py, and hg_defaults.py for the
original HG + lens-tilt convenience implementation).
"""
from dataclasses import dataclass, field
from typing import Any, List, Literal, Optional


@dataclass
class TransformSpec:
    """One physical transformation that produced one measurement/image.
    The `theta`/`z`/`flip_lens` fields are only meaningful if your
    BasisProvider's transform rule uses them (e.g. hg_defaults' lens-tilt
    model) -- a different transform_fn is free to read them differently or
    ignore them."""
    name: str
    theta: float = 0.0
    z: float = 0.0
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

    # ---- basis-agnostic optimisation parameters ---------------------------
    N: Optional[int] = None      # informational only; see class docstring
    param_scan: List[Any] = field(default_factory=lambda: [None])
    # ^ values handed one-by-one to BasisProvider.build_stacked_basis(param, ...).
    #   Use this for e.g. a beam-waist scan; leave as [None] if your basis
    #   doesn't need a scan parameter.

    valid_indices: Optional[List[int]] = None
    # ^ which basis columns (modes) to actually fit, as flat indices into
    #   n_modes. None -> use every mode the basis provides. For the HG
    #   triangular n+m<=N truncation, see hg_defaults.hg_valid_indices(N).

    n_restarts: int = 10
    sigma: float = 1.0
    maxiter: int = 5000
    tol: float = 1e-12

    # ---- output ------------------------------------------------------------
    output_pattern: str = "results/results_{r}.npz"

    def __post_init__(self):
        if self.n_transforms < 2:
            raise ValueError(f"n_transforms must be >= 2, got {self.n_transforms}")
        if len(self.transforms) < self.n_transforms:
            raise ValueError(
                f"n_transforms={self.n_transforms} but only "
                f"{len(self.transforms)} entries given in `transforms`. "
                "Add more TransformSpec entries (or lower n_transforms)."
            )
        if self.N is not None and self.N < 1:
            raise ValueError(f"N must be >= 1, got {self.N}")

    def active_transforms(self) -> List[TransformSpec]:
        return self.transforms[: self.n_transforms]
