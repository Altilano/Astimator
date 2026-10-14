"""
Two ways to get the (n_transforms * n_pix, n_modes) stacked basis matrix
the optimizer needs, chosen by which class you instantiate:

1. PrecomputedBasisProvider
   You already have the fully-transformed basis for each measurement
   (any basis -- HG, Laguerre-Gaussian, an arbitrary measured/simulated
   response, whatever -- and any transform physics, not necessarily a lens
   tilt). Just hand over one matrix per TransformSpec.name.

2. TransformBasisProvider
   You have one "base" (untransformed) basis, plus a rule describing how
   each TransformSpec changes it (e.g. the Gouy/tilt phase model used for
   the lens in your experiment -- see hg_defaults.py for that one, or write
   your own for a different physical model).

Both return the same shape of matrix, so the rest of the pipeline
(Estimator, backends, loss) doesn't care which one you used.
"""
from abc import ABC, abstractmethod
import numpy as np


class BasisProvider(ABC):
    @abstractmethod
    def build_stacked_basis(self, param, transforms) -> np.ndarray:
        """
        param:      whatever scalar/object your basis needs to vary over
                    (e.g. a candidate beam waist for a waist scan). Pass
                    None if your basis doesn't need one.
        transforms: list[TransformSpec], already sliced to n_transforms
                    (config.active_transforms()).
        returns:    complex np.ndarray, shape (n_transforms * n_pix, n_modes)
        """
        ...


class PrecomputedBasisProvider(BasisProvider):
    """Use when the transformed basis for each measurement is already computed."""

    def __init__(self, basis_by_name: dict):
        """
        basis_by_name: {transform_name: np.ndarray of shape (n_pix, n_modes)}
                       one entry per TransformSpec.name you plan to use,
                       already reflecting whatever that transformation does
                       to the basis (phase shift, mode mixing, anything).
        """
        self.basis_by_name = basis_by_name

    def build_stacked_basis(self, param, transforms):
        try:
            blocks = [self.basis_by_name[t.name] for t in transforms]
        except KeyError as e:
            raise KeyError(
                f"No precomputed basis found for transform {e}. "
                f"basis_by_name has entries for: {list(self.basis_by_name)}"
            )
        return np.vstack(blocks)


class TransformBasisProvider(BasisProvider):
    """Use when you have one base basis and a rule for how each TransformSpec modifies it."""

    def __init__(self, base_basis_fn, transform_fn):
        """
        base_basis_fn(param) -> np.ndarray (n_pix, n_modes) complex, untransformed basis.
                                 If your basis doesn't depend on `param`, just ignore the
                                 argument in your function.
        transform_fn(base_basis, transform_spec, param) -> np.ndarray (n_pix, n_modes),
                                 the basis as it appears in that particular measurement.
        """
        self.base_basis_fn = base_basis_fn
        self.transform_fn = transform_fn

    def build_stacked_basis(self, param, transforms):
        base = self.base_basis_fn(param)
        blocks = [self.transform_fn(base, t, param) for t in transforms]
        return np.vstack(blocks)
