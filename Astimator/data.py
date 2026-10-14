"""
Anything that can hand back "the raveled, per-transform-stacked, normalized
intensity vector for image i" works as a data source.

- NpzDataset:          your original processed_images*.npz format (legacy)
- H5Dataset:            the new h5py-based format (see h5_dataset.py) -- preferred
- ArrayDataProvider:    an in-memory numpy array you already loaded/generated
- CallableDataProvider: total flexibility -- wrap any function you want

All of these take exactly one file path (or one in-memory object) -- no
separate folder/subfolder/filename pieces to keep in sync.
"""
from abc import ABC, abstractmethod
import numpy as np


class DataProvider(ABC):
    @abstractmethod
    def get_image(self, index: int, n_transforms: int) -> np.ndarray:
        """Return a real-valued 1D array: n_transforms sub-images, each
        normalized and raveled, concatenated in the same order as
        config.active_transforms()."""
        ...


class NpzDataset(DataProvider):
    """Legacy: loads the .npz produced by your original image-processing step."""

    def __init__(self, path, normalize_fn=None):
        if normalize_fn is None:
            from utils import normalize as normalize_fn  # your existing utils.py
        self._normalize = normalize_fn

        data = np.load(path, allow_pickle=True)
        self.images = data["images"]
        self.index = data["index"]
        self.coefs = data["coefs"] if "coefs" in data else None

    def get_image(self, index, n_transforms):
        raw = self.images[index] / 255
        im_size = raw.shape[1]

        raw = raw[: n_transforms * im_size]
        parts = []
        for k in range(n_transforms):
            seg = raw[k * im_size:(k + 1) * im_size]
            seg = self._normalize(seg)
            seg = np.flip(seg, axis=0)
            parts.append(seg)

        img = np.concatenate(parts)
        return img.ravel()

    @property
    def im_size(self):
        return self.images[0].shape[1]


class ArrayDataProvider(DataProvider):
    """
    Wrap image data you already have in memory.

    images: array-like, indexable by `index`, where each element is either
      - shape (n_transforms, H, W): already split per transform, or
      - shape (n_transforms * H, W): sub-images stacked vertically, like the
        raw .npz format (H is inferred as shape[0] // n_transforms).
    """

    def __init__(self, images, normalize_fn=None, flip=True):
        self.images = images
        self.normalize_fn = normalize_fn or (lambda x: x / x.max())
        self.flip = flip

    def get_image(self, index, n_transforms):
        raw = np.asarray(self.images[index])

        if raw.ndim == 3:
            parts = [raw[k] for k in range(n_transforms)]
        elif raw.ndim == 2:
            im_size = raw.shape[0] // n_transforms
            parts = [raw[k * im_size:(k + 1) * im_size] for k in range(n_transforms)]
        else:
            raise ValueError(f"Expected images[index] to be 2D or 3D, got shape {raw.shape}")

        out = []
        for seg in parts:
            seg = self.normalize_fn(seg)
            if self.flip:
                seg = np.flip(seg, axis=0)
            out.append(seg)

        img = np.concatenate(out)
        return img.ravel()


class CallableDataProvider(DataProvider):
    """Wrap an arbitrary function: index, n_transforms -> raveled real array."""

    def __init__(self, fn):
        self.fn = fn

    def get_image(self, index, n_transforms):
        return np.asarray(self.fn(index, n_transforms))
