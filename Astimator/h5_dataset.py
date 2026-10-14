"""
Build pipeline: take individual picture files (one list per transform: "here
are all the forw pictures, here are all the tilt pictures, ...") and pack
them into a single .h5 file, computing and storing ax/ay for each transform
along the way.

Layout written by `build_h5_dataset`:

    /<transform_name>/images   dataset, shape (n_samples, H, W)
    /<transform_name>.attrs['ax']    float
    /<transform_name>.attrs['ay']    float
    /<transform_name>.attrs['theta'] float
    /<transform_name>.attrs['z']     float

`H5Dataset` (a DataProvider) reads it back, taking just the one file path.
"""
import numpy as np

from .data import DataProvider


def default_image_loader(path):
    path = str(path)
    if path.endswith(".npy"):
        return np.load(path)
    from PIL import Image
    return np.array(Image.open(path))


def build_h5_dataset(picture_paths, transforms, output_path,
                      waist, lamb, z0, f, loader=None):
    """
    picture_paths: list of lists, aligned with `transforms`:
                       picture_paths[i] = [path_sample0, path_sample1, ...]
                       for transforms[i]. All inner lists must be the same
                       length (one entry per sample/image index).
    transforms:    list[TransformSpec] -- theta/z/flip_lens used to compute
                       ax, ay for each transform (via calculate_ax_ay).
    output_path:   the single .h5 file to write.
    waist, lamb, z0, f: passed to calculate_ax_ay.
    loader:        callable(path) -> np.ndarray. Defaults to
                       default_image_loader (handles .npy, else PIL).
    """
    import h5py
    from utils import calculate_ax_ay  # your existing optics function

    if len(picture_paths) != len(transforms):
        raise ValueError(
            f"picture_paths has {len(picture_paths)} entries but "
            f"transforms has {len(transforms)}; they must align 1:1."
        )
    n_samples = len(picture_paths[0])
    for paths, t in zip(picture_paths, transforms):
        if len(paths) != n_samples:
            raise ValueError(
                f"transform '{t.name}' has {len(paths)} picture paths, "
                f"expected {n_samples} (same as the first transform)."
            )

    loader = loader or default_image_loader

    with h5py.File(output_path, "w") as h5:
        # h5py's default key order is alphabetical, not insertion order --
        # store the intended order explicitly so H5Dataset doesn't have to
        # guess (and doesn't silently reorder your transforms).
        h5.attrs["transform_order"] = np.array(
            [t.name for t in transforms], dtype=h5py.string_dtype()
        )

        for paths, t in zip(picture_paths, transforms):
            ax, ay = calculate_ax_ay(
                waist=waist, lamb=lamb, z0=z0, f=f,
                theta=t.theta, z=t.z, flip_lens=t.flip_lens,
            )
            imgs = np.stack([loader(p) for p in paths])

            grp = h5.create_group(t.name)
            grp.create_dataset("images", data=imgs)
            grp.attrs["ax"] = ax
            grp.attrs["ay"] = ay
            grp.attrs["theta"] = t.theta
            grp.attrs["z"] = t.z


class H5Dataset(DataProvider):
    """Reads back the file written by build_h5_dataset. Takes one file path."""

    def __init__(self, path, normalize_fn=None, flip=True):
        import h5py
        if normalize_fn is None:
            from utils import normalize as normalize_fn  # your existing utils.py
        self._normalize = normalize_fn
        self.flip = flip
        self.h5 = h5py.File(path, "r")

        if "transform_order" in self.h5.attrs:
            self.transform_names = [
                n.decode() if isinstance(n, bytes) else n
                for n in self.h5.attrs["transform_order"]
            ]
        else:
            # older file written without the explicit order -- fall back to
            # whatever h5py gives us (alphabetical), with a warning.
            import warnings
            warnings.warn(
                "This .h5 file has no 'transform_order' attribute; falling "
                "back to h5py's (alphabetical) key order, which may not "
                "match your intended transform order."
            )
            self.transform_names = list(self.h5.keys())

    def get_image(self, index, n_transforms):
        names = self.transform_names[:n_transforms]
        parts = []
        for name in names:
            raw = self.h5[name]["images"][index]
            seg = self._normalize(raw)
            if self.flip:
                seg = np.flip(seg, axis=0)
            parts.append(seg)
        return np.concatenate(parts).ravel()

    def get_ax_ay(self, transform_name):
        """ax, ay as stored for this transform -- use these instead of
        recomputing via calculate_ax_ay if you trust the values baked into
        the file (e.g. they came from a calibration you did separately)."""
        grp = self.h5[transform_name]
        return float(grp.attrs["ax"]), float(grp.attrs["ay"])

    @property
    def im_size(self):
        first = self.transform_names[0]
        return self.h5[first]["images"].shape[-1]

    def close(self):
        self.h5.close()
