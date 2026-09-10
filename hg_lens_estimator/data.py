import numpy as np


class Dataset:
    """Loads the .npz produced by your image-processing step (images + index [+ coefs])."""

    def __init__(self, config):
        from utils import normalize  # your existing utils.py

        self._normalize = normalize
        path = config.parent_folder + config.mode_folder + config.mode_file
        data = np.load(path, allow_pickle=True)
        self.images = data["images"]
        self.index = data["index"]
        self.coefs = data["coefs"] if "coefs" in data else None

    def get_image(self, i, n_transforms):
        """
        Equivalent of get_pic(i) in the original notebooks, but returns a plain
        object instead of mutating globals, and stacks exactly `n_transforms`
        sub-images (2 or 3) instead of a hardcoded 3.
        """
        idx = self.index[i]
        raw = self.images[i] / 255
        im_size = raw.shape[1]

        raw = raw[: n_transforms * im_size]
        parts = []
        for k in range(n_transforms):
            seg = raw[k * im_size:(k + 1) * im_size]
            seg = self._normalize(seg)
            seg = np.flip(seg, axis=0)
            parts.append(seg)

        img = np.concatenate(parts)
        return ImageSample(index=i, idx_meta=idx, im_size=im_size,
                            img=img, img_ravel=img.ravel())


class ImageSample:
    def __init__(self, index, idx_meta, im_size, img, img_ravel):
        self.index = index
        self.idx_meta = idx_meta
        self.im_size = im_size
        self.img = img
        self.img_ravel = img_ravel
