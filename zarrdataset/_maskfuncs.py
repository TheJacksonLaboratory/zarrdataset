import numpy as np
from skimage import morphology, color, filters, transform


class MaskGenerator(object):
    """Base class to define transformation functions on ImageBase objects.
    """
    def __init__(self, axes):
        self.axes = axes

    def _compute_transform(self, image: np.ndarray):
        raise NotImplementedError("This is a virtual method and has to be "
                                  "implemented by a derived image "
                                  "transformation class")

    def __call__(self, image: np.ndarray):
        return self._compute_transform(image)


class WSITissueMaskGenerator(MaskGenerator):
    def __init__(self, mask_scale=1/16, min_size=16, area_threshold=128,
                 thresh=None):
        super(WSITissueMaskGenerator, self).__init__(axes="YX")

        self._mask_scale = mask_scale
        self._min_size = min_size
        self._area_threshold_2 = area_threshold ** 2
        self._thresh = thresh

    def _compute_transform(self, image: np.ndarray):
        gray = color.rgb2gray(image)
        scaled_gray = transform.rescale(gray, scale=self._mask_scale, order=0,
                                        preserve_range=True)

        if self._thresh is None:
            self._thresh = filters.threshold_otsu(image)

        chunk_mask = scaled_gray > self._thresh
        chunk_mask = morphology.remove_small_objects(
            chunk_mask == 0, min_size=self._min_size ** 2, connectivity=2)
        mask = morphology.remove_small_holes(
            chunk_mask, area_threshold=self._area_threshold_2)
        mask = morphology.binary_dilation(
            chunk_mask, morphology.disk(self._min_size))

        return mask
