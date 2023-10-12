from typing import Union
import numpy as np
from skimage import morphology, color, filters, transform, measure


class MaskGenerator(object):
    """Base class to define transformation functions on ImageBase objects.
    """
    def __init__(self, axes):
        self.axes = axes

    def _compute_transform(self, image: np.ndarray) -> np.ndarray:
        raise NotImplementedError("This is a virtual method and has to be "
                                  "implemented by a derived image "
                                  "transformation class")

    def __call__(self, image: np.ndarray) -> np.ndarray:
        return self._compute_transform(image)


class WSITissueMaskGenerator(MaskGenerator):
    """Mask generator for tissue objects in Whole Slide Images (WSI).

    This will compute a mask of scale `mask_scale` from the input image where
    tissue (darker pixels than white background) are present.

    Parameters
    ----------
    mask_scale : float
    min_size : int
    area_threshold : int,
    thresh : Union[float, None]
    axes : str
    """
    def __init__(self, mask_scale : float = 1/16, min_size : int = 16,
                 area_threshold : int = 128,
                 thresh : Union[float, None] = None,
                 axes : str = "YX"):
        super(WSITissueMaskGenerator, self).__init__(axes=axes)

        self._mask_scale = mask_scale
        self._min_size = min_size
        self._area_threshold_2 = area_threshold ** 2
        self._thresh = thresh

    def _compute_transform(self, image: np.ndarray) -> np.ndarray:
        gray = color.rgb2gray(image)
        scaled_gray = transform.rescale(gray, scale=self._mask_scale, order=0,
                                        preserve_range=True)

        if self._thresh is None:
            self._thresh = filters.threshold_otsu(scaled_gray)

        chunk_mask = scaled_gray < self._thresh
        chunk_mask = morphology.remove_small_objects(
            chunk_mask == 0, min_size=self._min_size ** 2, connectivity=2)
        chunk_mask = morphology.remove_small_holes(
            chunk_mask, area_threshold=self._area_threshold_2)
        chunk_mask = morphology.binary_erosion(
            chunk_mask, morphology.disk(max(1, self._min_size // 2)))
        mask = morphology.binary_dilation(
            chunk_mask, morphology.disk(self._min_size))

        if "Z" in self.axes:
            mask = mask[None, ...]

        return mask


class LabelMaskGenerator(MaskGenerator):
    """Label generator for masked images.

    This function can be used as pre-processing step to use the
    CenteredPatchSampler method and will retrieve only the object centroids by
    default, unless `only_centroids` is set to False.

    Parameters
    ----------
    require_label : bool
        When True, the connected components algorithm will be aplied to the
        input image. Otherwise, the image will be used as it is.
    only_centroids : bool
        If True, the labeled image will contain only a single pixel at the
        centroid of each object, the label will be used as the intensity of
        that pixel.
    axes : str
        The order of the axes of the transformed image that this function
        generates.
    """
    def __init__(self, require_label: bool = True,
                 only_centroids: bool = True,
                 axes : str = "YX"):
        super(LabelMaskGenerator, self).__init__(axes=axes)
        self._require_label = require_label
        self._only_centroids = only_centroids

    def _compute_transform(self, image: np.ndarray) -> np.ndarray:
        # Take a binary mask and convert it to a labeled mask ready for use
        # with the CenteredPatchSampler.
        if self._require_label:
            label = measure.label(image)
        else:
            label = image

        centroids = np.zeros_like(label)
        if self._only_centroids:
            properties = measure.regionprops(label)

            for o in properties:
                centroids[tuple(round(cc) for cc in o.centroid)] = o.label
        else:
            centroids = label

        return centroids
