import math
import numpy as np

from skimage import transform, measure
from matplotlib.path import Path
from bridson import poisson_disc_samples



class PatchSampler(object):
    """Base class to implement patches sampler from images.

    If the image used for extracting patches has a mask associated to it, only
    patches from masked regions of the image are retrieved.

    Parameters
    ----------
    patch_size : int
        Size in pixels of the patches extracted. Only squared patches are
        supported by now.
    """
    def __init__(self, patch_size):
        self._patch_size = patch_size

    def _sampling_method(self, *args, **kwargs):
        raise NotImplementedError("This is a virtual class and has to be "
                                  "inherited from other class implementing the"
                                  " actual sampling method.")

    def compute_toplefts(self, image):
        """Compute the top-left and bottom-right positions that define each of
        the possible patches that can be extracted from the image.

        Parameters
        ----------
        image : zarrdataset.ImageLoader
            The image from where the patches are extracted.

        Returns
        -------
        tl_brs : numpy.ndarray
            An array containing the coordinates of the patches that can be
            extracted from `image`. Each row in the array has the coordinates
            in the following format.
            (top-left y, top-left x, bottom-right y, bottom-right x)
        """
        tls = self._sampling_method(image.mask, image.shape)
        brs = tls + self._patch_size
        tl_brs = np.hstack((tls, brs))

        return tl_brs


class GridPatchSampler(PatchSampler):
    """Patch sampler that retrieves non-overlapping patches from a grid of
    coordinates that cover all the masked image.

    Parameters
    ----------
    patch_size : int
        Size in pixels of the patches extracted. Only squared patches are
        supported by now.
    min_object_presence : int
        Minimum presence of the masked object inside each patch. Only patches
        containing at least `min_object_presence` are retrieved.
    """
    def __init__(self, patch_size, min_object_presence=0.1, **kwargs):
        super(GridPatchSampler, self).__init__(patch_size)
        self._min_object_presence = min_object_presence

    def _sampling_method(self, mask, shape):
        # Use the mask as base to determine the valid patches that can be
        # retrieved from the image.
        mask_scale = mask.shape[-1] / shape[-1]
        scaled_ps = self._patch_size * mask_scale

        if scaled_ps < 1:
            mask_scale = 1.0
            scaled_h = shape[-2] // self._patch_size
            scaled_w = shape[-1] // self._patch_size

            # Upscale the mask to a size where each pixel represents a patch.
            dwn_valid_mask = transform.resize(mask, (scaled_h, scaled_w),
                                              order=0,
                                              mode="edge",
                                              anti_aliasing=False)

        else:
            scaled_ps = int(math.floor(scaled_ps))
            scaled_h = mask.shape[-2] // scaled_ps
            scaled_w = mask.shape[-1] // scaled_ps

            # Downscale the mask to a size where each pixel represents a patch.
            dwn_valid_mask = transform.downscale_local_mean(
                mask, factors=(scaled_ps, scaled_ps))

            dwn_valid_mask = dwn_valid_mask[:scaled_h, :scaled_w]

        # Retrieve patches that contain at least the requested minimum presence
        # of the masked object.
        valid_mask = dwn_valid_mask > self._min_object_presence

        toplefts = np.nonzero(valid_mask)
        toplefts = np.stack(toplefts, axis=1)

        # Map the positions back to the original scale of the image.
        toplefts = toplefts * self._patch_size

        return toplefts


class BlueNoisePatchSampler(PatchSampler):
    """Patch sampler that retrieves patches from coordinates sampled using the
    Bridson sampling algorithm, also known as Blue-noise sampling algorithm.

    Parameters
    ----------
    patch_size : int
        Size in pixels of the patches extracted. Only squared patches are
        supported by now.
    allow_overlap : bool
        Whether overlapping of patches is allowed or not.
    """
    def __init__(self, patch_size, allow_overlap=True, chunk=1000, **kwargs):
        super(BlueNoisePatchSampler, self).__init__(patch_size)
        self._overlap = math.sqrt(2) ** (not allow_overlap)
        self._chunk = chunk

    def _sampling_method(self, mask, shape):
        mask_scale =  mask.shape[-1] / shape[-1]
        rad = self._patch_size * mask_scale

        H, W = mask.shape
        chunk = min(self._chunk * mask_scale, H, W)

        # If the image is smaller than the scaled patch, retrieve the full
        # image instead.
        if H <= rad or W <= rad:
            sample_tls = np.array([[0, 0]], dtype=np.float32)

        else:
            chunk_sample_tls = np.array(
                poisson_disc_samples(height=chunk - rad,
                                     width=chunk - rad,
                                     r=rad,
                                     k=30),
                dtype=np.float32)

            # Upscale the mask to a size where each pixel represents a patch.
            dwn_valid_mask = transform.resize(mask.astype(np.float32),
                                              (round(H / chunk),
                                               round(W / chunk)),
                                              order=1,
                                              mode="edge",
                                              anti_aliasing=True)

            dwn_valid_mask = dwn_valid_mask > 0.25

            sample_tls = []
            for offset_y, offset_x in zip(*np.nonzero(dwn_valid_mask)):
                sample_tls.append(chunk_sample_tls
                                  + np.array((offset_x * chunk,
                                              offset_y * chunk)).reshape(1, 2))
            sample_tls = np.vstack(sample_tls)

        toplefts = np.round(sample_tls[:, (1, 0)]
                            / mask_scale).astype(np.int64)

        valid_samples = np.bitwise_and(
            toplefts[:, 0] < shape[-2] - self._patch_size,
            toplefts[:, 1] < shape[-1] - self._patch_size)

        toplefts = toplefts[valid_samples]

        return toplefts
