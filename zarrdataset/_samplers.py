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
    def __init__(self, patch_size, allow_overlap=True, **kwargs):
        super(BlueNoisePatchSampler, self).__init__(patch_size)
        self._overlap = math.sqrt(2) ** (not allow_overlap)

    def _sampling_method(self, mask, shape):
        mask_scale =  mask.shape[-1] / shape[-1]

        rad = self._patch_size * mask_scale
        H, W = mask.shape

        # If the image is smaller than the scaled patch, retrieve the full
        # image instead.
        if H <= rad or W <= rad:
            sample_tls = np.array([[0, 0]], dtype=np.float32)

        else:
            sample_tls = np.array(poisson_disc_samples(height=H - rad,
                                                       width=W - rad,
                                                       r=rad * self._overlap,
                                                       k=30),
                                  dtype=np.float32)

        # If there are ROIs in the mask, only take sampling coordinates inside
        # them.
        if np.any(np.bitwise_not(mask)):
            validsample_tls = np.zeros(len(sample_tls), dtype=bool)

            mask_conts = measure.find_contours(np.pad(mask, 1),
                                               fully_connected='low',
                                               level=0.999)

            mask_paths = [Path(cont[:, (1, 0)] - 1) for cont in mask_conts]

            # Check for holes in the mask
            mask_paths_hierarchy = [[] for _ in range(len(mask_paths))]

            for m, mask_path in enumerate(mask_paths[:-1]):
                for n, test_mask_path in enumerate(mask_paths[m+1:]):
                    n = n + m + 1
                    if mask_path.contains_path(test_mask_path):
                        mask_paths_hierarchy[n].append(m)

                    if test_mask_path.contains_path(mask_path):
                        mask_paths_hierarchy[m].append(n)

            for mask_path in mask_paths:
                active_samples = mask_path.contains_points(sample_tls
                                                           + rad / 2,
                                                           radius=rad)
                validsample_tls[active_samples] = True

            # Remove samples inside holes
            hole_paths = [path
                          for path, hier in zip(mask_paths,
                                                mask_paths_hierarchy)
                          if len(hier) % 2 != 0]

            for hole_path in hole_paths:
                inactive_samples = hole_path.contains_points(sample_tls
                                                             + rad / 2,
                                                             radius=rad)
                validsample_tls[inactive_samples] = False

            toplefts = sample_tls[validsample_tls]

        else:
            toplefts = sample_tls
        
        toplefts = np.round(toplefts[:, (1, 0)] / mask_scale).astype(np.int64)

        return toplefts
