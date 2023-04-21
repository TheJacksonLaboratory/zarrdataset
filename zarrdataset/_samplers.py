import math
import numpy as np

from skimage import transform, measure
from matplotlib.path import Path
from bridson import poisson_disc_samples



class PatchSampler(object):
    def __init__(self, patch_size):
        self._patch_size = patch_size

    def _sampling_method(self, *args, **kwargs):
        raise NotImplementedError("This is a virtual class and has to be "
                                  "inherited from other class implementing the"
                                  " actual sampling method.")

    def compute_toplefts(self, image):
        curr_tls = self._sampling_method(image.mask, image.shape)
        curr_brs = curr_tls + self._patch_size
        toplefts = np.hstack((curr_tls, curr_brs))

        return toplefts


class GridPatchSampler(PatchSampler):
    def __init__(self, patch_size, min_object_presence=0.1, **kwargs):
        super(GridPatchSampler, self).__init__(patch_size)
        self._min_object_presence = min_object_presence

    def _sampling_method(self, mask, shape):
        mask_scale = mask.shape[-1] / shape[-1]
        scaled_ps = self._patch_size * mask_scale

        if scaled_ps < 1:
            mask_scale = 1.0
            scaled_h = round(shape[-2] / self._patch_size)
            scaled_w = round(shape[-1] / self._patch_size)
            dwn_valid_mask = transform.resize(mask, (scaled_h, scaled_w),
                                              order=0,
                                              mode="edge",
                                              anti_aliasing=False)

        else:
            scaled_ps = round(scaled_ps)
            scaled_h = mask.shape[-2] // scaled_ps
            scaled_w = mask.shape[-1] // scaled_ps

            dwn_valid_mask = transform.downscale_local_mean(
                mask, factors=(scaled_ps, scaled_ps))

            dwn_valid_mask = dwn_valid_mask[:scaled_h, :scaled_w]

        valid_mask = dwn_valid_mask > self._min_object_presence

        toplefts = np.nonzero(valid_mask)
        toplefts = np.stack(toplefts, axis=1)
        toplefts = toplefts * self._patch_size

        return toplefts


class BlueNoisePatchSampler(PatchSampler):
    def __init__(self, patch_size, allow_overlap=True, **kwargs):
        super(BlueNoisePatchSampler, self).__init__(patch_size)
        self._overlap = math.sqrt(2) ** (not allow_overlap)

    def _sampling_method(self, mask, shape):
        mask_scale =  mask.shape[-1] / shape[-1]

        rad = self._patch_size * mask_scale
        H, W = mask.shape

        if H <= rad or W <= rad:
            sample_tls = np.array([[0, 0]], dtype=np.float32)

        else:
            sample_tls = np.array(poisson_disc_samples(height=H - rad,
                                                       width=W - rad,
                                                       r=rad * self._overlap,
                                                       k=30),
                                  dtype=np.float32)

        # If there are ROIs in the mask, take the sampling positions that
        # are inside them.
        if np.any(np.bitwise_not(mask)):
            validsample_tls = np.zeros(len(sample_tls), dtype=bool)
            mask_conts = measure.find_contours(np.pad(mask, 1),
                                               fully_connected='low',
                                               level=0.999)

            for cont in mask_conts:
                mask_path = Path(cont[:, (1, 0)] - 1)
                validsample_tls = np.bitwise_or(
                    validsample_tls,
                    mask_path.contains_points(sample_tls + rad / 2, 
                                              radius=rad))

            toplefts = sample_tls[validsample_tls]

        else:
            toplefts = sample_tls

        toplefts = np.round(toplefts / mask_scale).astype(np.int64)

        return toplefts
