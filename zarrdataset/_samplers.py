import math
import numpy as np

from skimage import transform
from scipy import interpolate
from bridson import poisson_disc_samples

from ._utils import map_axes_order


class PatchSampler(object):
    """Base class to implement patches sampler from images.

    If the image used for extracting patches has a mask associated to it, only
    patches from masked regions of the image are retrieved.

    Parameters
    ----------
    patch_size : int
        Size in pixels of the patches extracted. Only squared patches are
        supported by now.
    chunk: tuple of ints
        Size in pixels of the chunks the image is sub-divided before extracting
        the patches.
    """
    def __init__(self, patch_size):
        self._patch_size = patch_size

    def compute_patches(self, *args, **kwargs):
        raise NotImplementedError("This is a virtual class and has to be "
                                  "inherited from other class implementing the"
                                  " actual sampling method.")

    def compute_chunks(self, image):
        """Compute the top-left positions of the chunks in which the image is
        divided before taking patches from them.

        The chunk size should match the zarr's chunk sizes to make the patch
        extraction more efficient.

        Parameters
        ----------
        image : zarrdataset.ImageLoader
            The image from where the patches are extracted.

        Returns
        -------
        tl_brs : numpy.ndarray
            An array containing the coordinates of the chunks that can be
            extracted from `image`. Each row in the array has the coordinates
            in the following format.
            (top-left y, top-left x)
        """
        # Use the mask as base to determine the valid patches that can be
        # retrieved from the image.
        ax_ref_ord = map_axes_order(image.data_axes, "YX")

        H = image.shape[ax_ref_ord[-2]]
        W = image.shape[ax_ref_ord[-1]]

        chk_H = max(image.chunk_size[ax_ref_ord[-2]], self._patch_size)
        chk_W = max(image.chunk_size[ax_ref_ord[-1]], self._patch_size)

        valid_mask = transform.resize(image.mask, (round(H / chk_H),
                                                   round(W / chk_W)),
                                      order=0,
                                      mode="edge",
                                      anti_aliasing=False)

        # Retrieve patches that contain at least the requested minimum presence
        # of the masked object.
        toplefts = np.nonzero(valid_mask)
        toplefts = np.stack(toplefts, axis=1)

        # Map the positions back to the original scale of the image.
        toplefts[:, 0] = toplefts[:, 0] * chk_H
        toplefts[:, 1] = toplefts[:, 1] * chk_W

        bottomrights_y = np.minimum(toplefts[:, 0] + chk_H, H).reshape(-1, 1)
        bottomrights_x = np.minimum(toplefts[:, 1] + chk_W, W).reshape(-1, 1)

        toplefts = np.hstack((toplefts, bottomrights_y, bottomrights_x))

        return toplefts

    @staticmethod
    def _get_chunk_mask(image, chunk_tlbr):
        mask_roi = [slice(0, 1, None)] * (len(image.mask_data_axes) - 2)
        mask_roi += [slice(round(chunk_tlbr[0] * image.mask_scale),
                            round(chunk_tlbr[2] * image.mask_scale),
                            None),
                        slice(round(chunk_tlbr[1] * image.mask_scale),
                            round(chunk_tlbr[3] * image.mask_scale),
                            None)]
        mask_roi = tuple(mask_roi[a]
                            for a in map_axes_order(image.mask_data_axes,
                                                    "YX"))

        # To get samples only inside valid areas of the mask
        return image.mask[mask_roi]


class GridPatchSampler(PatchSampler):
    """Patch sampler that retrieves non-overlapping patches from a grid of
    coordinates that cover all the masked image.

    Parameters
    ----------
    patch_size : int
        Size in pixels of the patches extracted. Only squared patches are
        supported by now.
    """
    def __init__(self, patch_size, **kwargs):
        super(GridPatchSampler, self).__init__(patch_size)

    def compute_patches(self, image, chunk_tlbr):
        chk_H = chunk_tlbr[2] - chunk_tlbr[0]
        chk_W = chunk_tlbr[3] - chunk_tlbr[1]

        chunk_valid_mask = self._get_chunk_mask(image, chunk_tlbr)

        if chk_H < self._patch_size or chk_W < self._patch_size:
            # If the chunk area is smaller than the patch size, return an empty
            # set of topleft positions.
            toplefts = np.empty((0, 4), dtype=np.int64)

        else:
            valid_mask = transform.resize(chunk_valid_mask,
                                        (chk_H // self._patch_size,
                                        chk_W // self._patch_size),
                                        order=0,
                                        mode="edge",
                                        anti_aliasing=False)

            toplefts = np.nonzero(valid_mask)
            toplefts = np.stack(toplefts, axis=1) * self._patch_size
            toplefts = np.hstack((toplefts, toplefts + self._patch_size))

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

    def compute_patches(self, image, chunk_tlbr):
        chk_H = chunk_tlbr[2] - chunk_tlbr[0]
        chk_W = chunk_tlbr[3] - chunk_tlbr[1]

        rad = self._patch_size * self._overlap

        if chk_H < self._patch_size or chk_W < self._patch_size:
            # If the chunk area is smaller than the patch size, return an empty
            # set of topleft positions.
            toplefts = np.empty((0, 4), dtype=np.int64)

        elif chk_H <= rad or chk_W <= rad:
            # If the chunk area is smaller than the radious of a single patch,
            # return the position to the topleft corner of the chunk.
            toplefts = np.array([[0, 0, self._patch_size, self._patch_size]],
                                dtype=np.int64)

        else:
            chunk_valid_mask = self._get_chunk_mask(image, chunk_tlbr)
            chunk_valid_mask = np.pad(chunk_valid_mask, 1)

            chunk_valid_mask_grid = np.meshgrid(
                np.arange(chunk_valid_mask.shape[0]),
                np.arange(chunk_valid_mask.shape[1]))

            chunk_tls = np.array(poisson_disc_samples(width=chk_W - rad,
                                                      height=chk_H - rad,
                                                      r=rad,
                                                      k=30),
                                 dtype=np.float32)

            chunk_patches = np.hstack(
                (chunk_tls,
                 chunk_tls[:, :1] + self._patch_size,
                 chunk_tls[:, 1:],
                 chunk_tls[:, :1] + self._patch_size,
                 chunk_tls[:, 1:] + self._patch_size,
                 chunk_tls[:, :1],
                 chunk_tls[:, 1:] + self._patch_size)).reshape(-1, 2)

            samples_validity = interpolate.griddata(
                tuple(ax.flatten() for ax in chunk_valid_mask_grid),
                chunk_valid_mask.flatten(),
                chunk_patches * image.mask_scale + 1,
                method='nearest'
                ).reshape(-1, 4)

            samples_validity = samples_validity.any(axis=1)

            toplefts = np.round(chunk_tls[samples_validity][:, (1, 0)])
            toplefts = toplefts.astype(np.int64).reshape(-1, 2)

            toplefts = np.hstack((toplefts, toplefts + self._patch_size))

        return toplefts
