import math
import random
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

        # The maximum chunk sizes (H, W) are used to generate a reference
        # sampling position array used fo every sampled chunk.
        self._max_chk_H = 0
        self._max_chk_W = 0

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

        im_chk_H = image.chunk_size[ax_ref_ord[-2]]
        im_chk_W = image.chunk_size[ax_ref_ord[-1]]

        self._max_chk_H = max(self._max_chk_H, im_chk_H)
        self._max_chk_W = max(self._max_chk_W, im_chk_W)

        if self._patch_size >= im_chk_H:
            im_chk_H = self._patch_size
        
        if self._patch_size >= im_chk_W:
            im_chk_W = self._patch_size

        scaled_H = round(im_chk_H * image.mask_scale)
        scaled_W = round(im_chk_W * image.mask_scale)

        valid_mask = transform.downscale_local_mean(image.mask, 
                                                    factors=(scaled_H,
                                                             scaled_W),
                                                    cval=0)

        chunks_grid_y, chunks_grid_x = np.nonzero(valid_mask)
        chunks_grid_y = chunks_grid_y.reshape(-1, 1) * im_chk_H
        chunks_grid_x = chunks_grid_x.reshape(-1, 1) * im_chk_W

        chunks_grid = np.hstack(
            (chunks_grid_y, chunks_grid_x,
             chunks_grid_y + im_chk_H, chunks_grid_x + im_chk_W))

        return chunks_grid

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
                         for a in map_axes_order(image.mask_data_axes, "YX"))

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
    def __init__(self, patch_size, sampling_freq=8, **kwargs):
        super(BlueNoisePatchSampler, self).__init__(patch_size)
        self._base_chunk_tls = None
        self._sampling_positions = None
        self._sampling_freq = sampling_freq

    def _get_positions_array(self, chk_H, chk_W, force=False):
        if self._sampling_positions is None:
            sampling_pos_y, sampling_pos_x = np.meshgrid(
                np.linspace(0, self._patch_size, self._sampling_freq),
                np.linspace(0, self._patch_size, self._sampling_freq)
            )
            self._sampling_positions = np.hstack(
                (sampling_pos_y.reshape(-1, 1),
                 sampling_pos_x.reshape(-1, 1))).reshape(1, -1, 2)

        if self._base_chunk_tls is None or force:
            self._max_chk_H = max(self._max_chk_H, chk_H)
            self._max_chk_W = max(self._max_chk_W, chk_W)

            if (self._max_chk_H < self._patch_size
              or self._max_chk_W < self._patch_size):
                self._base_chunk_tls = None
                return None, None

            elif (self._max_chk_H == self._patch_size
              or self._max_chk_W == self._patch_size):
                grid_y = np.arange(0, self._max_chk_H, self._patch_size,
                                dtype=np.int64)
                grid_x = np.arange(0, self._max_chk_W, self._patch_size,
                                dtype=np.int64)
                self._base_chunk_tls = np.hstack((
                    grid_y.reshape(-1, 1),
                    grid_x.reshape(-1, 1)))
            else:
                self._base_chunk_tls = np.array(
                    poisson_disc_samples(width=self._max_chk_W
                                               - self._patch_size,
                                         height=self._max_chk_H
                                                - self._patch_size,
                                         r=self._patch_size,
                                         k=30),
                    dtype=np.int64)
                self._base_chunk_tls = self._base_chunk_tls[:, (1, 0)]

        valid_in_chunk = np.bitwise_and(
            self._base_chunk_tls[:, 0] + self._patch_size <= chk_H,
            self._base_chunk_tls[:, 1] + self._patch_size <= chk_W)

        base_chunk_tls = self._base_chunk_tls[valid_in_chunk]

        base_chunk_patches = (base_chunk_tls[:, None, :]
                              + self._sampling_positions)

        base_chunk_patches = base_chunk_patches.reshape(-1, 2)

        return base_chunk_tls, base_chunk_patches

    def compute_patches(self, image, chunk_tlbr):
        chk_H = chunk_tlbr[2] - chunk_tlbr[0]
        chk_W = chunk_tlbr[3] - chunk_tlbr[1]

        if chk_H < self._patch_size or chk_W < self._patch_size:
            # If the chunk area is smaller than the patch size, return an empty
            # set of topleft positions.
            toplefts = np.empty((0, 4), dtype=np.int64)

        else:
            (chunk_tls,
             chunk_patches) = self._get_positions_array(chk_H=chk_H,
                                                        chk_W=chk_W,
                                                        force=False)
            chunk_valid_mask = self._get_chunk_mask(image, chunk_tlbr)

            chunk_valid_mask_grid = np.meshgrid(
                [-1]
                + list(np.linspace(0, chk_H - 1, chunk_valid_mask.shape[0]))
                + [chk_H],
                [-1]
                + list(np.linspace(0, chk_W - 1, chunk_valid_mask.shape[1]))
                + [chk_W])

            chunk_valid_mask = np.pad(chunk_valid_mask, 1)
 
            samples_validity = interpolate.griddata(
                tuple(ax.flatten() for ax in chunk_valid_mask_grid),
                chunk_valid_mask.flatten(),
                chunk_patches,
                method='nearest'
                ).reshape(-1, self._sampling_freq ** 2)

            samples_validity = samples_validity.any(axis=1)

            toplefts = np.round(chunk_tls[samples_validity])
            toplefts = toplefts.astype(np.int64).reshape(-1, 2)

            toplefts = np.hstack((toplefts, toplefts + self._patch_size))

        return toplefts
