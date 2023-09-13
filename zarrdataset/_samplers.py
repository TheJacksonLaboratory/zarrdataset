from typing import Iterable
import math
import numpy as np
import operator
from itertools import repeat

from skimage import transform
from scipy import interpolate
import poisson_disc as pd

from ._imageloaders import ImageCollection


class PatchSampler(object):
    """Base class to implement patches sampler from images.

    If the image used for extracting patches has a mask associated to it, only
    patches from masked regions of the image are retrieved.

    Parameters
    ----------
    patch_size : int or iterator of ints, dict
        Size in pixels of the patches extracted on each axis. Only rectangular
        patches (hyper-cuboids) are supported by now. If a single int is
        passed, that size is used for all dimensions. If an iterable (list,
        tuple) is passed, each value will be assigned to the corresponding axes
         in `spatial_axes`, the size of `patch_size` must match the lenght of 
        `spatial_axes'. If a dict is passed, this should have at least the size
         of the patch of the axes listed in `spatial_axes`.  Use the same 
        convention as how Zarr structure array chunks in order to handle patch 
        shapes and channels correctly.
    spatial_axes : str
        The spatial axes from where patches can be extracted.
    """
    def __init__(self, patch_size: (int, Iterable[int], dict),
                 spatial_axes="ZYX"):
        # The maximum chunk sizes are used to generate a reference sampling
        # position array used fo every sampled chunk.
        self._max_chunk_size = dict((ax, 0) for ax in spatial_axes)

        if isinstance(patch_size, (list, tuple)):
            patch_size = dict((ax, ps)
                              for ax, ps in zip(spatial_axes, patch_size))

        elif isinstance(patch_size, int):
            patch_size = dict((ax, patch_size) for ax in spatial_axes)

        elif not isinstance(patch_size, dict):
            raise ValueError(f"Patch size must be a dictionary specifying the"
                             f" patch size of each axes, an iterable (list, "
                             f"tuple) with the same order as the spatial axes,"
                             f" or an integer for a cubic patch. Received "
                             f"{patch_size} of type {type(patch_size)}")

        self.spatial_axes = spatial_axes

        self._patch_size = dict(
            (ax, patch_size.get(ax, 1))
            for ax in spatial_axes
        )

    def compute_chunks(self, image_collection: ImageCollection) -> np.ndarray:
        """Compute the top-left positions of the chunks in which the image is
        divided before taking patches from them.

        The chunk size should match the zarr's chunk sizes to make the patch
        extraction more efficient.

        Parameters
        ----------
        image : zarrdataset.ImageCollection
            The image from where the patches are extracted.

        Returns
        -------
        chunks_grids : list of tuples of slices
            Each valid chunk is returned in form of a slice that can be
            extracted from the image. The slices are stored as tuples, with one
            slice per axis.
        """
        # Use the mask as base to determine the valid patches that can be
        # retrieved from the image.
        image = image_collection.collection["images"]
        mask = image_collection.collection["masks"]

        if isinstance(self._max_chunk_size, int):
            self._max_chunk_size = [
                self._max_chunk_size
                for ax in image.axes if ax in self.spatial_axes
                ]

        spatial_chunk_sizes = dict(
            (ax, chk)
            for ax, chk in zip(image.axes, image.chunk_size)
            if ax in self.spatial_axes
        )

        spatial_shape = dict(
            (ax, s)
            for ax, s in zip(image.axes, image.shape)
            if ax in self.spatial_axes
        )

        self._max_chunk_size = dict(
            (ax, (min(max(self._max_chunk_size[ax],
                          spatial_chunk_sizes[ax],
                          self._patch_size[ax]),
                      spatial_shape[ax]))
             if ax in image.axes else 1)
            for ax in self.spatial_axes
        )

        spatial_chunks_ratios = tuple(
            int(math.ceil(spatial_shape[ax] / self._max_chunk_size[ax]))
            for ax in mask.axes
        )

        spatial_mask_ratios = np.array(
            [[spatial_shape[ax] / m_s
              for ax, m_s in zip(mask.axes, mask.shape)
              if ax in self.spatial_axes
             ]]
        )

        spatial_chunk_sizes = np.array(
            [[self._max_chunk_size[ax]
              for ax in mask.axes
              if ax in self.spatial_axes
             ]]
        )

        # Expand the mask array to treat the positions as corners
        mask_corners = transform.rescale(mask[:], scale=2)

        # Scale the positions of the corners to the size of the reference image
        chunks_grids = np.stack(np.nonzero(mask_corners)).T / 2
        chunks_grids = chunks_grids * spatial_mask_ratios

        # Rescale the positions to determine the chunks that contain any
        # portion of the mask
        chunks_grids = np.floor(chunks_grids / spatial_chunk_sizes)
        chunks_grids = chunks_grids.astype(np.int64)
        chunks_grids = np.ravel_multi_index(chunks_grids.T,
                                            spatial_chunks_ratios)

        # Remove any repeated position. This can happen if more than one mask
        # corner is inside a chunk
        chunks_grids = np.unique(chunks_grids)
        chunks_grids = np.unravel_index(chunks_grids, spatial_chunks_ratios)

        chunks_grids = tuple(
            chunks_grids[mask.axes.index(ax)]
            if ax in mask.axes else repeat(0)
            for ax in self.spatial_axes
        )

        # Generate the slices for each masked chunk
        chunks_grids = [
            dict((ax, slice(tl * self._max_chunk_size[ax],
                            (tl + 1) * self._max_chunk_size[ax],
                            None)
                 )
                 for ax, tl in zip(self.spatial_axes, tls))
            for tls in zip(*chunks_grids)
        ]

        return chunks_grids

    def _compute_toplefts(self, image_collection: ImageCollection, chunk_tlbr):
        raise NotImplementedError("This is a virtual method and has to be "
                                  "implemented by the sampling algorithm of "
                                  "the derived class.")

    def compute_patches(self, image_collection: ImageCollection, chunk_tlbr):
        image = image_collection.collection["images"]

        toplefts = self._compute_toplefts(image_collection, chunk_tlbr)

        chk_offsets = dict(
            (ax,
             chunk_tlbr[ax].start if chunk_tlbr[ax].start is not None else 0)
            for ax in self.spatial_axes
        )

        spatial_shape = dict(
            (ax, s)
            for ax, s in zip(image.axes, image.shape)
            if ax in self.spatial_axes
        )

        toplefts = [
            dict(
                (ax, slice(tls[ax] + chk_offsets[ax],
                           tls[ax] + self._patch_size[ax] + chk_offsets[ax],
                           None))
                 for ax in self.spatial_axes
            )
            for tls in toplefts
            if all(
                (tls[ax] + self._patch_size[ax] + chk_offsets[ax])
                <= spatial_shape[ax]
                for ax in self.spatial_axes
                if ax in image.axes
               )
        ]

        return toplefts


class GridPatchSampler(PatchSampler):
    """Patch sampler that retrieves non-overlapping patches from a grid of
    coordinates that cover all the masked image.

    Parameters
    ----------
    patch_size : int
        Size in pixels of the patches extracted. Only squared patches are
        supported by now.
    """
    def __init__(self, patch_size: (int, Iterable[int], dict), **kwargs):
        super(GridPatchSampler, self).__init__(patch_size)

    def _compute_toplefts(self, image_collection: ImageCollection, chunk_tlbr):
        image = image_collection.collection["images"]
        mask = image_collection.collection["masks"]

        spatial_shape = dict(
            (ax, s)
            for ax, s in zip(image.axes, image.shape)
            if ax in self.spatial_axes
        )

        spatial_patch_ratios = tuple(
            int(math.ceil(spatial_shape[ax] / self._patch_size[ax]))
            for ax in mask.axes
        )

        spatial_mask_ratios = np.array(
            [[spatial_shape[ax] / m_s
              for ax, m_s in zip(mask.axes, mask.shape)
              if ax in self.spatial_axes
             ]]
        )

        spatial_patch_sizes = np.array(
            [[self._patch_size[ax]
              for ax in mask.axes
              if ax in self.spatial_axes
             ]]
        )

        # Expand the mask array to treat the positions as corners
        mask_corners = transform.rescale(mask[chunk_tlbr], scale=2)

        # Scale the positions of the corners to the size of the reference image
        toplefts = np.stack(np.nonzero(mask_corners)).T / 2
        toplefts = toplefts * spatial_mask_ratios

        # Rescale the positions to determine the chunks that contain any
        # portion of the mask
        toplefts = np.floor(toplefts / spatial_patch_sizes)
        toplefts = toplefts.astype(np.int64)
        toplefts = np.ravel_multi_index(toplefts.T,
                                        spatial_patch_ratios)

        # Remove any repeated position. This can happen if more than one mask
        # corner is inside a chunk
        toplefts = np.unique(toplefts)
        toplefts = np.unravel_index(toplefts, spatial_patch_ratios)
        toplefts = tuple(
            toplefts[mask.axes.index(ax)]
            if ax in mask.axes else repeat(0)
            for ax in self.spatial_axes
        )

        # Generate the top-left positions for each patch in the chunk
        toplefts = [
            dict(
                (ax, tl * self._patch_size[ax])
                for ax, tl in zip(self.spatial_axes, tls)
            )
            for tls in zip(*toplefts)
        ]

        return toplefts


class BlueNoisePatchSampler(PatchSampler):
    """Patch sampler that retrieves patches from coordinates sampled using the
    Bridson sampling algorithm, also known as Blue-noise sampling algorithm.

    Parameters
    ----------
    patch_size : int, iterable, dict
        Size in pixels of the patches extracted. Only squared patches are
        supported by now.
    allow_overlap : bool
        Whether overlapping of patches is allowed or not.
    """
    def __init__(self, patch_size: (int, Iterable[int], dict),
                 sampling_freq: int = 8,
                 **kwargs):
        super(BlueNoisePatchSampler, self).__init__(patch_size)
        self._base_chunk_tls = None
        self._sampling_positions = None

        if isinstance(sampling_freq, (list, tuple)):
            sampling_freq = dict(
                (ax, sf)
                for ax, sf in zip(self.spatial_axes, sampling_freq)
            )

        elif isinstance(sampling_freq, int):
            sampling_freq = dict(
                (ax, sampling_freq)
                for ax in self.spatial_axes
            )

        elif not isinstance(sampling_freq, dict):
            raise ValueError(f"Sampling frequency must be a dictionary "
                             f"specifying the sampling frequency for each axes"
                             f", an iterable (list, tuple) with the same order"
                             f" as the spatial axes, or an integer for a cubic"
                             f" patch. Received {sampling_freq} of type "
                             f"{type(sampling_freq)}")

        self._sampling_freq = dict(
            (ax, sampling_freq.get(ax, 1))
            for ax in self.spatial_axes
        )

    def _get_positions_array(self, image_axes, force=False):
        if self._base_chunk_tls is None or force:
            self._sampling_positions = np.meshgrid(
                *[np.linspace(0, self._patch_size[ax] - 1,
                              self._sampling_freq[ax])
                  for ax in self.spatial_axes]
            )

            self._sampling_positions = np.stack(self._sampling_positions)
            self._sampling_positions = \
                self._sampling_positions.reshape(len(self.spatial_axes), -1).T

            radius = max(self._patch_size.values())
            samplable_axes = [ax for ax in self.spatial_axes
                              if self._max_chunk_size[ax] > radius]
            dims = [self._max_chunk_size[ax] - radius for ax in samplable_axes]

            if len(samplable_axes):
                base_chunk_tls = pd.Bridson_sampling(dims=np.array(dims),
                                                     radius=radius,
                                                     k=30)

                base_chunk_tls = [base_chunk_tls[:, samplable_axes.index(ax)]
                                  if ax in samplable_axes else
                                  np.zeros(base_chunk_tls.shape[0],
                                           dtype=np.float32)
                                  for ax in self.spatial_axes]
                base_chunk_tls = np.stack(base_chunk_tls).T

                self._base_chunk_tls = base_chunk_tls.astype(np.int64)

            else:
                self._base_chunk_tls = np.zeros((1, len(self.spatial_axes)),
                                                dtype=np.int64)

        base_chunk_patches = self._base_chunk_tls[None, :, :]\
            + self._sampling_positions[:, None, :]

        base_chunk_patches = base_chunk_patches.reshape(-1,
                                                        len(self.spatial_axes))

        base_chunk_patches = np.stack([
            base_chunk_patches[:, self.spatial_axes.index(ax)]
            for ax in image_axes
            if ax in self.spatial_axes
        ]).T

        return base_chunk_patches

    def _compute_toplefts(self, image_collection: ImageCollection, chunk_tlbr):
        image = image_collection.collection["images"]
        mask = image_collection.collection["masks"]

        chunk_patches = self._get_positions_array(image.axes, force=False)

        chunk_valid_mask = mask[chunk_tlbr]

        mask_shape = dict(
            (ax,  chunk_valid_mask.shape[mask.axes.index(ax)]
                  if ax in mask.axes else 1
            )
            for ax in self.spatial_axes
        )

        chunk_valid_mask_grid = tuple(
            np.linspace(0, self._max_chunk_size[ax], mask_shape[ax])
            for ax in image.axes
            if ax in self.spatial_axes
        )

        chunk_valid_mask_grid = np.meshgrid(*chunk_valid_mask_grid,
                                            indexing='ij')

        chunk_valid_mask_grid = [
            np.pad(grid, 1, mode='constant', constant_values=1e12)
            for grid in chunk_valid_mask_grid
        ]

        chunk_valid_mask_grid = tuple(
            ax.flatten() for ax in chunk_valid_mask_grid)

        chunk_valid_mask = np.pad(chunk_valid_mask, 1, mode='constant',
                                  constant_values=0)

        samples_validity = interpolate.griddata(
            chunk_valid_mask_grid,
            chunk_valid_mask.flatten(),
            chunk_patches,
            method='nearest'
        )

        samples_validity = samples_validity.reshape(
            self._base_chunk_tls.shape[0], -1)

        samples_validity = np.any(samples_validity, axis=1)

        toplefts = self._base_chunk_tls[samples_validity]

        # Generate the top-left positions for each patch in the chunk
        toplefts = [
            dict(
                (ax, tls[mask.axes.index(ax)]
                     if ax in mask.axes else 0
                )
                for ax in self.spatial_axes
            )
            for tls in toplefts
        ]

        return toplefts
