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
    patch_size : int or iterator of ints
        Size in pixels of the patches extracted on each axis. Only squared
        patches (hyper-cubes) are supported by now. If a single int is passed,
        that size is used for each dimension. Use the same convention as dask
        and zarr chunks structure to handle shape and channels correctly.
    """
    def __init__(self, patch_size, spatial_axes="ZYX"):
        # The maximum chunk sizes are used to generate a reference sampling
        # position array used fo every sampled chunk.
        self._max_chunk_size = 0
        self._patch_size = patch_size
        self.spatial_axes = spatial_axes

    def compute_patches(self, *args, **kwargs):
        raise NotImplementedError("This is a virtual method and has to be "
                                  "implemented by the sampling algorithm of "
                                  "the derived class.")

    def compute_chunks(self, image_collection: ImageCollection):
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

        if isinstance(self._patch_size, int):
            self._patch_size = [
                self._patch_size
                for ax in image.axes if ax in self.spatial_axes
                ]

        if isinstance(self._max_chunk_size, int):
            self._max_chunk_size = [
                self._max_chunk_size
                for ax in image.axes if ax in self.spatial_axes
                ]

        spatial_chunk_sizes = [
            chk
            for ax, chk in zip(image.axes, image.chunk_size)
            if ax in self.spatial_axes
            ]

        spatial_shape = [
            s
            for ax, s in zip(image.axes, image.shape)
            if ax in self.spatial_axes
            ]

        spatial_axes = [
            ax
            for ax in image.axes
            if ax in self.spatial_axes
            ]

        self._max_chunk_size = [
            min(max(max_chk, im_chk, ps), s)
            for max_chk, im_chk, ps, s in zip(self._max_chunk_size,
                                              spatial_chunk_sizes,
                                              self._patch_size,
                                              spatial_shape)
            ]

        scaled_chunk_size = tuple(
            [int(self._max_chunk_size[spatial_axes.index(ax)] * scl)
             if ax in spatial_axes else 1
             for ax, scl in zip(mask.axes, mask.scale)])

        full_mask = mask[:]
        valid_mask = transform.downscale_local_mean(full_mask,
                                                    factors=scaled_chunk_size)

        chunks_grids = np.nonzero(valid_mask)

        chunks_grids = [tuple((slice(tl * chk, min(s, (tl + 1) * chk), None)
                              for tl, chk, s in zip(tls, self._max_chunk_size,
                                                    spatial_shape)))
                        for tls in zip(*chunks_grids)]

        return chunks_grids


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

    def compute_patches(self, image_collection: ImageCollection, chunk_tlbr):
        image = image_collection.collection["images"]
        mask = image_collection.collection["masks"]

        mask_spatial_scale = tuple(
            scl
            for scl, ax in zip(mask.scale, mask.axes)
            if ax in self.spatial_axes
            )

        spatial_axes = [
            ax
            for ax in image.axes
            if ax in self.spatial_axes
            ]

        scaled_patch_size = tuple(
            [int(self._max_chunk_size[spatial_axes.index(ax)] * scl)
             if ax in spatial_axes else 1
             for ax, scl in zip(mask.axes, mask.scale)])

        if all(map(operator.ge, scaled_patch_size, repeat(1))):
            valid_mask = transform.downscale_local_mean(
                mask[chunk_tlbr], factors=scaled_patch_size)
        else:
            scaled_patch_size = tuple(map(lambda ps, scl: 1 / (scl * ps),
                                          self._patch_size,
                                          mask_spatial_scale))
            valid_mask = transform.rescale(
                mask[chunk_tlbr], scale=scaled_patch_size)

        toplefts = np.nonzero(valid_mask)

        chk_offsets = [chk.start if chk.start is not None else 0
                       for chk in chunk_tlbr]
        chk_offsets = np.array([chk_offsets], dtype=np.int64)

        toplefts = np.stack(toplefts).T * np.array([self._patch_size],
                                                   dtype=np.int64)
        toplefts = toplefts + chk_offsets
        bottomrights = toplefts + np.array([self._patch_size], dtype=np.int64)

        spatial_shape = [s for ax, s in zip(image.axes, image.shape)
                         if ax in self.spatial_axes]

        spatial_limits = np.array([spatial_shape], dtype=np.int64)

        samples_validity = np.all(bottomrights <= spatial_limits, axis=1)

        toplefts = toplefts[samples_validity, ...]
        bottomrights = bottomrights[samples_validity, ...]

        toplefts = list(map(lambda tls, brs:
                            tuple(map(lambda tl, br:
                                      slice(tl, br, None), tls, brs)),
                            toplefts,
                            bottomrights))

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

    def _get_positions_array(self, force=False):
        if self._base_chunk_tls is None or force:
            self._sampling_positions = np.meshgrid(
                *[np.linspace(0, ps - 1, self._sampling_freq)
                  for ps in self._patch_size])
            self._sampling_positions = np.stack(self._sampling_positions)
            self._sampling_positions = \
                self._sampling_positions.reshape(len(self._patch_size), -1).T

            dims = np.array(self._max_chunk_size)
            radius = max(self._patch_size)

            if radius < min(dims):
                self._base_chunk_tls = pd.Bridson_sampling(dims=dims - radius,
                                                           radius=radius,
                                                           k=30)
                self._base_chunk_tls = self._base_chunk_tls.astype(np.int64)

            else:
                self._base_chunk_tls = np.zeros((len(self._patch_size), 1),
                                                dtype=np.int64)

        base_chunk_patches = self._base_chunk_tls[None, :, :]\
            + self._sampling_positions[:, None, :]

        base_chunk_patches = base_chunk_patches.reshape(-1,
                                                        len(self._patch_size))

        return base_chunk_patches

    def compute_patches(self, image_collection: ImageCollection, chunk_tlbr):
        image = image_collection.collection["images"]
        mask = image_collection.collection["masks"]

        chunk_patches = self._get_positions_array(force=False)

        chunk_valid_mask = mask[chunk_tlbr]
        chunk_valid_mask_grid = map(lambda chk, s:
                                    np.linspace(0, chk, s),
                                    self._max_chunk_size,
                                    chunk_valid_mask.shape)
        chunk_valid_mask_grid = np.meshgrid(
            *reversed(tuple(chunk_valid_mask_grid))
            )
    
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
            ).reshape(-1, self._sampling_freq ** 2)

        samples_validity = np.any(samples_validity, axis=1)
        toplefts = self._base_chunk_tls[samples_validity, ...]

        chk_offsets = [chk.start if chk.start is not None else 0
                       for chk in chunk_tlbr]
        chk_offsets = np.array([chk_offsets], dtype=np.int64)

        toplefts = toplefts + chk_offsets

        bottomrights = toplefts + np.array([self._patch_size], dtype=np.int64)

        spatial_shape = [s for ax, s in zip(image.axes, image.shape)
                         if ax in self.spatial_axes]

        spatial_limits = np.array([spatial_shape], dtype=np.int64)

        samples_validity = np.all(bottomrights <= spatial_limits, axis=1)

        toplefts = toplefts[samples_validity, ...]
        bottomrights = bottomrights[samples_validity, ...]

        toplefts = list(map(lambda tls, brs:
                            tuple(map(lambda tl, br:
                                      slice(tl, br, None), tls, brs)),
                            toplefts,
                            bottomrights))
        return toplefts
