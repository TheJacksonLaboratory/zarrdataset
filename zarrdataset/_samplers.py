import numpy as np

from skimage import transform
from scipy import interpolate
# from bridson import poisson_disc_samples
import poisson_disc as pd

from ._utils import map_axes_order
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
        self._max_chk = 0
        self.patch_size = patch_size
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

        if isinstance(self.patch_size, int):
            self.patch_size = [self.patch_size
                               for ax in image.axes if ax in self.spatial_axes]

        if isinstance(self._max_chk, int):
            self._max_chk = [self._max_chk
                             for ax in image.axes if ax in self.spatial_axes]

        image_spatial_axes = [ax
                              for ax in image.axes if ax in self.spatial_axes]

        spatial_chunk_sizes = [chk
                               for ax, chk in zip(image.axes, image.chunk_size)
                               if ax in image_spatial_axes]

        spatial_shape = [s
                         for ax, s in zip(image.axes, image.shape)
                         if ax in image_spatial_axes]

        self._max_chk = [
            min(max(max_chk, im_chk), s)
            for max_chk, im_chk, s in zip(self._max_chk, spatial_chunk_sizes,
                                          spatial_shape)
            ]

        chunk_sizes = tuple(
            map(lambda ps, chk, s: min(ps, s) if ps >= chk else chk,
                self.patch_size,
                spatial_chunk_sizes,
                spatial_shape))

        mask_factor = tuple(
            max(1, round(chk * mask.scale[mask.axes.index(ax)]))
            for chk, ax in zip(self._max_chk, image_spatial_axes))

        valid_mask = transform.downscale_local_mean(mask[:],
                                                    factors=mask_factor)

        chunks_grids = np.nonzero(valid_mask)

        chunks_grids = (chk_grid.reshape(-1) * chk
                        for chk_grid, chk in zip(chunks_grids, chunk_sizes))

        chunks_grids = (
            zip(chk_grid, np.minimum(chk_grid + chk, s))
            for chk_grid, chk, s in zip(chunks_grids, chunk_sizes, image.shape)
            )

        chunks_grids = (
            map(lambda start_stop: slice(start_stop[0], start_stop[1], None),
                start_stop)
            for start_stop in chunks_grids
            )

        chunks_grids = [tuple(ax_slice) for ax_slice in zip(*chunks_grids)]

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

        image_spatial_axes = [ax
                              for ax in image.axes if ax in self.spatial_axes]

        mask_factor = tuple(max(1, round(ps * mask.scale[mask.axes.index(ax)]))
                            for ps, ax in zip(self.patch_size,
                                              image_spatial_axes))

        valid_mask = transform.downscale_local_mean(mask[chunk_tlbr],
                                                    factors=mask_factor)

        toplefts = np.nonzero(valid_mask)

        toplefts = (zip(tl * ps, (tl + 1) * ps)
                    for tl, ps in zip(toplefts, self.patch_size))
        toplefts = ([slice(start, stop, None) for start, stop in start_stop]
                    for start_stop in toplefts)

        toplefts = [tuple(ax_slice) for ax_slice in zip(*toplefts)]

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
        self._samp_pos = None
        self._sampling_freq = sampling_freq

    def _get_positions_array(self, chunk_sizes, axes, force=False):
        spt_ps = []
        spt_mx = []
        spt_sz = []
        for ps, mx, sz, ax in zip(self.patch_size, self._max_chk, chunk_sizes,
                                  axes):
            if ax in self.spatial_axes:
                spt_ps.append(ps)
                spt_mx.append(mx)
                spt_sz.append(sz)

        if self._samp_pos is None:
            samp_pos = np.meshgrid(
                *[np.linspace(0, ps, self._sampling_freq)
                  for ps in spt_ps]
            )
            self._samp_pos = np.hstack(samp_pos).reshape(1, -1, len(spt_ps))

        if self._base_chunk_tls is None or force:
            self._max_chk = [
                max(max_chk, im_chk)
                for max_chk, im_chk in zip(self._max_chk, chunk_sizes)]
            spt_mx = [
                mx
                for mx, ax in zip(self._max_chk, axes)
                if ax in self.spatial_axes
                ]

            if any(map(lambda chk, ps: chk < ps, spt_mx, spt_ps)):
                self._base_chunk_tls = None
                return None, None

            elif any(map(lambda chk, ps: chk == ps, spt_mx, spt_ps)):
                self._base_chunk_tls = np.hstack(
                    [np.arange(0, chk, ps, dtype=np.int64).reshape(-1, 1)
                     for chk, ps in zip(spt_sz, spt_ps)])

            else:
                dims = np.array(spt_mx) - np.array(spt_ps)
                radius = max(spt_ps)

                self._base_chunk_tls = pd.Bridson_sampling(dims=dims,
                                                           radius=radius,
                                                           k=30)
                self._base_chunk_tls = self._base_chunk_tls.astype(np.int64)

        base_chunk_patches = self._base_chunk_tls[:, None, :] + self._samp_pos

        base_chunk_patches = base_chunk_patches.reshape(-1, len(spt_ps))

        return base_chunk_patches

    def compute_patches(self, image_collection: ImageCollection, chunk_tlbr):
        image = image_collection.collection[image_collection.reference_mode]
        scale = image_collection.scale
        shape = image_collection.shape
        axes = image_collection.axes

        if isinstance(self.patch_size, int):
            self.patch_size = [
                self.patch_size if ax in self.spatial_axes else s
                for ax, scl, s in zip(axes, scale, shape)
                ]

        if isinstance(self._max_chk, int):
            self._max_chk = [
                self._max_chk if ax in self.spatial_axes else s
                for ax, s in zip(axes, shape)
                ]

        chunk_sizes = [tlbr.stop - tlbr.start
                       if tlbr.stop is not None else s
                       for tlbr, s in zip(chunk_tlbr, shape)]

        if any(map(lambda chk, ps, scl: chk < round(ps * scl),
                   chunk_sizes,
                   self.patch_size,
                   scale)):
            # If the chunk area is smaller than the patch size, return an empty
            # set of topleft positions.
            toplefts = []

        elif all(map(lambda chk, ps, scl: chk == (ps * scl),
                     chunk_sizes,
                     self.patch_size,
                     scale)):
            # If the chunk area is smaller than the patch size, return an empty
            # set of topleft positions.
            toplefts = [slice(None)]

        else:
            chunk_patches = self._get_positions_array(
                chunk_sizes=chunk_sizes,
                axes=image.axes,
                force=False)

            chunk_valid_mask = image[chunk_tlbr]

            chunk_valid_mask_grid = np.meshgrid(
                *[[-1] + list(np.linspace(0, chk_sz - 1, chk_msk)) + [chk_sz]
                  for chk_sz, chk_msk, ax in zip(chunk_sizes,
                                                 chunk_valid_mask.shape,
                                                 axes)
                  if ax in self.spatial_axes
                  ]
                 )

            spt_pad = [
                (1, 1) if ax in self.spatial_axes else (0, 0)
                for ax in axes
            ]
            chunk_valid_mask = np.pad(chunk_valid_mask, spt_pad)

            samples_validity = interpolate.griddata(
                tuple(ax.flatten() for ax in chunk_valid_mask_grid),
                chunk_valid_mask.flatten(),
                chunk_patches,
                method='nearest'
                ).reshape(-1, self._sampling_freq ** 2)

            samples_validity = samples_validity.any(axis=1)

            toplefts = np.round(chunk_tls[samples_validity])
            toplefts = toplefts.astype(np.int64).reshape(-1, 2)

        toplefts = (zip(tl * round(ps * scl), (tl + 1) * round(ps * scl))
                    for tl, ps, scl in zip(toplefts, self.patch_size, scale))
        toplefts = [[slice(start, stop, None) for start, stop in start_stop]
                    for start_stop in toplefts]

        toplefts = [tuple(ax_slice) for ax_slice in zip(*toplefts)]

        return toplefts
