from typing import Iterable, Union
import math
import numpy as np
from itertools import repeat
from functools import reduce
import operator

import poisson_disc

from ._imageloaders import ImageCollection, ImageBase


class PatchSampler(object):
    """Patch sampling algorithm to retrieve image patches/windows from images.

    If the image used for extracting patches has a mask associated to it, only
    patches from masked regions of the image are retrieved.

    Parameters
    ----------
    patch_size : Union[int, Iterable[int], dict]
        Size in pixels of the patches extracted on each axis. Only rectangular
        patches (hyper-cuboids) are supported by now. If a single int is
        passed, that size is used for all dimensions. If an iterable (list,
        tuple) is passed, each value will be assigned to the corresponding axes
         in `spatial_axes`, the size of `patch_size` must match the lenght of 
        `spatial_axes'. If a dict is passed, this should have at least the size
         of the patch of the axes listed in `spatial_axes`.  Use the same 
        convention as how Zarr structure array chunks in order to handle patch 
        shapes and channels correctly.
    min_area : Union[int, float]
        Minimum patch area covered by the mask to consider it samplable. A
        number in range [0, 1) will be used as percentage of the patch size. A
        number >= 1 will be considered as the number of minimum patch pixels
        covered by the mask.
    spatial_axes : str
        The spatial axes from where patches can be extracted.
    """
    def __init__(self, patch_size: Union[int, Iterable[int], dict],
                 min_area: Union[int, float] = 1,
                 spatial_axes: str ="ZYX"):
        # The maximum chunk sizes are used to generate a reference sampling
        # position array used fo every sampled chunk.
        self._max_chunk_size = dict((ax, 0) for ax in spatial_axes)

        if isinstance(patch_size, (list, tuple)):
            if len(patch_size) != len(spatial_axes):
                raise ValueError(f"The size of `patch_size` must match the "
                                 f"number of axes in `spatial_axes`, got "
                                 f"{len(patch_size)} for {spatial_axes}")

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

        self._min_area = min_area

    def _get_samplable_positions(self, non_zero_chunk_mask: tuple,
                                 mask_axes: str,
                                 mask_scale: dict) -> np.ndarray:
        # Scale the positions of the corners to the size of the reference image
        mask_toplefts = np.stack(non_zero_chunk_mask).T
        mask_toplefts = mask_toplefts.astype(np.float32)

        # Add the coordinates of the corners of the mask pixels
        mask_toplefts_corners = []

        mask_dim = len(mask_axes)
        factors = 2 ** np.arange(mask_dim + 1)
        for d in range(2 ** mask_dim):
            corner_value =  np.array((d % factors[1:]) // factors[:-1],
                                     dtype=np.float32)
            mask_toplefts_corners.append(
                mask_toplefts + (1 - 1e-4) * corner_value
            )

        mask_corners = np.stack(mask_toplefts_corners)

        spatial_mask_ratios = np.array(
            [[[1 / m_s
               for ax, m_s in mask_scale.items()
               if ax in self.spatial_axes
            ]]],
            dtype=np.float32
        )

        mask_corners = mask_corners * spatial_mask_ratios

        return mask_corners

    def _compute_coverages(self, corners: np.ndarray,
                           spatial_sizes: np.ndarray,
                           spatial_patch_ratios: np.ndarray):
        # Get the intersection of the masks with the patch grid
        tls_scaled = corners / spatial_sizes[None, None]
        tls_idx = np.floor(tls_scaled).astype(np.int64)

        corners_cut = np.maximum(corners[0], tls_idx[-1] * spatial_sizes[None])

        coverage = np.prod(np.abs(corners - corners_cut[None]), axis=-1).T

        return coverage, tls_idx


    def _compute_grid_mask_less(self,
                                mask_corners: np.ndarray,
                                spatial_patch_sizes: np.ndarray,
                                spatial_patch_ratios: np.ndarray,
                                min_area: float,
                                mask_axes: str,
                                patch_size: dict):
        # Rescale the positions to determine the chunks that contain any
        # portion of the mask

        non_zero_patch_pos = np.meshgrid(*[
            np.arange(patch_size[ax])
            for ax in mask_axes
            if ax in self.spatial_axes
        ])

        non_zero_patch_pos = tuple(
            mesh_pos.flatten()
            for mesh_pos in non_zero_patch_pos
        )

        mask_coverage, mask_tls = self._compute_coverages(mask_corners,
                                                          spatial_patch_sizes,
                                                          spatial_patch_ratios)

        mask_tls = np.ravel_multi_index(
            tuple(mask_tls.astype(np.int64).T),
            spatial_patch_ratios,
            mode="clip"
        )

        # A position can appear multiple times when more than one pixel of the
        # mask is inside the same patch. Removing these duplicates will leave
        # only those patches that contain masked regions.
        unique_mask_tls = np.unique(mask_tls)

        # Remove patches that are not covered by the minimum masked area.
        mask_coverage = np.bincount(mask_tls.flatten(),
                                    weights=mask_coverage.flatten())

        mask_coverage = np.take(mask_coverage, unique_mask_tls)

        mask_covered = mask_coverage > min_area

        mask_tls_idx = np.unravel_index(unique_mask_tls[mask_covered],
                                        spatial_patch_ratios)

        mask_toplefts = np.stack(mask_tls_idx).T * spatial_patch_sizes[None]

        return mask_toplefts.astype(np.int64)

    def _compute_grid_mask_greater(self,
                                   chunk_mask: np.ndarray,
                                   spatial_patch_sizes: np.ndarray,
                                   spatial_patch_ratios: np.ndarray,
                                   spatial_mask_sizes: np.ndarray,
                                   spatial_mask_ratios: np.ndarray,
                                   min_area: float,
                                   mask_axes: str,
                                   patch_size: dict):
        # Rescale the positions to determine the chunks that contain any
        # portion of the mask

        non_zero_patch_pos = np.meshgrid(*[
            np.arange(ps_ratio)
            for ps_ratio in spatial_patch_ratios
        ])

        non_zero_patch_pos = tuple(
            mesh_pos.flatten()
            for mesh_pos in non_zero_patch_pos
        )

        patch_axes = "".join([
            ax
            for ax in mask_axes
            if ax in self.spatial_axes
        ])

        patch_scale = dict(
            (ax, 1 / patch_size[ax])
            for ax in mask_axes
            if ax in self.spatial_axes
        )

        patch_corners = self._get_samplable_positions(non_zero_patch_pos,
                                                      patch_axes,
                                                      patch_scale)

        (patch_coverage,
         patch_tls_idx) = self._compute_coverages(patch_corners,
                                                  spatial_mask_sizes,
                                                  spatial_mask_ratios)

        patch_mask_overlay_idx = np.ravel_multi_index(tuple(patch_tls_idx.T),
                                                      chunk_mask.shape)

        mask_values = np.take(chunk_mask, patch_mask_overlay_idx)

        patch_tls_idx = np.arange(spatial_patch_ratios.prod())
        patch_coverage = np.bincount(
            patch_tls_idx,
            weights=np.sum(patch_coverage * mask_values, axis=1)
        )

        patch_tls_idx = tuple(
            non_zero_patch_pos_axis[patch_coverage > min_area]
            for non_zero_patch_pos_axis in non_zero_patch_pos
        )

        mask_toplefts = np.stack(patch_tls_idx).T * spatial_patch_sizes[None]

        return mask_toplefts.astype(np.int64)

    def _compute_grid(self, chunk_mask: np.ndarray,
                      mask_corners: np.ndarray,
                      mask_axes: str,
                      mask_scale: dict,
                      image_shape: dict,
                      patch_size: dict):

        spatial_patch_sizes = np.array(
            [patch_size[ax]
             for ax in mask_axes
             if ax in self.spatial_axes
            ],
            dtype=np.float32
        )

        spatial_patch_ratios = np.array(
            [math.ceil(image_shape[ax] / patch_size[ax]) for ax in mask_axes],
            dtype=np.int64
        )

        # Compute minimum area covered by masked regions to sample a patch.
        min_area = self._min_area
        if min_area < 1:
            min_area *= spatial_patch_sizes.prod()

        spatial_mask_ratios = np.array(
            chunk_mask.shape,
            dtype=np.int64
        )

        spatial_mask_sizes = np.array(
            [1 / m_scl
             for ax, m_scl in mask_scale.items()
             if ax in self.spatial_axes
            ],
            dtype=np.float32
        )

        if all(map(operator.ge, spatial_mask_sizes, spatial_patch_sizes)):
            return self._compute_grid_mask_greater(
                chunk_mask,
                spatial_patch_sizes,
                spatial_patch_ratios,
                spatial_mask_sizes,
                spatial_mask_ratios,
                min_area,
                mask_axes,
                patch_size
            )
        else:
            return self._compute_grid_mask_less(
                mask_corners,
                spatial_patch_sizes,
                spatial_patch_ratios,
                min_area,
                mask_axes,
                patch_size
            )


    def _compute_valid_toplefts(self, chunk_mask: np.ndarray,
                                mask_corners: np.ndarray,
                                mask_axes: str,
                                mask_scale: dict,
                                image_shape: dict,
                                patch_size: dict):
        return self._compute_grid(chunk_mask,
                                  mask_corners,
                                  mask_axes,
                                  mask_scale,
                                  image_shape,
                                  patch_size
                                )

    def _compute_toplefts_slices(self, mask: ImageBase, image_shape: dict,
                                 patch_size: dict,
                                 valid_mask_toplefts: np.ndarray,
                                 chunk_tlbr: dict):
        toplefts = []
        for tls in valid_mask_toplefts:
            curr_tl = []

            for ax in self.spatial_axes:
                if ax in mask.axes:
                    tl = ((chunk_tlbr[ax].start
                           if chunk_tlbr[ax].start is not None else 0)
                          + tls[mask.axes.index(ax)])
                    br = ((chunk_tlbr[ax].start
                           if chunk_tlbr[ax].start is not None else 0)
                          + tls[mask.axes.index(ax)] + patch_size[ax])
                    if br <= image_shape[ax]:
                        curr_tl.append((ax, slice(tl, br)))
                    else:
                        break
                else:
                    curr_tl.append((ax, slice(0, 1)))

            else:
                toplefts.append(dict(curr_tl))

        return toplefts

    def compute_chunks(self,
                       image_collection: ImageCollection) -> Iterable[dict]:
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
        image = image_collection.collection[image_collection.reference_mode]
        mask = image_collection.collection[image_collection.mask_mode]

        spatial_chunk_sizes = dict(
            (ax, chk)
            for ax, chk in zip(image.axes, image.chunk_size)
            if ax in self.spatial_axes
        )

        image_shape = dict(map(tuple, zip(image.axes, image.shape)))

        self._max_chunk_size = dict(
            (ax, (min(max(self._max_chunk_size[ax],
                          spatial_chunk_sizes[ax],
                          self._patch_size[ax]),
                      image_shape[ax]))
                 if ax in image.axes else 1)
            for ax in self.spatial_axes
        )

        chunk_tlbr = dict(
            map(tuple, zip(self.spatial_axes, repeat(slice(None))))
        )

        chunk_mask = mask[chunk_tlbr]
        non_zero_chunk_mask = np.nonzero(chunk_mask)

        mask_corners = self._get_samplable_positions(non_zero_chunk_mask,
                                                     mask_axes=mask.axes,
                                                     mask_scale=mask.scale)

        valid_mask_toplefts = self._compute_grid(
            chunk_mask,
            mask_corners,
            mask.axes,
            mask.scale,
            image_shape,
            self._max_chunk_size
        )

        chunks_slices = self._compute_toplefts_slices(
            mask,
            image_shape=image_shape,
            patch_size=self._max_chunk_size,
            valid_mask_toplefts=valid_mask_toplefts,
            chunk_tlbr=chunk_tlbr
        )

        return chunks_slices

    def compute_patches(self, image_collection: ImageCollection,
                        chunk_tlbr: dict) -> Iterable[dict]:
        image = image_collection.collection[image_collection.reference_mode]
        mask = image_collection.collection[image_collection.mask_mode]
        image_shape = dict(map(tuple, zip(image.axes, image.shape)))
        curr_chunk_shape = dict(
            (ax, (ctb.stop - (ctb.start if ctb.start is not None else 0)) \
                 if ctb.stop is not None else image_shape.get(ax, 1)
            )
            for ax, ctb in chunk_tlbr.items()
        )

        chunk_mask = mask[chunk_tlbr]
        non_zero_chunk_mask = np.nonzero(chunk_mask)

        mask_corners = self._get_samplable_positions(non_zero_chunk_mask,
                                                     mask_axes=mask.axes,
                                                     mask_scale=mask.scale)

        valid_mask_toplefts = self._compute_valid_toplefts(
            chunk_mask,
            mask_corners,
            mask.axes,
            mask.scale,
            curr_chunk_shape,
            self._patch_size
        )

        patches_slices = self._compute_toplefts_slices(
            mask,
            image_shape=image_shape,
            patch_size=self._patch_size,
            valid_mask_toplefts=valid_mask_toplefts,
            chunk_tlbr=chunk_tlbr
        )

        return patches_slices

    def __repr__(self) -> str:
        """String representation of classes derived from PatchSampler.
        """
        return (f"{type(self)} for sampling patches of size "
                f"{self._patch_size}.")


class BlueNoisePatchSampler(PatchSampler):
    """Patch sampler that retrieves patches from coordinates sampled using the
    Bridson sampling algorithm, also known as Blue-noise sampling algorithm.

    Parameters
    ----------
    patch_size : int, iterable, dict
        Size in pixels of the patches extracted. Only squared patches are
        supported by now.
    resample_positions : bool
        Whether positions should be resampled for each new chunk or not.
    allow_overlap : bool
        Whether overlapping of patches is allowed or not.
    """
    def __init__(self, patch_size: Union[int, Iterable[int], dict],
                 resample_positions=False,
                 allow_overlap=False,
                 **kwargs):
        super(BlueNoisePatchSampler, self).__init__(patch_size)
        self._base_chunk_tls = None
        self._resample_positions = resample_positions
        self._allow_overlap = allow_overlap
 
    def compute_sampling_positions(self, force=False) -> None:
        """Compute the sampling positions using blue-noise sampling.

        Parameters
        ----------
        force: bool
            Whether force resampling positions, or reuse existing positions.
        """
        if self._base_chunk_tls is not None and not force:
            return

        radius = max(self._patch_size.values())
        if self._allow_overlap:
            radius *= math.sqrt(2)

        samplable_axes = [ax for ax in self.spatial_axes
                          if self._max_chunk_size[ax] > radius]
        dims = [self._max_chunk_size[ax] - radius for ax in samplable_axes]

        if len(samplable_axes):
            base_chunk_tls = poisson_disc.Bridson_sampling(dims=np.array(dims),
                                                           radius=radius,
                                                           k=30)
            base_chunk_tls = base_chunk_tls.astype(np.int64)

            base_chunk_tls = [
                base_chunk_tls[:, samplable_axes.index(ax)]
                if ax in samplable_axes else np.zeros(len(base_chunk_tls),
                                                      dtype=np.int64)
                for ax in self.spatial_axes
            ]
            self._base_chunk_tls = np.stack(base_chunk_tls).T

        else:
            self._base_chunk_tls = np.zeros((1, len(self.spatial_axes)),
                                            dtype=np.int64)

    def _compute_valid_toplefts(self,
                                chunk_mask: np.ndarray,
                                mask_corners: np.ndarray,
                                mask_axes: str,
                                mask_scale: dict,
                                image_shape: dict,
                                patch_size: dict):
        self.compute_sampling_positions(force=self._resample_positions)

        # Filter sampling positions that does not contain any mask portion.
        sampling_pos = np.hstack(
            tuple(
                self._base_chunk_tls[:, self.spatial_axes.index(ax), None]
                if ax in self.spatial_axes else
                np.zeros((len(self._base_chunk_tls), 1), dtype=np.float32)
                for ax in mask_axes
            )
        )

        spatial_patch_sizes = np.array([
            patch_size[ax]
            for ax in mask_axes
            if ax in self.spatial_axes
        ])

        dist = (mask_corners[None, :, :, :]
                - sampling_pos[:, None, None, :].astype(np.float32)
                - spatial_patch_sizes[None, None, None, :] / 2)

        mask_samplable_pos, = np.nonzero(
            np.any(
                np.all(dist < spatial_patch_sizes[None, None, None], axis=-1),
                axis=(1, 2)
            )
        )

        toplefts = sampling_pos[mask_samplable_pos]

        return toplefts
