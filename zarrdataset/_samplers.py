from typing import Iterable, Union, Tuple
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

    def _compute_corners(self, non_zero_pos: tuple, axes: str) -> np.ndarray:
        toplefts = np.stack(non_zero_pos).T
        toplefts = toplefts.astype(np.float32)

        toplefts_corners = []

        dim = len(axes)
        factors = 2 ** np.arange(dim + 1)
        for d in range(2 ** dim):
            corner_value =  np.array((d % factors[1:]) // factors[:-1],
                                     dtype=np.float32)
            toplefts_corners.append(
                toplefts + (1 - 1e-4) * corner_value
            )

        corners = np.stack(toplefts_corners)

        return corners

    def _compute_overlap(self, corners: np.ndarray, shape: np.ndarray,
                         ref_shape: np.ndarray) -> Tuple[np.ndarray,
                                                         np.ndarray]:
        scaled_corners = corners * shape[None, None]
        tls_scaled = scaled_corners / ref_shape[None, None]
        tls_idx = np.floor(tls_scaled)

        corners_cut = np.maximum(tls_scaled[0], tls_idx[-1])

        dist2cut = np.fabs(scaled_corners - corners_cut[None])
        coverage = np.prod(dist2cut, axis=-1)

        return coverage, tls_idx.astype(np.int64)

    def _compute_grid(self, chunk_mask: np.ndarray,
                      mask_axes: str,
                      mask_scale: dict,
                      patch_size: dict,
                      image_size: dict):

        mask_relative_shape = np.array(
            [1 / m_scl
             for ax, m_scl in mask_scale.items()
             if ax in self.spatial_axes
            ],
            dtype=np.float32
        )

        patch_shape = np.array(
            [patch_size[ax]
             for ax in mask_axes
             if ax in self.spatial_axes
            ],
            dtype=np.float32
        )

        # If the patch sizes are greater than the relative shape of the mask
        # with respect to the input image, use the mask coordinates as
        # reference to overlap the coordinates of the sampling patches.
        # Otherwise, use the patches coordinates instead.
        if all(map(operator.gt, patch_shape, mask_relative_shape)):
            active_coordinates = np.nonzero(chunk_mask)
            ref_axes = mask_axes

            ref_shape = patch_shape
            shape = mask_relative_shape

            mask_is_greater = False

            patch_ratio = [
                    round(image_size[ax] / ps)
                    for ax, ps in zip(mask_axes, patch_shape.astype(np.int64))
                    if ax in self.spatial_axes
            ]

            if not all(patch_ratio):
                return np.empty(
                    [0] * len(set(mask_axes).intersection(self.spatial_axes)),
                    dtype=np.int64
                )

        else:
            active_coordinates = np.meshgrid(
                *[np.arange(np.ceil(image_size[ax] / ps))
                  for ax, ps in zip(mask_axes, patch_shape)
                  if ax in self.spatial_axes]
            )

            active_coordinates = tuple(
                coord_ax.flatten()
                for coord_ax in active_coordinates
            )

            ref_axes = "".join([
                ax for ax in self.spatial_axes if ax in mask_axes
            ])

            ref_shape = mask_relative_shape
            shape = patch_shape

            mask_is_greater = True

        corners = self._compute_corners(active_coordinates, axes=ref_axes)

        (coverage,
         corners_idx)= self._compute_overlap(corners, shape, ref_shape)

        if mask_is_greater:
            # The mask ratio is greater than the patches size
            mask_values = chunk_mask[tuple(corners_idx[0].T)].T
            patches_coverage = coverage * mask_values

            covered_tls = corners[0, ...].astype(np.int64)

        else:
            # The mask ratio is less than the patches size
            patch_coordinates = np.ravel_multi_index(tuple(corners_idx[0].T),
                                                     chunk_mask.shape)
            patches_coverage = np.bincount(patch_coordinates.flatten(),
                                           weights=coverage.flatten())
            patches_coverage = np.take(patches_coverage, patch_coordinates).T

            covered_tls = corners_idx[0, ...]

        patches_coverage = np.sum(patches_coverage, axis=0)

        # Compute minimum area covered by masked regions to sample a patch.
        min_area = self._min_area
        if min_area < 1:
            min_area *= patch_shape.prod()

        minumum_covered_tls = covered_tls[patches_coverage > min_area]

        if not mask_is_greater:
            # Collapse to unique coordinates since there will be multiple
            # instances of the same patch.
            minumum_covered_tls = np.ravel_multi_index(
                tuple(minumum_covered_tls.T),
                patch_ratio,
                mode="clip"
            )

            minumum_covered_tls = np.unique(minumum_covered_tls)

            minumum_covered_tls = np.unravel_index(
                minumum_covered_tls,
                patch_ratio
            )

            minumum_covered_tls = np.stack(minumum_covered_tls).T

        return minumum_covered_tls * patch_shape[None].astype(np.int64)

    def _compute_valid_toplefts(self, chunk_mask: np.ndarray, mask_axes: str,
                                mask_scale: dict,
                                patch_size: dict,
                                image_size: dict):
        return self._compute_grid(chunk_mask, mask_axes, mask_scale,
                                  patch_size,
                                  image_size)

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
            (ax,
             self._patch_size[ax] * max(1, round(chk / self._patch_size[ax])))
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

        valid_mask_toplefts = self._compute_grid(
            chunk_mask,
            mask.axes,
            mask.scale,
            self._max_chunk_size,
            image_shape
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
        chunk_size = dict(
            (ax,
             (ctb.stop if ctb.stop is not None else image_shape[ax])
             - (ctb.start if ctb.start is not None else 0)
            )
            for ax, ctb in chunk_tlbr.items()
        )

        chunk_mask = mask[chunk_tlbr]

        valid_mask_toplefts = self._compute_valid_toplefts(
            chunk_mask,
            mask.axes,
            mask.scale,
            self._patch_size,
            chunk_size
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
                                mask_axes: str,
                                mask_scale: dict,
                                patch_size: dict,
                                image_shape: dict):
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

        mask_corners = self._compute_corners(np.nonzero(chunk_mask),
                                             mask_axes)

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
