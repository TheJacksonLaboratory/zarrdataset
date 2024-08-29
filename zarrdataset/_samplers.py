from typing import Iterable, Union, Tuple, List
import math
import numpy as np
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
    stride : Union[int, Iterable[int], dict, None]
        Distance in pixels of the movement of the sampling sliding window.
        If `stride` is less than `patch_size` for an axis, patches will have an
        overlap between them. This is usuful in inference mode for avoiding
        edge artifacts. If None is passed, the `patch_size` will be used as
        `stride`.
    pad : Union[int, Iterable[int], dict, None]
        Padding in pixels added to the extracted patch at each specificed axis.
    min_area : Union[int, float]
        Minimum patch area covered by the mask to consider it samplable. A
        number in range [0, 1) will be used as percentage of the patch size. A
        number >= 1 will be considered as the number of minimum patch pixels
        covered by the mask.
    spatial_axes : str
        The spatial axes from where patches can be extracted.
    allow_incomplete_patches : bool
        Allow to retrieve patches that are smaller than the patch size. This is
        the case of samples at the edge of the image that are usually smaller
        than the specified patch size.
    """
    def __init__(self, patch_size: Union[int, Iterable[int], dict],
                 stride: Union[int, Iterable[int], dict, None] = None,
                 pad: Union[int, Iterable[int], dict, None] = None,
                 min_area: Union[int, float] = 1,
                 spatial_axes: str = "ZYX",
                 allow_incomplete_patches: bool = False):
        # The maximum chunk sizes are used to generate a reference sampling
        # position array used fo every sampled chunk.
        self._max_chunk_size = {ax: 0 for ax in spatial_axes}

        if isinstance(patch_size, (list, tuple)):
            if len(patch_size) != len(spatial_axes):
                raise ValueError(f"The size of `patch_size` must match the "
                                 f"number of axes in `spatial_axes`, got "
                                 f"{len(patch_size)} for {spatial_axes}")

            patch_size = {ax: ps for ax, ps in zip(spatial_axes, patch_size)}

        elif isinstance(patch_size, int):
            patch_size = {ax: patch_size for ax in spatial_axes}

        elif not isinstance(patch_size, dict):
            raise ValueError(f"Patch size must be a dictionary specifying the"
                             f" patch size of each axes, an iterable (list, "
                             f"tuple) with the same order as the spatial axes,"
                             f" or an integer for a cubic patch. Received "
                             f"{patch_size} of type {type(patch_size)}")

        if isinstance(stride, (list, tuple)):
            if len(stride) != len(spatial_axes):
                raise ValueError(f"The size of `stride` must match the "
                                 f"number of axes in `spatial_axes`, got "
                                 f"{len(stride)} for {spatial_axes}")

            stride = {ax: st for ax, st in zip(spatial_axes, stride)}

        elif isinstance(stride, int):
            stride = {ax: stride for ax in spatial_axes}

        elif stride is None:
            stride = patch_size

        elif not isinstance(stride, dict):
            raise ValueError(f"Stride size must be a dictionary specifying the"
                             f" stride step size of each axes, an iterable ("
                             f"list, tuple) with the same order as the spatial"
                             f" axes, or an integer for a cubic patch. "
                             f"Received {stride} of type {type(stride)}")

        if pad is None:
            pad = 0

        if isinstance(pad, (list, tuple)):
            if len(pad) != len(spatial_axes):
                raise ValueError(f"The size of `pad` must match the "
                                 f"number of axes in `spatial_axes`, got "
                                 f"{len(pad)} for {spatial_axes}")

            pad = {ax: st for ax, st in zip(spatial_axes, pad)}

        elif isinstance(pad, int):
            pad = {ax: pad for ax in spatial_axes}

        elif not isinstance(pad, dict):
            raise ValueError(f"Pad size must be a dictionary specifying the"
                             f" numer of pixels added to each axes, an "
                             f"iterable (list, tuple) with the same order as "
                             f"the spatial axes, or an integer for a cubic "
                             f"patch. Received {pad} of type {type(pad)}")

        self.spatial_axes = spatial_axes

        self._patch_size = {ax: patch_size.get(ax, 1) for ax in spatial_axes}
        self._stride = {ax: stride.get(ax, 1) for ax in spatial_axes}
        self._pad = {ax: pad.get(ax, 0) for ax in spatial_axes}

        self._min_area = min_area
        self._allow_incomplete_patches = allow_incomplete_patches

    def _compute_corners(self, coordinates: np.ndarray, scale: np.ndarray
                         ) -> np.ndarray:
        toplefts_corners = []

        dim = coordinates.shape[-1]
        factors = 2 ** np.arange(dim + 1)
        for d in range(2 ** dim):
            corner_value = np.array((d % factors[1:]) // factors[:-1],
                                    dtype=np.float32)
            toplefts_corners.append(
                coordinates + scale * (1 - 1e-4) * corner_value
            )

        corners = np.stack(toplefts_corners)

        return corners

    def _compute_reference_indices(self, reference_coordinates: np.ndarray,
                                   reference_axes_sizes: np.ndarray
                                   ) -> Tuple[List[np.ndarray],
                                              List[Tuple[int]]]:
        reference_per_axis = list(map(
            lambda coords, axis_size: np.concatenate((
                np.full((1, ), fill_value=-float("inf")),
                np.unique(coords),
                np.full((1, ), fill_value=np.max(coords) + axis_size))),
            reference_coordinates.T,
            reference_axes_sizes
        ))

        reference_idx = map(
            lambda coord_axis, ref_axis:
            np.max(np.arange(ref_axis.size)
                   * (coord_axis.reshape(-1, 1) >= ref_axis[None, ...]),
                   axis=1),
            reference_coordinates.T,
            reference_per_axis
        )
        reference_idx = np.stack(tuple(reference_idx), axis=-1)
        reference_idx = reference_idx.reshape(reference_coordinates.T.shape)

        reference_idx = [
            tuple(tls_coord)
            for tls_coord in reference_idx.reshape(-1, len(reference_per_axis))
        ]

        return reference_per_axis, reference_idx

    def _compute_overlap(self, corners_coordinates: np.ndarray,
                         reference_per_axis: np.ndarray) -> Tuple[np.ndarray,
                                                                  np.ndarray]:
        tls_idx = map(
            lambda coord_axis, ref_axis:
            np.max(np.arange(ref_axis.size)
                   * (coord_axis.reshape(-1, 1) >= ref_axis[None, ...]),
                   axis=1),
            np.moveaxis(corners_coordinates, -1, 0),
            reference_per_axis
        )
        tls_idx = np.stack(tuple(tls_idx), axis=-1)
        tls_idx = tls_idx.reshape(corners_coordinates.shape)

        tls_coordinates = map(
            lambda tls_coord, ref_axis: ref_axis[tls_coord],
            np.moveaxis(tls_idx, -1, 0),
            reference_per_axis
        )
        tls_coordinates = np.stack(tuple(tls_coordinates), axis=-1)

        corners_cut = np.maximum(corners_coordinates[0], tls_coordinates[-1])

        dist2cut = np.fabs(corners_coordinates - corners_cut[None])
        coverage = np.prod(dist2cut, axis=-1)

        return coverage, tls_idx

    def _compute_grid(self, chunk_tlbr: dict, mask: ImageBase,
                      patch_size: dict,
                      image_size: dict,
                      min_area: float,
                      allow_incomplete_patches: bool = False):
        mask_scale = np.array([mask.scale.get(ax, 1)
                               for ax in self.spatial_axes],
                              dtype=np.float32)

        image_scale = np.array([image_size.get(ax, 1) / patch_size.get(ax, 1)
                                for ax in self.spatial_axes],
                               dtype=np.float32)

        round_fn = math.ceil if allow_incomplete_patches else math.floor

        image_blocks = [
            round_fn(
                (
                    min(image_size.get(ax, 1),
                        chunk_tlbr[ax].stop
                        if chunk_tlbr[ax].stop is not None
                        else float("inf"))
                    - (chunk_tlbr[ax].start
                       if chunk_tlbr[ax].start is not None
                       else 0)
                ) / patch_size.get(ax, 1))
            for ax in self.spatial_axes
        ]

        if min(image_blocks) == 0:
            return []

        image_scale = np.array([patch_size.get(ax, 1)
                                for ax in self.spatial_axes],
                               dtype=np.float32)
        image_coordinates = np.array(list(np.ndindex(*image_blocks)),
                                     dtype=np.float32)

        image_coordinates *= image_scale

        mask_scale = 1 / np.array([mask.scale.get(ax, 1)
                                   for ax in self.spatial_axes],
                                  dtype=np.float32)

        mask_coordinates = list(np.nonzero(mask[:]))
        for ax_i, ax in enumerate(self.spatial_axes):
            if ax not in mask.axes:
                mask_coordinates.insert(
                    ax_i,
                    np.zeros(mask_coordinates[0].size)
                )

        mask_coordinates = np.stack(mask_coordinates, dtype=np.float32).T
        mask_coordinates *= mask_scale[None, ...]

        # Filter out mask coordinates outside the current selected chunk
        chunk_tl_coordinates = np.array(
            [chunk_tlbr[ax].start if chunk_tlbr[ax].start is not None else 0
             for ax in self.spatial_axes],
            dtype=np.float32
        )

        chunk_br_coordinates = np.array(
            [chunk_tlbr[ax].stop
             if chunk_tlbr[ax].stop is not None
             else float('inf')
             for ax in self.spatial_axes],
            dtype=np.float32
        )

        in_chunk = np.all(
            np.bitwise_and(
                mask_coordinates > (chunk_tl_coordinates - mask_scale - 1e-4),
                mask_coordinates < chunk_br_coordinates + 1e-4
            ),
            axis=1
        )
        mask_coordinates = mask_coordinates[in_chunk]

        # Translate the mask coordinates to the origin for comparison with
        # image coordinates.
        mask_coordinates -= chunk_tl_coordinates

        if all(map(operator.ge, image_scale, mask_scale)):
            mask_corners = self._compute_corners(mask_coordinates, mask_scale)

            (reference_per_axis,
             reference_idx) =\
                self._compute_reference_indices(image_coordinates, image_scale)

            (coverage,
             corners_idx) = self._compute_overlap(mask_corners,
                                                  reference_per_axis)

            covered_indices = [
                reference_idx.index(tuple(idx))
                if tuple(idx) in reference_idx else len(reference_idx)
                for idx in corners_idx.reshape(-1, len(self.spatial_axes))
            ]

            patches_coverage = np.bincount(covered_indices,
                                           weights=coverage.flatten(),
                                           minlength=len(reference_idx) + 1)
            patches_coverage = patches_coverage[:-1]

        else:
            image_corners = self._compute_corners(image_coordinates,
                                                  image_scale)

            (reference_per_axis,
             reference_idx) = self._compute_reference_indices(mask_coordinates,
                                                              mask_scale)

            (coverage,
             corners_idx) = self._compute_overlap(image_corners,
                                                  reference_per_axis)

            covered_indices = np.array([
                tuple(idx) in reference_idx
                for idx in corners_idx.reshape(-1, len(self.spatial_axes))
            ]).reshape(coverage.shape)

            patches_coverage = np.sum(covered_indices * coverage, axis=0)

        minimum_covered_tls = image_coordinates[patches_coverage > min_area]
        minimum_covered_tls = minimum_covered_tls.astype(np.int64)

        return minimum_covered_tls

    def _compute_valid_toplefts(self, chunk_tlbr: dict, mask: ImageBase,
                                patch_size: dict,
                                **kwargs):
        return self._compute_grid(chunk_tlbr, mask, patch_size, **kwargs)

    def _compute_toplefts_slices(self, chunk_tlbr: dict,
                                 valid_mask_toplefts: np.ndarray,
                                 patch_size: dict,
                                 pad: Union[dict, None] = None):
        if pad is None:
            pad = {ax: 0 for ax in self.spatial_axes}

        toplefts = [
            {ax: slice(
                (chunk_tlbr[ax].start if chunk_tlbr[ax].start is not None
                 else 0) + tls[self.spatial_axes.index(ax)]
                - pad[ax],
                (chunk_tlbr[ax].start if chunk_tlbr[ax].start is not None
                 else 0) + tls[self.spatial_axes.index(ax)] + patch_size[ax]
                + pad[ax])
             for ax in self.spatial_axes
             }
            for tls in valid_mask_toplefts
        ]

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

        # This computes a chunk size in terms of the patch size instead of the
        # original array chunk size.
        spatial_chunk_sizes = {
            ax: (self._stride[ax]
                 * max(1, math.ceil(chk / self._stride[ax])))
            for ax, chk in zip(image.axes, image.chunk_size)
            if ax in self.spatial_axes
        }

        image_size = {ax: s for ax, s in zip(image.axes, image.shape)}

        self._max_chunk_size = {
            ax: (min(max(self._max_chunk_size[ax],
                         spatial_chunk_sizes[ax]),
                     image_size[ax]))
            if ax in image.axes else 1
            for ax in self.spatial_axes
        }

        chunk_tlbr = {ax: slice(None) for ax in self.spatial_axes}

        valid_mask_toplefts = self._compute_grid(
            chunk_tlbr,
            mask,
            self._max_chunk_size,
            image_size,
            min_area=1,
            allow_incomplete_patches=True
        )

        chunks_slices = self._compute_toplefts_slices(
            chunk_tlbr,
            valid_mask_toplefts=valid_mask_toplefts,
            patch_size=self._max_chunk_size
        )

        return chunks_slices

    def compute_patches(self, image_collection: ImageCollection,
                        chunk_tlbr: dict) -> Iterable[dict]:
        image = image_collection.collection[image_collection.reference_mode]
        mask = image_collection.collection[image_collection.mask_mode]
        image_size = {ax: s for ax, s in zip(image.axes, image.shape)}

        stride = {
            ax: self._stride.get(ax, 1) if image_size.get(ax, 1) > 1 else 1
            for ax in self.spatial_axes
        }

        patch_size = {
            ax: self._patch_size.get(ax, 1) if image_size.get(ax, 1) > 1 else 1
            for ax in self.spatial_axes
        }

        pad = {
            ax: self._pad.get(ax, 0) if image_size.get(ax, 1) > 1 else 0
            for ax in self.spatial_axes
        }

        min_area = self._min_area
        if min_area < 1:
            min_area *= np.prod(list(patch_size.values()))

        valid_mask_toplefts = self._compute_valid_toplefts(
            chunk_tlbr,
            mask,
            stride,
            image_size=image_size,
            min_area=min_area,
            allow_incomplete_patches=self._allow_incomplete_patches
        )

        patches_slices = self._compute_toplefts_slices(
            chunk_tlbr,
            valid_mask_toplefts=valid_mask_toplefts,
            patch_size=patch_size,
            pad=pad
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
        super(BlueNoisePatchSampler, self).__init__(patch_size, **kwargs)
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

    def _compute_valid_toplefts(self, chunk_tlbr: dict, mask: ImageBase,
                                patch_size: dict,
                                **kwargs):
        self.compute_sampling_positions(force=self._resample_positions)

        # Filter sampling positions that does not contain any mask portion.
        spatial_patch_sizes = np.array([
            patch_size.get(ax, 1) for ax in self.spatial_axes
        ])

        mask_scale = np.array([mask.scale.get(ax, 1)
                               for ax in self.spatial_axes],
                              dtype=np.float32)

        mask_scale = 1 / np.array([mask.scale.get(ax, 1)
                                   for ax in self.spatial_axes],
                                  dtype=np.float32)

        mask_coordinates = list(np.nonzero(mask[:]))
        for ax_i, ax in enumerate(self.spatial_axes):
            if ax not in mask.axes:
                mask_coordinates.insert(
                    ax_i,
                    np.zeros(mask_coordinates[0].size)
                )

        mask_coordinates = np.stack(mask_coordinates, dtype=np.float32).T
        mask_coordinates *= mask_scale[None, ...]

        # Filter out mask coordinates outside the current selected chunk
        chunk_tl_coordinates = np.array(
            [chunk_tlbr[ax].start if chunk_tlbr[ax].start is not None else 0
             for ax in self.spatial_axes],
            dtype=np.float32
        )
        chunk_br_coordinates = np.array(
            [chunk_tlbr[ax].stop
             if chunk_tlbr[ax].stop is not None
             else float('inf')
             for ax in self.spatial_axes],
            dtype=np.float32
        )

        in_chunk = np.all(
            np.bitwise_and(
                mask_coordinates > (chunk_tl_coordinates - mask_scale - 1e-4),
                mask_coordinates < chunk_br_coordinates + 1e-4
            ),
            axis=1
        )
        mask_coordinates = mask_coordinates[in_chunk]

        mask_corners = self._compute_corners(mask_coordinates, mask_scale)

        dist = (mask_corners[None, :, :, :]
                - self._base_chunk_tls[:, None, None, :].astype(np.float32)
                - spatial_patch_sizes[None, None, None, :] / 2)

        mask_samplable_pos, = np.nonzero(
            np.any(
                np.all(dist < spatial_patch_sizes[None, None, None], axis=-1),
                axis=(1, 2)
            )
        )

        toplefts = self._base_chunk_tls[mask_samplable_pos]

        return toplefts
