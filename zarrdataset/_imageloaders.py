from typing import Union, Iterable, Callable

import math
import zarr
import numpy as np

import PIL
from PIL import Image
from io import BytesIO

from ._utils import (map_axes_order, connect_s3, scale_coords, select_axes,
                     parse_rois,
                     translate2roi)

try:
    import tifffile
    TIFFFILE_SUPPORT = True

except ModuleNotFoundError:
    TIFFFILE_SUPPORT = False


def image2array(arr_src: Union[str, zarr.Group, zarr.Array, np.ndarray],
                data_group: Union[str, int, None] = None,
                zarr_store: Union[zarr.storage.Store, None] = None):
    """Open images stored in zarr format or any image format that can be opened
    by PIL as an array.

    Parameters
    ----------
    arr_src : Union[str, zarr.Group, zarr.Array, np.ndarray]
        The image filename, or zarr object, to be loaded as a zarr array.
    data_group : Union[str, int, None]
        The group within the zarr file from where the array is loaded. This is
        used only when the input file is a zarr object.
    zarr_store : Union[zarr.storage.Store, None]
        The class used to open the zarr file. Leave it as None to let this
        function to use the most suitable depending to the data location
        (s3/remote: FSStore, local disk: DirectoryStore).

    Returns
    -------
    arr : zarr.Array
        The image as a zarr array.
    store: None, zarr.storage.Store, PIL.Image, tifffile.ZarrTiffStore
        A connection to the image file that is kept open until the dataset is
        not needed, so this connection can be closed properly.
    """
    if isinstance(arr_src, zarr.Group):
        if data_group is not None and len(data_group):
            arr = arr_src[data_group]
        else:
            raise ValueError(f"Cannot use {arr_src} because it is a Zarr Group"
                             f" and a Zarr Array was expected. Please specify"
                             f" the group where the array is stored.")
        return arr, None

    elif isinstance(arr_src, zarr.Array):
        return arr_src, None

    elif isinstance(arr_src, str) and ".zarr" in arr_src:
        s3_obj = connect_s3(arr_src)

        if zarr_store is None:
            # If zarr_store is not set by the user, assign the most suitable
            # according to the image location (remote: FSStore,
            # local: DirectoryStore).
            if s3_obj is not None:
                zarr_store = zarr.storage.FSStore
            else:
                zarr_store = zarr.storage.DirectoryStore

        store = zarr_store(arr_src)
        grp = zarr.open(store, mode="r")

        if isinstance(grp, zarr.Group):
            if data_group is not None and len(data_group):
                arr = grp[data_group]
            else:
                raise ValueError(f"Cannot use {arr_src} because it is a Zarr "
                                 f"Group and the group where the array is "
                                 f"stored was not specified.")
        else:
            arr = grp

        return arr, None

    elif isinstance(arr_src, np.ndarray):
        arr = zarr.array(data=arr_src, shape=arr_src.shape,
                         chunks=arr_src.shape)
        return arr, None

    # Try to create a connection with the file, to determine if it is a remote
    # resource or local file.
    s3_obj = connect_s3(arr_src)

    if TIFFFILE_SUPPORT and s3_obj is None:
        # Try to open the input file with tifffile (if installed).
        try:
            if (data_group is None
               or (isinstance(data_group, str) and not len(data_group))):
                tiff_args = dict(
                    key=None,
                    level=None,
                    series=None
                )
            elif isinstance(data_group, str) and len(data_group):
                data_group_split = data_group.split("/")

                if len(data_group_split) > 1:
                    tiff_args = dict(
                        key=int(data_group_split[0]),
                        level=int(data_group_split[1]),
                        series=None
                    )
                else:
                    tiff_args = dict(
                        key=int(data_group_split[0]),
                        level=None,
                        series=None
                    )

            elif isinstance(data_group, int):
                tiff_args = dict(
                    key=data_group,
                    level=None,
                    series=None
                )
            else:
                raise ValueError(f"Incorrect data group format "
                                 f"{type(data_group)}")

            store = tifffile.imread(arr_src, aszarr=True, **tiff_args)
            arr = zarr.open(store, mode="r")

            return arr, store

        except tifffile.tifffile.TiffFileError:
            pass

    # If the input is a path to an image stored in a format
    # supported by PIL, open it and use it as a numpy array.
    try:
        if s3_obj is not None:
            # The image is stored in a S3 bucket
            filename = arr_src.split(s3_obj["endpoint_url"]
                                     + "/"
                                     + s3_obj["bucket_name"])[1][1:]
            im_bytes = s3_obj["s3"].get_object(
                Bucket=s3_obj["bucket_name"],
                Key=filename)["Body"].read()
            store = Image.open(BytesIO(im_bytes))

        else:
            # The image is stored locally
            store = Image.open(arr_src, mode="r")

        channels = len(store.getbands())
        height = store.size[1]
        width = store.size[0]

        arr_shape = [height, width] + ([channels] if channels > 1 else [])
        arr = zarr.array(data=np.array(store),
                         shape=arr_shape,
                         chunks=arr_shape,
                         dtype=np.uint8)
        return arr, store

    except PIL.UnidentifiedImageError:
        pass

    raise ValueError(f"The file/object {arr_src} cannot be opened by Zarr "
                     f"{', TiffFile, ' if TIFFFILE_SUPPORT else ''} or PIL")


class ImageBase(object):
    arr = None
    spatial_axes = "ZYX"
    source_axes = None
    axes = None
    mode = ""
    permute_order = None
    _store = None
    _new_axes = ""
    _drop_axes = ""
    _scale = None
    _shape = None
    _spatial_reference_shape = None
    _spatial_reference_axes = None
    _chunk_size = None
    _cached_coords = None
    _image_func = None

    def __init__(self, shape: Iterable[int],
                 chunk_size: Union[Iterable[int], None] = None,
                 source_axes: str = "",
                 mode: str = ""):
        if chunk_size is None:
            chunk_size = shape

        self.source_axes = source_axes
        self.axes = source_axes
        self.permute_order = list(range(len(source_axes)))
        self.arr = zarr.ones(shape=shape, dtype=bool, chunks=chunk_size)
        self.roi = tuple([slice(None)] * len(source_axes))
        self.mode = mode

        self._chunk_size = chunk_size

    def _iscached(self, coords):
        if self._cached_coords is None:
            return False
        else:
            return all(
                map(lambda coord, cache, s:
                    (cache.start if cache.start is not None else 0)
                    <= (coord.start if coord.start is not None else 0)
                    and (coord.stop if coord.stop is not None else s)
                    <= (cache.stop if cache.stop is not None else s),
                    coords,
                    self._cached_coords,
                    self.arr.shape))

    def _cache_chunk(self, index):
        if not self._iscached(index):
            self._cached_coords = tuple(
                map(lambda i, chk, s:
                    slice(max(0, chk * int(i.start / chk))
                          if i.start is not None else 0,
                          min(s, chk * int(math.ceil(i.stop / chk)))
                          if i.stop is not None else s),
                    index,
                    self.arr.chunks,
                    self.arr.shape)
            )

            padding = tuple(
                (cc.start - i.start if i.start is not None and i.start < 0 else 0,
                 i.stop - cc.stop if i.stop is not None and i.stop > s else 0)
                for cc, i, s in zip(self._cached_coords, index, self.arr.shape)
            )

            self._cache = self.arr[self._cached_coords]

            if any([any(pad) for pad in padding]):
                self._cache = np.pad(self._cache, padding)
                self._cached_coords = tuple(
                    slice(cc.start - p_low, cc.stop + p_high)
                    for (p_low, p_high), cc in zip(padding, self._cached_coords)
                )

        cached_index = tuple(
            map(lambda cache, i:
                slice((i.start - cache.start) if i.start is not None else 0,
                      (i.stop - cache.start) if i.stop is not None else None,
                      None),
                self._cached_coords, index)
        )

        return cached_index

    def __getitem__(self, index : Union[slice, tuple, dict]) -> np.ndarray:
        if self._spatial_reference_axes is not None:
            spatial_reference_axes = self._spatial_reference_axes
        else:
            spatial_reference_axes = self.axes

        if isinstance(index, slice):
            index = [index] * len(spatial_reference_axes)

        if not isinstance(index, dict):
            # Arrange the indices requested using the reference image axes
            # ordering.
            index = {ax: sel for ax, sel in zip(spatial_reference_axes, index)}

        mode_index, _ = select_axes(self.axes, index)
        mode_scales = tuple(self.scale[ax] for ax in self.axes)

        mode_index = scale_coords(mode_index, mode_scales)

        mode_index = {ax: sel for ax, sel in zip(self.axes, mode_index)}

        # Locate the mode_index within the ROI:
        roi_mode_index = translate2roi(mode_index, self.roi, self.source_axes,
                                       self.axes)

        # Save the corresponding cache of this patch for faster access.
        cached_index = self._cache_chunk(roi_mode_index)
        selection = self._cache[cached_index]

        # Permute the axes order to match `axes`
        selection = selection.transpose(self.permute_order)

        # Drop axes with length 1 that are not in `axes`.
        out_shape = [s
                     for s, p_a in zip(selection.shape, self.permute_order)
                     if self.source_axes[p_a] not in self._drop_axes]
        selection = selection.reshape(out_shape)

        # Add axes requested in `axes` that does not exist on `source_axes`.
        selection = np.expand_dims(selection, tuple(self.axes.index(a)
                                                    for a in self._new_axes))

        if self._image_func is not None:
            selection = self._image_func(selection)

        return selection

    def _compute_shapes(self):
        if self._shape is not None:
            return

        self._shape = []
        self._chunk_size = []

        for a in self.axes:
            if a in self.source_axes:
                a_i = self.source_axes.index(a)
                r = self.roi[a_i]
                s = self.arr.shape[a_i]
                c = min(self.arr.chunks[a_i], s)

                r_start = 0 if r.start is None else r.start
                r_stop = s if r.stop is None else r.stop

                self._shape.append(r_stop - r_start)
                self._chunk_size.append(c)

            else:
                self._shape.append(1)
                self._chunk_size.append(1)

        if self._spatial_reference_axes is None:
            self._spatial_reference_axes = [
                ax for ax in self.axes if ax in self.spatial_axes
                ]

        if self._spatial_reference_shape is None:
            self._spatial_reference_shape = [
                s for s, ax in zip(self._shape, self.axes)
                if ax in self.spatial_axes
                ]

    def rescale(self, spatial_reference_shape: Union[Iterable[int], None]=None,
                spatial_reference_axes: Union[str, None]=None) -> None:
        """Rescale this image using the `spatial_reference_shape` as reference.

        Parameters
        ----------
        spatial_reference_shape: Union[Iterable[int], None]
            Reference image shape used to match extracted regions from this
            image (e.g., when calling __getitem__, or ImageBase[slice(...)])
        spatial_reference_axes: Union[str, None]
            Rescale only this axes from the image, keeping the rest unscaled.
        """
        if (self._scale is not None
           and spatial_reference_shape is None
           and spatial_reference_axes is None):
            return

        if spatial_reference_shape is not None:
            self._spatial_reference_shape = spatial_reference_shape

        if spatial_reference_axes is not None:
            self._spatial_reference_axes = spatial_reference_axes

        self._compute_shapes()

        self._scale = {}

        for ax, s in zip(self.axes, self.shape):
            if ax in self._spatial_reference_axes:
                ax_i = self._spatial_reference_axes.index(ax)
                ref_s = self._spatial_reference_shape[ax_i]

                self._scale[ax] = s / ref_s
            else:
                self._scale[ax] = 1.0

    @property
    def shape(self) -> Iterable[int]:
        self._compute_shapes()
        return self._shape

    @property
    def chunk_size(self) -> Iterable[int]:
        self._compute_shapes()
        return self._chunk_size

    @property
    def scale(self) -> dict:
        self.rescale()
        return self._scale


class ImageLoader(ImageBase):
    """Image lazy loader class.

    Opens the zarr file, or any image that can be open by TiffFile or PIL, as a
    Zarr array.

    Parameters
    ----------
    filename: str
    source_axes: str
    data_group: Union[str, None]
    axes: Union[str, None]
    roi: Union[str, slice, Iterable[slice], None]
    image_func: Union[Callable, None]
    zarr_store: Union[zarr.storage.Store, None]
    spatial_axes: str
    mode: str
    """
    def __init__(self, filename: str, source_axes: str,
                 data_group: Union[str, None] = None,
                 axes: Union[str, None] = None,
                 roi: Union[str, slice, Iterable[slice], None] = None,
                 image_func: Union[Callable, None] = None,
                 zarr_store: Union[zarr.storage.Store, None] = None,
                 spatial_axes: str = "ZYX",
                 mode: str = ""):
        self.mode = mode
        self.spatial_axes = spatial_axes

        if roi is None:
            parsed_roi = [slice(None)] * len(source_axes)
        elif isinstance(roi, str):
            parsed_roi = parse_rois([roi])[0]
        elif isinstance(roi, (list, tuple)):
            if len(roi) != len(source_axes):
                raise ValueError(f"ROIs does not match the number of the array"
                                 f" axes. Expected {len(source_axes)}, got "
                                 f"{len(roi)}")
            elif not all([isinstance(roi_ax, slice) for roi_ax in roi]):
                raise ValueError(f"ROIs must be slices, but got "
                                 f"{[type(roi_ax) for roi_ax in roi]}")
            else:
                parsed_roi = roi
        elif isinstance(roi, slice):
            if (len(source_axes) > 1
               and not (roi.start is None and roi.stop is None)):
                raise ValueError(f"ROIs must specify a slice per axes. "
                                 f"Expected {len(source_axes)} slices, got "
                                 f"only {roi}")
            else:
                parsed_roi = [roi] * len(source_axes)
        else:
            raise ValueError(f"Incorrect ROI format, expected a list of "
                             f"slices, or a parsable string, got {roi}")

        roi_slices = [
            slice(r.start if r.start is not None else 0, r.stop, None)
            for r in parsed_roi
        ]

        (self.arr,
         self._store) = image2array(filename, data_group=data_group,
                                    zarr_store=zarr_store)

        self.roi = roi_slices
        self.source_axes = source_axes
        self.axes = source_axes

        if axes is not None and axes != source_axes:
            source_axes_list = list(source_axes)
            self._drop_axes = list(set(source_axes) - set(axes))
            for d_a in sorted((source_axes_list.index(a)
                               for a in self._drop_axes),
                              reverse=True):
                if self.roi[d_a].stop is not None:
                    roi_len = self.roi[d_a].stop
                else:
                    roi_len = self.arr.shape[d_a]

                roi_len -= self.roi[d_a].start

                if roi_len > 1:
                    raise ValueError(f"Cannot drop axis `{source_axes[d_a]}` "
                                     f"(from `source_axes={source_axes}`) "
                                     f"because no selection was made for it, "
                                     f"and it is not being used as image axes "
                                     f"thereafter (`axes={axes}`).")

            self.permute_order = map_axes_order(source_axes, axes)
            self._new_axes = list(set(axes) - set(source_axes))
            self.axes = axes
        else:
            self.permute_order = list(range(len(self.axes)))

        self._image_func = image_func

        if image_func is not None:
            self.axes = image_func.axes

    def __del__(self):
        # Close the connection to the image file
        if self._store is not None:
            self._store.close()


class ImageCollection(object):
    """A class to contain a collection of inputs from different modalities.

    This is used to match images with their respective labels and masks.

    Parameters
    ----------
    collection_args : dict
        Collection arguments containing specifications to open `images`,
        `masks`, `labels`, etc.
    spatial_axes : str
        Spatial axes of the dataset, which are used to match different
        modalities using as reference these axes from the `images` collection.
    """
    def __init__(self, collection_args : dict,
                 spatial_axes: str = "ZYX"):

        self.reference_mode = list(collection_args.keys())[0]

        self.spatial_axes = spatial_axes

        self.collection = {
            mode: ImageLoader(spatial_axes=spatial_axes, mode=mode,
                              **mode_args)
            for mode, mode_args in collection_args.items()
        }

        self._generate_mask()
        self.reset_scales()

    def _generate_mask(self):
        mask_modes = list(
            filter(lambda k: "mask" in k,
                   self.collection.keys()
            )
        )

        if len(mask_modes) > 0:
            self.mask_mode = mask_modes[0]
            return

        ref_axes = self.collection[self.reference_mode].axes
        ref_shape = self.collection[self.reference_mode].shape
        ref_chunk_size = self.collection[self.reference_mode].chunk_size

        mask_axes = list(set(self.spatial_axes).intersection(ref_axes))
        mask_axes_ord = map_axes_order(mask_axes, ref_axes)
        mask_axes = [mask_axes[a] for a in mask_axes_ord]

        mask_chunk_size = [
            int(math.ceil(s / c))
            for s, c, a in zip(ref_shape, ref_chunk_size, ref_axes)
            if a in mask_axes
            ]

        self.collection["masks"] = ImageBase(shape=mask_chunk_size,
                                             chunk_size=mask_chunk_size,
                                             source_axes=mask_axes,
                                             mode="masks")
        self.mask_mode = "masks"

    def reset_scales(self) -> None:
        """Reset the scales between data modalities to match the `images`
        collection shape on the `spatial_axes` only.
        """
        img_shape = self.collection[self.reference_mode].shape
        img_source_axes = self.collection[self.reference_mode].source_axes
        img_axes = self.collection[self.reference_mode].axes

        spatial_reference_axes = [
            ax
            for ax in img_axes if ax in self.spatial_axes
        ]
        spatial_reference_shape = [
            img_shape[img_axes.index(ax)]
            if ax in img_source_axes else 1
            for ax in spatial_reference_axes
        ]

        for img in self.collection.values():
            img.rescale(spatial_reference_shape, spatial_reference_axes)

    def __getitem__(self, index):
        collection_set = {mode: img[index]
                          for mode, img in self.collection.items()}

        return collection_set
