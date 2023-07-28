import os
import math
import zarr
import numpy as np
import dask
import dask.array as da

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


def image2array(arr_src, data_group=None, s3_obj=None,
                zarr_store=zarr.storage.FSStore):
    """Open images stored in zarr format or any image format that can be opened
    by PIL as an array.

    Parameters
    ----------
    arr_src : str, zarr.Group, or zarr.Array
        The image filename, or zarr object, to be loaded as a dask array.
    data_group : src or None
        The group within the zarr file from where the array is loaded. This is
        used only when the input file is a zarr object.
    s3_obj : dict or None
        A dictionary containing the bucket name and a boto3 client connection
        to a S3 bucket, or None if the file is stored locally.

    Returns
    -------
    arr : zarr.Array
        The image as a zarr array.
    """
    if isinstance(arr_src, zarr.Group):
        arr = arr_src[data_group]
        return arr, None

    elif isinstance(arr_src, zarr.Array):
        return arr_src, None

    elif isinstance(arr_src, str) and ".zarr" in arr_src:
        store = zarr_store(os.path.join(arr_src, data_group))
        arr = zarr.open(store, mode="r")
        return arr, None

    if TIFFFILE_SUPPORT:
        # Try first to open the input file with tifffile (if installed).
        # If that fails, try to open it with PIL.
        try:
            data_group = data_group.split("/")[-1]
            store = tifffile.imread(arr_src, aszarr=True,
                                    key=int(data_group))
            arr = zarr.open(store, mode="r")
            return arr, store

        except tifffile.tifffile.TiffFileError:
            pass

    # If the input is a path to an image stored in a format
    # supported by PIL, open it and use it as a numpy array.
    try:
        if s3_obj is not None:
            # The image is stored in a S3 bucket
            filename = arr_src.split(s3_obj["endpoint"]
                                     + "/"
                                     + s3_obj["bucket_name"])[1][1:]
            im_bytes = s3_obj["s3"].get_object(Bucket=s3_obj["bucket_name"],
                                               Key=filename)["Body"].read()
            store = Image.open(BytesIO(im_bytes))

        else:
            # The image is stored locally
            store = Image.open(arr_src, mode="r")

    except PIL.UnidentifiedImageError:
        raise ValueError(f"The file/object {arr_src} cannot be opened by "
                         f"zarr, dask, TiffFile, or PIL")

    channels = len(store.getbands())
    height = store.size[1]
    width = store.size[0]

    arr = zarr.array(data=np.array(store),
                     shape=(height, width, channels),
                     dtype=np.uint8)

    return arr, store


class ImageBase(object):
    spatial_axes = "ZYX"
    source_axes = None
    axes = None
    chunk_size = None
    scale = 1
    spatial_reference_axes = None
    mode = ""
    image_func = None
    image_func_args = {}
    _permute_order = None
    _new_axes = ""
    _drop_axes = ""
    _arr = None
    _shape = None
    _chunk_size = None
    _cached_coords = None

    def __init__(self, shape, chunk_size=None, axes=None):
        if chunk_size is None:
            chunk_size = shape

        self.source_axes = axes
        self.axes = axes
        self._permute_order = list(range(len(axes)))
        self._arr = zarr.ones(shape=shape, dtype=bool, chunks=chunk_size)
        self.roi = tuple([slice(None)] * len(axes))

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
                    self._arr.shape))

    def _cache_chunk(self, index):
        if not self._iscached(index):
            self._cached_coords = tuple(
                map(lambda i, chk, s:
                    slice(chk * int(i.start / chk)
                          if i.start is not None else 0,
                          min(s, chk * int(math.ceil(i.stop / chk)))
                          if i.stop is not None else None,
                          None),
                    index,
                    self._arr.chunks,
                    self._arr.shape)
            )
            self._cache = self._arr[self._cached_coords]

        cached_index = tuple(
            map(lambda cache, i:
                slice((i.start - cache.start) if i.start is not None else 0,
                      (i.stop - cache.start) if i.stop is not None else None,
                      None),
                self._cached_coords, index)
        )

        return cached_index

    def __getitem__(self, index):
        if isinstance(index, slice):
            index = [index] * len(self.axes)

        # Scale the spatial axes of the index according to this image scale.
        index = dict(((ax, sel)
                      for ax, sel in zip(self.spatial_reference_axes, index)))

        mode_index, _ = select_axes(self.axes, index)
        mode_index = scale_coords(mode_index, self.scale)

        # Locate the mode_index within the ROI:
        roi_mode_index = translate2roi(mode_index, self.roi, self.source_axes,
                                       self.axes)

        # Save the corresponding cache of this patch for faster access.
        cached_index = self._cache_chunk(roi_mode_index)
        selection = self._cache[cached_index]

        # Permute the axes order to match `axes`
        selection = selection.transpose(self._permute_order)

        # Drop axes with length 1 that are not in `axes`.
        out_shape = [s
                     for s, p_a in zip(selection.shape, self._permute_order)
                     if self.source_axes[p_a] not in self._drop_axes]
        selection = selection.reshape(out_shape)

        # Add axes requested in `axes` that does not exist on `source_axes`.
        selection = np.expand_dims(selection, tuple(self.axes.index(a)
                                                    for a in self._new_axes))

        # Apply the transformation to the selection.
        if self.image_func is not None:
            selection = self.image_func(selection, **self.image_func_args)

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
                s = self._arr.shape[a_i]
                c = self._arr.chunks[a_i]

                r_start = 0 if r.start is None else r.start
                r_stop = s if r.stop is None else r.stop

                if c > s:
                    c = s

                self._shape.append(r_stop - r_start)
                self._chunk_size.append(c)

            else:
                self._shape.append(1)
                self._chunk_size.append(1)

    @property
    def shape(self):
        self._compute_shapes()
        return self._shape

    @property
    def chunk_size(self):
        self._compute_shapes()
        return self._chunk_size


class ImageLoader(ImageBase):
    """Image lazy loader class.

    Opens the zarr file, or any image that can be open by TiffFile or PIL, as a
    Dask array.
    """
    def __init__(self, filename, source_axes, data_group=None, axes=None,
                 roi=None,
                 image_func=None,
                 image_func_args=None,
                 zarr_store=zarr.storage.FSStore,
                 spatial_axes="ZYX"):

        self.spatial_axes = spatial_axes
        self._s3_obj = connect_s3(filename)

        (self._arr,
         self._store) = image2array(filename,
                                    data_group=data_group,
                                    s3_obj=self._s3_obj,
                                    zarr_store=zarr_store)

        if roi is None:
            roi = [slice(None)] * len(source_axes)
        elif isinstance(roi, str):
            roi = parse_rois([roi])[0]

        roi = list(map(lambda r:
                       slice(r.start if r.start is not None else 0, r.stop,
                             None),
                       roi))

        self.roi = roi
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
                    roi_len = self._arr.shape[d_a]

                roi_len -= self.roi[d_a].start

                if roi_len > 1:
                    raise ValueError(f"Cannot drop axis `{source_axes[d_a]}` "
                                     f"(from `source_axes={source_axes}`) "
                                     f"because no selection was made for it, "
                                     f"and it is not being used as image axes "
                                     f"thereafter (`axes={axes}`).")

            self._permute_order = map_axes_order(source_axes, axes)
            self._new_axes = list(set(axes) - set(source_axes))
            self.axes = axes
        else:
            self._permute_order = list(range(len(axes)))

        self.image_func = image_func
        if image_func_args is None:
            image_func_args = {}
        self.image_func_args = image_func_args

    def __del__(self):
        # Close the connection to the image file
        if self._store is not None:
            self._store.close()


class ImageCollection(object):
    def __init__(self, collection_args,
                 spatial_axes="ZYX"):

        self.spatial_axes = spatial_axes
        self._cached_coords = None
        self._cache = None

        self.collection = dict((
            (mode, ImageLoader(spatial_axes=spatial_axes, **mode_args))
            for mode, mode_args in collection_args.items()
        ))

        self._generate_mask()
        self._compute_scales()

    def _generate_mask(self):
        if "masks" in self.collection.keys():
            return

        ref_axes = self.collection["images"].axes
        ref_shape = self.collection["images"].shape
        ref_chunk_size = self.collection["images"].chunk_size

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
                                             axes=mask_axes)

    def _compute_scales(self):
        img_shape = self.collection["images"].shape
        img_axes = self.collection["images"].axes
        spatial_reference_axes = [
            ax
            for ax in self.collection["images"].axes if ax in self.spatial_axes
        ]

        for mode, img in self.collection.items():
            curr_axes = img.axes
            curr_shape = img.shape

            img.scale = [
                2 ** -round(math.log2(img_shape[img_axes.index(a)] / s))
                if a in self.spatial_axes else 1.0
                for a, s in zip(curr_axes, curr_shape)
                ]
            img.spatial_reference_axes = spatial_reference_axes
            img.mode = mode

    def __getitem__(self, index):
        collection_set = dict((mode, img[index])
                              for mode, img in self.collection.items()
                              if mode != "masks")

        return collection_set

    def free_cache(self):
        for img in self.collection.values():
            img.free_cache()