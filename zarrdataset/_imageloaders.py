import os
import math
import zarr
import numpy as np
import dask
import dask.array as da

import PIL
from PIL import Image
from io import BytesIO

from ._utils import map_axes_order, connect_s3, scale_coords, select_axes

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
    arr : dask.array.core.Array or zarr.Array
        A dask array for lazy loading the source array.
    """
    if isinstance(arr_src, zarr.Group):
        arr = da.from_zarr(arr_src, component=data_group)
        return arr, None

    elif isinstance(arr_src, zarr.Array):
        arr = da.from_zarr(arr_src)
        return arr, None

    elif isinstance(arr_src, str) and ".zarr" in arr_src:
        store = zarr_store(os.path.join(arr_src, data_group))
        # z_arr = zarr.open(store, mode="r")
        arr = da.from_zarr(store)
        return arr, None

    if TIFFFILE_SUPPORT:
        # Try first to open the input file with tifffile (if installed).
        # If that fails, try to open it with PIL.
        try:
            data_group = data_group.split("/")[-1]
            store = tifffile.imread(arr_src, aszarr=True,
                                    key=int(data_group))
            arr = da.from_zarr(store)
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

    arr = da.from_delayed(dask.delayed(np.array)(store),
                          shape=(height, width, channels),
                          dtype=np.uint8)

    return arr, store


class ImageBase(object):
    spatial_axes = "ZYX"
    axes = None
    chunk_size = None
    scale = 1
    reference_axes = None
    mode = ""
    _arr = None
    _cached_coords = None
    _cache = None

    def __init__(self, shape, chunk_size=None, axes=None):
        if chunk_size is None:
            chunk_size = shape

        self.shape = shape
        self.chunk_size = chunk_size
        self.axes = axes

        self._arr = da.from_array(np.ones(shape=shape, dtype=bool))
        self._arr = da.rechunk(self._arr, chunk_size)

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
                    self.shape))

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
                    self.chunk_size,
                    self.shape)
            )
            self._cache = self._arr[self._cached_coords].compute(
                scheduler="synchronous")

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

        index = dict(((ax, sel)
                      for ax, sel in zip(self.reference_axes, index)))

        mode_index, _ = select_axes(self.axes, index)

        mode_index = scale_coords(mode_index, self.scale)
        cached_index = self._cache_chunk(mode_index)
        return self._cache[cached_index]

    def rechunk(self):
        spatial_chunks = [
            chk if a in self.spatial_axes else s
            for chk, a, s in zip(self.chunk_size, self.axes, self._arr.shape)
            ]

        self._arr = self._arr.rechunk(spatial_chunks)
        self.chunk_size = self._arr.chunksize

    def free_cache(self):
        self._cached_coords = None
        self._cache = None


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

        if roi is not None:
            self._arr = self._arr[roi]

        self.axes = source_axes

        if axes is not None and axes != source_axes:
            source_axes_list = list(source_axes)
            drop_axes = list(set(source_axes) - set(axes))
            for d_a in sorted((source_axes_list.index(a) for a in drop_axes),
                              reverse=True):
                try:
                    self._arr = self._arr.squeeze(d_a)
                except ValueError:
                    raise ValueError(f"Cannot drop axis `{source_axes[d_a]}` "
                                     f"(from `source_axes={source_axes}`) "
                                     f"because no selection was made for it, "
                                     f"and it is not being used as image axes "
                                     f"thereafter (`axes={axes}`).")

                source_axes_list.pop(d_a)

            permute_order = map_axes_order(source_axes_list, axes)
            self._arr = self._arr.transpose(permute_order)

            new_axes = list(set(axes) - set(source_axes_list))
            self._arr = np.expand_dims(self._arr, tuple(axes.index(a)
                                                        for a in new_axes))

            self.axes = axes

        if image_func is not None:
            if image_func_args is None:
                image_func_args = {}

            self._arr, self.axes = image_func(self._arr, **image_func_args)

        self.shape = self._arr.shape
        self.chunk_size = self._arr.chunksize

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
        self._rechunk()

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

        for mode, img in self.collection.items():
            curr_axes = img.axes
            curr_shape = img.shape

            img.scale = [
                2 ** -round(math.log2(img_shape[img_axes.index(a)] / s))
                if a in self.spatial_axes else 1.0
                for a, s in zip(curr_axes, curr_shape)
                ]
            img.reference_axes = img_axes
            img.mode = mode

    def _rechunk(self):
        for mode, img in self.collection.items():
            img.rechunk()

    def __getitem__(self, index):
        collection_set = dict((mode, img[index])
                              for mode, img in self.collection.items()
                              if mode != "masks")

        return collection_set

    def free_cache(self):
        for img in self.collection.values():
            img.free_cache()
