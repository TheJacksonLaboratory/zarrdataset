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


def image2array(arr_src, data_group=None, s3_obj=None):
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
    if (isinstance(arr_src, (zarr.Group, zarr.Array))
       or (isinstance(arr_src, str) and ".zarr" in arr_src)):
        arr = da.from_zarr(arr_src, component=data_group)
        return arr, None

    if TIFFFILE_SUPPORT:
        # Try first to open the input file with tifffile (if installed).
        # If that fails, try to open it with PIL.
        try:
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

    def __getitem__(self, index):
        if self.reference_axes is None:
            self.reference_axes = self.axes

        if isinstance(index, slice):
            index = tuple([index] * len(self.axes))

        index = dict(((ax, sel)
                      for ax, sel in zip(self.reference_axes, index)))

        index, _ = select_axes(self.axes, index)
        index = scale_coords(index, self.scale)

        cached_index = self._cache_chunk(index)
        return self._cache[cached_index]

    def _iscached(self, coords):
        if self._cached_coords is None:
            return False
        else:
            return (cache.start <= coord.start < coord.stop <= cache.stop
                    if cache.start and cache.stop else True
                    for coord, cache in zip(coords, self._cached_coords))

    def _cache_chunk(self, index):
        cached_coords = tuple((
            slice(chk * (i.start // chk)
                  if i.start is not None else 0,
                  chk * int(math.ceil(i.stop / chk))
                  if i.stop is not None else None,
                  None)
            for chk, i in zip(self.chunk_size, index)
            ))

        cached_index = tuple((
            slice(i.start - cached.start
                  if i.start is not None else 0,
                  i.stop - cached.start
                  if i.stop is not None else None)
            for cached, i in zip(cached_coords, index)
            ))

        if not self._iscached(cached_coords):
            with dask.diagnostics.ProgressBar():
                self._cache = self._arr[cached_coords].compute(
                    scheduler="synchronous")
            self._cached_coords = cached_coords

        return cached_index

    def _rechunk(self):
        spatial_chunks = [
            c if a in self.spatial_axes else s
            for c, a, s in zip(self._arr.chunksize, self.axes,
                               self._arr.shape)
            ]

        self._arr = self._arr.rechunk(spatial_chunks)
        self.chunk_size = self._arr.chunksize
        self.shape = self._arr.shape

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
                 spatial_axes="ZYX"):
        self.spatial_axes = spatial_axes
        self._s3_obj = connect_s3(filename)

        (self._arr,
         self._store) = image2array(filename,
                                    data_group=data_group,
                                    s3_obj=self._s3_obj)

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

        self._rechunk()

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

        for img in self.collection.values():
            curr_axes = img.axes
            curr_shape = img.shape

            img.scale = [
                (s / img_shape[img_axes.index(a)])
                if a in self.spatial_axes else 1.0
                for a, s in zip(curr_axes, curr_shape)
                ]
            img.reference_axes = img_axes

    def __getitem__(self, index):
        collection_set = dict((mode, img[index])
                              for mode, img in self.collection.items())

        return collection_set

    def free_cache(self):
        for img in self.collection.values():
            img.free_cache()
