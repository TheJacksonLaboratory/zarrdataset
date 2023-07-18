
import zarr
import numpy as np
import dask
import dask.array as da

import PIL
from PIL import Image
from io import BytesIO

from ._utils import map_axes_order

try:
    import tifffile
    TIFFFILE_SUPPORT = True

except ModuleNotFoundError:
    TIFFFILE_SUPPORT = False


def image2array(arr_src, data_group=None, s3_obj=None,
                use_dask=False):
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
            if use_dask:
                arr = da.from_zarr(store)
            else:
                arr = zarr.open(store, mode='r')

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


class ImageLoader(object):
    """Image lazy loader class.

    Opens the zarr file, or any image that can be open by TiffFile or PIL, as a
    Dask array.
    """
    def __init__(self, filename, data_group=None, source_axes=None,
                 axes=None,
                 s3_obj=None,
                 roi=None):
        self._s3_obj = s3_obj

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

        self.shape = self._arr.shape
        if isinstance(self._arr, zarr.Array):
            self.chunk_size = self._arr.chunks
        elif isinstance(self._arr, da.core.Array):
            self.chunk_size = self._arr.chunksize
        else:
            self.chunk_size = self.shape

    def __getitem__(self, index):
        return self._arr[index].compute(scheduler="synchronous")

    def __del__(self):
        # Close the connection to the image file
        if self._store is not None:
            self._store.close()


class MaskLoader(ImageLoader):
    def __init__(self, filename, mask_func=None, mask_func_opts=None,
                 **kwargs):
        super(MaskLoader, self).__init__(filename, **kwargs)

        # If no mask generation function was passed, use the loaded array.
        # Otherwise, compute the mask using the passed function from
        # `self._arr`.
        if mask_func is not None:
            if mask_func_opts is None:
                mask_func_opts = {}
            self._arr, self.axes = mask_func(self._arr, **mask_func_opts)

        with dask.diagnostics.ProgressBar():
            self._arr = self._arr.persist(scheduler="synchronous")
        self.shape = self._arr.shape
