import os
import math
import zarr
import numpy as np
import dask
import dask.array as da

from PIL import Image
from io import BytesIO

from skimage import morphology, color, filters, transform

from ._utils import parse_metadata, isconsolidated, map_axes_order, select_axes

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
    use_dask : bool
        Whether use dask to lazy load arrays or not. If False, return the array
        as zarr array.

    Returns
    -------
    arr : dask.array.core.Array or zarr.Array
        A dask array for lazy loading the source array.
    """
    if isinstance(arr_src, zarr.Group):
        if use_dask:
            arr = da.from_zarr(arr_src, component=data_group)
        else:
            arr = arr_src[data_group]

        return arr, None

    if isinstance(arr_src, zarr.Array):
        # The array was already open from a zarr file
        if use_dask:
            arr = da.from_zarr(arr_src)
        else:
            arr = arr_src

        return arr, None

    if isinstance(arr_src, str):
        source_format = os.path.basename(arr_src).split(".")[-1]
    else:
        raise ValueError(
            f"The file/object {arr_src} cannot be opened as a zarr/dask array")

    if "zarr" in source_format:
        if use_dask:
            arr = da.from_zarr(arr_src, component=data_group)
        else:
            if isconsolidated(arr_src):
                arr = zarr.open_consolidated(arr_src, mode="r")[data_group]
            else:
                arr = zarr.open(arr_src, mode="r")[data_group]

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

    channels = len(store.getbands())
    height = store.size[1]
    width = store.size[0]
    if use_dask:
        arr = da.from_delayed(dask.delayed(np.array)(store),
                              shape=(height, width, channels),
                              dtype=np.uint8)
    else:
        arr = zarr.array(data=np.array(store), shape=(height, width, channels),
                         dtype=np.uint8,
                         chunks=True)
    return arr, store


def compute_tissue_mask(filename, data_group, source_axes, min_size=16,
                        area_threshold=128):
    if isconsolidated(filename):
        arr = zarr.open_consolidated(filename, mode="r")
    else:
        arr = zarr.open(filename, mode="r")

    group_root = "/".join(data_group.split("/")[:-1])
    group_keys = list(arr[group_root].group_keys())

    if len(group_keys) == 0:
        # If it was not possible to load the group keys, carry out a full
        # search for them.
        last_key = 0
        while True:
            last_key = last_key + 1
            if arr[group_root].get(str(last_key), None) is not None:

                group_keys.append(str(last_key))
            else:
                break

    ax_ref_ord = map_axes_order(source_axes, "YX")
    H_ax = ax_ref_ord[-2]
    W_ax = ax_ref_ord[-1]

    H_max = arr["%s/%i" % (group_root, 0)].shape[H_ax]
    W_max = arr["%s/%i" % (group_root, 0)].shape[W_ax]

    all_Hs = [arr["%s/%s" % (group_root, gk)].shape[H_ax] for gk in group_keys]
    closest_group = min(zip(map(lambda H: math.fabs(H - H_max / 16), all_Hs),
                            group_keys))[1]

    # Use that reference image as base to compute the tissue mask.
    base_wsi = da.from_zarr(filename, component="%s/%s" % (group_root,
                                                           closest_group))

    # Reorder the base image axes to have an order of YXC, and fix the
    # remaining axes.
    (sel_slices,
     unfixed_axes) = select_axes(source_axes, axes_selection={"T": 0, "Z": 0})
    permute_order = map_axes_order(unfixed_axes, "YXC")
    base_wsi = base_wsi[sel_slices]
    base_wsi = base_wsi.transpose(permute_order)
    base_wsi = base_wsi.rechunk((2048, 2048, 3))

    scaled_wsi = base_wsi.map_blocks(transform.resize,
                                     output_shape=(H_max // 16, W_max // 16),
                                     dtype=np.uint8,
                                     meta=np.empty((), dtype=np.uint8))

    scaled_wsi = scaled_wsi.compute()

    gray = color.rgb2gray(scaled_wsi)
    thresh = filters.threshold_otsu(gray)
    mask = gray > thresh

    mask = morphology.remove_small_objects(mask == 0, min_size=min_size ** 2,
                                           connectivity=2)
    mask = morphology.remove_small_holes(mask,
                                         area_threshold=area_threshold ** 2)
    mask = morphology.binary_dilation(mask, morphology.disk(min_size))

    return mask


class ImageLoader(object):
    """Image lazy loader class.

    Opens the zarr file, or any image that can be open by TiffFile or PIL, as a
    Dask array.
    """
    def __init__(self, filename, data_group, source_axes, mask_data_group=None,
                 mask_source_axes=None,
                 s3_obj=None,
                 compute_valid_mask=False,
                 use_dask=False):

        # Separate the filename and any ROI passed as the name of the file
        (self._filename,
         data_group,
         source_axes,
         target_axes,
         rois) = parse_metadata(filename)
        self._s3_obj = s3_obj
        self.mask_source_axes = mask_source_axes
        self.source_axes = source_axes
        self.data_group = data_group
        self.mask_data_group = mask_data_group

        (self._arr,
         self._store) = image2array(self._filename,
                                    data_group=self.data_group,
                                    s3_obj=self._s3_obj,
                                    use_dask=use_dask)

        self.shape = self._arr.shape
        if isinstance(self._arr, zarr.Array):
            self.chunk_size = self._arr.chunks
        elif isinstance(self._arr, da.core.Array):
            self.chunk_size = self._arr.chunksize

        if len(rois) == 0:
            rois = [tuple([slice(0, s, None) for s in self.shape])]

        self._rois = rois

        (self.mask,
         self.mask_source_axes,
         self._mask_store,
         self.mask_scale) = self._get_valid_mask(compute_valid_mask)

    def _get_valid_mask(self, compute_valid_mask=False):
        ax_ref_ord = map_axes_order(self.source_axes, "YX")

        if "Y" in self.source_axes:
            H_ax = ax_ref_ord[-2]
            H = self.shape[H_ax]
            H_chk = self.chunk_size[H_ax]

        else:
            H = 1
            H_chk = 1

        if "X" in self.source_axes:
            W_ax = ax_ref_ord[-1]
            W = self.shape[W_ax]
            W_chk = self.chunk_size[W_ax]

        else:
            W = 1
            W_chk = 1

        mask_store = None
        if compute_valid_mask:
            if ".zarr" in self._filename:
                if self.mask_data_group is not None:
                    # If the input file is stored in zarr format, try to retrieve
                    # the object mask from the `mask_data_group`.
                    (mask,
                    mask_store) = image2array(self._filename,
                                              data_group=self.mask_data_group,
                                              s3_obj=self._s3_obj,
                                              use_dask=False)
                    mask_source_axes = self.mask_source_axes
                else:
                    # Get the closest image to 1:16 of the highest resolution image
                    # and compute the tissue mask on that.
                    mask = compute_tissue_mask(self._filename,
                                               data_group=self.data_group,
                                               source_axes=self.source_axes)
                    mask_source_axes = "YX"

                sel_ax = []
                spatial_mask_axes = ""
                for ax in self.mask_source_axes:
                    if ax in "YX":
                        sel_ax.append(slice(None))
                        spatial_mask_axes += ax
                    else:
                        sel_ax.append(0)
                
                mask = mask[tuple(sel_ax)]

                ax_ord = map_axes_order(source_axes=spatial_mask_axes,
                                        target_axes="YX")
                mask = mask.transpose(ax_ord)

            mask_scale_H = mask.shape[0] / H
            mask_scale_W = mask.shape[1] / W

        else:
            mask_source_axes = "YX"
            mask_store = None
            mask_scale_H = 1 / H_chk
            mask_scale_W = 1 / W_chk
            mask = np.ones((int(H / H_chk), int(W / W_chk)), dtype=bool)

        roi_mask = np.zeros_like(mask, dtype=bool)

        for roi in self._rois:
            if len(roi) >= 2:
                scaled_roi = (slice(round(roi[H_ax].start * mask_scale_H),
                                    round(roi[H_ax].stop * mask_scale_H),
                                    None),
                            slice(round(roi[W_ax].start * mask_scale_W),
                                    round(roi[W_ax].stop * mask_scale_W),
                                    None))
            else:
                scaled_roi = slice(None)

            roi_mask[scaled_roi] = True

        mask = np.bitwise_and(mask, roi_mask)

        return mask, mask_source_axes, mask_store, (mask_scale_H, mask_scale_W)

    def __getitem__(self, index):
        return self._arr[index]

    def __del__(self):
        # Close the connection to the image file 
        if self._store is not None:
            self._store.close()

        if self._mask_store is not None:
            self._mask_store.close()