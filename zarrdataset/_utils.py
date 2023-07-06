import os
import math
import zarr
import numpy as np
import dask
import dask.array as da

from urllib.parse import urlparse
import requests
from PIL import Image
import boto3
from io import BytesIO

from skimage import morphology, color, filters, transform

try:
    import tifffile
    TIFFFILE_SUPPORT=True

except ModuleNotFoundError:
    TIFFFILE_SUPPORT=False


def parse_roi(filename):
    """Parse the filename and ROIs from `filename`.

    The filename and ROIs must be separated by a semicolon (;).
    Any number of ROIs are accepted; however, this parser only supports
    rectangle shaped ROIs.

    Parameters
    ----------
    filename : str
        Path to the image. If the string does not contain any ROI in the format
        specified, the same filename is returned.

    Returns
    -------
    fn : str
        The parsed filename
    rois : list of tuples
        A list of regions of interest parsed from the input string.

    Notes
    -----
    ROIs are spected as filename;(start coordinate):(lenght of the ROI).
    Coordinates and lenghts are expected for each one of the axis in the image.
    The following is an exaple of a ROI defined for a test file in zarr format.

    test_file.zarr;(0, 12, 0, 0, 0):(10, 10, 1, 1, 1)

    This will parse a ROI from `test_file.zarr` from indices 0 to 10 in the
    first axis, indices from 12 to 22 in the second axis, and indices from 0 to
    1 in the third, fourth, and fifth axes.
    """
    rois = []
    if isinstance(filename, str):
        fn_rois_str = filename.split(";")
        if len(fn_rois_str) == 2:
            fn, rois_str = fn_rois_str
        else:
            fn = fn_rois_str[0]
            rois_str = []

        # Create a tuple of slices to define each ROI.
        for roi in rois_str:
            start_coords, axis_lengths = roi.split(":")

            start_coords = tuple([int(c.strip("\n\r ()"))
                                  for c in start_coords.split(",")])

            axis_lengths = tuple([int(ln.strip("\n\r ()"))
                                  for ln in axis_lengths.split(",")])

            roi_slices = tuple([slice(c_i, c_i + l_i, None) for c_i, l_i in
                                zip(start_coords, axis_lengths)])

            rois.append(roi_slices)

    return fn, rois


def map_axes_order(source_axes, target_axes="YX"):
    """Get the indices of a set of axes that reorders it to match another set
    of axes. This is can be used to transpose an array which coordinates
    systems is defined in a different ordering than the one needed.

    Parameters
    ----------
    source_axes : str
        Set of axes to be reordered to match `target_axes`.
    target_axes : str
        Set of axes in the desired order.

    Returns
    -------
    transpose_order : list of ints
        The indices order in `source_axes` that makes it match `target_axes`.
        If `source_axes` has more axes than `target_axes`, the unspecified axes
        will be moved to the front of the ordering, leaving the `target_axes`
        at the trailing positions.

    Notes
    -----
    The axes must be passed as a string in format `XYZ`, and cannot be appear
    more than once.
    """
    # Take only existing axes of the image
    target_axes = [t_ax for t_ax in target_axes if t_ax in source_axes]
    unused_axes = list(set(source_axes) - set(target_axes))
    transpose_order = [source_axes.index(a)
                       for a in unused_axes + list(target_axes)]

    return transpose_order


def select_axes(source_axes, axes_selection):
    """Get a sliced selection of axes from a zarr array.

    Parameters
    ----------
    source_axes : str
        Ordered set of axes on the original array.
    axes_selection : dict
        A relationship of what index to take from each specified axes.
        Unspecified axes are fully retrieved.

    Returns
    -------
    sel_slices : tuple of slices
        A tuple of slices that can be used to select the indices of the
        specified axes from an array.
    unfixed_axes : str
        A string with the ordered set of remaining axes of the array.

    Notes
    -----
    The axes must be passed as a string in format `XYZ`, and cannot be appear
    more than once.
    """
    sel_slices = []
    unfixed_axes = list(source_axes)

    for ax in source_axes:
        idx = axes_selection.get(ax, None)

        if idx is None:
            sel_slices.append(slice(None))

        else:
            # Remove this axis from the set of axes that the array has after
            # fixing this to index `idx`.
            unfixed_axes.remove(ax)
            sel_slices.append(idx)

    # These are the new set of data_axes of the array after fixing some axes to
    # the specified indices.
    unfixed_axes = "".join(unfixed_axes)
    sel_slices = tuple(sel_slices)

    return sel_slices, unfixed_axes


def connect_s3(filename_sample):
    """Stablish a connection with a S3 bucket.

    Parameters
    ----------
    filename_sample : str
        A sample filename containing the S3 end-point and bucket names.

    Returns
    -------
    s3_obj : dict of None
        A dictionary containing the bucket name and a boto3 client connected to
        the S3 bucket. If the filename does not points to a S3 bucket, this
        returns None.
    """
    if (filename_sample.startswith("s3")
       or filename_sample.startswith("http")):
        protocol = filename_sample.split(":")[0]
        endpoint = protocol + "://" + urlparse(filename_sample).netloc
        s3_obj = dict(
            bucket_name=filename_sample.split(endpoint)[1].split("/")[1],
            s3=boto3.client("s3", aws_access_key_id="",
                            aws_secret_access_key="",
                            region_name="us-east-2",
                            endpoint_url=endpoint))

        s3_obj["s3"]._request_signer.sign = (lambda *args, **kwargs: None)
    else:
        s3_obj = None
        
    return s3_obj


def isconsolidated(arr_src):
    """Check if the zarr file is consolidated so it is faster to open.

    Parameters
    ----------
    arr_src : str, zarr.Group, or zarr.Array
        The image filename, or zarr object, to be checked.
    """
    is_consolidated = False

    # TODO: Add more protocols to check, like gcp and s3
    if "http" in arr_src or "https" in arr_src:
        response = requests.get(arr_src + "/.zmetadata")
        is_consolidated = response.status_code == 200
    else:
        is_consolidated = os.path.exists(os.path.join(arr_src, ".zmetadata"))

    return is_consolidated


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
        arr = zarr.array(data=np.array(store),
                            shape=(height, width, channels),
                            dtype=np.uint8,
                            chunks=True)
    return arr, store


def compute_tissue_mask(filename, data_group, data_axes):
    # TODO: Make this function something customizable by the user

    # Find the closest to 1:16 of the highest resolution image in the zarr file
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

    ax_ref_ord = map_axes_order(data_axes, "CYX")
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
    # TODO: Consider cases of different number of dimensions.
    (sel_slices,
     unfixed_axes) = select_axes(data_axes, axes_selection={"T": 0, "Z": 0})
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

    mask = morphology.remove_small_objects(mask==0, min_size=16 * 16,
                                           connectivity=2)
    mask = morphology.remove_small_holes(mask, area_threshold=128 * 128)

    mask = morphology.binary_dilation(mask, morphology.disk(16))

    return mask


class ImageLoader(object):
    """Image lazy loader class.

    Opens the zarr file, or any image that can be open by TiffFile or PIL, as a
    Dask array.
    """
    def __init__(self, filename, data_group, data_axes, mask_data_group=None,
                 mask_data_axes=None,
                 s3_obj=None,
                 compute_valid_mask=False,
                 use_dask=False):

        # Separate the filename and any ROI passed as the name of the file
        self._filename, rois = parse_roi(filename)
        self._s3_obj = s3_obj
        self.mask_data_axes = mask_data_axes
        self.data_axes = data_axes
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
         self.mask_data_axes,
         self._mask_store,
         self.mask_scale) = self._get_valid_mask(compute_valid_mask)

    def _get_valid_mask(self, compute_valid_mask=False):
        ax_ref_ord = map_axes_order(self.data_axes, "YX")

        if "Y" in self.data_axes:
            H_ax = ax_ref_ord[-2]
            H = self.shape[H_ax]
            H_chk = self.chunk_size[H_ax]

        else:
            H = 1
            H_chk = 1

        if "X" in self.data_axes:
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
                    mask_data_axes = self.mask_data_axes
                else:
                    # Get the closest image to 1:16 of the highest resolution image
                    # and compute the tissue mask on that.
                    mask = compute_tissue_mask(self._filename,
                                               data_group=self.data_group,
                                               data_axes=self.data_axes)
                    mask_data_axes = "YX"

                sel_ax = []
                spatial_mask_axes = ""
                for ax in self.mask_data_axes:
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

        else:
            mask = np.ones((H_chk, W_chk), dtype=bool)
            mask_data_axes = "YX"
            mask_store = None
            mask_scale_H = H / H_chk
            mask_scale_W = W / W_chk

        return mask, mask_data_axes, mask_store, (mask_scale_H, mask_scale_W)

    def __getitem__(self, index):
        return self._arr[index]

    def __del__(self):
        # Close the connection to the image file 
        if self._store is not None:
            self._store.close()

        if self._mask_store is not None:
            self._mask_store.close()