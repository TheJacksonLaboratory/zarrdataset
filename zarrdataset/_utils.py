import os
import math
import zarr
import numpy as np
import dask
import dask.array as da

import requests
from PIL import Image
import boto3
from io import BytesIO


def parse_roi(filename, source_format):
    """Parse the filename and ROIs from `filename`.

    The filename and ROIs must be separated by a semicolon (;).
    Any number of ROIs are accepted; however, this parser only supports
    rectangle shaped ROIs.

    Parameters
    ----------
    filename : str
        Path to the image. If the string does not contain any ROI in the format
        specified, the same filename is returned.
    source_format : str
        Format of the input file.

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

    test_file.zarr;(0, 10, 0, 0, 0):(10, 10, 1, 1, 1)

    This will parse a ROI from `test_file.zarr` from indices 0 to 10 in the
    first axis, indices from 10 to 20 in the second axis, and indices from 0 to
    1 in the third, fourth, and fifth axes.
    """
    rois = []
    if isinstance(filename, str):
        split_pos = filename.lower().find(source_format)
        rois_str = filename[split_pos + len(source_format):]
        fn = filename[:split_pos + len(source_format)]
        rois_str = rois_str.split(";")[1:]

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
    unused_axes = list(set(source_axes) - set(target_axes))
    transpose_order = [source_axes.index(a)
                       for a in unused_axes + list(target_axes)]

    return transpose_order


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
        endpoint = "/".join(filename_sample.split("/")[:3])
        s3_obj = dict(bucket_name=filename_sample.split("/")[3],
                      s3=boto3.client("s3", aws_access_key_id="",
                                      aws_secret_access_key="",
                                      region_name="us-east-2",
                                      endpoint_url=endpoint))

        s3_obj["s3"]._request_signer.sign = (lambda *args, **kwargs: None)
    else:
        s3_obj = None
        
    return s3_obj


def load_image(filename, s3_obj=None):
    """Load an image stored locally or from a S3 bucket.

    Parameters
    ----------
    filename : str
        The image's filename.
    s3_obj : dict or None
        A dictionary containing the bucket name and a boto3 client connection
        to a S3 bucket, or None if the file is stored locally.

    Returns
    -------
    im : PIL.Image.Image
        The opened image object.

    Notes
    -----
    Only images with formats supported by PIL Image loader can be opened with
    this function.    
    """
    if s3_obj is not None:
        # Remove the end-point from the file name
        filename = "/".join(filename.split("/")[4:])
        im_bytes = s3_obj["s3"].get_object(Bucket=s3_obj["bucket_name"],
                                           Key=filename)["Body"].read()
        im = Image.open(BytesIO(im_bytes))

    else:
        im = Image.open(filename, mode="r")

    return im


def image2array(arr_src, source_format, data_group=None, s3_obj=None,
                use_dask=False):
    """Open images stored in zarr format or any image format that can be opened
    by PIL as an array.

    Parameters
    ----------
    arr_src : str, zarr.Group, or zarr.Array
        The image filename, or zarr object, to be loaded as a dask array.
    source_format : str
        The source format of the image to determine whether it is a zarr file
        or not.
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

    elif isinstance(arr_src, zarr.Array):
        # The array was already open from a zarr file
        if use_dask:
            arr = da.from_zarr(arr_src)
        else:
            arr = arr_src

    elif isinstance(arr_src, str) and ".zarr" in source_format:
        if use_dask:
            arr = da.from_zarr(arr_src, component=data_group)
        else:
            # Check if the zarr file is consolidated so it is faster to open.
            is_consolidated = False

            if "http" in arr_src:
                response = requests.get(arr_src + "/.zmetadata")
                is_consolidated = response.status_code == 200
            else:
                is_consolidated = os.path.exists(os.path.join(arr_src,
                                                              ".zmetadata"))

            if is_consolidated:
                arr = zarr.open_consolidated(arr_src, mode="r")[data_group]
            else:
                arr = zarr.open(arr_src, mode="r")[data_group]

    elif (isinstance(arr_src, str) and ".zarr" not in source_format):
        # If the input is a path to an image stored in a format
        # supported by PIL, open it and use it as a numpy array.
        im = load_image(arr_src, s3_obj=s3_obj)
        channels = len(im.getbands())
        if use_dask:
            arr = da.from_delayed(dask.delayed(np.array)(im),
                                  shape=(im.size[1], im.size[0], channels),
                                  dtype=np.uint8)
        else:
            arr = zarr.array(data=np.array(im),
                             shape=(im.size[1], im.size[0], channels),
                             dtype=np.uint8,
                             chunks=True)
    else:
        raise ValueError(
            f"The file/object {arr_src} cannot be opened as a zarr/dask array")

    return arr


class ImageLoader(object):
    """Image lazy loader class.

    Opens the zarr file, or any image that can be open by PIL, as a Dask array.
    """
    def __init__(self, filename, data_group, data_axes, mask_group=None,
                 mask_data_axes=None,
                 source_format=".zarr",
                 s3_obj=None,
                 compute_valid_mask=False,
                 use_dask=False):

        # Separate the filename and any ROI passed as the name of the file
        self._filename, rois = parse_roi(filename, source_format)
        self._s3_obj = s3_obj
        self.mask_data_axes = mask_data_axes
        self.data_axes = data_axes

        self._arr = image2array(self._filename, source_format=source_format,                                
                                data_group=data_group,
                                s3_obj=s3_obj,
                                use_dask=use_dask)

        self.shape = self._arr.shape
        if isinstance(self._arr, zarr.Array):
            self.chunk_size = self._arr.chunks
        elif isinstance(self._arr, da.core.Array):
            self.chunk_size = self._arr.chunksize

        if len(rois) == 0:
            rois = [tuple([slice(0, s, None) for s in self.shape])]

        self._rois = rois
        self.mask_group = mask_group

        if compute_valid_mask:
            self.mask, self.mask_scale = self._get_valid_mask()
        else:
            self.mask = None
            self.mask_scale = None

    def _get_valid_mask(self):
        ax_ref_ord = map_axes_order(self.data_axes, "YX")
        H_ax = ax_ref_ord[-2]
        W_ax = ax_ref_ord[-1]

        H = self.shape[H_ax]
        W = self.shape[W_ax]

        # If the input file is stored in zarr format, try to retrieve the object
        # mask from the `mask_group`.
        if self.mask_group is not None and ".zarr" in self._filename:
            mask = image2array(self._filename, source_format=".zarr",
                               data_group=self.mask_group,
                               s3_obj=self._s3_obj,
                               use_dask=False)

        else:
            mask = np.ones((round(H / self.chunk_size[H_ax]),
                            round(W / self.chunk_size[W_ax])), dtype=bool)
            self.mask_data_axes = "YX"

        mk_ax_ref_ord = map_axes_order(self.mask_data_axes, "Y")
        mask_scale = mask.shape[mk_ax_ref_ord[-1]] / H

        roi_mask = np.zeros_like(mask, dtype=bool)

        for roi in self._rois:
            if len(roi) >= 2:
                scaled_roi = (slice(round(roi[H_ax].start * mask_scale),
                                    round(roi[H_ax].stop * mask_scale),
                                    None),
                              slice(round(roi[W_ax].start * mask_scale),
                                    round(roi[W_ax].stop * mask_scale),
                                    None))
            else:
                scaled_roi = slice(None)

            roi_mask[scaled_roi] = True

        return np.bitwise_and(mask, roi_mask), mask_scale

    def __getitem__(self, index):
        return self._arr[index]
