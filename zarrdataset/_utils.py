import math
import zarr
import numpy as np
import dask
import dask.array as da

from PIL import Image
import boto3
from io import BytesIO


def parse_roi(filename, source_format):
    """Parse the filename and ROIs from `filename`.

    The filename and ROIs must be separated by a semicolon (;).
    Any number of ROIs are accepted. ROIs are expected to be passed as
    (start_coords:axis_lengths), in the axis order of the input data axes.

    Notes:
    ------
    An example of a ROI structure is the following.

    test_file.zarr;(0, 10, 0, 0, 0):(10, 10, 1, 1, 1)
    Will parse a ROI from \"test_file\" from 0:10 in the first axis, 10:20 in
    the second axis, 0:1 in the third to fifth axes.

    Parameters:
    ----------
    filename : str
        Path to the image.
    source_format : str
        Format of the input file.

    Returns
    -------
    fn : str
    rois : list of tuples
    """
    rois = []
    if isinstance(filename, str):
        split_pos = filename.lower().find(source_format)
        rois_str = filename[split_pos + len(source_format):]
        fn = filename[:split_pos + len(source_format)]
        rois_str = rois_str.split(";")[1:]

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


def get_spatial_axes_order(data_axes, spatial_axes="YX"):
    unused_axes = list(set(data_axes) - set(spatial_axes))
    transpose_order = [data_axes.index(a)
                       for a in unused_axes + list(spatial_axes)]

    return transpose_order


def connect_s3(filename_sample):
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
    if s3_obj is not None:
        # Remove the end-point from the file name
        filename = "/".join(filename.split("/")[4:])
        im_bytes = s3_obj["s3"].get_object(Bucket=s3_obj["bucket_name"],
                                           Key=filename)["Body"].read()
        with Image.open(BytesIO(im_bytes)) as im_s3:
            arr = im_s3.convert("RGB")
    else:
        im = Image.open(filename, mode="r").convert("RGB")
    return im


def image2dask(arr_src, source_format, data_group, s3_obj=None):
    if (isinstance(arr_src, zarr.Group) or (isinstance(arr_src, str)
       and ".zarr" in source_format)):
        arr = da.from_zarr(arr_src, component=data_group)

    elif isinstance(arr_src, zarr.Array):
        # The array was already open from a zarr file
        arr = da.from_zarr(arr_src)

    elif (isinstance(arr_src, str) and ".zarr" not in source_format):
        # If the input is a path to an image stored in a format
        # supported by PIL, open it and use it as a numpy array.
        im = load_image(arr_src, s3_obj=s3_obj)
        channels = len(im.getbands())
        arr = da.from_delayed(dask.delayed(np.array)(im),
                              shape=(im.size[1], im.size[0], channels),
                              dtype=np.uint8)

    return arr, arr.shape



class ImageLoader(object):
    """Image lazy loader class.

    Opens the zarr file, or any image that can be open by PIL, as a Dask array.
    """
    def __init__(self, filename, data_group, data_axes, mask_group=None,
                 mask_data_axes="YX",
                 source_format=".zarr",
                 s3_obj=None,
                 compute_valid_mask=False):

        # Separate the filename and any ROI passed as the name of the file
        self._filename, rois = parse_roi(filename, source_format)

        (self._arr,
         self.shape) = image2dask(self._filename, source_format, data_group,
                                  s3_obj)

        if len(rois) == 0:
            rois = [tuple([slice(0, s, None) for s in self.shape])]

        self._rois = rois

        self._data_axes = data_axes
        self._mask_group = mask_group
        self._mask_data_axes = mask_data_axes

        if compute_valid_mask:
            self.mask = self._get_valid_mask()
        else:
            self.mask = None

    def _get_valid_mask(self):
        spatial_axes = get_spatial_axes_order(self._data_axes, "YX")
        default_mask_scale = 1 / min(self.shape[spatial_axes[-1]],
                                     self.shape[spatial_axes[-2]])

        # If the input file is stored in zarr format, try to retrieve the object
        # mask from the `mask_group`.
        if self._mask_group is not None and ".zarr" in self._filename:
            mask = da.from_zarr(self._filename, component=self._mask_group)

            tr_ord = get_spatial_axes_order(self._mask_data_axes, "YX")
            mask = mask.transpose(tr_ord).squeeze()
            mask = mask.compute(scheduler="synchronous")

        else:
            scaled_h = int(math.floor(self.shape[-2] * default_mask_scale))
            scaled_w = int(math.floor(self.shape[-1] * default_mask_scale))

            mask = np.ones((scaled_h, scaled_w), dtype=bool)

        scale = mask.shape[-1] / self.shape[-1]
        roi_mask = np.zeros_like(mask, dtype=bool)
        tr_ord = get_spatial_axes_order(self._data_axes, "YX")

        for roi in self._rois:
            if len(roi) >= 2:
                roi = [roi[a] for a in tr_ord]
                scaled_roi = (slice(int(math.ceil(roi[-2].start * scale)),
                                    int(math.ceil(roi[-2].stop * scale)),
                                    None),
                            slice(int(math.ceil(roi[-1].start * scale)),
                                    int(math.ceil(roi[-1].stop * scale)),
                                    None))
            else:
                scaled_roi = slice(None)

            roi_mask[scaled_roi] = True

        return np.bitwise_and(mask, roi_mask)

    def __getitem__(self, index):
        return self._arr[index]
