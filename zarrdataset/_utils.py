from typing import Iterable, Union, List
import os
import math
from itertools import repeat
from urllib.parse import urlparse
import requests
import boto3
import zarr


def parse_rois(rois_str: Iterable[str]) -> List[slice]:
    """Parse a list of strings defining ROIs.

    Parameters
    ----------
    rois_str : Iterable[str]
        A list of regions of interest parsed from the input string.

    Returns
    -------
    rois : List[slice]
        A list of slices to select regions from arrays.
    """
    rois = []

    if isinstance(rois_str, str):
        rois_str = [rois_str]

    # Create a tuple of slices to define each ROI.
    for roi in rois_str:
        if ":" in roi:
            start_coords, axis_lengths = roi.split(":")

            start_coords = tuple([int(c.strip("\n\r ()"))
                                  for c in start_coords.split(",")])

            axis_lengths = tuple([int(ln.strip("\n\r ()"))
                                  for ln in axis_lengths.split(",")])

            roi_slices = []
            for c_i, l_i in zip(start_coords, axis_lengths):
                if l_i > 0:
                    roi_slices.append(slice(c_i, c_i + l_i, None))
                elif c_i > 0:
                    roi_slices.append(slice(c_i, None, None))
                else:
                    roi_slices.append(slice(None))

            roi_slices = tuple(roi_slices)
        else:
            roi_slices = slice(None)

        rois.append(roi_slices)

    return rois


def parse_metadata(filename: str, default_source_axes: str,
                   default_data_group: Union[str, int, None]=None,
                   default_axes: Union[str, None]=None,
                   default_rois: Union[Iterable[slice], None]=None,
                   override_meta: bool=False) -> List[dict]:
    """Parse the filename, data groups, axes ordering, ROIs from `filename`.

    The different fields must be separated by a semicolon (;).
    After parsed the filename, data group, and axes, any number of ROIs are
    accepted; however, this parser only supports rectangle shaped ROIs.

    Parameters
    ----------
    filename : str
        Path to the image.
    default_source_axes : src
        Default source axes ordering used when no axes are present in the
        filename string.
    default_data_group : Union[str, int, None]
        Default data group used when data group is present in the filename
        string.
    default_axes : Union[str, None]
        Default output axes ordering used when no target axes are present in
        the filename string.
    default_rois : Union[Iterable[slice], None]
        Default roi used when no rois are present in the filename string.
    override_meta : bool
        Whether to override the values parsed from the filename with the
        defaults or not.

    Returns
    -------
    parsed_metadata : list of dicts
        A list of parsed values from the filename. A single element is
        retrieved for each ROI in the filename.

    Notes
    -----
    The parsed metadata dictionary has the following structure.
    fn : str
        The parsed filename
    data_group: str or None
        The group for zarr images, or key for tiff files
    source_axes: str or None
        The orignal axes ordering
    axes: str or None
        The axes ordering as it being used from the array (may involve
        permuting, dropping unused axes, and creating new axes)
    rois : list of slices
        A list of regions of interest parsed from the input string.

    Data groups are expected for files with hierarchical structure, like zarr
    and tiff files. This is usually found on pyramid-like files, where the main
    image is stored in group `0/0`, and any downsampling version of that are
    stored in subgroups `0/1', '0/2`, etc. If the file does not have any
    hierarchical structure, leave the space in blank.

    Original axes and permuted desired order are expected as
    `source axes order`:`target axes order`. Axes identifiers can appear only
    once. Different number of target axes from the original number of axes can
    be requested. For more axes, dummy axes will be added in the position of
    the new requested axes. For less axes, the undefined axes will be
    positioned arbitrarly before those specified axes. It is responsability of
    the user to make remove unused axes with the a corresponding ROI.
    If no reordering is required, do not add the colon (:) on that section.

    ROIs are spected as (start coordinate):(lenght of the ROI). Coordinates and
    lenghts are expected for each one of the axes in the image, since all ROIs
    are taken before any axes reordering. Negative values can be used to select
    all indices of that axis.

    Example 1:
    test_file.zarr;0/0;TCZYX:YXC;(0,0,0,0,0):(1,-1,1,-1,-1)

    This will parse a ROI from `test_file.zarr` from the group `0/0`, which is
    expetect to have axes ordering as Time (T), Channels (C), Depth (Z),
    Height (Y), and Width (X), into an axes order of Height, Width and
    Channels. Index `0` is selected from both Time (T) and Depth(Z) axes.

    Example 2:
    test_file.zarr;1;CYX;(0,0,0):(-1,4,3);(0,5,5):(-1,2,2)

    This will parse a ROI from `test_file.zarr` from the array at group `1`,
    which is expetect to have axes ordering as Channels (C), Height (Y), and
    Width (X). From that array, two ROIs are extracted, one from coordinates
    (0, 0) with a shape (4, 3), and other from coordinates (5, 5) with shape
    (2, 2). Both ROIs use all the available channels.

    Example 3:
    test_array.zarr;;CYX:YXC

    The `test_array.zarr` file stores an array without any hierarchy, so no
    data group is required. Only the channels axes is required to be moved from
    the first position to the last one. Because no ROIs are defined, the full
    the array will be used.
    """
    source_axes = None
    target_axes = None
    data_group = None
    rois_str = []

    if default_axes is None:
        default_axes = default_source_axes

    if isinstance(filename, str):
        # There can be filenames with `;` on them, so take the dot separating
        # the filename and extension as reference to start spliting the
        # filename.
        fn_base_split = filename.split(".")
        ext_meta = fn_base_split[-1].split(";")
        fn_ext = ext_meta[0]
        fn = ".".join(fn_base_split[:-1]) + "." + fn_ext

        if len(ext_meta) > 1:
            data_group = ext_meta[1]
            axes_str = ext_meta[2].split(":")
            rois_str = ext_meta[3:]

            # Parse source and target axes ordering
            source_axes = axes_str[0]
            if len(axes_str) > 1:
                target_axes = axes_str[1]
    else:
        fn = filename

    if not source_axes or (override_meta and default_source_axes is not None):
        source_axes = default_source_axes

    if not data_group or (override_meta and default_data_group is not None):
        data_group = default_data_group

    if not target_axes or (override_meta and default_axes is not None):
        target_axes = default_axes

    rois = parse_rois(rois_str)
    if not rois or (override_meta and default_rois is not None):
        if not isinstance(default_rois, list):
            default_rois = [default_rois]
        rois = default_rois

    parsed_metadata = [
        {"filename": fn,
         "data_group": data_group,
         "source_axes": source_axes,
         "axes": target_axes,
         "roi": roi}
        for roi in rois
    ]

    return parsed_metadata


def map_axes_order(source_axes: str, target_axes: str="YX"):
    """Get the indices of a set of axes that reorders it to match another set
    of axes.

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


def select_axes(source_axes : str, axes_selection : dict):
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

        if idx is None or (idx.start is None and idx.stop is None):
            sel_slices.append(slice(None))

        else:
            # Remove this axis from the set of axes that the array has after
            # fixing this to index `idx`.
            unfixed_axes.remove(ax)
            sel_slices.append(idx)

    # These are the new set of source_axes of the array after fixing some axes
    # to the specified indices.
    unfixed_axes = "".join(unfixed_axes)
    sel_slices = tuple(sel_slices)

    return sel_slices, unfixed_axes


def connect_s3(filename_sample: str):
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
    if (isinstance(filename_sample, str)
        and (filename_sample.startswith("s3")
             or filename_sample.startswith("http"))):
        protocol = filename_sample.split(":")[0]
        endpoint = protocol + "://" + urlparse(filename_sample).netloc
        s3_obj = dict(
            bucket_name=filename_sample.split(endpoint)[1].split("/")[1],
            endpoint_url=endpoint,
            s3=boto3.client("s3", aws_access_key_id="",
                            aws_secret_access_key="",
                            region_name="us-east-2",
                            endpoint_url=endpoint))

        s3_obj["s3"]._request_signer.sign = (lambda *args, **kwargs: None)
    else:
        s3_obj = None

    return s3_obj


def isconsolidated(arr_src: Union[str, zarr.Group, zarr.Array]):
    """Check if the zarr file is consolidated so it is faster to open.

    Parameters
    ----------
    arr_src : Union[str, zarr.Group, zarr.Array]
        The image filename, or zarr object, to be checked.

    Returns
    -------
    is_consolidated : bool
        Whether zarr array in `arr_src` is consolidated or not.
    """
    is_consolidated = False

    # TODO: Add more protocols to check, like gcp and s3
    if "http" in arr_src or "https" in arr_src:
        response = requests.get(arr_src + "/.zmetadata")
        is_consolidated = response.status_code == 200
    else:
        is_consolidated = os.path.exists(os.path.join(arr_src, ".zmetadata"))

    return is_consolidated


def scale_coords(selection_range : Iterable[slice],
                 scale : Union[float, Iterable[float]] = 1.0):
    """Scale a set of top-lefts, bottom-rights coordinates, in any dimension,
    by `scale` factor.

    Parameters
    ----------
    selection_range : Iterable[slice]
        The selection range from an n-dimensional array to scale. This can be a
        range (start, end), single index, or None, defining the range of
        indices taken from each axis.
    scale : Union[float, Iterable[float]]
        The factor to rescale the selection range of each axes. If a single
        value is passed, all axes are rescaled by that factor.

    Returns
    -------
    scaled_selection_range : tuple of slices
        The scaled selection range as a tuple of slices
    """
    scaled_selection_range = []
    if not isinstance(scale, (list, tuple)):
        scale = repeat(scale)

    for ax_range, s in zip(selection_range, scale):
        if isinstance(ax_range, (slice, range)):
            if ax_range.start is None or ax_range.stop is None:
                scaled_selection_range.append(
                    slice(None)
                )
            else:
                scaled_selection_range.append(
                    slice(int(ax_range.start * s),
                          int(math.ceil(ax_range.stop * s)),
                          ax_range.step)
                )
        elif isinstance(ax_range, (tuple, list)):
            scaled_selection_range.append(
                slice(int(ax_range[0] * s), int(math.ceil(ax_range[1] * s)),
                      None)
            )
        elif isinstance(ax_range, int):
            scaled_selection_range.append(
                slice(int(ax_range * s), int(math.ceil((ax_range + 1) * s)),
                      None)
            )
        elif ax_range is None:
            scaled_selection_range.append(
                slice(None)
            )
        else:
            raise ValueError(f"Axis selection {ax_range} is not supported. "
                             f"Only ranges in form [start, end], "
                             f"single index, and None, are supported.")

    scaled_selection_range = tuple(scaled_selection_range)
    return scaled_selection_range


def translate2roi(index : dict, roi : tuple, source_axes : str, axes : str):
    roi_mode_index = {}
    for a_i, ax in enumerate(source_axes):
        r = roi[a_i]

        if ax in axes and ax in index:
            i = index[ax]

            i_start = i.start if i.start is not None else 0
            r_start = r.start if r.start is not None else 0

            i_start = i_start + r_start
            i_stop = (r_start + i.stop) if i.stop is not None else None

            if i_stop is not None and r.stop is not None:
                i_stop = min(i_stop, r.stop)
            elif i_stop is None:
                i_stop = r.stop

            if i_stop is None and i_start == 0:
                i_start = None

            roi_mode_index[ax] = slice(i_start, i_stop, None)
        else:
            roi_mode_index[ax] = r

    roi_mode_index, _ = select_axes(source_axes, roi_mode_index)

    return roi_mode_index
