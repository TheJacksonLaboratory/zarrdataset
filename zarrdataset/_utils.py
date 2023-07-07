import os

from urllib.parse import urlparse
import requests
import boto3


def parse_metadata(filename):
    """Parse the filename, data groups, axes ordering, ROIs from `filename`.

    The different fields must be separated by a semicolon (;).
    After parsed the filename, data group, and axes, any number of ROIs are
    accepted; however, this parser only supports rectangle shaped ROIs.

    Parameters
    ----------
    filename : str
        Path to the image.

    Returns
    -------
    fn : str
        The parsed filename
    data_group: str
        The group for zarr images, or key for tiff files
    source_axes: str
        The orignal axes ordering
    target_axes: str
        The permuted axes ordering
    rois : list of tuples
        A list of regions of interest parsed from the input string.

    Notes
    -----
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
    source_axes = ""
    target_axes = ""
    data_group = ""
    rois = []
    rois_str = []

    if isinstance(filename, str):
        # There can be filenames with `;` on them, so take the point as
        # reference to start spliting the filename.
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

        # Create a tuple of slices to define each ROI.
        for roi in rois_str:
            start_coords, axis_lengths = roi.split(":")

            start_coords = tuple([int(c.strip("\n\r ()"))
                                  for c in start_coords.split(",")])

            axis_lengths = tuple([int(ln.strip("\n\r ()"))
                                  for ln in axis_lengths.split(",")])
            # TODO: When passing -1 use all the axis
            roi_slices = tuple([slice(c_i, c_i + l_i, None) for c_i, l_i in
                                zip(start_coords, axis_lengths)])

            rois.append(roi_slices)

    return fn, data_group, source_axes, target_axes, rois


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

    # These are the new set of source_axes of the array after fixing some axes to
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
