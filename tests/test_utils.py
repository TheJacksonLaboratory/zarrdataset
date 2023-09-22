import pytest
import operator

import zarrdataset as zds

import shutil
from pathlib import Path
import zarr
import boto3


@pytest.mark.parametrize("rois_str, expected_rois", [
    ("", [slice(None)]),
    ([""], [slice(None)]),
    ("(0):(10)", [(slice(0, 10), )]),
    ("(2, 5):(-1, 6)", [(slice(2, None), slice(5, 11))]),
])
def test_parse_rois(rois_str, expected_rois):
    parsed_rois = zds.parse_rois(rois_str)

    assert all(map(operator.eq, parsed_rois, expected_rois)), \
        (f"Expected parsed ROIs to be {expected_rois}, got {parsed_rois} "
         f"instead")


@pytest.mark.parametrize("filename, default_source_axes, default_data_group,"
                         "default_axes, default_rois, override_meta,"
                         "expected_metadata", [
    ("test.png", "YX", "0/0", None, None, True, [
        dict(
            filename="test.png",
            source_axes="YX",
            axes="YX",
            data_group="0/0",
            roi=None)
    ]),
    ("file; test.png", "CYX", "", "WCYX", None, True, [
        dict(
            filename="file; test.png",
            source_axes="CYX",
            axes="WCYX",
            data_group="",
            roi=None)
    ]),
    ("test.png;0/0;CYX:YXC;(0,0,1):(-1,-1,3)", "YCX", None, None,
     [(slice(0, 2), slice(0, 2), slice(None)),
      (slice(None), slice(None), slice(None))], False, [
        dict(
            filename="test.png",
            source_axes="CYX",
            axes="YXC",
            data_group="0/0",
            roi=(slice(None), slice(None), slice(1, 4))
        ),
    ]),
    ("test.png;0/0;CYX:YXC;(0,0,1):(-1,-1,3)", "YCX", None, None,
     [(slice(0, 2), slice(0, 2), slice(None)),
      (slice(None), slice(None), slice(None))], True, [
        dict(
            filename="test.png",
            source_axes="YCX",
            axes="YCX",
            data_group="0/0",
            roi=(slice(0, 2), slice(0, 2), slice(None))
        ),
        dict(
            filename="test.png",
            source_axes="YCX",
            axes="YCX",
            data_group="0/0",
            roi=(slice(None), slice(None), slice(None))
        ),
    ]),
    ("test.png;0/0;CYX:YXC;(0,0,1):(-1,-1,3);(2,3,1):(2,2,-1)", None, None,
     None, None, False, [
        dict(
            filename="test.png",
            source_axes="CYX",
            axes="YXC",
            data_group="0/0",
            roi=(slice(None), slice(None), slice(1, 4))
        ),
        dict(
            filename="test.png",
            source_axes="CYX",
            axes="YXC",
            data_group="0/0",
            roi=(slice(2, 4), slice(3, 5), slice(1, None))
        )
    ]),
    ([0, 1, 2], "C", "", None, None, True, [
        dict(
            filename=[0, 1, 2],
            source_axes="C",
            axes="C",
            data_group="",
            roi=None)
        ]
    ),
])
def test_parse_metadata(filename, default_source_axes, default_data_group,
                        default_axes,
                        default_rois,
                        override_meta,
                        expected_metadata):
    parsed_metadata = zds.parse_metadata(filename, default_source_axes,
                                         default_data_group,
                                         default_axes,
                                         default_rois,
                                         override_meta)
    
    assert isinstance(parsed_metadata, list), \
        (f"Expected parsed metadata to be a list of dictionaries, got "
         f"{type(parsed_metadata)}")

    assert all(all(par_meta[ax] == exp_val for ax, exp_val in exp_meta.items())
               for par_meta, exp_meta in zip(parsed_metadata,
                                             expected_metadata)), \
        (f"Expected parsed metadata to be {expected_metadata}, got "
         f"{parsed_metadata} instead.")


@pytest.mark.parametrize("source_axes, target_axes, expected_order", [
    ("YX", "XY", [1, 0]),
])
def test_map_axes_order(source_axes, target_axes, expected_order):
    transpose_order = zds.map_axes_order(source_axes, target_axes)

    assert transpose_order == expected_order, \
        (f"Expected order to be {expected_order}, got {transpose_order} "
         f"instead")


@pytest.mark.parametrize("source_axes, axes_selection, expected_sel_slices, "
                         "expected_unfixed_axes", [
    ("YX", dict(Y=slice(None), X=slice(None)), (slice(None), slice(None)),
     "YX"),
    ("Y", dict(Y=None), (slice(None), ), "Y"),
])
def test_select_axes(source_axes, axes_selection, expected_sel_slices,
                     expected_unfixed_axes):
    sel_slices, unfixed_axes = zds.select_axes(source_axes, axes_selection)

    assert sel_slices == expected_sel_slices, \
        (f"Expected selection to be {expected_sel_slices}, got {sel_slices} "
         f"instead.")

    assert unfixed_axes == expected_unfixed_axes, \
        (f"Expected unfixed axes to be {expected_unfixed_axes}, got "
         f"{unfixed_axes} instead.")


@pytest.mark.parametrize("filename_sample, expected_s3_obj", [
    ("test.png", None),
    ("https://live.staticflickr.com/4908/31072787307_59f7943caa_o.jpg", dict(
        bucket_name="4908",
        endpoint_url="https://live.staticflickr.com",
        s3=boto3.client("s3", aws_access_key_id="", aws_secret_access_key="",
                        region_name="us-east-2",
                        endpoint_url="https://live.staticflickr.com"))),
])
def test_connect_s3(filename_sample, expected_s3_obj):
    s3_obj = zds.connect_s3(filename_sample)

    if expected_s3_obj is None:
        assert s3_obj is None, \
            (f"Expected connection to be None when passing a local filename.")
    else:
        assert s3_obj["bucket_name"] == expected_s3_obj["bucket_name"], \
            (f"Expected bucket name {expected_s3_obj['bucket_name']}, got "
             f"{s3_obj['bucket_name']} instead.")
        assert s3_obj["endpoint_url"] == expected_s3_obj["endpoint_url"], \
            (f"Expected endpoiint url {expected_s3_obj['endpoint_url']}, got "
             f"{s3_obj['endpoint_url']} instead.")
        assert (s3_obj["s3"]._endpoint.host
                == expected_s3_obj["s3"]._endpoint.host), \
            (f"Expected client endpoint {expected_s3_obj['s3']._endpoint.host}"
             f", got {s3_obj['s3']._endpoint.host} instead.")


def test_consolidated():
    root_path = Path("tests/test_zarrs")
    root_path.mkdir(parents=True, exist_ok=True)

    try:
        root = zarr.open(root_path, mode="w")
        root.create_dataset("test/test_arr", shape=10, dtype="i8")

        zarr.consolidate_metadata(root_path)

        isconsolidated = zds.isconsolidated(str(root_path))

        assert isconsolidated, \
            (f"Expected {root_path} be consolidated, got {isconsolidated} "
             f"instead.")

    finally:
        shutil.rmtree(root_path)


@pytest.mark.parametrize("arr_src, expected_consolidated", [
    ("test.zarr", False),
    ("https://uk1s3.embassy.ebi.ac.uk/idr/zarr/v0.1/6001237.zarr", False),
])
def test_no_consolidated(arr_src, expected_consolidated):
    isconsolidated = zds.isconsolidated(arr_src)

    assert isconsolidated == expected_consolidated, \
        (f"Expected {arr_src} be consolidated={expected_consolidated}, got"
         f"{isconsolidated} instead.")


@pytest.mark.parametrize("selection_range, scale, expected_coords", [
    ([slice(None), slice(None)], [1, 1], (slice(None), slice(None))),
    ([slice(None), slice(None)], 1, (slice(None), slice(None))),
    ([None, slice(None)], 1, (slice(None), slice(None))),
    ([slice(2, 4)], 0.5, (slice(1, 2), )),
    ([(1, 5)], 2.01, (slice(2, 11), )),
    ([range(4)], 2, (slice(0, 8, 1), )),
    ((8, ), 0.25, (slice(2, 3), )),
])
def test_scale_coords(selection_range, scale, expected_coords):
    scaled_coords = zds.scale_coords(selection_range, scale)
    assert scaled_coords == expected_coords, \
        (f"Expected scaled coords {expected_coords}, got {scaled_coords} "
         f"instead.")


@pytest.mark.parametrize("selection_range, scale", [
    ([dict(Y=slice(None))], [1]),
])
def test_unsupported_scale_coords(selection_range, scale):
    with pytest.raises(ValueError):
        scaled_coords = zds.scale_coords(selection_range, scale)


@pytest.mark.parametrize("index, roi, source_axes, axes, expected_roi", [
    (dict(Y=slice(None), X=slice(None)), (slice(None), slice(None)), "YX",
     "YX", (slice(None), slice(None))),
    (dict(Y=slice(0, 5)), (slice(None), ), "Y", "Y", (slice(0, 5), )),
    (dict(Y=slice(0, 5)), (slice(5, 10), ), "Y", "Y", (slice(5, 10), )),
    (dict(Y=slice(0, 5)), (slice(5, 7), ), "Y", "Y", (slice(5, 7), )),
    (dict(Y=slice(0, 5)), (slice(None), slice(None)), "YZ", "Y",
     (slice(0, 5), slice(None))),
    (dict(W=slice(0, 5)), (slice(None), slice(None)), "YZ", "Z",
     (slice(None), slice(None))),
])
def test_translate2roi(index, roi, source_axes, axes, expected_roi):
    translated_roi = zds.translate2roi(index, roi, source_axes, axes)

    assert translated_roi == expected_roi, \
        (f"Expected translated ROI to be {expected_roi}, got {translated_roi} "
         f"instead.")
