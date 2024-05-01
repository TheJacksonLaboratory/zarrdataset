import zarrdataset as zds
from unittest import mock
import pytest
import importlib
import os
import shutil
import tifffile
import random

from pathlib import Path
import zarr
import numpy as np
from tests.utils import IMAGE_SPECS, UNSUPPORTED_IMAGE_SPECS


@pytest.fixture
def dummy_array():
    source_data = np.ones(shape=(16, 16, 3), dtype=np.uint8)
    source_axes = "YXC"
    return source_data, source_axes


@pytest.fixture(scope="function")
def input_image(request):
    if isinstance(request.param["source"], str):
        yield request.param["source"], request.param["specs"]

    else:
        dst_dir = request.param["dst_dir"]

        if dst_dir is not None:
            dst_dir = Path(request.param["dst_dir"])
            dst_dir.mkdir(parents=True, exist_ok=True)

        yield (request.param["source"](request.param["dst_dir"],
                                       request.param["specs"])[0],
               request.param["specs"])

        if dst_dir is not None and os.path.isdir(dst_dir):
            shutil.rmtree(dst_dir)


@pytest.mark.parametrize("input_image", IMAGE_SPECS, indirect=["input_image"])
def test_image_formats(input_image):
    image_src, image_specs = input_image

    expected_shape = tuple(image_specs["shape"])
    expected_chunks = tuple(image_specs["chunks"])

    store = None

    try:
        arr, store = zds.image2array(
            image_src,
            data_group=image_specs["data_group"]
        )

        assert arr.shape == expected_shape, \
            (f"Expected image of shape {expected_shape}, got {arr.shape}")

        assert arr.chunks == expected_chunks, \
            (f"Expected chunks {expected_chunks}, got {arr.chunks}")

    finally:
        if store is not None:
            store.close()


@pytest.mark.parametrize("input_image", UNSUPPORTED_IMAGE_SPECS,
                         indirect=["input_image"])
def test_unsupported_image_formats(input_image):
    image_src, image_specs = input_image

    with pytest.raises(ValueError):
        _ = zds.image2array(image_src,
                            data_group=image_specs["wrong_data_group"])


@pytest.mark.parametrize("shape, chunk_size, source_axes, expected_chunk_size",
    [
        ([16, 16, 3], [8, 8, 2], "YXC", [8, 8, 2]),
        ([16, 16, 3], None, "YXC", [16, 16, 3]),
        ([16, 16, 3], None, "YXC", [16, 16, 3]),
    ]
)
def test_ImageBase(shape, chunk_size, source_axes, expected_chunk_size):
    img = zds.ImageBase(shape, chunk_size=chunk_size, source_axes=source_axes,
                        mode="images")
    assert img.shape == shape, \
        (f"Expected shape is {shape}, got {img.shape} instead")

    assert img.chunk_size == expected_chunk_size, \
        (f"Expected chunk size is {expected_chunk_size}, got {img.chunk_size}"
         f" instead")

    expected_scale = dict((ax, 1) for ax in source_axes)
    assert img.scale == expected_scale, \
        (f"Expected scale before rescaling to be {expected_scale}, "
         f"got {img.scale} instead")


def test_ImageBase_slicing():
    shape = (16, 16, 3)
    axes = "YXC"
    ref_shape = (8, 64, 90)
    ref_axes = "ZXY"
    img = zds.ImageBase(shape, chunk_size=None, source_axes=axes, mode="image")

    img.rescale(spatial_reference_shape=ref_shape,
                spatial_reference_axes=ref_axes)
    
    ref_shape_dict = dict(
        (ax, s)
        for ax, s in zip(ref_axes, ref_shape)
    )

    expected_scales = dict(
        (ax, (s / ref_shape_dict[ax]) if ax in ref_axes else 1.0)
        for ax, s in zip(axes, shape)
    )

    assert img.scale == expected_scales, \
        (f"Expected scale after rescaling the image to be {expected_scales}, "
         f"got {img.scale} instead.")

    rescaled_img_shape = [
        (ref_shape_dict[ax] * img.scale[ax]) if ax in ref_axes else s
        for ax, s in zip(img.axes, img.shape)
    ]

    assert all([int(s) == se for s, se in zip(rescaled_img_shape, shape)]), \
        (f"Expected selection slice(None), None, :, to be of shape {shape}, "
         f"got {rescaled_img_shape} instead.")

    assert img[:].shape == shape, \
        (f"Expected selection slice(None), None, :, to be of shape {shape}, "
         f"got {img[:].shape} instead.")

    random.seed(44512)
    selection_1 = dict(
        (ax, slice(0, random.randint(1, r_s)))
        for ax, r_s in zip(ref_axes, ref_shape)
    )

    expected_selection_shape = tuple(
        round(expected_scales[ax] * selection_1[ax].stop)
        if ax in ref_axes else s
        for ax, s in zip(axes, shape)
    )

    img_sel_1 = img[selection_1]

    assert img_sel_1.shape == expected_selection_shape, \
        (f"Expected selection {selection_1} to have shape "
         f"{expected_selection_shape}, got {img_sel_1.shape} instead")

    selection_2 = tuple(selection_1[ax] for ax in ref_axes)

    img_sel_2 = img[selection_2]

    assert img_sel_2.shape == expected_selection_shape, \
        (f"Expected selection {selection_2} to have shape "
         f"{expected_selection_shape}, got {img_sel_2.shape} instead")


def test_ImageBase_padding():
    shape = (16, 16, 3)
    axes = "YXC"
    img = zds.ImageBase(shape, chunk_size=None, source_axes=axes, mode="image")

    random.seed(44512)
    selection_1 = dict(
        (ax, slice(random.randint(-10, 0),
                   random.randint(1, r_s + 10)))
        for ax, r_s in zip(axes, shape)
    )

    expected_selection_shape = tuple(
        selection_1[ax].stop - selection_1[ax].start for ax in axes
    )

    img_sel_1 = img[selection_1]

    assert img_sel_1.shape == expected_selection_shape, \
        (f"Expected selection {selection_1} to have shape "
         f"{expected_selection_shape}, got {img_sel_1.shape} instead")


@pytest.mark.parametrize("axes, roi, expected_size", [
    (None, None, (16, 16, 3)),
    (None, slice(None), (16, 16, 3)),
    (None, (slice(2, None, None), slice(7, None, None), slice(1, 2, None)),
     (14, 9, 1)),
    (None, [slice(2, None, None), slice(7, None, None), slice(1, 2, None)],
     (14, 9, 1)),
    (None, "(10,2,0):(5,10,-1)", (5, 10, 3)),
    ("WYC", "(7,8,0):(5,1,-1)", (1, 5, 3)),
])
def test_ImageLoader_supported_rois(dummy_array, axes, roi, expected_size):
    source_data, source_axes = dummy_array

    img = zds.ImageLoader(source_data, source_axes, axes=axes, roi=roi)

    assert img[:].shape == expected_size, \
        (f"Expected size of image ROI is {expected_size}, got {img[:].shape} "
         f"instead.")


@pytest.mark.parametrize("axes, roi", [
    (None, (slice(None), 10, slice(1, 2, None))),
    ("YX", slice(0, 10)),
    ("YX", (slice(1, 2, None), )),
    (None, {"Y": slice(None), "X": slice(None)}),
    ("YC", None),
])
def test_ImageLoader_unsupported_rois(dummy_array, axes, roi):
    source_data, source_axes = dummy_array

    with pytest.raises(ValueError):
        img = zds.ImageLoader(source_data, source_axes, axes=axes, roi=roi)


@pytest.mark.parametrize("axes, permute_order", [
    ("YXC", [0, 1, 2]),
    ("CXY", [2, 1, 0]),
    ("WZCYX", [2, 0, 1])
])
def test_ImageLoader_axes(dummy_array, axes, permute_order):
    source_data, source_axes = dummy_array

    img = zds.ImageLoader(source_data, source_axes, axes=axes)

    assert img.permute_order == permute_order, \
        (f"Incorrect permute ordering, expected {permute_order}, got "
         f"{img.permute_order} instead.")


def test_ImageLoader_function(dummy_array):
    class TestFunc(zds.MaskGenerator):
        def __init__(self):
            super(TestFunc, self).__init__(axes="YX")

        def _compute_transform(self, image):
            return np.mean(image, axis=-1)

    source_data, source_axes = dummy_array

    img = zds.ImageLoader(source_data, source_axes, axes="YXC",
                          image_func=TestFunc())

    res = img[:]

    assert res.shape == (16, 16), \
        (f"Expected shape of transformed image to be (16, 16), got {res.shape}"
         f" instead.")


@pytest.mark.parametrize("input_image", IMAGE_SPECS, indirect=["input_image"])
def test_ImageLoader_formats(input_image):
    image_src, image_specs = input_image

    expected_shape = image_specs["shape"]

    img = zds.ImageLoader(image_src,
                          source_axes=image_specs["source_axes"],
                          data_group=image_specs["data_group"])

    assert img.shape == expected_shape, \
        (f"Expected image of shape {expected_shape}, got {img.shape}")


def test_ImageCollection():
    img = zarr.array(data=np.random.randn(15, 15), shape=(15, 15),
                     chunks=(5, 5))
    mask = zarr.array(data=np.random.randn(10, 10), shape=(10, 10),
                      chunks=(5, 5))

    image_collection = zds.ImageCollection(collection_args=dict(
            images=dict(
                filename=img,
                data_group=None,
                source_axes="YX",
                axes=None,
                roi=None
            ),
            masks=dict(
                filename=mask,
                data_group=None,
                source_axes="YX",
                axes=None,
                roi=None
            )
        )
    )

    assert np.array_equal(image_collection.collection["masks"][:], mask[:]),\
        (f"Original mask and mask in collection are not the same. Expected "
         f"{mask[0, 0]}, got {image_collection.collection['masks'][0, 0]}")

    image_collection = zds.ImageCollection(collection_args=dict(
            images=dict(
                filename=img,
                data_group=None,
                source_axes="YX",
                axes=None,
                roi=None
            )
        )
    )

    expected_mask_size = [s//c for s, c in zip(img.shape, img.chunks)]

    assert image_collection.collection["masks"].shape == expected_mask_size,\
        (f"Auto-generated mask is not the same size as of the image chunks")
    
    patch = image_collection[:]

    assert isinstance(patch, dict), \
        (f"Expected patch extracted from image collection to be a tuple, got "
         f"{type(patch)} instead.")

    assert patch["images"].shape == img.shape, \
        (f"Expected patch shape to be {img.shape}, got {patch['images'].shape}"
         f" instead.")


def test_compatibility_no_tifffile():
    with mock.patch.dict('sys.modules', {'tifffile': None}):
        importlib.reload(zds._imageloaders)

        assert not zds._imageloaders.TIFFFILE_SUPPORT,\
            (f"If TiffFile is not installed, image loading functionalities "
             f"depending on it should be disabled, but TIFFFILE_SUPPORT is "
             f"{zds._imageloaders.TIFFFILE_SUPPORT}")

    with mock.patch.dict('sys.modules', {'tifffile': tifffile}):
        importlib.reload(zds._imageloaders)
        importlib.reload(zds)
