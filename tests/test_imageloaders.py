import zarrdataset as zds
from unittest import mock
import pytest
import importlib
import tifffile

from pathlib import Path
import zarr
import numpy as np
from sample_images_generator import (IMAGE_SPECS,
                                     UNSUPPORTED_IMAGE_SPECS,
                                     remove_directory)


@pytest.mark.parametrize("image_specs", IMAGE_SPECS)
def test_image_formats(image_specs):
    dst_dir = None
    store = None

    if isinstance(image_specs["source"], str):
        img_src = image_specs["source"]

    else:
        if image_specs["dst_dir"] is not None:
            dst_dir = Path(image_specs["dst_dir"])
            dst_dir.mkdir(parents=True, exist_ok=True)

        img_src, _, _, _ = image_specs["source"](image_specs["dst_dir"],
                                                 image_specs["specs"])


    expected_shape = tuple(image_specs["specs"]["shape"])
    expected_chunks = tuple(image_specs["specs"]["chunks"])

    try:
        arr, store = zds.image2array(
            img_src,
            data_group=image_specs["specs"]["data_group"]
        )

        assert arr.shape == expected_shape, \
            (f"Expected image of shape {expected_shape}, got {arr.shape}")

        assert arr.chunks == expected_chunks, \
            (f"Expected chunks {expected_chunks}, got {arr.chunks}")

    finally:
        if store is not None:
            store.close()

        remove_directory(dir=dst_dir)


@pytest.mark.parametrize("image_specs", UNSUPPORTED_IMAGE_SPECS)
def test_unsupported_image_formats(image_specs):
    if image_specs["dst_dir"] is not None:
        dst_dir = Path(image_specs["dst_dir"])
        dst_dir.mkdir(parents=True, exist_ok=True)

    img_src, _, _, _ = image_specs["source"](image_specs["dst_dir"],
                                             image_specs["specs"])

    with pytest.raises(ValueError):
        arr, _ = zds.image2array(img_src,
                                 data_group=image_specs["wrong_data_group"])

    remove_directory(dir=dst_dir)


def test_ImageBase():
    pass


def test_ImageLoader():
    pass


def test_ImageCollection():
    img = zarr.array(data=np.random.randn(10, 10), shape=(10, 10),
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
