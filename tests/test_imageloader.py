from pathlib import Path
import pytest
from sample_images_generator import (IMAGE_SPECS,
                                     remove_directory,
                                     base_test_image_loader)


@pytest.mark.parametrize("random_roi", [True, False])
@pytest.mark.parametrize("random_axes", [True, False])
@pytest.mark.parametrize("apply_transform", [True, False])
def test_image_loaders(random_roi, random_axes, apply_transform):
    base_test_image_loader(IMAGE_SPECS[0], random_roi, random_axes,
                           apply_transform)


@pytest.mark.parametrize("image_specs", IMAGE_SPECS)
def test_image_formats(image_specs):
    base_test_image_loader(image_specs, False, False, False)


def test_incorrect_image_format():
    import zarrdataset as zds

    dst_dir = "tests/unsopported_images"
    filename = "tests/unsopported_images/image.unsopported"

    Path(dst_dir).mkdir(parents=True, exist_ok=True)
    fp = open(filename, "w")
    fp.close()

    with pytest.raises(ValueError):
        arr, store = zds.image2array(filename)

    remove_directory(dir=dst_dir)
