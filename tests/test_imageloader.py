from zarrdataset import *
import pytest
from sample_images_generator import (IMAGE_SPECS,
                                     generate_sample_image,
                                     ImageTransformTest)


@pytest.mark.parametrize("image_specs", IMAGE_SPECS[:1])
@pytest.mark.parametrize("random_roi", [True, False])
@pytest.mark.parametrize("random_axes", [True, False])
@pytest.mark.parametrize("apply_transform", [True, False])


def test_image_loader(image_specs, random_roi, random_axes, apply_transform):
    (image_args,
     expected_shape,
     destroy_func) = generate_sample_image(image_specs, random_roi=random_roi,
                                           random_axes=random_axes)

    if apply_transform:
        transform = ImageTransformTest(image_args["axes"])
    else:
        transform = None

    img = ImageLoader(
        filename=image_args["filename"],
        source_axes=image_args["source_axes"],
        data_group=image_args["data_group"],
        axes=image_args["axes"],
        roi=image_args["roi"],
        image_func=transform,
        zarr_store=None,
        spatial_axes="ZYX",
        mode="r")

    assert isinstance(img, ImageBase), (f"Image loader returned an incorrect"
                                        f" type of object, expected one based"
                                        f" in ImageBase, got {type(img)}")
    assert all(map(lambda s1, s2: s1 == s2, img.shape, expected_shape)),\
          (f"Expected image of shape {expected_shape}"
           f", got {img.shape}")

    del img

    destroy_func()
