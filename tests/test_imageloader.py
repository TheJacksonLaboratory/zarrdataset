from zarrdataset import ImageLoader, ImageBase
import pytest
from sample_images_generator import IMAGE_SPECS, generate_sample_image


@pytest.mark.parametrize("image_specs", IMAGE_SPECS)
@pytest.mark.parametrize("random_roi", [True, False])
@pytest.mark.parametrize("random_axes", [True, False])


def test_imageloader(image_specs, random_roi, random_axes):
    (image_args,
     expected_shape,
     destroy_func) = generate_sample_image(image_specs, random_roi=random_roi,
                                           random_axes=random_axes)

    img = ImageLoader(
        filename=image_args["filename"],
        source_axes=image_args["source_axes"],
        data_group=image_args["data_group"],
        axes=image_args["axes"],
        roi=image_args["roi"],
        image_func=None,
        zarr_store=None,
        spatial_axes="ZYX",
        mode="r")

    assert isinstance(img, ImageBase), (f"Image loader returned an incorrect"
                                        f" type of object, expected one based"
                                        f" in ImageBase, got {type(img)}")
    assert img.shape == expected_shape,\
          (f"Expected image of shape {expected_shape}"
           f", got {img.shape}")

    del img

    destroy_func()
