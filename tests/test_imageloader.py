from zarrdataset import ImageLoader, ImageBase
import pytest
from sample_images_generator import EXPECTED_SHAPES

image_data = [
    ("tests/test_tiffs/img_0.ome.tif", "", "YXC", None, "CYX", [EXPECTED_SHAPES[0]["C"], EXPECTED_SHAPES[0]["Y"], EXPECTED_SHAPES[0]["X"]]),
    ("tests/test_tiffs/img_1.ome.tif", "", "YXC", None, "CYX", [EXPECTED_SHAPES[1]["C"], EXPECTED_SHAPES[1]["Y"], EXPECTED_SHAPES[1]["X"]]),
    ("tests/test_tiffs/img_2.ome.tif", "", "YXC", None, "CYX", [EXPECTED_SHAPES[2]["C"], EXPECTED_SHAPES[2]["Y"], EXPECTED_SHAPES[2]["X"]]),
    ("tests/test_images/img_0.png", "", "YXC", None, "CYX", [EXPECTED_SHAPES[0]["C"], EXPECTED_SHAPES[0]["Y"], EXPECTED_SHAPES[0]["X"]]),
    ("tests/test_images/img_1.png", "", "YXC", None, "CYX", [EXPECTED_SHAPES[1]["C"], EXPECTED_SHAPES[1]["Y"], EXPECTED_SHAPES[1]["X"]]),
    ("tests/test_images/img_2.png", "", "YXC", "(100,250,0):(500,127,-1)", "CYX", [EXPECTED_SHAPES[2]["C"], 500, 127]),
    ("tests/test_zarrs/zarr_group_0.zarr", "0/0", "TCZYX", None, "CYXZT", [EXPECTED_SHAPES[0]["C"], EXPECTED_SHAPES[0]["Y"], EXPECTED_SHAPES[0]["X"], EXPECTED_SHAPES[0]["Z"], EXPECTED_SHAPES[0]["T"]]),
    ("tests/test_zarrs/zarr_group_1.zarr", "0/0", "TCZYX", "(0,0,0,0,0):(1,-1,1,-1,-1)", "CYWX", [EXPECTED_SHAPES[1]["C"], EXPECTED_SHAPES[1]["Y"], 1, EXPECTED_SHAPES[1]["X"]]),
    ("tests/test_zarrs/zarr_group_2.zarr", "0/0", "TCZYX", "(0,0,0,100,250):(1,-1,1,500,127)", "YXC", [500, 127, EXPECTED_SHAPES[2]["C"]]),
]


@pytest.mark.parametrize("image_url,data_group,source_axes,roi,axes,expected_shape", image_data)
def test_imageloader(image_url, data_group, source_axes, roi, axes,
                     expected_shape):

    img = ImageLoader(image_url, data_group=data_group,
                      source_axes=source_axes,
                      axes=axes,
                      roi=roi)

    assert isinstance(img, ImageBase), (f"Image loader returned an incorrect"
                                        f" type of object, expected one based"
                                        f" in ImageBase, got {type(img)}")
    assert img.shape == expected_shape, (f"Expected image of shape "
                                         f"{expected_shape}, got {img.shape}")
