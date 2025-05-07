import pytest

import zarrdataset as zds

import operator
import numpy as np
import math


class TestMaskGenerator(zds.MaskGenerator):
    def __init__(self):
        super().__init__(axes="YX")

    def _compute_transform(self, image: np.ndarray) -> np.ndarray:
        h, w, _ = image.shape
        return np.ones((h, w), dtype=bool)


@pytest.fixture(scope="function")
def input_image():
    np.random.seed(75146)
    shape = [np.random.randint(1, 256) if ax != "C" else 3
             for ax in "YXC"]
    img = np.random.randint(low=0, high=255, size=shape, dtype=np.uint8)
    axes = "YXC"
    shape = dict(
        (ax, s)
        for ax, s in zip(axes, shape)
    )
    return img, shape, axes


def test_MaskGenerator(input_image):
    mask_generator = zds.MaskGenerator(axes=None)

    img, _, _ = input_image

    with pytest.raises(NotImplementedError):
        _ = mask_generator(img)


def test_scaled_MaskGenerator(input_image):
    mask_generator = TestMaskGenerator()

    img, img_shape, axes = input_image

    mask = mask_generator(img)

    expected_shape = [
        img_shape[ax]
        for ax in axes
        if ax in mask_generator.axes
    ]

    assert all(map(operator.eq, mask.shape, expected_shape)), \
        (f"Expected mask has shape {expected_shape}, got {mask.shape}")


@pytest.mark.parametrize("mask_scale, min_size, area_threshold, thresh, axes",
[
    (0.5, 5, 10, None, "YX"),
    (2, 5, 10, 0.5, "ZYX"),
])
def test_WSITissueMaskGenerator(input_image, mask_scale, min_size,
                                area_threshold,
                                thresh,
                                axes):
    img, img_shape, img_axes = input_image

    tissue_mask_generator = zds.WSITissueMaskGenerator(
        mask_scale=mask_scale,
        min_size=min_size,
        area_threshold=area_threshold,
        thresh=thresh,
        axes=axes
    )

    mask = tissue_mask_generator(img)

    assert isinstance(mask, np.ndarray), \
        (f"Expected mask to be a Numpy NDArray, got {type(mask)} instead.")

    expected_shape = tuple(
        round(img_shape[ax] * mask_scale) if ax in img_axes else 1
        for ax in tissue_mask_generator.axes
    )

    assert mask.shape == expected_shape, \
        (f"Expected mask has shape {expected_shape}, got {mask.shape} "
         f"instead.")
