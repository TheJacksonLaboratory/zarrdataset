import pytest

import zarrdataset as zds

import numpy as np
from skimage.draw import disk


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


@pytest.fixture(scope="function")
def input_mask(request):
    np.random.seed(75146)
    axes, mask_objects, labeled, only_centers = request.param
    shape_dict = dict(
        (ax, np.random.randint(1, 256))
        for ax in axes
    )
    shape = list(shape_dict.values())

    mask = np.zeros(shape=shape, dtype=np.int64)

    for l in range(1, mask_objects + 1):
        pos = {}
        if only_centers:
            pos["Y"] = [np.random.randint(0, shape_dict["Y"])]
            pos["X"] = [np.random.randint(0, shape_dict["X"])]

        else:
            radius = np.random.randint(1, max(1, min(shape_dict["X"],
                                                     shape_dict["Y"]) // 4))

            pos["Y"], pos["X"] = disk((np.random.randint(0, shape_dict["Y"]),
                                       np.random.randint(0, shape_dict["X"])),
                                       radius=radius,
                                       shape=(shape_dict["Y"], shape_dict["X"])
                                    )

        pos = tuple(
            pos[ax] if ax in pos else [0] * len(pos["X"])
            for ax in axes
        )

        if labeled:
            mask[pos] = l
        else:
            mask[pos] = 1

    return mask, shape_dict, axes, mask_objects, not labeled, not only_centers


def test_MaskGenerator(input_image):
    mask_generator = zds.MaskGenerator(axes=None)

    img, _, _ = input_image

    with pytest.raises(NotImplementedError):
        _ = mask_generator(img)


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


@pytest.mark.parametrize("input_mask",
[
    ("XY", 5, True, True),
    ("XY", 5, True, False),
    ("XY", 5, False, True),
    ("ZXY", 10, True, True),
], indirect=["input_mask"])
def test_LabelMaskGenerator(input_mask):
    in_mask, mask_shape, axes, mask_objects, labeled, only_centers = input_mask

    label_mask_generator = zds.LabelMaskGenerator(labeled,
                                                  only_centers,
                                                  axes)

    mask = label_mask_generator(in_mask)

    assert isinstance(mask, np.ndarray), \
        (f"Expected mask to be a Numpy NDArray, got {type(mask)} instead.")

    expected_shape = tuple(
        mask_shape[ax] if ax in mask_shape else 1
        for ax in label_mask_generator.axes
    )

    assert mask.shape == expected_shape, \
        (f"Expected mask has shape {expected_shape}, got {mask.shape} "
         f"instead.")

    assert np.sum(mask > 0) == mask_objects, \
        (f"Expected mask have {mask_objects}, got {np.sum(mask > 0)} instead.")


if __name__ == "__main__":
    class Request():
        def __init__(self, param):
            self.param = param

    parameters = [
        ("XY", 5, True, True),
        ("XY", 5, True, False),
        ("XY", 5, False, True),
        ("ZXY", 10, True, True),
    ]

    for param in parameters:
        test_LabelMaskGenerator(input_mask(Request(param)))
