from itertools import product
from types import GeneratorType
import numpy as np
from functools import partial
from zarrdataset import (ZarrDataset,
                         LabeledZarrDataset,
                         MaskedZarrDataset,
                         MaskedLabeledZarrDataset,
                         GridPatchSampler,
                         BlueNoisePatchSampler)
import pytest

from sample_images_generator import generate_group, EXPECTED_SHAPES


def transform_func(x, dtype=np.float32):
    return x.astype(dtype)


def transform_pair_func(x, t, dtype=np.float32):
    return x.astype(dtype), t.astype(dtype)


image_data = [
    (dict(
        filenames=[
            "https://uk1s3.embassy.ebi.ac.uk/idr/zarr/v0.1/6001237.zarr"
            ],
        data_group="0",
        source_axes="TCZYX",
        axes="YXC",
        roi="(0,0,30,0,0):(1,-1,1,-1,-1)",
        transform=partial(transform_func, dtype=np.float64)
        ),
     [(1024, 1024, 4)],
     None,
     (128, 128),
     (128, 128, 4),
     None,
     np.float64,
     None
    ),
    (dict(
        filenames=[
            "https://r0k.us/graphics/kodak/kodak/kodim01.png",
            "https://r0k.us/graphics/kodak/kodak/kodim02.png",
            "https://r0k.us/graphics/kodak/kodak/kodim03.png",
            "https://r0k.us/graphics/kodak/kodak/kodim04.png",
            ],
        data_group="",
        source_axes="YXC",
        axes="CYX",
        roi=None,
        transform=None
        ),
     [
         (3, 512, 768),
         (3, 512, 768),
         (3, 512, 768),
         (3, 768, 512),
     ],
     None,
     (75, 64),
     (3, 75, 64),
     None,
     np.uint8,
     None,
    ),
]

labeled_data = [
    (dict(
        filenames=[
            generate_group(2048, 1024, 512, 512, 2, 2, 0, 127)
            ],
        data_group="0/0",
        source_axes="TCZYX",
        axes="YXC",
        mask_data_group="masks/0/0",
        mask_source_axes="YX",
        mask_axes="YX",
        labels_data_group="labels/0/0",
        labels_source_axes="YX",
        labels_axes="XY",
        roi="(0,0,0,0,0):(1,-1,1,-1,-1)",
        transform=partial(transform_func, dtype=np.float32),
        target_transform=partial(transform_func, dtype=np.float32),
        input_target_transform=None
        ),
     [(2048, 1024, 3)],
     [(1024, 2048)],
     (128, 256),
     (128, 256, 3),
     (256, 128),
     np.float32,
     np.float32,
    ),
    (dict(
        filenames=[
            generate_group(2048, 2048, 512, 512, 2, 2, 0, 127),
            generate_group(1107, 404, 512, 256, 2, 3, 1, 127)
            ],
        data_group="0/0",
        source_axes="TCZYX",
        axes="YXC",
        roi="(0,0,0,0,0):(1,-1,1,-1,-1)",
        mask_data_group="masks/0/0",
        mask_source_axes="YX",
        mask_axes="YX",
        labels_data_group="labels/0/0",
        labels_source_axes="YX",
        labels_axes="YX",
        transform=partial(transform_func, dtype=np.float32),
        target_transform=partial(transform_func, dtype=np.float32),
        input_target_transform=None
        ),
     [(2048, 2048, 3),
      (1107, 404, 3)],
     [(2048, 2048),
      (1107, 404)],
     (128, 128),
     (128, 128, EXPECTED_SHAPES[0]["C"]),
     (128, 128),
     np.float32,
     np.float32,
    ),
    (dict(
        filenames=[
            generate_group(2048, 2048, 512, 512, 2, 2, 0, 127)["0/0"],
            generate_group(1107, 404, 512, 256, 2, 3, 1, 127)["0/0"]
            ],
        data_group="",
        source_axes="TCZYX",
        axes="YXC",
        roi="(0,0,0,0,0):(1,-1,1,-1,-1)",
        mask_data_group="masks/0/0",
        mask_source_axes="YX",
        mask_axes="YX",
        labels_data_group="labels/0/0",
        labels_source_axes="YX",
        labels_axes="WYX",
        transform=partial(transform_func, dtype=np.float32),
        target_transform=partial(transform_func, dtype=np.float32),
        input_target_transform=None,
        ),
     [(2048, 2048, 3),
      (1107, 404, 3)],
     [(1, 2048, 2048),
      (1, 1107, 404)],
     (256, 128),
     (256, 128, EXPECTED_SHAPES[0]["C"]),
     (1, 256, 128),
     np.float32,
     np.float32,
    ),
    (dict(
        filenames=[
            generate_group(2048, 2048, 512, 512, 2, 2, 0, 127)["0/0"][:],
            generate_group(1107, 404, 512, 256, 2, 3, 1, 127)["0/0"][:]
            ],
        data_group="",
        source_axes="TCZYX",
        axes="YXC",
        roi="(0,0,0,0,0):(1,-1,1,-1,-1)",
        mask_data_group="masks/0/0",
        mask_source_axes="YX",
        mask_axes="YX",
        labels_data_group="labels/0/0",
        labels_source_axes="YX",
        labels_axes="YXW",
        transform=partial(transform_func, dtype=np.float32),
        target_transform=partial(transform_func, dtype=np.float32),
        input_target_transform=None,
        ),
     [(2048, 2048, 3),
      (1107, 404, 3)],
     [(2048, 2048, 1),
      (1107, 404, 1)],
     (226, 128),
     (226, 128, EXPECTED_SHAPES[0]["C"]),
     (226, 128, 1),
     np.float32,
     np.float32,
    ),
    (dict(
        filenames=[
            "tests/test_zarrs/zarr_group_0.zarr",
            "tests/test_zarrs/zarr_group_1.zarr",
            "tests/test_zarrs/zarr_group_2.zarr",
            ],
        data_group="0/0",
        source_axes="TCZYX",
        axes="YXC",
        roi="(0,0,0,0,0):(1,-1,1,-1,-1)",
        mask_data_group="masks/0/0",
        mask_source_axes="YX",
        mask_axes="YX",
        labels_data_group="labels/0/0",
        labels_source_axes="YX",
        labels_axes="YX",
        transform=None,
        target_transform=partial(transform_func, dtype=np.float32),
        input_target_transform=partial(transform_pair_func, dtype=np.float64),
        ),
     [
         (EXPECTED_SHAPES[0]["Y"], EXPECTED_SHAPES[0]["X"], EXPECTED_SHAPES[0]["C"]),
         (EXPECTED_SHAPES[1]["Y"], EXPECTED_SHAPES[1]["X"], EXPECTED_SHAPES[1]["C"]),
         (EXPECTED_SHAPES[2]["Y"], EXPECTED_SHAPES[2]["X"], EXPECTED_SHAPES[2]["C"]),
     ],
     [
         (EXPECTED_SHAPES[0]["Y"], EXPECTED_SHAPES[0]["X"]),
         (EXPECTED_SHAPES[1]["Y"], EXPECTED_SHAPES[1]["X"]),
         (EXPECTED_SHAPES[2]["Y"], EXPECTED_SHAPES[2]["X"]),
     ],
     (275, 364),
     (275, 364, EXPECTED_SHAPES[0]["C"]),
     (275, 364),
     np.float64,
     np.float32,
    ),
    (dict(
        filenames=[
            "tests/test_images/img_0.png",
            "tests/test_images/img_1.png",
            "tests/test_images/img_2.png",
            ],
        data_group="",
        source_axes="YXC",
        axes="CYX",
        roi=None,
        mask_filenames=[
            "tests/test_images/mask_0.png",
            "tests/test_images/mask_1.png",
            "tests/test_images/mask_2.png",
            ],
        maks_data_group="",
        maks_source_axes="YX",
        maks_axes="YX",
        labels_filenames=[
            "tests/test_images/labels_0.png",
            "tests/test_images/labels_1.png",
            "tests/test_images/labels_2.png",
            ],
        labels_data_group="",
        labels_source_axes="YX",
        labels_axes="YX",
        transform=partial(transform_func, dtype=np.float32),
        target_transform=partial(transform_func, dtype=np.float32),
        input_target_transform=None,
        ),
     [
         (EXPECTED_SHAPES[0]["C"], EXPECTED_SHAPES[0]["Y"], EXPECTED_SHAPES[0]["X"]),
         (EXPECTED_SHAPES[1]["C"], EXPECTED_SHAPES[1]["Y"], EXPECTED_SHAPES[1]["X"]),
         (EXPECTED_SHAPES[2]["C"], EXPECTED_SHAPES[2]["Y"], EXPECTED_SHAPES[2]["X"]),
     ],
     [
         (EXPECTED_SHAPES[0]["Y"], EXPECTED_SHAPES[0]["X"]),
         (EXPECTED_SHAPES[1]["Y"], EXPECTED_SHAPES[1]["X"]),
         (EXPECTED_SHAPES[2]["Y"], EXPECTED_SHAPES[2]["X"]),
     ],
     (175, 180),
     (EXPECTED_SHAPES[0]["C"], 175, 180),
     (175, 180),
     np.float32,
     np.float32,
    ),
    (dict(
        filenames=[
            "tests/test_tiffs/img_0.ome.tif",
            "tests/test_tiffs/img_1.ome.tif",
            "tests/test_tiffs/img_2.ome.tif",
            ],
        data_group="0",
        source_axes="YXC",
        axes="CYX",
        roi=None,
        mask_filenames=[
            "tests/test_images/mask_0.png",
            "tests/test_images/mask_1.png",
            "tests/test_images/mask_2.png",
            ],
        maks_data_group="",
        maks_source_axes="YX",
        maks_axes="YX",
        labels_filenames=[
            "tests/test_images/labels_0.png",
            "tests/test_images/labels_1.png",
            "tests/test_images/labels_2.png",
            ],
        labels_data_group="",
        labels_source_axes="YX",
        labels_axes="YX",
        transform=partial(transform_func, dtype=np.float64),
        target_transform=None,
        input_target_transform=None,
        ),
     [
         (EXPECTED_SHAPES[0]["C"], EXPECTED_SHAPES[0]["Y"], EXPECTED_SHAPES[0]["X"]),
         (EXPECTED_SHAPES[1]["C"], EXPECTED_SHAPES[1]["Y"], EXPECTED_SHAPES[1]["X"]),
         (EXPECTED_SHAPES[2]["C"], EXPECTED_SHAPES[2]["Y"], EXPECTED_SHAPES[2]["X"]),
     ],
     [
         (EXPECTED_SHAPES[0]["Y"], EXPECTED_SHAPES[0]["X"]),
         (EXPECTED_SHAPES[1]["Y"], EXPECTED_SHAPES[1]["X"]),
         (EXPECTED_SHAPES[2]["Y"], EXPECTED_SHAPES[2]["X"]),
     ],
     (175, 164),
     (EXPECTED_SHAPES[0]["C"], 175, 164),
     (175, 164),
     np.float64,
     np.uint8,
    ),
]


@pytest.mark.parametrize("groups_data,expected_shapes,expected_label_shapes,patch_size,expected_patch_shape,expected_label_patch_shape,dtype,label_dtype", image_data + labeled_data)
def test_zarrdataset(groups_data, expected_shapes, expected_label_shapes,
                     patch_size,
                     expected_patch_shape,
                     expected_label_patch_shape,
                     dtype,
                     label_dtype):

    np.random.seed(15795)
    patch_samplers = [
        None,
        BlueNoisePatchSampler(patch_size=patch_size),
        GridPatchSampler(patch_size=patch_size)
    ]

    dataset_classes = [
        (ZarrDataset, False),
        (MaskedZarrDataset, False)
    ]

    if expected_label_shapes is not None:
        dataset_classes.append((LabeledZarrDataset, True))
        dataset_classes.append((MaskedLabeledZarrDataset, True))

    for (dataset_class, is_labeled), patch_sampler in product(dataset_classes,
                                                              patch_samplers):
        test_ds = dataset_class(
            **groups_data,
            patch_sampler=patch_sampler)

        for s, sample in enumerate(test_ds):
            if patch_sampler is None:
                actual_shape = expected_shapes[s]
            else:
                actual_shape = expected_patch_shape

            assert isinstance(sample, tuple),\
                (f"Elements yielded by the ZarrDataset should be "
                f"tuples not {type(sample)}")
            assert isinstance(sample[0], np.ndarray),\
                (f"Expected sample {s} to be a numpy NDArray, not "
                f"{type(sample[0])}")
            assert tuple(sample[0].shape) == actual_shape,\
                (f"Generated sample {s} does not have the correct shape, "
                 f"expected it to be {actual_shape}, not "
                 f"{sample[0].shape}")
            assert sample[0].dtype == dtype,\
                (f"Sample {s} must be of type {dtype}, not {sample[0].dtype}")

            if is_labeled:
                if patch_sampler is None:
                    actual_label_shape = expected_label_shapes[s]
                else:
                    actual_label_shape = expected_label_patch_shape

                assert isinstance(sample[1], np.ndarray),\
                    (f"Expected label {s} to be a numpy NDArray, not "
                    f"{type(sample[1])}")
                assert tuple(sample[1].shape) == actual_label_shape,\
                    (f"Generated label {s} does not have the correct shape, "
                    f"expected it to be {actual_label_shape}, not "
                    f"{sample[1].shape}")
                assert sample[1].dtype == label_dtype,\
                    (f"Label {s} must be of type {label_dtype}, not "
                    f"{sample[1].dtype}")
