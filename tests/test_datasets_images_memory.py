from types import GeneratorType
from PIL import Image
import zarr
import numpy as np
from functools import partial
from zarrdataset import (ZarrDataset,
                         MaskedZarrDataset,
                         LabeledZarrDataset,
                         MaskedLabeledZarrDataset,
                         GridPatchSampler,
                         BlueNoisePatchSampler,
                         WSITissueMaskGenerator)
import pytest


def transform_func(x, dtype=np.float32):
    return x.astype(dtype)


@pytest.fixture
def generate_input_transform():
    return partial(transform_func, dtype=np.float32), np.float32


@pytest.fixture
def generate_target_transform():
    return partial(transform_func, dtype=np.float32), np.float32



@pytest.fixture
def generate_groups():
    grps = [
        "tests/test_images/img_0.png",
        "tests/test_images/img_1.png",
        "tests/test_images/img_2.png"
        ]

    mask_grps = [
        "tests/test_images/mask_0.png",
        "tests/test_images/mask_1.png",
        "tests/test_images/mask_2.png"
        ]

    labels_grps = [
        "tests/test_images/label_0.png",
        "tests/test_images/label_1.png",
        "tests/test_images/label_2.png"
        ]

    grps_arr = []
    for fn in grps:
        with Image.open(fn) as im:
            grps_arr.append(np.array(im))

    mask_grps_arr = []
    for fn in mask_grps:
        with Image.open(fn) as im:
            mask_grps_arr.append(np.array(im))

    labels_grps_arr = []
    for fn in labels_grps:
        with Image.open(fn) as im:
            labels_grps_arr.append(np.array(im))

    return grps_arr, mask_grps_arr, labels_grps_arr


def test_zarrdataset(generate_groups, generate_input_transform):
    image_groups, _, _ = generate_groups
    dtype_transform, dtype = generate_input_transform

    data_group = ""
    source_axes = "YXC"
    target_axes = "YCX"

    test_ds = ZarrDataset(
        filenames=image_groups,
        data_group=data_group,
        source_axes=source_axes,
        axes=target_axes,
        transform=dtype_transform,
        return_any_label=True)

    test_ds_iter = iter(test_ds)
    assert isinstance(test_ds_iter, GeneratorType),\
        f"Expected a generator, not {type(test_ds_iter)}"

    for s in range(len(image_groups)):        
        data_shape = image_groups[s].shape

        expected_shape = tuple(data_shape[source_axes.index(ax)]
                               for ax in target_axes)

        sample = next(test_ds_iter)
        assert isinstance(sample, tuple),\
            (f"Elements yielded by the ZarrDataset should be "
             f"tuples not {type(sample)}")
        assert isinstance(sample[0], np.ndarray),\
            (f"Expected sample {s} to be a numpy NDArray, not "
             f"{type(sample[0])}")
        assert sample[0].shape == expected_shape,\
            (f"Generated sample {s} does not have the correct shape, expected "
             f"it to be {expected_shape}, not {sample[0].shape}")
        assert sample[0].dtype == dtype,\
            (f"Sample {s} must be of type {dtype}, not {sample[0].dtype}")


def test_fully_labeled_zarrdataset(generate_groups, generate_input_transform,
                                   generate_target_transform):
    image_groups, _, labels_groups = generate_groups
    dtype_transform, dtype = generate_input_transform
    labels_dtype_transform, labels_dtype = generate_target_transform

    data_group = ""
    labels_data_group = ""
    source_axes = "YXC"
    target_axes = "YCX"

    labels_source_axes = "YX"
    labels_target_axes = "XCY"

    test_ds = LabeledZarrDataset(
        filenames=image_groups,
        data_group=data_group,
        source_axes=source_axes,
        axes=target_axes,
        labels_filenames=labels_groups,
        labels_data_group=labels_data_group,
        labels_source_axes=labels_source_axes,
        labels_axes=labels_target_axes,
        transform=dtype_transform,
        target_transform=labels_dtype_transform,
        return_any_label=True)

    test_ds_iter = iter(test_ds)
    assert isinstance(test_ds_iter, GeneratorType),\
        f"Expected a generator, not {type(test_ds_iter)}"

    for s in range(len(image_groups)):
        data_shape = image_groups[s].shape
        labels_shape = labels_groups[s].shape

        expected_shape = tuple(data_shape[source_axes.index(ax)]
                               for ax in target_axes)

        expected_labels_shape = tuple(
            labels_shape[labels_source_axes.index(ax)]
            if ax in labels_source_axes else 1
            for ax in labels_target_axes)

        sample = next(test_ds_iter)
        assert isinstance(sample, tuple),\
            (f"Elements yielded by the LabeledZarrDataset should be "
             f"tuples not {type(sample)}")
        assert isinstance(sample[0], np.ndarray),\
            (f"Expected sample {s} to be a numpy NDArray, not "
             f"{type(sample[0])}")
        assert sample[0].shape == expected_shape,\
            (f"Generated sample {s} does not have the correct shape, expected "
             f"it to be {expected_shape}, not {sample[0].shape}")
        assert sample[0].dtype == dtype,\
            (f"Sample {s} must be of type {dtype}, not {sample[0].dtype}")
        assert sample[1].shape == expected_labels_shape,\
            (f"Generated label {s} does not have the correct shape, expected "
             f" it to be {expected_labels_shape}, not "
             f"{sample[1].shape}")
        assert sample[1].dtype == labels_dtype,\
            (f"Label from sample {s} must be of type "
             f"{labels_dtype}, not "
             f"{sample[1].dtype}")


def test_masked_zarrdataset(generate_groups, generate_input_transform):
    image_groups, mask_groups, _ = generate_groups

    dtype_transform, dtype = generate_input_transform
    data_group = ""
    mask_data_group = ""
    source_axes = "YXC"
    target_axes = "YCX"

    data_shape = image_groups[0].shape

    channels = data_shape[-1]
    patch_height = 51
    patch_width = 31

    patch_shape = [patch_height, patch_width, channels]

    expected_shape = tuple(
        patch_shape[source_axes.index(ax)] for ax in target_axes
        )

    np.random.seed(15795)
    patch_sampler = GridPatchSampler(patch_size=(patch_height, patch_width))

    test_ds = MaskedZarrDataset(
        filenames=image_groups,
        data_group=data_group,
        source_axes=source_axes,
        axes=target_axes,
        mask_filenames=mask_groups,
        mask_data_group=mask_data_group,
        mask_source_axes="YX",
        mask_axes="YX",
        transform=dtype_transform,
        patch_sampler=patch_sampler,
        return_any_label=True)

    test_ds_iter = iter(test_ds)
    assert isinstance(test_ds_iter, GeneratorType),\
        f"Expected a generator, not {type(test_ds_iter)}"

    for s in range(len(image_groups)):
        sample = next(test_ds_iter)
        assert isinstance(sample, tuple),\
            (f"Elements yielded by the MaskedZarrDataset should be "
             f"tuples not {type(sample)}")
        assert isinstance(sample[0], np.ndarray),\
            (f"Expected sample {s} to be a numpy NDArray, not "
             f"{type(sample[0])}")
        assert sample[0].shape == expected_shape,\
            (f"Generated sample {s} does not have the correct shape, expected "
             f"it to be {expected_shape}, not {sample[0].shape}")
        assert sample[0].dtype == dtype,\
            (f"Sample {s} must be of type {dtype}, not {sample[0].dtype}")


def test_masked_labeled_zarrdataset(generate_groups, generate_input_transform,
                                    generate_target_transform):
    image_groups, mask_groups, labels_groups = generate_groups
    
    dtype_transform, dtype = generate_input_transform
    labels_dtype_transform, labels_dtype = generate_target_transform

    data_group = ""
    mask_data_group = ""
    labels_data_group = ""
    source_axes = "YXC"
    target_axes = "YCX"

    labels_source_axes = "YX"
    labels_target_axes = "CXY"

    data_shape = image_groups[0].shape

    channels = data_shape[-1]
    patch_height = 17
    patch_width = 23

    patch_shape = [patch_height, patch_width, channels]
    labels_patch_shape = [patch_height, patch_width]

    expected_shape = tuple(
        patch_shape[source_axes.index(ax)] for ax in target_axes
        )

    expected_labels_shape = tuple(
        labels_patch_shape[labels_source_axes.index(ax)]
        if ax in labels_source_axes else 1
        for ax in labels_target_axes)

    np.random.seed(15795)
    patch_sampler = BlueNoisePatchSampler(patch_size=(patch_height,
                                                      patch_width))

    test_ds = MaskedLabeledZarrDataset(
        filenames=image_groups,
        data_group=data_group,
        source_axes=source_axes,
        axes=target_axes,
        mask_filenames=mask_groups,
        mask_data_group=mask_data_group,
        mask_source_axes="YX",
        mask_axes="YX",
        labels_filenames=labels_groups,
        labels_data_group=labels_data_group,
        labels_source_axes=labels_source_axes,
        labels_axes=labels_target_axes,
        transform=dtype_transform,
        target_transform=labels_dtype_transform,
        patch_sampler=patch_sampler,
        return_any_label=True)

    test_ds_iter = iter(test_ds)
    assert isinstance(test_ds_iter, GeneratorType),\
        f"Expected a generator, not {type(test_ds_iter)}"

    for s in range(len(image_groups)):
        sample = next(test_ds_iter)
        assert isinstance(sample, tuple),\
            (f"Elements yielded by the MaskedLabeledZarrDataset should be "
             f"tuples not {type(sample)}")
        assert isinstance(sample[0], np.ndarray),\
            (f"Expected sample {s} to be a numpy NDArray, not "
             f"{type(sample[0])}")
        assert sample[0].shape == expected_shape,\
            (f"Generated sample {s} does not have the correct shape, expected "
             f"it to be {expected_shape}, not {sample[0].shape}")
        assert sample[0].dtype == dtype,\
            (f"Sample {s} must be of type {dtype}, not {sample[0].dtype}")
        assert sample[1].shape == expected_labels_shape,\
            (f"Generated label {s} does not have the correct shape, expected "
             f" it to be {expected_labels_shape}, not "
             f"{sample[1].shape}")
        assert sample[1].dtype == labels_dtype,\
            (f"Label from sample {s} must be of type "
             f"{labels_dtype}, not "
             f"{sample[1].dtype}")


def test_generated_mask_zarrdataset(generate_groups, generate_input_transform):
    image_groups, _, _ = generate_groups

    dtype_transform, dtype = generate_input_transform

    data_group = ""
    mask_data_group = data_group
    mask_generator = WSITissueMaskGenerator(mask_scale=1/16,
                                            min_size=16,
                                            area_threshold=128)
    source_axes = "YXC"
    target_axes = "YCX"

    data_shape = image_groups[0].shape

    channels = data_shape[-1]
    patch_height = 15
    patch_width = 13

    patch_shape = [patch_height, patch_width, channels]

    expected_shape = tuple(
        patch_shape[source_axes.index(ax)] for ax in target_axes
        )

    np.random.seed(15795)
    patch_sampler = GridPatchSampler(patch_size=(patch_height, patch_width))

    test_ds = MaskedZarrDataset(
        filenames=image_groups,
        data_group=data_group,
        source_axes=source_axes,
        axes=target_axes,
        mask_data_group=mask_data_group,
        mask_source_axes=source_axes,
        mask_axes="YXC",
        mask_func=mask_generator,
        transform=dtype_transform,
        patch_sampler=patch_sampler,
        return_any_label=True)

    test_ds_iter = iter(test_ds)
    assert isinstance(test_ds_iter, GeneratorType),\
        f"Expected a generator, not {type(test_ds_iter)}"

    for s in range(len(image_groups)):
        sample = next(test_ds_iter)
        assert isinstance(sample, tuple),\
            (f"Elements yielded by the MaskedZarrDataset should be "
             f"tuples not {type(sample)}")
        assert isinstance(sample[0], np.ndarray),\
            (f"Expected sample {s} to be a numpy NDArray, not "
             f"{type(sample[0])}")
        assert sample[0].shape == expected_shape,\
            (f"Generated sample {s} does not have the correct shape, expected "
             f"it to be {expected_shape}, not {sample[0].shape}")
        assert sample[0].dtype == dtype,\
            (f"Sample {s} must be of type {dtype}, not {sample[0].dtype}")


def test_generated_mask_labeled_zarrdataset(generate_groups,
                                            generate_input_transform,
                                            generate_target_transform):
    image_groups, _, labels_groups = generate_groups
    dtype_transform, dtype = generate_input_transform
    labels_dtype_transform, labels_dtype = generate_target_transform

    data_group = ""
    mask_data_group = data_group
    mask_generator = WSITissueMaskGenerator(mask_scale=1/16,
                                            min_size=16,
                                            area_threshold=128)
    labels_data_group = ""
    source_axes = "YXC"
    target_axes = "YCX"

    labels_source_axes = "YX"
    labels_target_axes = "XY"

    data_shape = image_groups[0].shape

    channels = data_shape[-1]
    patch_height = 15
    patch_width = 13

    patch_shape = [patch_height, patch_width, channels]
    labels_patch_shape = [patch_height, patch_width, 1]

    expected_shape = tuple(
        patch_shape[source_axes.index(ax)] for ax in target_axes
        )

    expected_labels_shape = tuple(
        labels_patch_shape[labels_source_axes.index(ax)]
        if ax in labels_source_axes else 1
        for ax in labels_target_axes)

    np.random.seed(15795)
    patch_sampler = BlueNoisePatchSampler(patch_size=(patch_height,
                                                      patch_width))

    test_ds = MaskedLabeledZarrDataset(
        filenames=image_groups,
        data_group=data_group,
        source_axes=source_axes,
        axes=target_axes,
        mask_data_group=mask_data_group,
        mask_source_axes=source_axes,
        mask_axes="YXC",
        mask_func=mask_generator,
        labels_filenames=labels_groups,
        labels_data_group=labels_data_group,
        labels_source_axes=labels_source_axes,
        labels_axes=labels_target_axes,
        transform=dtype_transform,
        target_transform=labels_dtype_transform,
        patch_sampler=patch_sampler,
        return_any_label=True)

    test_ds_iter = iter(test_ds)
    assert isinstance(test_ds_iter, GeneratorType),\
        f"Expected a generator, not {type(test_ds_iter)}"

    for s in range(len(image_groups)):
        sample = next(test_ds_iter)
        assert isinstance(sample, tuple),\
            (f"Elements yielded by the MaskedLabeledZarrDataset should be "
             f"tuples not {type(sample)}")
        assert isinstance(sample[0], np.ndarray),\
            (f"Expected sample {s} to be a numpy NDArray, not "
             f"{type(sample[0])}")
        assert sample[0].shape == expected_shape,\
            (f"Generated sample {s} does not have the correct shape, expected "
             f"it to be {expected_shape}, not {sample[0].shape}")
        assert sample[0].dtype == dtype,\
            (f"Sample {s} must be of type {dtype}, not {sample[0].dtype}")
        assert sample[1].shape == expected_labels_shape,\
            (f"Generated label {s} does not have the correct shape, expected "
             f" it to be {expected_labels_shape}, not "
             f"{sample[1].shape}")
        assert sample[1].dtype == labels_dtype,\
            (f"Label from sample {s} must be of type "
             f"{labels_dtype}, not "
             f"{sample[1].dtype}")
