from types import GeneratorType
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
        "tests/test_zarrs/zarr_group_1.zarr",
        "tests/test_zarrs/zarr_group_2.zarr",
        "tests/test_zarrs/zarr_group_3.zarr"
        ]

    return grps


def test_zarrdataset(generate_groups, generate_input_transform):
    dtype_transform, dtype = generate_input_transform

    data_group = "0/0"
    source_axes = "CYX"
    target_axes = "YXC"

    test_ds = ZarrDataset(
        filenames=generate_groups,
        data_group=data_group,
        source_axes=source_axes,
        axes=target_axes,
        transform=dtype_transform,
        return_any_label=True)

    test_ds_iter = iter(test_ds)
    assert isinstance(test_ds_iter, GeneratorType),\
        f"Expected a generator, not {type(test_ds_iter)}"

    for s in range(len(generate_groups)):        
        data_shape = zarr.open(generate_groups[s], mode="r")[data_group].shape

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
    dtype_transform, dtype = generate_input_transform
    labels_dtype_transform, labels_dtype = generate_target_transform

    data_group = "0/0"
    labels_data_group = "labels/0/0"
    source_axes = "CYX"
    target_axes = "YXC"

    labels_source_axes = "YX"
    labels_target_axes = "XY"

    test_ds = LabeledZarrDataset(
        filenames=generate_groups,
        data_group=data_group,
        source_axes=source_axes,
        axes=target_axes,
        labels_data_group=labels_data_group,
        labels_source_axes=labels_source_axes,
        labels_axes=labels_target_axes,
        transform=dtype_transform,
        target_transform=labels_dtype_transform,
        return_any_label=True)

    test_ds_iter = iter(test_ds)
    assert isinstance(test_ds_iter, GeneratorType),\
        f"Expected a generator, not {type(test_ds_iter)}"

    for s in range(len(generate_groups)):
        zarr_grp = zarr.open(generate_groups[s], mode="r")
        data_shape = zarr_grp[data_group].shape
        labels_shape = zarr_grp[labels_data_group].shape

        expected_shape = tuple(data_shape[source_axes.index(ax)]
                               for ax in target_axes)

        expected_labels_shape = tuple(
            labels_shape[labels_source_axes.index(ax)]
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


def test_weakly_labeled_zarrdataset(generate_groups, generate_input_transform,
                     generate_target_transform):
    dtype_transform, dtype = generate_input_transform
    labels_dtype_transform, labels_dtype = generate_target_transform

    data_group = "0/0"
    labels_data_group = "classes/0/0"
    source_axes = "CYX"
    target_axes = "YXC"

    labels_source_axes = "C"
    labels_target_axes = "C"

    test_ds = LabeledZarrDataset(
        filenames=generate_groups,
        data_group=data_group,
        source_axes=source_axes,
        axes=target_axes,
        labels_data_group=labels_data_group,
        labels_source_axes=labels_source_axes,
        labels_axes=labels_target_axes,
        transform=dtype_transform,
        target_transform=labels_dtype_transform,
        return_any_label=True)

    test_ds_iter = iter(test_ds)
    assert isinstance(test_ds_iter, GeneratorType),\
        f"Expected a generator, not {type(test_ds_iter)}"

    for s in range(len(generate_groups)):
        zarr_grp = zarr.open(generate_groups[s], mode="r")
        data_shape = zarr_grp[data_group].shape
        labels_shape = zarr_grp[labels_data_group].shape

        expected_shape = tuple(data_shape[source_axes.index(ax)]
                               for ax in target_axes)

        expected_labels_shape = tuple(
            labels_shape[labels_source_axes.index(ax)]
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
    dtype_transform, dtype = generate_input_transform
    data_group = "0/0"
    mask_data_group = "masks/0/0"
    source_axes = "CYX"
    target_axes = "YXC"

    data_shape = zarr.open(generate_groups[0], mode="r")[data_group].shape

    channels = data_shape[0]
    patch_height = 51
    patch_width = 25

    patch_shape = [channels, patch_height, patch_width]

    expected_shape = tuple(
        patch_shape[source_axes.index(ax)] for ax in target_axes
        )

    np.random.seed(15795)
    patch_sampler = GridPatchSampler(patch_size=(patch_height, patch_width))

    test_ds = MaskedZarrDataset(
        filenames=generate_groups,
        data_group=data_group,
        source_axes=source_axes,
        axes=target_axes,
        mask_data_group=mask_data_group,
        mask_source_axes="YX",
        mask_axes="YX",
        transform=dtype_transform,
        patch_sampler=patch_sampler,
        return_any_label=True)

    test_ds_iter = iter(test_ds)
    assert isinstance(test_ds_iter, GeneratorType),\
        f"Expected a generator, not {type(test_ds_iter)}"

    for s in range(len(generate_groups)):
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
    dtype_transform, dtype = generate_input_transform
    labels_dtype_transform, labels_dtype = generate_target_transform

    data_group = "0/2"
    mask_data_group = "masks/0/0"
    labels_data_group = "labels/0/0"
    source_axes = "CYX"
    target_axes = "YXC"

    labels_source_axes = "YX"
    labels_target_axes = "XY"

    data_shape = zarr.open(generate_groups[0], mode="r")[data_group].shape

    channels = data_shape[0]
    patch_height = 5
    patch_width = 3

    patch_shape = [channels, patch_height, patch_width]
    labels_patch_shape = [4 * patch_height, 4 * patch_width]

    expected_shape = tuple(
        patch_shape[source_axes.index(ax)] for ax in target_axes
        )

    expected_labels_shape = tuple(
        labels_patch_shape[labels_source_axes.index(ax)]
        for ax in labels_target_axes)

    np.random.seed(15795)
    patch_sampler = BlueNoisePatchSampler(patch_size=(patch_height,
                                                      patch_width))

    test_ds = MaskedLabeledZarrDataset(
        filenames=generate_groups,
        data_group=data_group,
        source_axes=source_axes,
        axes=target_axes,
        mask_data_group=mask_data_group,
        mask_source_axes="YX",
        mask_axes="YX",
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

    for s in range(len(generate_groups)):
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
    dtype_transform, dtype = generate_input_transform

    data_group = "0/1"
    mask_data_group = data_group
    mask_generator = WSITissueMaskGenerator(mask_scale=0.5,
                                            min_size=16,
                                            area_threshold=128)
    source_axes = "CYX"
    target_axes = "YXC"

    data_shape = zarr.open(generate_groups[0], mode="r")[data_group].shape

    channels = data_shape[0]
    patch_height = 4
    patch_width = 8

    patch_shape = [channels, patch_height, patch_width]

    expected_shape = tuple(
        patch_shape[source_axes.index(ax)] for ax in target_axes
        )

    np.random.seed(15795)
    patch_sampler = GridPatchSampler(patch_size=(patch_height, patch_width))

    test_ds = MaskedZarrDataset(
        filenames=generate_groups,
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

    for s in range(len(generate_groups)):
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
    dtype_transform, dtype = generate_input_transform
    labels_dtype_transform, labels_dtype = generate_target_transform

    data_group = "0/2"
    mask_data_group = data_group
    mask_generator = WSITissueMaskGenerator(mask_scale=1,
                                            min_size=16,
                                            area_threshold=128)
    labels_data_group = "labels/0/0"
    source_axes = "CYX"
    target_axes = "YXC"

    labels_source_axes = "YX"
    labels_target_axes = "XY"

    data_shape = zarr.open(generate_groups[0], mode="r")[data_group].shape

    channels = data_shape[0]
    patch_height = 5
    patch_width = 3

    patch_shape = [channels, patch_height, patch_width]
    labels_patch_shape = [4 * patch_height, 4 * patch_width]

    expected_shape = tuple(
        patch_shape[source_axes.index(ax)] for ax in target_axes
        )

    expected_labels_shape = tuple(
        labels_patch_shape[labels_source_axes.index(ax)]
        for ax in labels_target_axes)

    np.random.seed(15795)
    patch_sampler = BlueNoisePatchSampler(patch_size=(patch_height,
                                                      patch_width))

    test_ds = MaskedLabeledZarrDataset(
        filenames=generate_groups,
        data_group=data_group,
        source_axes=source_axes,
        axes=target_axes,
        mask_data_group=mask_data_group,
        mask_source_axes=source_axes,
        mask_axes="YXC",
        mask_func=mask_generator,
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

    for s in range(len(generate_groups)):
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
