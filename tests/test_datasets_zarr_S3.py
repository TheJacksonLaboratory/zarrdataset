from types import GeneratorType
import zarr
import numpy as np
from functools import partial
from zarrdataset import (ZarrDataset,
                         GridPatchSampler,
                         BlueNoisePatchSampler)
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
    groups = dict(
        filenames=[
            "https://uk1s3.embassy.ebi.ac.uk/idr/zarr/v0.1/6001237.zarr"
            ],
        data_group="0",
        source_axes="TCZYX",
        axes="YXC",
        roi="(0,0,30,0,0):(1,-1,1,-1,-1)",
        )
    return groups


def test_zarrdataset(generate_groups, generate_input_transform):
    dtype_transform, dtype = generate_input_transform
    test_ds = ZarrDataset(
        **generate_groups,
        transform=dtype_transform,
        return_any_label=True)

    test_ds_iter = iter(test_ds)
    assert isinstance(test_ds_iter, GeneratorType),\
        f"Expected a generator, not {type(test_ds_iter)}"

    for s in range(len(generate_groups["filenames"])):
        z_grp = zarr.open(generate_groups["filenames"][s], mode="r")
        data_shape = z_grp[generate_groups["data_group"]].shape

        expected_shape = tuple(
            data_shape[generate_groups["source_axes"].index(ax)]
            for ax in generate_groups["axes"])

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


def test_grid_patched_zarrdataset(generate_groups, generate_input_transform):
    dtype_transform, dtype = generate_input_transform

    patch_height = 512
    patch_width = 328
    patch_sampler = GridPatchSampler(patch_size=(patch_height, patch_width))

    test_ds = ZarrDataset(
        **generate_groups,
        transform=dtype_transform,
        patch_sampler=patch_sampler,
        return_any_label=True)

    z_grp = zarr.open(generate_groups["filenames"][0], mode="r")
    data_shape = z_grp[generate_groups["data_group"]].shape

    channels = data_shape[generate_groups["source_axes"].index("C")]
    patch_shape = [channels, patch_height, patch_width]
    patch_axes = "CYX"

    patch_shape = [
        patch_shape[patch_axes.index(ax)] if ax in patch_axes else 1
        for ax in generate_groups["source_axes"]
        ]

    expected_shape = tuple(
        patch_shape[generate_groups["source_axes"].index(ax)]
        for ax in generate_groups["axes"]
        )

    for s, sample in enumerate(test_ds):
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


def test_blue_noise_patched_zarrdataset(generate_groups,
                                        generate_input_transform):
    dtype_transform, dtype = generate_input_transform

    patch_height = 128
    patch_width = 128
    patch_sampler = BlueNoisePatchSampler(patch_size=(patch_height, 
                                                      patch_width))

    test_ds = ZarrDataset(
        **generate_groups,
        transform=dtype_transform,
        patch_sampler=patch_sampler,
        return_any_label=True)

    z_grp = zarr.open(generate_groups["filenames"][0], mode="r")
    data_shape = z_grp[generate_groups["data_group"]].shape

    channels = data_shape[generate_groups["source_axes"].index("C")]
    patch_shape = [channels, patch_height, patch_width]
    patch_axes = "CYX"

    patch_shape = [
        patch_shape[patch_axes.index(ax)] if ax in patch_axes else 1
        for ax in generate_groups["source_axes"]
        ]

    expected_shape = tuple(
        patch_shape[generate_groups["source_axes"].index(ax)]
        for ax in generate_groups["axes"]
        )

    for s, sample in enumerate(test_ds):
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
