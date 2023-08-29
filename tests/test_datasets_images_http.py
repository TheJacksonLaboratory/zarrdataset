from types import GeneratorType
from PIL import Image

from io import BytesIO
import numpy as np
from functools import partial
from zarrdataset import (ZarrDataset,
                         GridPatchSampler,
                         BlueNoisePatchSampler,
                         connect_s3)
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
            "https://r0k.us/graphics/kodak/kodak/kodim01.png",
            "https://r0k.us/graphics/kodak/kodak/kodim02.png",
            "https://r0k.us/graphics/kodak/kodak/kodim03.png"
            ],
        data_group="",
        source_axes="YXC",
        axes="CYX"
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
        s3_obj = connect_s3(generate_groups["filenames"][s])
        assert s3_obj is not None,\
             (f"Could not connect to remote object at "
              f"{generate_groups['filenames'][s]}")

        filename = generate_groups["filenames"][s].split(
            s3_obj["endpoint_url"] + "/" + s3_obj["bucket_name"])[1][1:]
        im_bytes = s3_obj["s3"].get_object(Bucket=s3_obj["bucket_name"],
                                            Key=filename)["Body"].read()
        with Image.open(BytesIO(im_bytes)) as im:
            data_shape = (im.height, im.width, len(im.getbands()))

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
    patch_width = 512
    patch_sampler = GridPatchSampler(patch_size=(patch_height, patch_width))

    test_ds = ZarrDataset(
        **generate_groups,
        transform=dtype_transform,
        patch_sampler=patch_sampler,
        return_any_label=True)
    
    s3_obj = connect_s3(generate_groups["filenames"][0])
    assert s3_obj is not None,\
            (f"Could not connect to remote object at "
            f"{generate_groups['filenames'][0]}")

    filename = generate_groups["filenames"][0].split(
        s3_obj["endpoint_url"] + "/" + s3_obj["bucket_name"])[1][1:]
    im_bytes = s3_obj["s3"].get_object(Bucket=s3_obj["bucket_name"],
                                        Key=filename)["Body"].read()
    with Image.open(BytesIO(im_bytes)) as im:
        data_shape = (im.height, im.width, len(im.getbands()))

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

    patch_height = 256
    patch_width = 512
    patch_sampler = BlueNoisePatchSampler(patch_size=(patch_height, 
                                                      patch_width))

    test_ds = ZarrDataset(
        **generate_groups,
        transform=dtype_transform,
        patch_sampler=patch_sampler,
        return_any_label=True)

    s3_obj = connect_s3(generate_groups["filenames"][0])
    assert s3_obj is not None,\
            (f"Could not connect to remote object at "
            f"{generate_groups['filenames'][0]}")

    filename = generate_groups["filenames"][0].split(
        s3_obj["endpoint_url"] + "/" + s3_obj["bucket_name"])[1][1:]
    im_bytes = s3_obj["s3"].get_object(Bucket=s3_obj["bucket_name"],
                                        Key=filename)["Body"].read()
    with Image.open(BytesIO(im_bytes)) as im:
        data_shape = (im.height, im.width, len(im.getbands()))

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
