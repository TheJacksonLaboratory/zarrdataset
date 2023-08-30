from PIL import Image

from io import BytesIO
import numpy as np
from functools import partial
from zarrdataset import (ZarrDataset,
                         GridPatchSampler,
                         BlueNoisePatchSampler,
                         zarrdataset_worker_init,
                         connect_s3)
import pytest

try:
    import torch
    from torch.utils.data import DataLoader

    def transform_func(x, dtype=torch.float32):
        return x.astype(dtype)


    @pytest.fixture
    def generate_input_transform():
        return partial(transform_func, dtype=np.float32), torch.float32


    @pytest.fixture
    def generate_groups():
        groups = dict(
            filenames=[
                "https://r0k.us/graphics/kodak/kodak/kodim01.png",
                "https://r0k.us/graphics/kodak/kodak/kodim02.png",
                "https://r0k.us/graphics/kodak/kodak/kodim03.png",
                "https://r0k.us/graphics/kodak/kodak/kodim04.png"
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
            return_any_label=True,
            progress_bar=True)

        test_dl = DataLoader(test_ds, num_workers=2, pin_memory=True,
                             worker_init_fn=zarrdataset_worker_init)

        for s, sample in enumerate(test_dl):
            assert isinstance(sample, list),\
                (f"Elements yielded by the ZarrDataset should be "
                 f"lists not {type(sample)}")
            assert isinstance(sample[0], torch.Tensor),\
                (f"Expected sample {s} to be a torch Tensor, not "
                 f"{type(sample[0])}")
            assert sample[0].dtype == dtype,\
                (f"Sample {s} must be of type {dtype}, not {sample[0].dtype}")


    def test_grid_patched_zarrdataset(generate_groups,
                                      generate_input_transform):
        dtype_transform, dtype = generate_input_transform

        patch_height = 512
        patch_width = 512
        patch_sampler = GridPatchSampler(patch_size=(patch_height,
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

        # The first axis from the expected output is the batch size
        expected_shape = tuple(
            [1] +
            [patch_shape[generate_groups["source_axes"].index(ax)]
             for ax in generate_groups["axes"]]
            )

        test_dl = DataLoader(test_ds, num_workers=2, pin_memory=True,
                             worker_init_fn=zarrdataset_worker_init)

        for s, sample in enumerate(test_dl):
            assert isinstance(sample, list),\
                (f"Elements yielded by the ZarrDataset should be "
                 f"lists not {type(sample)}")
            assert isinstance(sample[0], torch.Tensor),\
                (f"Expected sample {s} to be a torch Tensor, not "
                 f"{type(sample[0])}")
            assert sample[0].shape == expected_shape,\
                (f"Generated sample {s} does not have the correct shape, "
                 f"expected it to be {expected_shape}, not {sample[0].shape}")
            assert sample[0].dtype == dtype,\
                (f"Sample {s} must be of type {dtype}, not {sample[0].dtype}")


    def test_blue_noise_patched_zarrdataset(generate_groups,
                                            generate_input_transform):
        dtype_transform, dtype = generate_input_transform

        patch_height = 25
        patch_width = 51
        
        np.random.seed(15795)
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

        # The first axis from the expected output is the batch size
        expected_shape = tuple(
            [1] +
            [patch_shape[generate_groups["source_axes"].index(ax)]
             for ax in generate_groups["axes"]]
            )

        test_dl = DataLoader(test_ds, num_workers=2, pin_memory=True,
                             worker_init_fn=zarrdataset_worker_init)

        for s, sample in enumerate(test_dl):
            assert isinstance(sample, list),\
                (f"Elements yielded by the ZarrDataset should be "
                 f"lists not {type(sample)}")
            assert isinstance(sample[0], torch.Tensor),\
                (f"Expected sample {s} to be a torch Tensor, not "
                 f"{type(sample[0])}")
            assert sample[0].shape == expected_shape,\
                (f"Generated sample {s} does not have the correct shape, "
                 f"expected it to be {expected_shape}, not {sample[0].shape}")
            assert sample[0].dtype == dtype,\
                (f"Sample {s} must be of type {dtype}, not {sample[0].dtype}")

except ModuleNotFoundError:
    # Do not test pytorch DataLoader functionalities if pytorch is not
    # installed.
    pass
