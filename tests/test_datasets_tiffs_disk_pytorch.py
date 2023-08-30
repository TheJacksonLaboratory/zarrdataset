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
    import tifffile
    import torch
    from torch.utils.data import DataLoader

    def transform_func(x, dtype=torch.float32):
        return x.astype(dtype)


    @pytest.fixture
    def generate_input_transform():
        return partial(transform_func, dtype=np.float32), torch.float32


    @pytest.fixture
    def generate_groups():
        filenames=[
            "tests/test_tiffs/img_1.ome.tif",
            "tests/test_tiffs/img_2.ome.tif",
            "tests/test_tiffs/img_3.ome.tif",
            ]

        expected_shapes = [tifffile.imread(fn).shape for fn in filenames]

        groups = dict(
            filenames=filenames,
            data_group="",
            source_axes="CYX",
            axes="CYX",
            expected_shapes=expected_shapes,
            patch_height = 49,
            patch_width = 52
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
            assert (tuple(sample[0].shape)
                    == tuple(generate_groups["expected_shapes"][s])),\
                (f"Expected sample {s} to be a torch Tensor, not "
                 f"{type(sample[0])}")
            assert sample[0].dtype == dtype,\
                (f"Sample {s} must be of type {dtype}, not {sample[0].dtype}")


    def test_grid_patched_zarrdataset(generate_groups,
                                      generate_input_transform):
        dtype_transform, dtype = generate_input_transform

        patch_height = generate_groups["patch_height"]
        patch_width = generate_groups["patch_width"]
        patch_sampler = GridPatchSampler(patch_size=(patch_height,
                                                     patch_width))

        test_ds = ZarrDataset(
            **generate_groups,
            transform=dtype_transform,
            patch_sampler=patch_sampler,
            return_any_label=True)

        data_shape = generate_groups["expected_shapes"][0]

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

        patch_height = 15
        patch_width = 19
        
        np.random.seed(15795)
        patch_sampler = BlueNoisePatchSampler(patch_size=(patch_height, 
                                                          patch_width))

        test_ds = ZarrDataset(
            **generate_groups,
            transform=dtype_transform,
            patch_sampler=patch_sampler,
            return_any_label=True)

        data_shape = generate_groups["expected_shapes"][0]

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
