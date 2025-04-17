import pytest
import os
import shutil
from pathlib import Path
from unittest import mock
import zarrdataset as zds
import importlib
import tqdm
import numpy as np

import torch

from tests.utils import IMAGE_SPECS


def input_target_dtype_fn(input, target):
    return input.astype(np.float64), target.astype(np.float64)


@pytest.fixture(scope="function")
def image_dataset_specs(request):
    if not isinstance(request.param, list):
        params = [request.param]
    else:
        params = request.param

    dst_dirs = []

    filenames = []
    labels_filenames = []
    mask_filenames = []
    specs = []

    data_group = None
    source_axes = None
    mask_group = None
    labels_group = None

    for par in params:
        if isinstance(par["source"], str):
            dst_dirs.append(None)

        else:
            dst_dirs.append(par["dst_dir"])

        if dst_dirs[-1] is not None:
            dst_dir = Path(par["dst_dir"])
            dst_dir.mkdir(parents=True, exist_ok=True)

        (image_filename,
         mask_filename,
         labels_filename,
         _) = par["source"](par["dst_dir"], par["specs"])

        data_group = par["specs"]["data_group"]
        source_axes = par["specs"]["source_axes"]
        filenames.append(image_filename)

        if mask_filename is not None:
            mask_filenames.append(mask_filename)
            mask_group = par["specs"]["mask_group"]

        if labels_filename is not None:
            input_label_transform = input_target_dtype_fn
            target_transform = zds.ToDtype(np.int64)
            labels_filenames.append(labels_filename)
            labels_group = par["specs"]["labels_group"]

        specs.append(par["specs"])

    dataset_specs = [
        zds.ImagesDatasetSpecs(
            filenames=filenames,
            data_group=data_group,
            source_axes=source_axes,
        )
    ]

    if mask_filenames:
        dataset_specs.append(
            zds.MasksDatasetSpecs(
                filenames=mask_filenames,
                data_group=mask_group,
                source_axes="YX",
            )
        )

    if labels_filenames:
        dataset_specs[0]["transforms"].append(
            (("images", ), zds.ToDtype(np.float32))
        )

        dataset_specs.append(
            zds.LabelsDatasetSpecs(
                filenames=labels_filenames,
                data_group=labels_group,
                source_axes="YX",
                transform=target_transform,
                input_label_transform=input_label_transform,
            )
        )

    if len(dataset_specs) == 1:
        dataset_specs = dataset_specs[0]

    yield dataset_specs, specs[0]

    for dst_dir in dst_dirs:
        if dst_dir is not None and os.path.isdir(dst_dir):
            shutil.rmtree(dst_dir)


@pytest.fixture(scope="function")
def patch_sampler_specs(request):
    patch_size, allow_incomplete_patches = request.param
    patch_sampler = zds.PatchSampler(
        patch_size=patch_size,
        allow_incomplete_patches=allow_incomplete_patches
    )
    return patch_sampler, patch_size, allow_incomplete_patches


@pytest.mark.parametrize("image_dataset_specs", [
    IMAGE_SPECS[10],
], indirect=["image_dataset_specs"])
def test_compatibility_no_tqdm(image_dataset_specs):
    with mock.patch.dict('sys.modules', {'tqdm': None}):
        importlib.reload(zds._zarrdataset)

        assert isinstance(object, type(zds._zarrdataset.tqdm).__bases__), \
            (f"When `tqdm` is not installed, progress bars should be disabled,"
             f" and a dummy interface derived from object should be used by "
             f"ZarrDataset. Got an interface derived from "
             f"{type(zds._zarrdataset.tqdm).__bases__} instead")

        dataset_specs, _ = image_dataset_specs

        dataset = zds._zarrdataset.ZarrDataset(
            dataset_specs=dataset_specs,
            progress_bar=True
        )

        try:
            next(iter(dataset))

        except Exception as e:
            raise AssertionError(f"No exceptions where expected, got {e} "
                                 f"instead.")

    with mock.patch.dict('sys.modules', {'tqdm': tqdm}):
        importlib.reload(zds._zarrdataset)
        importlib.reload(zds)


def test_compatibility_no_pytroch():
    with mock.patch.dict('sys.modules', {'torch': None}):
        importlib.reload(zds._zarrdataset)

        dataset = zds._zarrdataset.ZarrDataset()

        assert isinstance(object, type(dataset).__bases__), \
            (f"When pytorch is not installed, ZarrDataset should be inherited"
             f" from object, not {type(dataset).__bases__}")

        try:
            zds._zarrdataset.zarrdataset_worker_init_fn(None)

        except Exception as e:
            raise AssertionError(f"No exceptions where expected when using "
                                 f"`zarrdataset_worker_init_fn` without "
                                 f"pytorch installed, got {e} instead.")

        try:
            zds._zarrdataset.chained_zarrdataset_worker_init_fn(None)

        except Exception as e:
            raise AssertionError(f"No exceptions where expected when using "
                                 f"`chained_zarrdataset_worker_init_fn` "
                                 f"without pytorch installed, got {e} "
                                 f"instead.")
    with mock.patch.dict('sys.modules', {'torch': torch}):
        importlib.reload(zds._zarrdataset)
        importlib.reload(zds)
