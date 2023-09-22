import pytest
import os
import shutil
from pathlib import Path
from unittest import mock
from collections import OrderedDict
from _collections_abc import dict_items

import operator

import zarrdataset as zds
import importlib
import tqdm
import numpy as np

from sample_images_generator import IMAGE_SPECS

import torch
from torch.utils.data import DataLoader, ChainDataset


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
            input_target_transform = input_target_dtype_fn
            target_transform = zds.ToDtype(np.int64)
            labels_filenames.append(labels_filename)
            labels_group = par["specs"]["labels_group"]

        specs.append(par["specs"])

    dataset_specs=OrderedDict(
        images=zds.ImagesDatasetSpecs(
            filenames=filenames,
            data_group=data_group,
            source_axes=source_axes,
        )
    )

    if mask_filenames:
        dataset_specs["masks"] = zds.MasksDatasetSpecs(
            filenames=mask_filenames,
            data_group=mask_group,
            source_axes="YX",
        )

    if labels_filenames:
        dataset_specs["images"]["transforms"][("images", )] =\
                zds.ToDtype(np.float32)

        dataset_specs["labels"] = zds.LabelsDatasetSpecs(
            filenames=labels_filenames,
            data_group=labels_group,
            source_axes="YX",
            transform=target_transform,
            input_target_transform=input_target_transform,
        )

    yield dataset_specs, specs[0]

    for dst_dir in dst_dirs:
        if dst_dir is not None and os.path.isdir(dst_dir):
            shutil.rmtree(dst_dir)


@pytest.fixture(scope="function")
def patch_sampler_specs(request):
    patch_sampler = zds.PatchSampler(patch_size=request.param)
    return patch_sampler, request.param


def test_compatibility_no_pytroch():
    with mock.patch.dict('sys.modules', {'torch': None}):
        importlib.reload(zds._zarrdataset)

        dataset = zds._zarrdataset.BaseZarrDataset()

        assert isinstance(object, type(dataset).__bases__), \
            (f"When pytorch is not installed, ZarrDataset should be inherited"
             f" from object, not {type(dataset).__bases__}")

        try:
            zds._zarrdataset.zarrdataset_worker_init_fn()
            
        except Exception as e:
            raise AssertionError(f"No exceptions where expected when using "
                                 f"`zarrdataset_worker_init_fn` without pytorch "
                                 f"installed, got {e} instead.")

        try:
            zds._zarrdataset.chained_zarrdataset_worker_init_fn()
            
        except Exception as e:
            raise AssertionError(f"No exceptions where expected when using "
                                 f"`chained_zarrdataset_worker_init_fn` without "
                                 f"pytorch installed, got {e} instead.")

    with mock.patch.dict('sys.modules', {'torch': torch}):
        importlib.reload(zds._zarrdataset)
        importlib.reload(zds)


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
                                 "instead.")

    with mock.patch.dict('sys.modules', {'tqdm': tqdm}):
        importlib.reload(zds._zarrdataset)
        importlib.reload(zds)


def test_DataSpecs():
    generic_specs = zds.DatasetSpecs(
        mode="generic",
        filenames=None,
        source_axes=""
    )

    assert generic_specs["mode"] == "generic",\
        (f"Expected mode to be `generic`, as assigned during initialization of"
         f" the dataset specification, got {generic_specs['mode']} "
         f"instead.")

    assert generic_specs["filenames"] is None,\
        (f"Expected filenames to be None, as assigned during initialization of"
         f" the dataset specification, got {generic_specs['filenames']} "
         f"instead.")

    assert isinstance(generic_specs["transforms"], OrderedDict),\
        (f"Expected transforms to be an ordered dicionary, got "
         f"{type(generic_specs['transforms'])} instead.")
    
    assert isinstance(generic_specs.items(), dict_items),\
        (f"Expected DatasetSpecs to return `dict_items` when called `.items()`"
         f", got {type(generic_specs.items())} instead.")


@pytest.mark.parametrize("image_dataset_specs", [
    IMAGE_SPECS[10],
], indirect=["image_dataset_specs"])
def test_no_images_ZarrDataset(image_dataset_specs):
    dataset_specs, _ = image_dataset_specs

    dataset_specs.pop("images")

    with pytest.raises(ValueError):
        ds = zds.ZarrDataset(dataset_specs=dataset_specs)


@pytest.mark.parametrize(
    "image_dataset_specs, shuffle, return_positions, return_any_label,"
    "return_worker_id", [
        (IMAGE_SPECS[10], False, False, False, False),
        (IMAGE_SPECS[10], True, True, True, True),
        (IMAGE_SPECS[11], False, False, False, False),
        (IMAGE_SPECS[11], False, False, True, False),
    ],
    indirect=["image_dataset_specs"]
)
def test_ZarrDataset(image_dataset_specs, shuffle, return_positions,
                     return_any_label,
                     return_worker_id):
    dataset_specs, specs = image_dataset_specs

    ds = zds.ZarrDataset(
        dataset_specs=dataset_specs,
        return_any_label=return_any_label,
        return_positions=return_positions,
        return_worker_id=return_worker_id,
        shuffle=shuffle,
    )

    array_idx = 0
    label_idx = 1

    if return_positions:
        array_idx += 1
        label_idx += 1

    if return_worker_id:
        array_idx += 1
        label_idx += 1

    for sample in ds:
        if ("labels" in dataset_specs or return_any_label or return_positions
          or return_worker_id):
            assert isinstance(sample, tuple), \
                (f"When `return_any_label`, `return_positions`, "
                 f"`return_worker_id` or a labels dataset specification is "
                 f"passed to ZarrDataset, samples should be a tuple, got "
                 f"{type(sample)} instead.")

            sample_array = sample[array_idx]

        else:
            sample_array = sample

        assert isinstance(sample_array, np.ndarray), \
            (f"Sample should be a Numpy NDArray, got {type(sample[array_idx])}"
             f" instead.")

        assert sample_array.shape == tuple(specs["shape"]), \
            (f"Sample expected to have shape {tuple(specs['shape'])}, got "
            f"{sample_array.shape} instead")

        if "labels" in dataset_specs:
            expected_labels_shape = tuple(
                specs["shape"][specs["source_axes"].index(ax)]
                for ax in "YX"
            )

            labels_array = sample[label_idx]

            assert labels_array.shape == expected_labels_shape, \
                (f"Labels expected to have shape {expected_labels_shape}, got "
                f"{labels_array.shape} instead")

        elif return_any_label:
            assert isinstance(sample[label_idx], int), \
                (f"Expected label returned when `return_any_labels` and no "
                 f"actual labels dataset specifications were passed to "
                 f"ZarrDataset to be {int}, got {type(sample[label_idx])}.")


@pytest.mark.parametrize(
    "image_dataset_specs, patch_sampler_specs, shuffle, draw_same_chunk", [
        (IMAGE_SPECS[10], 32, True, False),
        (IMAGE_SPECS[10], 32, True, True),
        (IMAGE_SPECS[10], 32, False, True),
        (IMAGE_SPECS[10], 1024, True, True),
    ],
    indirect=["image_dataset_specs", "patch_sampler_specs"]
)
def test_patched_ZarrDataset(image_dataset_specs, patch_sampler_specs,
                             shuffle,
                             draw_same_chunk):
    dataset_specs, specs = image_dataset_specs
    patch_sampler, patch_size = patch_sampler_specs

    ds = zds.ZarrDataset(
        dataset_specs=dataset_specs,
        shuffle=shuffle,
        patch_sampler=patch_sampler,
        draw_same_chunk=draw_same_chunk
    )

    array_idx = 0
    label_idx = 1

    for sample in ds:
        assert isinstance(sample, tuple), \
            (f"When `return_any_label`, `return_positions`, `return_worker_id`"
             f", or a labels dataset specification is passed to ZarrDataset, "
             f"samples should be a tuple, got {type(sample)} instead.")

        assert isinstance(sample[0], np.ndarray), \
            (f"Sample should be a Numpy NDArray, got {type(sample[0])} "
             f"instead.")

        sample_array = sample[array_idx]
        labels_array = sample[label_idx]

        expected_shape = tuple(
            min(patch_size, specs["shape"][specs["source_axes"].index(ax)])
            if ax in patch_sampler.spatial_axes else
            specs["shape"][specs["source_axes"].index(ax)]
            for ax in specs["source_axes"]
        )

        expected_labels_shape = tuple(
            patch_size if ax in patch_sampler.spatial_axes else
            specs["shape"]["YX".index(ax)]
            for ax in "YX"
        )

        assert sample_array.shape == expected_shape, \
            (f"Sample expected to have shape {expected_shape}, got "
             f"{sample_array.shape} instead")

        assert labels_array.shape == expected_labels_shape, \
            (f"Labels expected to have shape {expected_labels_shape}, got "
             f"{labels_array.shape} instead")

    # Second iteration
    for sample in ds:
        assert isinstance(sample, tuple), \
            (f"When `return_any_label`, `return_positions`, `return_worker_id`"
             f", or a labels dataset specification is passed to ZarrDataset, "
             f"samples should be a tuple, got {type(sample)} instead.")

        assert isinstance(sample[0], np.ndarray), \
            (f"Sample should be a Numpy NDArray, got {type(sample[0])} "
             f"instead.")

        sample_array = sample[array_idx]
        labels_array = sample[label_idx]

        expected_shape = tuple(
            min(patch_size, specs["shape"][specs["source_axes"].index(ax)])
            if ax in patch_sampler.spatial_axes else
            specs["shape"][specs["source_axes"].index(ax)]
            for ax in specs["source_axes"]
        )

        expected_labels_shape = tuple(
            patch_size if ax in patch_sampler.spatial_axes else
            specs["shape"]["YX".index(ax)]
            for ax in "YX"
        )

        assert sample_array.shape == expected_shape, \
            (f"Sample expected to have shape {expected_shape}, got "
             f"{sample_array.shape} instead")

        assert labels_array.shape == expected_labels_shape, \
            (f"Labels expected to have shape {expected_labels_shape}, got "
             f"{labels_array.shape} instead")


@pytest.mark.parametrize(
    "image_dataset_specs, patch_sampler_specs, shuffle, draw_same_chunk,"
    "batch_size, num_workers", [
        (IMAGE_SPECS[10], 32, True, False, 2, 2),
        ([IMAGE_SPECS[10]] * 4, 32, True, True, 2, 3),
        ([IMAGE_SPECS[10]] * 2, 32, True, True, 2, 3),
    ],
    indirect=["image_dataset_specs", "patch_sampler_specs"]
)
def test_multithread_ZarrDataset(image_dataset_specs, patch_sampler_specs,
                                 shuffle,
                                 draw_same_chunk,
                                 batch_size,
                                 num_workers):
    dataset_specs, specs = image_dataset_specs

    patch_sampler, patch_size = patch_sampler_specs

    ds = zds.ZarrDataset(
        dataset_specs=dataset_specs,
        shuffle=shuffle,
        patch_sampler=patch_sampler,
        draw_same_chunk=draw_same_chunk
    )

    dl = DataLoader(
        ds,
        batch_size=batch_size,
        num_workers=num_workers,
        worker_init_fn=zds.zarrdataset_worker_init_fn,
        drop_last=True
    )

    array_idx = 0
    label_idx = 1

    for sample in dl:
        assert isinstance(sample, list), \
            (f"When a labels dataset specification is passed to ZarrDataset "
             f"and used with PyTorch DataLoader, samples should be a list, got"
             f" {type(sample)} instead.")

        assert isinstance(sample[0], torch.Tensor), \
            (f"Sample should be a PyTorch Tensor, got {type(sample[0])} "
             f"instead.")

        sample_array = sample[array_idx]
        labels_array = sample[label_idx]

        expected_shape = [batch_size] + [
            min(patch_size, specs["shape"][specs["source_axes"].index(ax)])
            if ax in patch_sampler.spatial_axes else
            specs["shape"][specs["source_axes"].index(ax)]
            for ax in specs["source_axes"]
        ]

        expected_labels_shape = [batch_size] + [
            patch_size if ax in patch_sampler.spatial_axes else
            specs["shape"]["YX".index(ax)]
            for ax in "YX"
        ]

        assert all(map(operator.eq,
                       sample_array.shape,
                       expected_shape)), \
            (f"Sample expected to have shape {expected_shape}, got "
             f"{sample_array.shape} instead")

        assert all(map(operator.eq,
                       labels_array.shape,
                       expected_labels_shape)), \
            (f"Labels expected to have shape {expected_labels_shape}, got "
             f"{labels_array.shape} instead")


@pytest.mark.parametrize(
    "image_dataset_specs, patch_sampler_specs, shuffle, draw_same_chunk,"
    "batch_size, num_workers, repeat_dataset", [
        (IMAGE_SPECS[10:12], 32, True, False, 2, 2, 1),
        (IMAGE_SPECS[10:12], 32, True, False, 2, 2, 2),
        (IMAGE_SPECS[10:12], 32, True, False, 2, 2, 3),
    ],
    indirect=["image_dataset_specs", "patch_sampler_specs"]
)
def test_multithread_chained_ZarrDataset(image_dataset_specs,
                                          patch_sampler_specs,
                                          shuffle,
                                          draw_same_chunk,
                                          batch_size,
                                          num_workers,
                                          repeat_dataset):
    dataset_specs, specs = image_dataset_specs
    patch_sampler, patch_size = patch_sampler_specs

    ds = [zds.ZarrDataset(dataset_specs=dataset_specs,
                          shuffle=shuffle,
                          patch_sampler=patch_sampler,
                          draw_same_chunk=draw_same_chunk
        )] * repeat_dataset

    chained_ds = ChainDataset(ds)

    dl = DataLoader(
        chained_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        worker_init_fn=zds.chained_zarrdataset_worker_init_fn,
        drop_last=True
    )

    array_idx = 0
    label_idx = 1

    for sample in dl:
        assert isinstance(sample, list), \
            (f"When a labels dataset specification is passed to ZarrDataset "
             f"and used with PyTorch DataLoader, samples should be a list, got"
             f" {type(sample)} instead.")

        assert isinstance(sample[0], torch.Tensor), \
            (f"Sample should be a PyTorch Tensor, got {type(sample[0])} "
             f"instead.")

        sample_array = sample[array_idx]
        labels_array = sample[label_idx]

        expected_shape = [batch_size] + [
            min(patch_size, specs["shape"][specs["source_axes"].index(ax)])
            if ax in patch_sampler.spatial_axes else
            specs["shape"][specs["source_axes"].index(ax)]
            for ax in specs["source_axes"]
        ]

        expected_labels_shape = [batch_size] + [
            patch_size if ax in patch_sampler.spatial_axes else
            specs["shape"]["YX".index(ax)]
            for ax in "YX"
        ]

        assert all(map(operator.eq,
                       sample_array.shape,
                       expected_shape)), \
            (f"Sample expected to have shape {expected_shape}, got "
             f"{sample_array.shape} instead")

        assert all(map(operator.eq,
                       labels_array.shape,
                       expected_labels_shape)), \
            (f"Labels expected to have shape {expected_labels_shape}, got "
             f"{labels_array.shape} instead")
