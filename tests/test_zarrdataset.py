import pytest
import os
import shutil
from pathlib import Path
from _collections_abc import dict_items

import zarrdataset as zds
import numpy as np

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


@pytest.mark.parametrize("dataset_spec_class", [
    zds.DatasetSpecs,
    zds.ImagesDatasetSpecs,
    zds.MasksDatasetSpecs,
    zds.LabelsDatasetSpecs
])
def test_DataSpecs(dataset_spec_class):
    generic_specs = dataset_spec_class(
        modality="generic",
        filenames="",
        source_axes=""
    )

    assert generic_specs["modality"] == "generic", \
        (f"Expected mode to be `generic`, as assigned during initialization of"
         f" the dataset specification, got {generic_specs['modality']} "
         f"instead.")

    assert generic_specs["filenames"] == "", \
        (f"Expected filenames to be ````, as assigned during initialization of"
         f" the dataset specification, got {generic_specs['filenames']} "
         f"instead.")

    assert isinstance(generic_specs["transforms"], list), \
        (f"Expected transforms to be an ordered dicionary, got "
         f"{type(generic_specs['transforms'])} instead.")

    assert isinstance(generic_specs.items(), dict_items), \
        (f"Expected DatasetSpecs to return `dict_items` when called `.items()`"
         f", got {type(generic_specs.items())} instead.")

    ds = zds.ZarrDataset()
    try:
        ds.add_modality(**generic_specs)

    except Exception as e:
        raise AssertionError(f"No exceptions where expected, got {e} "
                             f"instead.")


def test_no_images_ZarrDataset():
    ds = zds.ZarrDataset()

    with pytest.raises(ValueError):
        ds_iter = iter(ds)
        sample = next(ds_iter)


@pytest.mark.parametrize("image_dataset_specs", [
    IMAGE_SPECS[10],
], indirect=["image_dataset_specs"])
def test_string_ZarrDataset(image_dataset_specs):
    dataset_specs, specs = image_dataset_specs

    patch_sampler = zds.PatchSampler(0)
    ds = zds.ZarrDataset(dataset_specs,
                         patch_sampler=patch_sampler)

    ds_str_repr = str(ds)
    assert (ds_str_repr.startswith("ZarrDataset")
            and all([mode['modality'] in ds_str_repr
                     for mode in dataset_specs])
            and f"Using {dataset_specs[0]['modality']}" in ds_str_repr
            and f"{patch_sampler}" in ds_str_repr), \
        (f"Spected string representation start with `ZarrDataset`, list all "
         f"modalities, and state what modality is being used as reference. "
         f"Got `{ds_str_repr}` instead.")


@pytest.mark.parametrize(
    "image_dataset_specs, transforms", [
        (IMAGE_SPECS[6], (("images", ), zds.ToDtype(np.float64))),
        (IMAGE_SPECS[6], [(("images", ), zds.ToDtype(np.float32)),
                          (("images", ), zds.ToDtype(np.float64))]),
        (IMAGE_SPECS[6], [(("images", ), [zds.ToDtype(np.float32),
                                          zds.ToDtype(np.float64)])]),
    ],
    indirect=["image_dataset_specs"]
)
def test_ZarrDataset_transforms(image_dataset_specs, transforms):
    dataset_specs, _ = image_dataset_specs

    ds = zds.ZarrDataset(dataset_specs=dataset_specs)

    if not isinstance(dataset_specs, list):
        dataset_specs = [dataset_specs]

    if not isinstance(transforms, list):
        transforms = [transforms]

    expected_transforms_per_mode = {}
    for mode, mode_transform in transforms:
        expected_transforms_per_mode[mode] =\
            expected_transforms_per_mode.get(mode, 0) + 1

        ds.add_transform(mode, mode_transform)

    transforms_per_mode = {}
    for mode, mode_transform in ds._transforms:
        transforms_per_mode[mode] = transforms_per_mode.get(mode, 0) + 1

    assert all(map(lambda mode_n_transforms:
                   transforms_per_mode.get(mode_n_transforms[0], 0)
                   == mode_n_transforms[1],
                   expected_transforms_per_mode.items())), \
           (f"Expected number of transforms to be "
            f"{expected_transforms_per_mode}, got "
            f"{transforms_per_mode} instead.")

    for sample in ds:
        if any(["labels" in mode["modality"] for mode in dataset_specs]):
            sample_array = sample[0]

        else:
            sample_array = sample

        assert sample_array.dtype == np.float64, \
            (f"Sample data type should be numpy.float64, got "
             f"{sample_array.dtype} instead.")


@pytest.mark.parametrize(
    "image_dataset_specs, shuffle, return_positions, return_worker_id", [
        (IMAGE_SPECS[10], False, False, False),
        (IMAGE_SPECS[10], True, True, True),
        (IMAGE_SPECS[11], False, False, False),
        (IMAGE_SPECS[11], False, False, False),
    ],
    indirect=["image_dataset_specs"]
)
def test_ZarrDataset(image_dataset_specs, shuffle, return_positions,
                     return_worker_id):
    dataset_specs, specs = image_dataset_specs

    ds = zds.ZarrDataset(
        dataset_specs=dataset_specs,
        return_positions=return_positions,
        return_worker_id=return_worker_id,
        shuffle=shuffle,
    )

    if not isinstance(dataset_specs, list):
        dataset_specs = [dataset_specs]

    ds.add_transform(dataset_specs[0]["modality"], zds.ToDtype(np.float64))

    array_idx = 0
    label_idx = 1

    if return_positions:
        array_idx += 1
        label_idx += 1

    if return_worker_id:
        array_idx += 1
        label_idx += 1

    n_samples = 0

    for sample in ds:
        n_samples += 1
        if (any(["labels" in mode["modality"] for mode in dataset_specs])
           or return_positions
           or return_worker_id):
            assert isinstance(sample, tuple), \
                (f"When `return_positions`, `return_worker_id` or a labels "
                 f"dataset specification is passed to ZarrDataset, retrieved "
                 f"samples should be a tuple, got {type(sample)} instead.")

            sample_array = sample[array_idx]

        else:
            sample_array = sample

        assert isinstance(sample_array, np.ndarray), \
            (f"Sample should be a Numpy NDArray, got {type(sample_array)}"
             f" instead.")

        assert sample_array.dtype == np.float64, \
            (f"Sample data type should be numpy.float64, got "
             f"{sample_array.dtype} instead.")

        assert tuple(sample_array.shape) == tuple(specs["shape"]), \
            (f"Sample expected to have shape {tuple(specs['shape'])}, got "
             f"{sample_array.shape} ({sample_array.dtype}) instead")

        if "labels" in dataset_specs:
            expected_labels_shape = tuple(
                specs["shape"][specs["source_axes"].index(ax)]
                for ax in "YX"
            )

            labels_array = sample[label_idx]

            assert tuple(labels_array.shape) == expected_labels_shape, \
                (f"Labels expected to have shape {expected_labels_shape}, got "
                 f"{labels_array.shape} ({labels_array.dtype}) instead")

    assert n_samples > 0, ("Expected more than zero samples extracted from "
                           "this experiment.")
