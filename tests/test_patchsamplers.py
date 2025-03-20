import pytest
import os
import shutil
from pathlib import Path
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
        dataset_specs[0]["transforms"][("images", )] = zds.ToDtype(np.float32)

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


@pytest.mark.parametrize(
    "image_dataset_specs, patch_sampler_specs, shuffle, draw_same_chunk", [
        (IMAGE_SPECS[10], (32, False), True, False),
        (IMAGE_SPECS[10], (32, False), True, True),
        (IMAGE_SPECS[10], (32, False), False, True),
    ],
    indirect=["image_dataset_specs", "patch_sampler_specs"]
)
def test_patched_ZarrDataset(image_dataset_specs, patch_sampler_specs,
                             shuffle,
                             draw_same_chunk):
    dataset_specs, specs = image_dataset_specs
    patch_sampler, patch_size, allow_incomplete_patches = patch_sampler_specs

    ds = zds.ZarrDataset(
        dataset_specs=dataset_specs,
        shuffle=shuffle,
        patch_sampler=patch_sampler,
        draw_same_chunk=draw_same_chunk
    )

    array_idx = 0
    label_idx = 1

    n_samples = 0

    for sample in ds:
        n_samples += 1
        assert isinstance(sample, tuple), \
            (f"When `return_positions`, `return_worker_id` or a labels dataset"
             f" specification is passed to ZarrDataset, retrieved samples "
             f"should be a tuple, got {type(sample)} instead.")

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

        assert tuple(sample_array.shape) == expected_shape, \
            (f"Sample expected to have shape {expected_shape}, got "
             f"{sample_array.shape} instead")

        assert tuple(labels_array.shape) == expected_labels_shape, \
            (f"Labels expected to have shape {expected_labels_shape}, got "
             f"{labels_array.shape} instead")

    assert n_samples > 0, ("Expected more than zero samples extracted from "
                           "this experiment.")

    # Second iteration
    n_samples = 0
    for sample in ds:
        n_samples += 1
        assert isinstance(sample, tuple), \
            (f"When `return_positions`, `return_worker_id` or a labels dataset"
             f" specification is passed to ZarrDataset, retrieved samples "
             f"should be a tuple, got {type(sample)} instead.")

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

        assert tuple(sample_array.shape) == expected_shape, \
            (f"Sample expected to have shape {expected_shape}, got "
             f"{sample_array.shape} instead")

        assert tuple(labels_array.shape) == expected_labels_shape, \
            (f"Labels expected to have shape {expected_labels_shape}, got "
             f"{labels_array.shape} instead")

    assert n_samples > 0, ("Expected more than zero samples extracted from "
                           "this experiment.")


@pytest.mark.parametrize(
    "image_dataset_specs, patch_sampler_specs", [
        (IMAGE_SPECS[10], (1024, True)),
        (IMAGE_SPECS[10], (1024, False)),
    ],
    indirect=["image_dataset_specs", "patch_sampler_specs"]
)
def test_greater_patch_ZarrDataset(image_dataset_specs, patch_sampler_specs):
    dataset_specs, specs = image_dataset_specs
    patch_sampler, patch_size, allow_incomplete_patches = patch_sampler_specs

    ds = zds.ZarrDataset(
        dataset_specs=dataset_specs,
        patch_sampler=patch_sampler
    )

    n_samples = 0
    for _ in ds:
        n_samples += 1

    if allow_incomplete_patches:
        assert n_samples > 0, ("Expected at elast one sample when patch"
                               " size is greater than the image size, and"
                               " `allow_incomplete_patches` is True.")
    else:
        assert n_samples == 0, ("Expected zero samples since requested patch"
                                " size is greater than the image size, and"
                                " `allow_incomplete_patches` is False.")
