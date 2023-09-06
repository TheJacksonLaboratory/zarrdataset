import torch
import numpy as np
from zarrdataset import *
import pytest

from sample_images_generator import (LABELED_DATASET_SPECS,
                                     generate_sample_dataset)

@pytest.mark.parametrize("labeled_dataset_specs", LABELED_DATASET_SPECS)

@pytest.mark.parametrize("random_roi", [True])

@pytest.mark.parametrize("random_axes", [True])

@pytest.mark.parametrize(
    "labeled_dataset_class",
    [
        MaskedZarrDataset,
        LabeledZarrDataset,
        MaskedLabeledZarrDataset,
    ]
)

@pytest.mark.parametrize(
    "patch_sampler_class",
    [
        lambda patch_size: None,
        lambda patch_size: GridPatchSampler(patch_size),
    ]
)

@pytest.mark.parametrize(
    "apply_transform",
    [True]
)

@pytest.mark.parametrize(
    "input_dtype",
    [np.uint16]
)

@pytest.mark.parametrize(
    "target_dtype",
    [np.float32]
)

@pytest.mark.parametrize(
    "input_target_dtype",
    [np.float64]
)

def test_pytorch_zarrdataset(labeled_dataset_specs, patch_sampler_class,
                             labeled_dataset_class,
                             random_roi,
                             random_axes,
                             apply_transform,
                             input_dtype,
                             target_dtype,
                             input_target_dtype):
    (dataset_args,
        expected_shapes,
        expected_labels_shapes,
        destroy_funcs) = generate_sample_dataset(labeled_dataset_specs,
                                                random_axes=random_axes,
                                                random_roi=random_roi,
                                                apply_transform=apply_transform,
                                                input_dtype=input_dtype,
                                                target_dtype=target_dtype,
                                                input_target_dtype=input_target_dtype)

    min_img_shape = np.array(expected_shapes).min(axis=0)

    patch_size = dict(
        (ax, np.random.randint(s // 8, s // 4) if s > 8 else 1)
        for ax, s in zip(dataset_args["axes"], min_img_shape)
        if ax in "ZYX"
    )

    patch_sampler = patch_sampler_class(patch_size)

    ds = labeled_dataset_class(**dataset_args,
                            patch_sampler=patch_sampler,
                            return_any_label=True)

    dl = torch.utils.data.DataLoader(ds,
                                     batch_size=4,
                                     worker_init_fn=zarrdataset_worker_init,
                                     num_workers=2)

    if not apply_transform:
        input_dtype = labeled_dataset_specs[0]["dtype"]
        label_dtype = labeled_dataset_specs[0]["dtype"]
    else:
        input_dtype = input_target_dtype
        label_dtype = target_dtype

    for i, (img_pair, expected_shape, expected_labels_shape) in enumerate(
        zip(dl, expected_shapes, expected_labels_shapes)):
        img, label = img_pair

        if patch_sampler is not None:
            expected_shape = dict(
                (ax, patch_size[ax] if ax in patch_size else s)
                for ax, s in zip(dataset_args["axes"], expected_shape)
            )

            expected_labels_shape = dict(
                (ax, patch_size[ax] if ax in patch_size else s)
                for ax, s in zip(dataset_args["labels_axes"],
                                expected_labels_shape)
            )

        else:
            expected_shape = dict(
                (ax, s)
                for ax, s in zip(dataset_args["axes"], expected_shape)
            )

            expected_labels_shape = dict(
                (ax, s)
                for ax, s in zip(dataset_args["labels_axes"],
                                expected_labels_shape)
            )

        assert isinstance(img, torch.Tensor),\
            (f"Sample {i}, expected to be a pytorch Tensor, got {type(img)}")

        assert img.dtype == input_dtype,\
            (f"Expected sample {i} be of type {input_dtype}, got {img.dtype}")

        assert all(map(lambda s, ax:
                    s == expected_shape[ax],
                    img.shape,
                    dataset_args["axes"])),\
            (f"Expected sample {i} of shape {expected_shape}, got {img.shape}"
            f" [{dataset_args['axes']}]")

        assert all(map(lambda s, ax:
                        s == expected_labels_shape[ax],
                        label.shape,
                        dataset_args["labels_axes"])),\
            (f"Expected label {i} of shape {expected_labels_shape}, "
            f"got {label.shape}")

        assert label.dtype == label_dtype,\
            (f"Expected sample {i} be of type {label_dtype}, got {label.dtype}")

    for destroy_func in destroy_funcs:
        destroy_func()
