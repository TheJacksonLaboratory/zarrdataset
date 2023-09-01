import numpy as np
from zarrdataset import (ZarrDataset,
                         LabeledZarrDataset,
                         MaskedZarrDataset,
                         MaskedLabeledZarrDataset,
                         GridPatchSampler,
                         BlueNoisePatchSampler)
import pytest

from sample_images_generator import IMAGE_SPECS, generate_sample_dataset


@pytest.mark.parametrize("image_specs", IMAGE_SPECS)


def transform_func(x, dtype=np.float32):
    return x.astype(dtype)


def transform_pair_func(x, t, dtype=np.float32):
    return x.astype(dtype), t.astype(dtype)


@pytest.mark.parametrize(
    "patch_sampler_class",
    [
        lambda patch_size: None,
        lambda patch_size: BlueNoisePatchSampler(patch_size=patch_size),
        lambda patch_size: GridPatchSampler(patch_size=patch_size)
    ]
)


@pytest.mark.parametrize(
    "dataset_class",
    [
        ZarrDataset,
        MaskedZarrDataset,
        LabeledZarrDataset,
        MaskedLabeledZarrDataset,
    ]
)


def test_zarrdataset(image_specs, patch_sampler_class, dataset_class):
    is_labeled = False

    if (len(labels_pars)
      and dataset_class in [LabeledZarrDataset, MaskedLabeledZarrDataset]):
        is_labeled = True
    
    if (not len(mask_pars)
      and dataset_class in [MaskedZarrDataset, MaskedLabeledZarrDataset]):
        # Do not test Masked-based ZarrDatasets if there is no mask
        # specifications
        # TODO: Raise an error in these classes when this happends
        return

    np.random.seed(15795)
    patch_sampler = patch_sampler_class(patch_size)

    collection_pars = image_pars
    collection_pars.update(mask_pars)
    collection_pars.update(labels_pars)

    test_ds = dataset_class(
        **collection_pars,
        patch_sampler=patch_sampler)

    for s, sample in enumerate(test_ds):
        if patch_sampler is None:
            actual_shape = expected_shapes[s]
        else:
            actual_shape = expected_patch_shape

        assert isinstance(sample, tuple),\
            (f"Elements yielded by the ZarrDataset should be "
            f"tuples not {type(sample)}")
        assert isinstance(sample[0], np.ndarray),\
            (f"Expected sample {s} to be a numpy NDArray, not "
            f"{type(sample[0])}")
        assert tuple(sample[0].shape) == actual_shape,\
            (f"Generated sample {s} does not have the correct shape, "
                f"expected it to be {actual_shape}, not "
                f"{sample[0].shape}")
        assert sample[0].dtype == dtype,\
            (f"Sample {s} must be of type {dtype}, not {sample[0].dtype}")

        if is_labeled:
            if patch_sampler is None:
                actual_label_shape = expected_label_shapes[s]
            else:
                actual_label_shape = expected_label_patch_shape

            assert isinstance(sample[1], np.ndarray),\
                (f"Expected label {s} to be a numpy NDArray, not "
                f"{type(sample[1])}")
            assert tuple(sample[1].shape) == actual_label_shape,\
                (f"Generated label {s} does not have the correct shape, "
                f"expected it to be {actual_label_shape}, not "
                f"{sample[1].shape}")
            assert sample[1].dtype == label_dtype,\
                (f"Label {s} must be of type {label_dtype}, not "
                f"{sample[1].dtype}")
