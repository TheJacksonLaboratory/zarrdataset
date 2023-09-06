import numpy as np
from zarrdataset import *
import pytest

from sample_images_generator import (UNLABELED_DATASET_SPECS,
                                     LABELED_DATASET_SPECS,
                                     base_test_zarrdataset)


@pytest.mark.parametrize("dataset_specs", UNLABELED_DATASET_SPECS)
@pytest.mark.parametrize("random_roi", [True])
@pytest.mark.parametrize("random_axes", [True])
@pytest.mark.parametrize("dataset_class", [ZarrDataset])
@pytest.mark.parametrize("patch_sampler_class", [lambda patch_size: BlueNoisePatchSampler(patch_size)])
@pytest.mark.parametrize("apply_transform", [True])
@pytest.mark.parametrize("input_dtype", [np.uint16])
@pytest.mark.parametrize("target_dtype", [np.float32])
@pytest.mark.parametrize("input_target_dtype", [np.float64])
def test_zarrdataset(dataset_specs, patch_sampler_class,
                     dataset_class,
                     random_roi,
                     random_axes,
                     apply_transform,
                     input_dtype,
                     target_dtype,
                     input_target_dtype):

    base_test_zarrdataset(dataset_specs, patch_sampler_class, dataset_class,
                          random_roi,
                          random_axes,
                          apply_transform,
                          input_dtype,
                          target_dtype,
                          input_target_dtype)



@pytest.mark.parametrize("dataset_specs", LABELED_DATASET_SPECS)
@pytest.mark.parametrize("random_roi", [True])
@pytest.mark.parametrize("random_axes", [True])
@pytest.mark.parametrize("dataset_class", [LabeledZarrDataset])
@pytest.mark.parametrize("patch_sampler_class", [lambda patch_size: None])
@pytest.mark.parametrize("apply_transform", [False])
@pytest.mark.parametrize("input_dtype", [np.uint16])
@pytest.mark.parametrize("target_dtype", [np.float32])
@pytest.mark.parametrize("input_target_dtype", [np.float64])
def test_labeled_zarrdataset(dataset_specs, patch_sampler_class,
                             dataset_class,
                             random_roi,
                             random_axes,
                             apply_transform,
                             input_dtype,
                             target_dtype,
                             input_target_dtype):

    base_test_zarrdataset(dataset_specs, patch_sampler_class, dataset_class,
                          random_roi,
                          random_axes,
                          apply_transform,
                          input_dtype,
                          target_dtype,
                          input_target_dtype)


@pytest.mark.parametrize("dataset_specs", LABELED_DATASET_SPECS)
@pytest.mark.parametrize("random_roi", [True])
@pytest.mark.parametrize("random_axes", [True])
@pytest.mark.parametrize("dataset_class", [MaskedZarrDataset])
@pytest.mark.parametrize("patch_sampler_class", [lambda patch_size: GridPatchSampler(patch_size)])
@pytest.mark.parametrize("apply_transform", [False, True])
@pytest.mark.parametrize("input_dtype", [np.uint16])
@pytest.mark.parametrize("target_dtype", [np.float32])
@pytest.mark.parametrize("input_target_dtype", [np.float64])
def test_masked_zarrdataset(dataset_specs, patch_sampler_class,
                            dataset_class,
                            random_roi,
                            random_axes,
                            apply_transform,
                            input_dtype,
                            target_dtype,
                            input_target_dtype):

    base_test_zarrdataset(dataset_specs, patch_sampler_class, dataset_class,
                          random_roi,
                          random_axes,
                          apply_transform,
                          input_dtype,
                          target_dtype,
                          input_target_dtype)


@pytest.mark.parametrize("dataset_specs", LABELED_DATASET_SPECS)
@pytest.mark.parametrize("random_roi", [True])
@pytest.mark.parametrize("random_axes", [True])
@pytest.mark.parametrize("dataset_class", [MaskedLabeledZarrDataset])
@pytest.mark.parametrize("patch_sampler_class", [lambda patch_size: GridPatchSampler(patch_size)])
@pytest.mark.parametrize("apply_transform", [False, True])
@pytest.mark.parametrize("input_dtype", [np.uint16])
@pytest.mark.parametrize("target_dtype", [np.float32])
@pytest.mark.parametrize("input_target_dtype", [np.float64])
def test_masked_labeled_zarrdataset(dataset_specs, patch_sampler_class,
                                    dataset_class,
                                    random_roi,
                                    random_axes,
                                    apply_transform,
                                    input_dtype,
                                    target_dtype,
                                    input_target_dtype):

    base_test_zarrdataset(dataset_specs, patch_sampler_class, dataset_class,
                          random_roi,
                          random_axes,
                          apply_transform,
                          input_dtype,
                          target_dtype,
                          input_target_dtype)
