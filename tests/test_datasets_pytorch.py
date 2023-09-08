from zarrdataset import *
import numpy as np
import pytest

try:
    import torch
    from sample_images_generator import (LABELED_DATASET_SPECS,
                                         UNLABELED_3D_DATASET_SPECS,
                                         base_test_zarrdataset_pytorch,
                                         base_test_zarrdataset_chain_pytorch)

    @pytest.mark.parametrize("dataset_specs", LABELED_DATASET_SPECS)
    @pytest.mark.parametrize("random_roi", [True])
    @pytest.mark.parametrize("random_axes", [True])
    @pytest.mark.parametrize("dataset_class", [MaskedZarrDataset])
    @pytest.mark.parametrize("patch_sampler_class", [lambda patch_size: None])
    @pytest.mark.parametrize("apply_transform", [True])
    @pytest.mark.parametrize("input_dtype", [np.float32])
    @pytest.mark.parametrize("target_dtype", [np.int64])
    @pytest.mark.parametrize("input_target_dtype", [np.float64])
    @pytest.mark.parametrize("num_workers", [0])
    def test_pytorch_zarrdataset(dataset_specs, patch_sampler_class,
                                dataset_class,
                                random_roi,
                                random_axes,
                                apply_transform,
                                input_dtype,
                                target_dtype,
                                input_target_dtype,
                                num_workers):
        base_test_zarrdataset_pytorch(dataset_specs, patch_sampler_class,
                                    dataset_class,
                                    random_roi,
                                    random_axes,
                                    apply_transform,
                                    input_dtype,
                                    target_dtype,
                                    input_target_dtype,
                                    num_workers)


    @pytest.mark.parametrize("dataset_specs", LABELED_DATASET_SPECS)
    @pytest.mark.parametrize("random_roi", [True])
    @pytest.mark.parametrize("random_axes", [True])
    @pytest.mark.parametrize("dataset_class", [MaskedZarrDataset])
    @pytest.mark.parametrize("patch_sampler_class", [
        lambda patch_size: GridPatchSampler(patch_size=patch_size)
    ])
    @pytest.mark.parametrize("apply_transform", [True])
    @pytest.mark.parametrize("input_dtype", [np.float32])
    @pytest.mark.parametrize("target_dtype", [np.int64])
    @pytest.mark.parametrize("input_target_dtype", [np.float64])
    @pytest.mark.parametrize("num_workers", [0, 1, 2, 3])
    def test_pytorch_multithread_zarrdataset(dataset_specs, patch_sampler_class,
                                            dataset_class,
                                            random_roi,
                                            random_axes,
                                            apply_transform,
                                            input_dtype,
                                            target_dtype,
                                            input_target_dtype,
                                            num_workers):
        base_test_zarrdataset_pytorch(dataset_specs, patch_sampler_class,
                                    dataset_class,
                                    random_roi,
                                    random_axes,
                                    apply_transform,
                                    input_dtype,
                                    target_dtype,
                                    input_target_dtype,
                                    num_workers)


    @pytest.mark.parametrize("dataset_specs", UNLABELED_3D_DATASET_SPECS)
    @pytest.mark.parametrize("random_roi", [True])
    @pytest.mark.parametrize("random_axes", [True])
    @pytest.mark.parametrize("dataset_class", [ZarrDataset])
    @pytest.mark.parametrize("patch_sampler_class", [
        lambda patch_size: GridPatchSampler(patch_size=patch_size)
    ])
    @pytest.mark.parametrize("apply_transform", [True])
    @pytest.mark.parametrize("input_dtype", [np.float32])
    @pytest.mark.parametrize("target_dtype", [np.int64])
    @pytest.mark.parametrize("input_target_dtype", [np.float64])
    @pytest.mark.parametrize("num_workers", [0, 2])
    def test_pytorch_multithread_chain_zarrdataset(dataset_specs,
                                                   patch_sampler_class,
                                                   dataset_class,
                                                   random_roi,
                                                   random_axes,
                                                   apply_transform,
                                                   input_dtype,
                                                   target_dtype,
                                                   input_target_dtype,
                                                   num_workers):
        base_test_zarrdataset_chain_pytorch(dataset_specs, patch_sampler_class,
                                            dataset_class,
                                            random_roi,
                                            random_axes,
                                            apply_transform,
                                            input_dtype,
                                            target_dtype,
                                            input_target_dtype,
                                            num_workers)

except ImportError:
    pass
