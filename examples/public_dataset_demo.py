import numpy as np
import zarrdataset as zds


try:
    import torch
    import torchvision
    from torch.utils.data import DataLoader

    if __name__ == "__main__":
        print("Zarr-based data loader demo")
        # These are images from the Image Data Resource (IDR) 
        # https://idr.openmicroscopy.org/ that are publicly available and were 
        # converted to the OME-NGFF (Zarr) format by the OME group. More
        # examples can be found at Public OME-Zarr data (Nov. 2020)
        # https://www.openmicroscopy.org/2020/11/04/zarr-data.html
        # filenames = [
        #     "https://uk1s3.embassy.ebi.ac.uk/idr/zarr/v0.1/9836839.zarr",
        #     "https://uk1s3.embassy.ebi.ac.uk/idr/zarr/v0.1/9836840.zarr",
        #     "https://uk1s3.embassy.ebi.ac.uk/idr/zarr/v0.1/9836841.zarr",
        #     "https://uk1s3.embassy.ebi.ac.uk/idr/zarr/v0.1/9836842.zarr"]
        # filenames = ["https://uk1s3.embassy.ebi.ac.uk/idr/zarr/v0.4/idr0073A/9798462.zarr",
        #              "https://uk1s3.embassy.ebi.ac.uk/idr/zarr/v0.4/idr0073A/9798462.zarr"]
        # filenames = ["https://uk1s3.embassy.ebi.ac.uk/idr/zarr/v0.4/idr0073A/9798462.zarr"]
        filenames = ["http://s3-far.jax.org/cptacdata/cptac_ccrcc/C3L-01963/e82993ad-e42b-48f7-86f7-89d12cbd95c2.zarr"]
        source_axes = "TCZYX"
        patch_size = 1024
        batch_size = 4
        num_workers = 0

        torch.manual_seed(777)
        patch_sampler = zds.GridPatchSampler(patch_size)
        # patch_sampler = zds.BlueNoisePatchSampler(patch_size)
        # patch_sampler = None

        transform_fn = torchvision.transforms.Compose([
            zds.ZarrToArray(np.float64)
        ])

        my_dataset = zds.MaskedZarrDataset(filenames,
                                            transform=transform_fn,
                                     data_group="0/1",
                                     source_axes=source_axes,
                                     axes="YXC",
                                     patch_sampler=patch_sampler,
                                     return_positions=True,
                                     shuffle=True,
                                     progress_bar=True,
                                     labels_data_group="masks/1/1",
                                     labels_source_axes="C",
                                     mask_data_group="0/4",
                                     mask_roi="(0,0,0,0,0):(1,-1,1,-1,-1)",
                                     mask_source_axes="TCZYX",
                                     mask_axes="YXC",
                                     mask_func=zds.compute_tissue_mask,
                                     mask_func_args={"mask_scale": 1.0}
                                     )

        my_dataloader = DataLoader(my_dataset, batch_size=batch_size,
                                   num_workers=num_workers,
                                   worker_init_fn=zds.zarrdataset_worker_init,
                                   persistent_workers=num_workers > 0)

        for i, (p, x, t) in enumerate(my_dataloader):
            print("Sample %i" % i, p.shape, p.dtype, x.shape, x.dtype, t.shape, t.dtype)


except ModuleNotFoundError:
    import logging
    logging.warning("PyTorch is not installed and this demo requires it to "
                    "work, please use 'general_dataset_demo.py instead'")
