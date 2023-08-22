import time
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
        filenames = ["https://uk1s3.embassy.ebi.ac.uk/idr/zarr/v0.4/idr0101A/13457537.zarr"]

        patch_size = (512, 512)
        batch_size = 4
        num_workers = 2

        torch.manual_seed(777)
        patch_sampler = zds.GridPatchSampler(patch_size)

        mask_generator = zds.WSITissueMaskGenerator(mask_scale=1)

        transform_fn = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor()
        ])

        my_dataset = zds.LabeledZarrDataset(filenames, transform=transform_fn,
                                           data_group="2",
                                           source_axes="TCZYX",
                                           roi="(0,0,0,0,0):(1,-1,1,-1,-1)",
                                           axes="YXC",
                                           mask_data_group="4",
                                           mask_source_axes="TCZYX",
                                           mask_roi="(0,0,0,0,0):(1,-1,1,-1,-1)",
                                           mask_axes="YXC",
                                           mask_func=mask_generator,
                                           patch_sampler=patch_sampler)

        my_dataloader = DataLoader(my_dataset, batch_size=batch_size,
                                   num_workers=num_workers,
                                   worker_init_fn=zds.zarrdataset_worker_init,
                                   persistent_workers=False)
        etimes = 0
        etime = time.perf_counter()
        for i, (x, t) in enumerate(my_dataloader):
            etime = time.perf_counter() - etime
            etimes += etime
            print("Sample %i" % i, x.shape, x.dtype, t.shape, t.dtype, etime,
                  etimes / (i + 1))
            etime = time.perf_counter()


except ModuleNotFoundError:
    import logging
    logging.warning("PyTorch is not installed and this demo requires it to "
                    "work, please use 'general_dataset_demo.py instead'")
