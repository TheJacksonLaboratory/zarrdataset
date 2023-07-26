import os
import numpy as np
import zarrdataset as zds

from numcodecs import register_codec
from caecodec import ConvolutionalAutoencoder, BottleneckStore

register_codec(ConvolutionalAutoencoder)

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
        filenames = [
            "https://uk1s3.embassy.ebi.ac.uk/idr/zarr/v0.1/9836839.zarr",
            "https://uk1s3.embassy.ebi.ac.uk/idr/zarr/v0.1/9836840.zarr",
            "https://uk1s3.embassy.ebi.ac.uk/idr/zarr/v0.1/9836841.zarr",
            "https://uk1s3.embassy.ebi.ac.uk/idr/zarr/v0.1/9836842.zarr"
            ]

        patch_size = 256
        batch_size = 4
        num_workers = 4

        torch.manual_seed(777)
        patch_sampler = zds.GridPatchSampler(patch_size)

        transform_fn = torchvision.transforms.Compose([
            zds.ZarrToArray(np.float64)
        ])

        my_dataset = zds.ZarrDataset(filenames,
                                     transform=transform_fn,
                                     data_group="0",
                                     source_axes="TCZYX",
                                     axes="YXC",
                                     roi="(0,0,0,0,0):(1,-1,1,-1,-1)",
                                     patch_sampler=patch_sampler)

        my_dataloader = DataLoader(my_dataset, batch_size=batch_size,
                                   num_workers=num_workers,
                                   worker_init_fn=zds.zarrdataset_worker_init,
                                   persistent_workers=False)

        for i, (x, t) in enumerate(my_dataloader):
            print("Sample %i" % i, x.shape, x.dtype, t.shape, t.dtype)


except ModuleNotFoundError:
    import logging
    logging.warning("PyTorch is not installed and this demo requires it to "
                    "work, please use 'general_dataset_demo.py instead'")
