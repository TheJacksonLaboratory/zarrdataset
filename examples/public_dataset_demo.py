import numpy as np
import torch
import torchvision
from torch.utils.data import DataLoader

import zarrdataset as zds


if __name__ == "__main__":
    print("Zarr-based data loader demo")
    # These are images from the Image Data Resource (IDR) 
    # https://idr.openmicroscopy.org/ that are publicly available and were 
    # converted to the OME-NGFF (Zarr) format by the OME group. More examples
    # can be found at Public OME-Zarr data (Nov. 2020)
    # https://www.openmicroscopy.org/2020/11/04/zarr-data.html
    filenames = ["https://uk1s3.embassy.ebi.ac.uk/idr/zarr/v0.1/9836839.zarr",
                 "https://uk1s3.embassy.ebi.ac.uk/idr/zarr/v0.1/9836840.zarr",
                 "https://uk1s3.embassy.ebi.ac.uk/idr/zarr/v0.1/9836841.zarr",
                 "https://uk1s3.embassy.ebi.ac.uk/idr/zarr/v0.1/9836842.zarr"]

    data_group = "0"
    data_axes = "TCZYX"
    patch_size = 512
    batch_size = 3
    num_workers = 0

    torch.manual_seed(777)
    patch_sampler = zds.GridPatchSampler(patch_size)

    transform_fn = torchvision.transforms.Compose([
        zds.SelectAxes(source_axes=data_axes,
                       axes_selection={"T": 0, "Z": 0},
                       target_axes="CYX"),
        zds.ZarrToArray(np.float64)
    ])

    my_dataset = zds.ZarrDataset(filenames, transform=transform_fn,
                                 data_group=data_group,
                                 data_axes=data_axes,
                                 patch_sampler=patch_sampler,
                                 shuffle=True,
                                 progress_bar=True)

    my_dataloader = DataLoader(my_dataset, batch_size=batch_size,
                               num_workers=num_workers,
                               worker_init_fn=zds.zarrdataset_worker_init,
                               persistent_workers=num_workers > 0)

    for i, (x, t) in enumerate(my_dataloader):
        print("Sample %i" % i, x.shape, x.dtype, t.shape, t.dtype)
