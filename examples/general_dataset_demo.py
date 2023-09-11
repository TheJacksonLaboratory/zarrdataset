import zarrdataset as zds

import random
import torch
import torchvision
from torch.utils.data import DataLoader, ChainDataset
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

class AsType(object):
    def __init__(self, dtype):
        self.dtype = dtype

    def __call__(self, image):
        return image.astype(self.dtype)


if __name__ == "__main__":
    np.random.seed(478963)
    random.seed(478963)
    torch.random.manual_seed(478963)

    # These are images from the Image Data Resource (IDR) 
    # https://idr.openmicroscopy.org/ that are publicly available and were 
    # converted to the OME-NGFF (Zarr) format by the OME group. More examples
    # can be found at Public OME-Zarr data (Nov. 2020)
    # https://www.openmicroscopy.org/2020/11/04/zarr-data.html

    filenames = [
        "https://uk1s3.embassy.ebi.ac.uk/idr/zarr/v0.1/6001240.zarr",
        "https://uk1s3.embassy.ebi.ac.uk/idr/zarr/v0.1/6001241.zarr",
        "https://uk1s3.embassy.ebi.ac.uk/idr/zarr/v0.1/6001242.zarr",
        "https://uk1s3.embassy.ebi.ac.uk/idr/zarr/v0.1/6001243.zarr",
    ]

    patch_size = dict(Z=32, Y=128, X=128)
    patch_sampler = zds.GridPatchSampler(patch_size=patch_size)

    img_preprocessing = torchvision.transforms.Compose([
        zds.ToDtype(dtype=np.float16)
    ])
    my_datasets = [zds.ZarrDataset(fn,
                                transform=img_preprocessing,
                                data_group="0",
                                source_axes="TCZYX",
                                patch_sampler=patch_sampler,
                                shuffle=True,
                                return_any_label=False,
                                return_positions=True,
                                return_worker_id=True)
                for fn in filenames]

    my_chained_dataset = ChainDataset(my_datasets)

    my_dataloader = DataLoader(my_chained_dataset,
                               num_workers=2,
                               batch_size=4,
                               worker_init_fn=zds.chained_zarrdataset_worker_init)

    samples = []
    positions = []
    wids = []
    for i, (wid, pos, sample) in enumerate(my_dataloader):
        wids += [w for w in wid]
        positions += [p for p in pos]
        samples += [s for s in sample]    

        print(f"Sample {i+1} with size {sample.shape} extracted by worker {wid}")
        # print("-" * 10 + f" Difference={torch.sum(sample[0].float() - sample[1].float())}")

    samples = torch.cat(samples, dim=0)

    samples_grid = torchvision.utils.make_grid(samples[:, :, 0, ...], nrow=4)

    plt.imshow(samples_grid[0], cmap="gray")
    plt.axis('off')

    samples_grid = samples_grid[1].mul(255.0/4096.0).to(torch.uint8).numpy()

    im = Image.fromarray(samples_grid)
    im.save("tests/grid_sample.png")
