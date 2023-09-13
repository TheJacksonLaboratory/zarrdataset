import zarrdataset as zds
import zarr
import numpy as np
import random
import matplotlib.pyplot as plt
from PIL import Image
from skimage import transform
import torch
import torchvision


if __name__ == "__main__":
    filenames = [
        "https://uk1s3.embassy.ebi.ac.uk/idr/zarr/v0.1/6001240.zarr",
        "https://uk1s3.embassy.ebi.ac.uk/idr/zarr/v0.1/6001241.zarr",
        "https://uk1s3.embassy.ebi.ac.uk/idr/zarr/v0.1/6001242.zarr",
        "https://uk1s3.embassy.ebi.ac.uk/idr/zarr/v0.1/6001243.zarr",
    ]
        
    patch_size = dict(Z=32, Y=128, X=128)
    patch_sampler = zds.BlueNoisePatchSampler(patch_size=patch_size)


    img_preprocessing = torchvision.transforms.Compose([
        zds.ToDtype(dtype=np.float16)
    ])

    mask = np.ones((12, 24), dtype=bool)

    my_datasets = [zds.MaskedZarrDataset(fn,
                                transform=img_preprocessing,
                                data_group="0",
                                source_axes="TCZYX",
                                axes="CYX",
                                roi="(0,0,0,0,0):(1,-1,1,-1,-1)",
                                mask_filenames=[mask],
                                mask_source_axes="YX",
                                patch_sampler=patch_sampler,
                                shuffle=True,
                                return_any_label=False,
                                return_positions=True,
                                return_worker_id=True)
                for fn in filenames]

    my_chain_dataset = torch.utils.data.ChainDataset(my_datasets)

    my_dataloader = torch.utils.data.DataLoader(my_chain_dataset,
                            num_workers=0,
                            worker_init_fn=zds.chained_zarrdataset_worker_init,
                            batch_size=4
                            )

    np.random.seed(478963)

    samples = []
    positions = []
    wids = []
    for i, (wid, pos, sample) in enumerate(my_dataloader):
        wids += [w for w in wid]
        positions += [p for p in pos]
        samples += [s for s in sample]    

        print(f"Sample {i+1} with size {sample.shape} extracted by worker {wid} from position {pos}")
        # print("-" * 10 + f" Difference={torch.sum(sample[0].float() - sample[1].float())}")

        if i > 3:
            # Take only five samples for illustration purposes
            break

