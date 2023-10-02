# ZarrDataset
A class for handling large-volume datasets stored in OME-NGFF Zarr format.
This can be used primarly with PyTorch's DataLoader in machine learning training workflows.

## Usage
```
import zarrdataset as zds

# Open a set of Zarr files stored locally or in a S3 bucket. Must specify the
# group/component were the arrays are stored within the zarr file, and the 
# order of the axes of the dataset.
my_dataset = zds.ZarrDataset(
  dict(
    modality="images",
    filenames=["https://uk1s3.embassy.ebi.ac.uk/idr/zarr/v0.1/9836839.zarr"],
    source_axes="TCZYX",
    data_group="0"
  ),
)
```

## Integration
The **ZarrDataset** class is derived from PyTorch's _IterableDataset_ class, and can be used with a _DataLoader_ object to generate batches of inputs for machine learning training workflows.

```
from torch.utils.data import DataLoader
import zarrdataset as zds

my_dataset = zds.ZarrDataset(...)

# Generate batches of 16 images, uisng four threads.
# Pass the worker initialization function from zarrdataset to the DataLoader
my_dataloader = DataLoader(my_dataset,
                           batch_size=16,
                           num_workers=4,
                           worker_init_fn=zds.zarrdataset_worker_init_fn)

for x, t in my_dataloader:
    # The training loop
    ...
    output = model(x)
    loss = criterion(output, t)
    ...
```
### Multithread data loading
Use of multiple workers through multithread requires the use of the **zarrdataset_worker_init_fn** function provided in this package.
This allows to load only a fraction of the dataset on each worker instead of the full dataset.

### Patch sampling
**ZarrDataset** retrieve the whole array contained in *data_group* by default.
To retrieve patches from that array instead, use any of the two samplers provided within this package, or implement a custom one derived from the **PatchSampler** class.

The two existing samplers are **PatchSampler** and **BlueNoisePatchSampler**.
**PatchSampler** retrieves patches from an evenly distributed grid of non-overlapping squared patches of side *patch_size*.
**BlueNoisePatchSampler** retrieves patches of side *patch_size* from random locations following [blue-noise sampling](https://blog.demofox.org/2017/10/20/generating-blue-noise-sample-points-with-mitchells-best-candidate-algorithm/).
The patch sampler can be integated into a **ZarrDataset** object as follows.
```
import zarrdataset as zds

# Retrieve patches of size patch_size in an evenly spaced grid from the image.
my_patch_sampler = zds.PatchSampler(patch_size)

my_dataset = zds.ZarrDataset(...,
                             patch_sampler=my_patch_sampler)
```

Examples of integration of the **ZarrDataset** class with the PyTorch's **DataLoader** can be found in the [documentation](https://thejacksonlaboratory.github.io/zarrdataset/index.html).

## Installation
This package can be installed from PyPI with the following command
```
pip install zarrdataset
```
