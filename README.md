# zarrdataset
A dataset for loading images stored in zarr format to be used primarly with PyTorch's DataLoader in machine learning training workflows.

## Usage
```
import zarrdataset as zds

# Open a set of Zarr files stored locally or in a S3 bucket.
filenames = ["https://uk1s3.embassy.ebi.ac.uk/idr/zarr/v0.1/9836839.zarr"]

# Specify the group/component were the arrays are stored within the zarr file,
# and the order of the axes of the dataset.
my_dataset = zds.ZarrDataset(filenames,
                             data_group="0",
                             data_axes="TCZYX",
                             transform=None,
                             patch_sampler=None,
                             shuffle=False,
                             progress_bar=False)
```

## Integration
The **ZarrDataset** and **LabeledZarrDataset** classes are derived from PyTorch's _IterableDataset_ class, and can be used with a _DataLoader_ object to generate batches of inputs for machine learning training workflows.

```
from torch.utils.data import DataLoader
import zarrdataset as zds

my_dataset = zds.ZarrDataset(...)

# Generate batches of 16 images, uisng four threads.
# Pass the worker initialization function from zarrdataset to the DataLoader
my_dataloader = DataLoader(my_dataset,
                           batch_size=16,
                           num_workers=4,
                           worker_init_fn=zds.zarrdataset_worker_init)

for x, t in my_dataloader:
    # The training loop
    ...
    output = model(x)
    loss = criterion(output, t)
    ...
```
### Multithread data loading
Use of multiple workers through multithread requires the use of the **zarrdataset_worker_init** function provided in this package.
This allows to load only a fraction of the dataset on each worker instead of the full dataset.

### Labeled datasets
Data containing labels associated to each image are supported if these labels are stored within the same zarr file.
The following structure illustrated the kind of groups structure within zarr files are supported by the  **LabeledZarrDataset** class.
```
image.zarr
|-0
| |-0 // Where the pixels data of image.zarr are stored (image.zarr["0/0"])
|
|-labels
  |-0
    |-0 // Where the labels of image.zarr are stored (image.zarr["labels/0/0"])
```

These files can be loaded as follows.
```
import zarrdataset as zds

# Specify the group/component were the labels are stored within the zarr file,
# and the order of their axes. In this case, labels are stored as a
# numpy array which axes are ordered as Class, Y-axis, X-axis
my_dataset = zds.LabeledZarrDataset(filenames,
                                    data_group="0",
                                    data_axes="TCZYX",
                                    labels_data_group="labels/0/0",
                                    labels_data_axes="CYX",
                                    transform=None,
                                    patch_sampler=None,
                                    shuffle=False,
                                    progress_bar=False)
```

### Transforms
Some transforms are provided to be integrated along torchvision transforms for data pre-processing and data augmentation.

```
from torchvision
import zarrdataset as zds

# Add a pre-process pipeline to select index 0 from axes T and Z, and reorder
# the remaining axes as Y-axis, X-axis, Color channels. Then convert the array
# to numpy float32 data type and finally to a PyTorch Tensor.
preprocess_funs = torchvision.transforms.Compose(
  [zds.SeletAxes(source_axes=args.data_axes,
                 axes_selection={"T": 0, "Z": 0},
                 target_axes="YXC"),
   zds.ZarrToArray(np.float32),
   torchvision.transforms.ToTensor()
  ]
)

my_dataset = zds.ZarrDataset(filenames,
                             data_group="0",
                             data_axes="TCZYX",
                             transform=preprocess_funs,
                             patch_sampler=None,
                             shuffle=False,
                             progress_bar=False)
```

### Patch sampling
The **ZarrDataset** and **LabeledZarrDataset** retrieve the full array contained in *data_group* by default.
To retrieve patches from that array instead, use any of the two samplers provided within this package, or implement a custom one derived from the **PatchSampler** class.

The two existing samplers are **GridPatchSampler** and **BlueNoisePatchSampler**.
**GridPatchSampler** generates a evenly distributed grid of non-overlapping squared patches of side *patch_size*. **BlueNoisePatchSampler** generates a random sample of non-overlapping squared patches of side *patch_size* that uniformly covers the image.
The patch sampler can be integated into a **ZarrDataset** or **LabeledZarrDataset** object as follows.
```
import zarrdataset as zds

# Retrieve patches of size patch_size in an evenly spaced grid from the image.
my_patch_sampler = zds.GridPatchSampler(patch_size)

my_dataset = zds.ZarrDataset(filenames,
                             data_group="0",
                             data_axes="TCZYX",
                             patch_sampler=my_patch_sampler,
                             shuffle=False,
                             progress_bar=False)
```

Examples of integration of the **ZarrDataset** class with the PyTorch's **DataLoader** can be found in the _examples_ folder.

## Installation
To install this package and have access to the zarr-based PyTorch Dataset (ZarrDataset and LabeledZarrDataset) and other functionalities, clone this repository and use the following command from the cloned repository location.
```
python setup.py -e .
```