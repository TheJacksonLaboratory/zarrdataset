---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.15.1
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

# Basic ZarrDataset usage example

+++

Import the "zarrdataset" package

```{code-cell} ipython3
import zarrdataset as zds
import zarr
```

Load data stored on S3 storage

```{code-cell} ipython3
# These are images from the Image Data Resource (IDR) 
# https://idr.openmicroscopy.org/ that are publicly available and were 
# converted to the OME-NGFF (Zarr) format by the OME group. More examples
# can be found at Public OME-Zarr data (Nov. 2020)
# https://www.openmicroscopy.org/2020/11/04/zarr-data.html

filenames = ["https://uk1s3.embassy.ebi.ac.uk/idr/zarr/v0.4/idr0073A/9798462.zarr"]
```

```{code-cell} ipython3
import random
import numpy as np

# For reproducibility
np.random.seed(478963)
random.seed(478965)
```

Inspect the image to sample

```{code-cell} ipython3
z_img = zarr.open(filenames[0], mode="r")
z_img["0"].info
```

Display a downsampled version of the image

```{code-cell} ipython3
import matplotlib.pyplot as plt

plt.imshow(np.moveaxis(z_img["5"][0, :, 0], 0, -1))
plt.show()
```

## Retrieving whole images

+++

Create a ZarrDataset to handle the image dataset instead of opening all the dataset images by separate and hold them in memory until they are not used anymore.

```{code-cell} ipython3
my_dataset = zds.ZarrDataset()
```

Start by retrieving whole images, from a subsampled (pyramid) group (e.g. group 6) within the zarr image file, instead the full resolution image at group "0".
The source array axes should be specified in order to handle images properly, in this case Time-Channel-Depth-Height-Width (TCZYX).

```{code-cell} ipython3
my_dataset.add_modality(
  modality="image",
  filenames=filenames,
  source_axes="TCZYX",
  data_group="6"
)
```

The ZarrDataset class can be used as a Python's generator, and can be accessed by `iter` and subsequently `next` operations.

```{code-cell} ipython3
ds_iterator = iter(my_dataset)
ds_iterator
```

```{code-cell} ipython3
sample = next(ds_iterator)

print(type(sample), sample.shape)
```

Compare the shape of the retreived sample with the shape of the original image in group "6"
```{code-cell} ipython3
z_img["6"].info
```

## Extracting patches of size 512x512 pixels from a Whole Slide Image (WSI)

+++

The PatchSampler class can be used along with ZarrDataset to retrieve patches from WSIs without having to tiling them in a pre-process step.

```{code-cell} ipython3
patch_size = dict(Y=512, X=512)
patch_sampler = zds.PatchSampler(patch_size=patch_size)

patch_sampler
```

Create a new dataset using the ZarrDataset class, and pass the PatchSampler as `patch_sampler` argument.
Because patches are being exracted instead of whole images, the full resolution image at group "0" can be used as input.

```{code-cell} ipython3
my_dataset = zds.ZarrDataset(patch_sampler=patch_sampler)

my_dataset.add_modality(
  modality="image",
  filenames=filenames,
  source_axes="TCZYX",
  data_group="0"
)

my_dataset
```

Create a generator from the dataset object and extract some patches

```{code-cell} ipython3
ds_iterator = iter(my_dataset)

sample = next(ds_iterator)
type(sample), sample.shape, sample.dtype

sample = next(ds_iterator)
type(sample), sample.shape, sample.dtype

sample = next(ds_iterator)
type(sample), sample.shape, sample.dtype
```

```{code-cell} ipython3
plt.imshow(np.moveaxis(sample[0, :, 0], 0, -1))
plt.show()
```

## Using ZarrDataset in a for loop

+++

ZarrDatasets can be used as generators, for example in for loops

```{code-cell} ipython3
samples = []
for i, sample in enumerate(my_dataset):
    samples.append(np.moveaxis(sample[0, :, 0], 0, -1))

    if i >= 4:
        # Take only five samples for illustration purposes
        break

samples_stack = np.hstack(samples)
```

```{code-cell} ipython3
plt.imshow(samples_stack)
plt.show()
```

## Create a ZarrDataset with all the dataset specifications.

Use a dictionary (or a list of them for multiple modalities) to define the dataset specifications.
Alternatively, use a list of DatasetSpecs (or derived classes) to define the dataset specifications that ZarrDataset requires.

For example, `ImagesDatasetSpecs` can be used to define an _image_ data modality. Other pre-defined modalities are `LabelsDatasetSpecs` for _labels_, and `MaskDatasetSpecs` for _masks_.

```{code-cell} ipython3
image_specs = zds.ImagesDatasetSpecs(
  filenames=filenames,
  data_group="0",
  source_axes="TCZYX",
)
```

Also, try sampling patches from random locations by setting `shuffle=True`.

```{code-cell} ipython3

my_dataset = zds.ZarrDataset(dataset_specs=[image_specs],
                             patch_sampler=patch_sampler,
                             shuffle=True)
```

```{code-cell} ipython3
samples = []
for i, sample in enumerate(my_dataset):
    samples.append(np.moveaxis(sample[0, :, 0], 0, -1))

    if i >= 4:
        # Take only five samples for illustration purposes
        break

samples_stack = np.hstack(samples)
```

```{code-cell} ipython3
plt.imshow(samples_stack)
plt.show()
```