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

## Extracting patches of size 512x512 pixels from a Whole Slide Image (WSI)

+++

Sample the image uniformly in a grid pattern

```{code-cell} ipython3
patch_size = dict(Y=512, X=512)
patch_sampler = zds.GridPatchSampler(patch_size=patch_size)
```

Create a dataset from the list of filenames. All those files should be stored within their respective group "0".

Also, specify that the axes order in the image is Time-Channel-Depth-Height-Width (TCZYX), so the data can be handled correctly

```{code-cell} ipython3
my_dataset = zds.ZarrDataset(filenames,
                             data_group="0",
                             source_axes="TCZYX",
                             patch_sampler=patch_sampler,
                             return_any_label=False)
```

Create a generator from the dataset object

```{code-cell} ipython3
ds_iterator = iter(my_dataset)
ds_iterator
```

Extract a sample from the image using the generator

```{code-cell} ipython3
sample = next(ds_iterator)
type(sample), sample.shape, sample.dtype
```

```{code-cell} ipython3
plt.imshow(np.moveaxis(sample[0, :, 0], 0, -1))
plt.show()
```

## Using ZarrDataset as a generator

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

Try now with sampling patches from random locations by setting `shuffle=True`

```{code-cell} ipython3
my_dataset = zds.ZarrDataset(filenames,
                             data_group="0",
                             source_axes="TCZYX",
                             patch_sampler=patch_sampler,
                             return_any_label=False,
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