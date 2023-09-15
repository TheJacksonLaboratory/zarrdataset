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

# Basic masked dataset loading with MaskedZarrDataset

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

```{code-cell} ipython3
z_img = zarr.open(filenames[0], mode="r")
z_img["0"].info
```

```{code-cell} ipython3
import matplotlib.pyplot as plt
plt.imshow(np.moveaxis(z_img["4"][0, :, 0], 0, -1))
plt.show()
```

## Define a mask from where patches will be extracted

+++

```{code-cell} ipython3
mask = np.array([
    [0, 0, 0, 0, 0],
    [0, 0, 1, 0, 0],
    [0, 1, 0, 0, 0],
    [0, 0, 0, 0, 0],
    [0, 0, 0, 0, 1],
], dtype=bool)
```

ZarrDataset will match the size of the mask t the size of the image that is being sampled.

For that reason, it is not necessary for the mask to be of the same size of the image.

```{code-cell} ipython3
_, d, _, h, w = z_img["4"].shape
m_h, m_w = mask.shape

factor_h = h / m_h
factor_w = w / m_w

plt.imshow(np.moveaxis(z_img["4"][0, :, 0], 0, -1))

sampling_region = np.array([
    [0, 0],
    [0, factor_w],
    [factor_h, factor_w],
    [factor_h, 0],
    [0, 0]
])

for m_y, m_x in zip(*np.nonzero(mask)):
    offset_y = m_y * factor_h
    offset_x = m_x * factor_w
    plt.plot(sampling_region[:, 1] + offset_x,
             sampling_region[:, 0] + offset_y)

plt.show()
```

## Extract patches of size 512x512 pixels from masked regiosn of a Whole Slide Image (WSI)

+++

Sample the image uniformly in a squared grid pattern

```{code-cell} ipython3
patch_size = dict(Y=512, X=512)
patch_sampler = zds.GridPatchSampler(patch_size=patch_size)
```

Use the MaskedZarrDataset class to enable extraction of samples from masked regions.

Enable sampling patched from random locations with `shuffle=True`

```{code-cell} ipython3
my_dataset = zds.MaskedZarrDataset(filenames,
                                   data_group="1",
                                   source_axes="TCZYX",
                                   patch_sampler=patch_sampler,
                                   draw_same_chunk=False,
                                   return_any_label=False,
                                   mask_filenames=[mask],
                                   mask_source_axes="YX",
                                   mask_data_group="",
                                   shuffle=True)
```

```{code-cell} ipython3
ds_iterator = iter(my_dataset)
```

```{code-cell} ipython3
sample = next(ds_iterator)
type(sample), sample.shape, sample.dtype
```

```{code-cell} ipython3
plt.imshow(np.moveaxis(sample[0, :, 0], 0, -1))
plt.show()
```

```{code-cell} ipython3
samples = []
for i, sample in enumerate(my_dataset):
    samples.append(np.pad(np.moveaxis(sample[0, :, 0], 0, -1),((1, 1), (1, 1), (0, 0))))

    # Obtain only 5 samples
    if i >= 4:
        break

grid_samples = np.hstack(samples)
```

```{code-cell} ipython3
plt.imshow(grid_samples)
plt.show()
```
