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

# Custom masks for sampling with MaskedZarrDataset

```{code-cell} ipython3
import zarrdataset as zds
import zarr
```

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
import numpy as np
import matplotlib.pyplot as plt
plt.imshow(np.moveaxis(z_img["4"][0, :, 0], 0, -1))
plt.show()
```

Define a mask from where patches can be extracted

```{code-cell} ipython3
from skimage import color, filters, morphology
```

```{code-cell} ipython3
im_gray = color.rgb2gray(z_img["4"][0, :, 0], channel_axis=0)
thresh = filters.threshold_otsu(im_gray)

mask = im_gray > thresh
mask = morphology.remove_small_objects(mask == 0, min_size=16 ** 2,
                                       connectivity=2)
mask = morphology.remove_small_holes(mask, area_threshold=128)
mask = morphology.binary_erosion(mask, morphology.disk(8))
mask = morphology.binary_dilation(mask, morphology.disk(8))
```

```{code-cell} ipython3
plt.imshow(mask)
plt.show()
```

```{code-cell} ipython3
plt.imshow(np.moveaxis(z_img["4"][0, :, 0], 0, -1))
plt.imshow(mask, cmap="gray", alpha=1.0*(mask < 1))
plt.show()
```

## Extract patches of size 512x512 pixels from a Whole Slide Image (WSI)

+++

Sample the image uniformly in a squared grid pattern

```{code-cell} ipython3
patch_size = dict(Y=512, X=512)
patch_sampler = zds.GridPatchSampler(patch_size=patch_size)
```

Use the MaskedZarrDataset class to enable extraction of samples from masked regions.

An extra dimension is added to the mask, so it matches the number of spatial axes in the image

```{code-cell} ipython3
my_dataset = zds.MaskedZarrDataset(filenames,
                                   data_group="1",
                                   source_axes="TCZYX",
                                   patch_sampler=patch_sampler,
                                   return_any_label=False,
                                   mask_filenames=[mask[None, ...]],
                                   mask_source_axes="ZYX",
                                   mask_axes="ZYX",
                                   mask_data_group="")
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
    samples.append(np.moveaxis(sample[0, :, 0], 0, -1))

    if i >= 4:
        # Take only five samples for illustration purposes
        break

samples = np.hstack(samples)
```

```{code-cell} ipython3
plt.imshow(samples)
plt.show()
```

## Use a function to generate the masks for each image in the dataset

```{code-cell} ipython3
patch_size = dict(Y=512, X=512)
patch_sampler = zds.GridPatchSampler(patch_size=patch_size)
```

Apply WSITissueMaskGenerator transform to each image in the dataset to define each sampling mask

```{code-cell} ipython3
mask_func = zds.WSITissueMaskGenerator(mask_scale=1,
                                       min_size=16,
                                       area_threshold=128,
                                       axes="ZYX")
```

Because the input image (zarr group "1") is large, computing the mask directly on that could require high computational resources.

For that reason, use a donwsampled version of that image instead by pointing `mask_data_group="4"` to use a 1:16 downsampled version of the input image.

The `mask_axes` should match the ones that WSITissueMaskGenerator requies as input ("YXC"). To do that, a ROI can be specified to take just the spatial and channel axes from the input image with `mask_roi="(0,0,0,0,0):(1,-1,1,-1,-1)"`, and rearrange the output axes with `mask_axes="YXC"`.

This is achieved by defining a ROI that extracts only the spatial axes, and color channels from the image.

```{code-cell} ipython3
my_dataset = zds.MaskedZarrDataset(filenames,
                                   data_group="1",
                                   source_axes="TCZYX",
                                   patch_sampler=patch_sampler,
                                   return_any_label=False,
                                   mask_func=mask_func,
                                   mask_filenames=None,
                                   mask_data_group="4",
                                   mask_source_axes="TCZYX",
                                   mask_roi="(0,0,0,0,0):(1,-1,1,-1,-1)",
                                   mask_axes="YXC",
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
    samples.append(np.moveaxis(sample[0, :, 0], 0, -1))

    if i >= 4:
        # Take only five samples for illustration purposes
        break

samples = np.hstack(samples)
```

```{code-cell} ipython3
plt.imshow(samples)
plt.show()
```