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

# Labeled dataset loading with LabeledZarrDataset

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

## Extract pair of patches of size 512x512 pixels and their respective label from a labeled Whole Slide Image (WSI)

+++

LabeledZarrDataset can retrieve the associated label to each patch extracted as a pair of input and target samples.

```{code-cell} ipython3
patch_size = dict(Y=512, X=512)
patch_sampler = zds.PatchSampler(patch_size=patch_size)
```

### Weakly labeled exmaple

+++

Weakly labeled means that there is a few labels (or only one) associated to the whole image.

These labels could be loaded directly from a list or arrays.

```{code-cell} ipython3
my_dataset = zds.LabeledZarrDataset(filenames,
                                    data_group="1",
                                    source_axes="TCZYX",
                                    patch_sampler=patch_sampler,
                                    labels_filenames=[np.array([1])],
                                    labels_source_axes="C",
                                    shuffle=True)
```

```{code-cell} ipython3
for i, (sample, label) in enumerate(my_dataset):
    print(f"Sample {i}, patch size: {sample.shape}, label: {label}")

    # Obtain only 5 samples
    if i >= 4:
        break
```

+++

### Densely labeled example

+++
Densely labeled images contain more spatial information about the image.

This could be the case when pixels of the image belong to a specific class, like in object segmentation problems.

The image label does not need to be of the same size of the image, since ZarrDataset will match the coordinates of the image and the label.

```{code-cell} ipython3
from skimage import color, filters, morphology

z_img = zarr.open(filenames[0], mode="r")

im_gray = color.rgb2gray(z_img["4"][0, :, 0], channel_axis=0)
thresh = filters.threshold_otsu(im_gray)

labels = im_gray > thresh
labels = morphology.remove_small_objects(labels == 0, min_size=16 ** 2,
                                         connectivity=2)
labels = morphology.remove_small_holes(labels, area_threshold=128)
labels = morphology.binary_erosion(labels, morphology.disk(3))
labels = morphology.binary_dilation(labels, morphology.disk(16))
```

The label image would be something like the following

```{code-cell} ipython3
import matplotlib.pyplot as plt
plt.subplot(1, 2, 1)
plt.imshow(np.moveaxis(z_img["4"][0, :, 0], 0, -1))
plt.subplot(1, 2, 1)
plt.imshow(labels)
plt.show()
```

In this case, the labels are passed as a list of Numpy NDArrays, but these could be also stored in Zarr, either locally or in a remote S3 bucket.

```{code-cell} ipython3
my_dataset = zds.LabeledZarrDataset(filenames,
                                    data_group="1",
                                    source_axes="TCZYX",
                                    patch_sampler=patch_sampler,
                                    labels_filenames=[labels],
                                    labels_source_axes="YX",
                                    shuffle=True)
```

```{code-cell} ipython3

fig, ax = plt.subplots(3, 6)
for i, (sample, label) in enumerate(my_dataset):
    print(f"Sample {i}, patch size: {sample.shape}, label: {label.shape}")

    ax[i // 3, 2 * (i % 3)].imshow(sample[0, :, 0].transpose(1, 2, 0))
    ax[i // 3, 2 * (i % 3)].set_title(f"Image {i}")
    ax[i // 3, 2 * (i % 3)].axis("off")

    ax[i // 3, 2 * (i % 3) + 1].imshow(label)
    ax[i // 3, 2 * (i % 3) + 1].set_title(f"Label {i}")
    ax[i // 3, 2 * (i % 3) + 1].axis("off")

    # Obtain only 9 samples
    if i >= 8:
        break

plt.show()
```
