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
execution:
  timeout: 600
---

# Integration of ZarrDataset with Tensorflow Datasets

```{code-cell} ipython3
import zarrdataset as zds
import tensorflow as tf
```

```{code-cell} ipython3
# These are images from the Image Data Resource (IDR) 
# https://idr.openmicroscopy.org/ that are publicly available and were 
# converted to the OME-NGFF (Zarr) format by the OME group. More examples
# can be found at Public OME-Zarr data (Nov. 2020)
# https://www.openmicroscopy.org/2020/11/04/zarr-data.html

filenames = [
    "https://uk1s3.embassy.ebi.ac.uk/idr/zarr/v0.4/idr0073A/9798462.zarr"
]
```

```{code-cell} ipython3
import random
import numpy as np

# For reproducibility
np.random.seed(478963)
random.seed(478965)
```

## Extracting patches of size 1024x1024 pixels from a Whole Slide Image (WSI)

+++

Sample the image randomly using a [Blue Noise](https://blog.demofox.org/2017/10/20/generating-blue-noise-sample-points-with-mitchells-best-candidate-algorithm/) sampling.

```{code-cell} ipython3
patch_size = dict(Y=1024, X=1024)
patch_sampler = zds.BlueNoisePatchSampler(patch_size=patch_size)
```

Create a dataset from the list of filenames. All those files should be stored within their respective group "0".

Also, specify that the axes order in the image is Time-Channel-Depth-Height-Width (TCZYX), so the data can be handled correctly

```{code-cell} ipython3
image_specs = zds.ImagesDatasetSpecs(
  filenames=filenames,
  data_group="3",
  source_axes="TCZYX",
)

# A list with a labeled image, for the single image in the dataset, is passed as `filenames` argument.
labels_specs = zds.LabelsDatasetSpecs(
  filenames=[np.ones(1)],
  source_axes="L",
)

my_dataset = zds.ZarrDataset([image_specs, labels_specs],
                             patch_sampler=patch_sampler,
                             shuffle=True)
```

## Create a Tensoflow Dataset from the ZarrDataset object

+++

When PyTorch is not present in the system, ZarrDataset will still work as a python generator.

This makes it easy to connect ZarrDataset with `tensorflow.data.Dataset` and create an iterable dataset.

```{code-cell} ipython3
my_dataloader = tf.data.Dataset.from_generator(
            my_dataset.__iter__,
            output_signature=(tf.TensorSpec(shape=(1, 3, 1, None, None),
                                            dtype=tf.float32),
                              tf.TensorSpec(shape=(1,),
                                            dtype=tf.int64)))

batched_dataset = my_dataloader.batch(4)
```

This data loader can be used within Tensorflow training pipelines.

```{code-cell} ipython3
samples = []
for i, (sample, target) in enumerate(my_dataloader):
    samples.append(np.moveaxis(sample[0, :, 0], 0, -1))

    print(f"Sample {i+1} with size {sample.shape}, and target {target}")

    if i >= 4:
        # Take only five samples for illustration purposes
        break

samples_stack = np.hstack(samples)
```

```{code-cell} ipython3
samples_stack.shape
```

```{code-cell} ipython3
import matplotlib.pyplot as plt

plt.imshow(samples_stack.astype(np.uint8))
plt.show()
```
