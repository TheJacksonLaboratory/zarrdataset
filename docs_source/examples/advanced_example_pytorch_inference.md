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
  timeout: 120
---

# Integration of ZarrDataset with PyTorch's DataLoader for inference (Advanced)

```python
import zarrdataset as zds

import torch
from torch.utils.data import DataLoader
```


```python
# These are images from the Image Data Resource (IDR) 
# https://idr.openmicroscopy.org/ that are publicly available and were 
# converted to the OME-NGFF (Zarr) format by the OME group. More examples
# can be found at Public OME-Zarr data (Nov. 2020)
# https://www.openmicroscopy.org/2020/11/04/zarr-data.html

filenames = [
    "https://uk1s3.embassy.ebi.ac.uk/idr/zarr/v0.4/idr0073A/9798462.zarr"
]
```


```python
import random
import numpy as np

# For reproducibility
np.random.seed(478963)
torch.manual_seed(478964)
random.seed(478965)
```

## Extracting patches of size 1024x1024 pixels from a Whole Slide Image (WSI)

Retrieve samples for inference. Add padding to each patch to avoid edge artifacts when stitching the inference result.
Finally, let the PatchSampler retrieve patches from the edge of the image that would be otherwise smaller than the patch size by setting `allow_incomplete_patches=True`.


```python
patch_size = dict(Y=128, X=128)
pad = dict(Y=16, X=16)
patch_sampler = zds.PatchSampler(patch_size=patch_size, pad=pad, allow_incomplete_patches=True)
```

Create a dataset from the list of filenames. All those files should be stored within their respective group "0".

Also, specify that the axes order in the image is Time-Channel-Depth-Height-Width (TCZYX), so the data can be handled correctly


```python
image_specs = zds.ImagesDatasetSpecs(
  filenames=filenames,
  data_group="4",
  source_axes="TCZYX",
  axes="YXC",
  roi="0,0,0,0,0:1,-1,1,-1,-1"
)

my_dataset = zds.ZarrDataset(image_specs,
                             patch_sampler=patch_sampler,
                             return_positions=True)
```


```python
my_dataset
```




    ZarrDataset (PyTorch support:True, tqdm support :True)
    Modalities: images
    Transforms order: []
    Using images modality as reference.
    Using <class 'zarrdataset._samplers.PatchSampler'> for sampling patches of size {'Z': 1, 'Y': 128, 'X': 128}.



Add a pre-processing step before creating the image batches, where the input arrays are casted from int16 to float32.


```python
import torchvision

img_preprocessing = torchvision.transforms.Compose([
    zds.ToDtype(dtype=np.float32),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(127, 255)
])

my_dataset.add_transform("images", img_preprocessing)
```


```python
my_dataset
```




    ZarrDataset (PyTorch support:True, tqdm support :True)
    Modalities: images
    Transforms order: [('images',)]
    Using images modality as reference.
    Using <class 'zarrdataset._samplers.PatchSampler'> for sampling patches of size {'Z': 1, 'Y': 128, 'X': 128}.



## Create a DataLoader from the dataset object

ZarrDataset is compatible with DataLoader from PyTorch since it is inherited from the IterableDataset class of the torch.utils.data module.


```python
my_dataloader = DataLoader(my_dataset, num_workers=0)
```


```python
import dask.array as da
import numpy as np
import zarr

z_arr = zarr.open("https://uk1s3.embassy.ebi.ac.uk/idr/zarr/v0.4/idr0073A/9798462.zarr/4", mode="r")

H = z_arr.shape[-2]
W = z_arr.shape[-1]

pad_H = (128 - H) % 128
pad_W = (128 - W) % 128
z_prediction = zarr.zeros((H + pad_H, W + pad_W), dtype=np.float32, chunks=(128, 128))
z_prediction
```




    <zarr.core.Array (1152, 1408) float32>



Set up a simple model for illustration purpose


```python
model = torch.nn.Sequential(
    torch.nn.Conv2d(in_channels=3, out_channels=1, kernel_size=1),
    torch.nn.ReLU()
)
```


```python
for i, (pos, sample) in enumerate(my_dataloader):
    pred_pos = (
        slice(pos[0, 0, 0].item() + 16,
              pos[0, 0, 1].item() - 16),
        slice(pos[0, 1, 0].item() + 16,
              pos[0, 1, 1].item() - 16)
    )
    pred = model(sample)
    z_prediction[pred_pos] = pred.detach().cpu().numpy()[0, 0, 16:-16, 16:-16]
```

## Visualize the result


```python
import matplotlib.pyplot as plt

plt.subplot(2, 1, 1)
plt.imshow(np.moveaxis(z_arr[0, :, 0, ...], 0, -1))
plt.subplot(2, 1, 2)
plt.imshow(z_prediction)
plt.show()
```
