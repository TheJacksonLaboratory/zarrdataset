import numpy as np
import zarrdataset as zds


def transform_fn(image):
    image = zds.ZarrToArray(np.float64)(image)
    return image


if __name__ == "__main__":
    print("Zarr-based data loader demo (without PyTorch functionalities)")
    # These are images from the Image Data Resource (IDR) 
    # https://idr.openmicroscopy.org/ that are publicly available and were 
    # converted to the OME-NGFF (Zarr) format by the OME group. More examples
    # can be found at Public OME-Zarr data (Nov. 2020)
    # https://www.openmicroscopy.org/2020/11/04/zarr-data.html
    filenames = ["https://uk1s3.embassy.ebi.ac.uk/idr/zarr/v0.4/idr0073A/9798462.zarr"]

    patch_size = (512, 512)

    patch_sampler = zds.GridPatchSampler(patch_size)
    my_dataset = zds.ZarrDataset(filenames, transform=transform_fn,
                                 data_group="2",
                                 source_axes="TCZYX",
                                 roi="(0,0,0,0,0):(1,-1,1,-1,-1)",
                                 axes="YXC",
                                 patch_sampler=patch_sampler)

    for i, (x, t) in enumerate(my_dataset):
        print("Sample %i" % i, x.shape, x.dtype, type(t), t)
