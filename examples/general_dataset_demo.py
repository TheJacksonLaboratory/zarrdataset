import numpy as np
import zarrdataset as zds


def transform_fn(image):
    image = zds.SelectAxes(source_axes=data_axes,
                           axes_selection={"T": 0, "Z": 0},
                           target_axes="CYX")(image)
    image = zds.ZarrToArray(np.float64)(image)

    return image


if __name__ == "__main__":
    print("Zarr-based data loader demo (without PyTorch functionalities)")
    # These are images from the Image Data Resource (IDR) 
    # https://idr.openmicroscopy.org/ that are publicly available and were 
    # converted to the OME-NGFF (Zarr) format by the OME group. More examples
    # can be found at Public OME-Zarr data (Nov. 2020)
    # https://www.openmicroscopy.org/2020/11/04/zarr-data.html
    filenames = ["https://uk1s3.embassy.ebi.ac.uk/idr/zarr/v0.1/9836839.zarr",
                 "https://uk1s3.embassy.ebi.ac.uk/idr/zarr/v0.1/9836840.zarr",
                 "https://uk1s3.embassy.ebi.ac.uk/idr/zarr/v0.1/9836841.zarr",
                 "https://uk1s3.embassy.ebi.ac.uk/idr/zarr/v0.1/9836842.zarr"]

    data_group = "0"
    data_axes = "TCZYX"
    patch_size = 1024
    batch_size = 16
    num_workers = 4

    patch_sampler = zds.GridPatchSampler(patch_size)

    my_dataset = zds.ZarrDataset(filenames, transform=transform_fn,
                                 data_group=data_group,
                                 data_axes=data_axes,
                                 patch_sampler=patch_sampler,
                                 shuffle=True,
                                 progress_bar=True)


    for i, (x, t) in enumerate(my_dataset):
        print("Sample %i" % i, x.shape, x.dtype, type(t), t)
