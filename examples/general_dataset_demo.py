import numpy as np
import zarrdataset as zds
import zarr
from PIL import Image
from torch.utils.data import DataLoader


def transform_fn(image):
    image = zds.ZarrToArray(np.float64)(image)
    return image


def generate_groups():
    grps = [
        "tests/test_zarrs/zarr_group_1.zarr",
        "tests/test_zarrs/zarr_group_2.zarr",
        "tests/test_zarrs/zarr_group_3.zarr"
    ]

    grps_arr = []
    mask_grps_arr = []
    labels_grps_arr = []

    for fn in grps:
        z_grp = zarr.open(fn, mode="r")
        grps_arr.append(z_grp["0/0"])
        mask_grps_arr.append(z_grp["masks/0/0"])
        labels_grps_arr.append(z_grp["labels/0/0"])

    return grps_arr, mask_grps_arr, labels_grps_arr


if __name__ == "__main__":
    print("Zarr-based data loader demo (without PyTorch functionalities)")
    # These are images from the Image Data Resource (IDR) 
    # https://idr.openmicroscopy.org/ that are publicly available and were 
    # converted to the OME-NGFF (Zarr) format by the OME group. More examples
    # can be found at Public OME-Zarr data (Nov. 2020)
    # https://www.openmicroscopy.org/2020/11/04/zarr-data.html
    # filenames = ["https://uk1s3.embassy.ebi.ac.uk/idr/zarr/v0.4/idr0073A/9798462.zarr"]
    filenames = [
        "../tests/test_zarrs/zarr_group_1.zarr",
        "../tests/test_zarrs/zarr_group_2.zarr",
        "../tests/test_zarrs/zarr_group_3.zarr"
        ]

    # patch_size = (512, 512)

    # patch_sampler = zds.GridPatchSampler(patch_size)
    # my_dataset = zds.ZarrDataset(filenames, transform=transform_fn,
    #                              data_group="0/0",
    #                              source_axes="CYX",
    #                              # source_axes="TCZYX",
    #                              # roi="(0,0,0,0,0):(1,-1,1,-1,-1)",
    #                              axes="YXC",
    #                              patch_sampler=patch_sampler)
    groups = dict(
        filenames=[
            "tests/test_tiffs/img_1.ome.tif",
            "tests/test_tiffs/img_2.ome.tif",
            "tests/test_tiffs/img_3.ome.tif",
            ],
        data_group="",
        source_axes="CYX",
        axes="CYX"
        )

    patch_height = 25
    patch_width = 51
    patch_sampler = zds.BlueNoisePatchSampler(patch_size=(patch_height, 
                                                        patch_width))

    test_ds = zds.ZarrDataset(
        **groups,
        transform=None,
        patch_sampler=patch_sampler,
        return_any_label=True)

    test_dl = DataLoader(test_ds, num_workers=2, pin_memory=True,
                            worker_init_fn=zds.zarrdataset_worker_init)

    for i, (x, t) in enumerate(test_ds):
        print("Sample %i" % i, x.shape, x.dtype, type(t), t)
