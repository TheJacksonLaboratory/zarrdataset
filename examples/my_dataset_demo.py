import argparse
import os
import numpy as np
import torch
import torchvision
from torch.utils.data import DataLoader

import zarrdataset as zds


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Zarr-based data loader demo")
    parser.add_argument("-dd", "--data-dir", dest="data_dir", nargs="+",
                        type=str,
                        help="Directory where the images are stored",
                        default=".")
    parser.add_argument("-sf", "--source-format", dest="source_format",
                        type=str,
                        help="Format of the images to be used as dataset. "
                             "If non-zarr images are used, these must be able "
                             "to be open by PIL",
                        default=".zarr")
    parser.add_argument("-dg", "--data-group", dest="data_group",
                        type=str,
                        help="Group within the zarr file to be used as input",
                        default="")
    parser.add_argument("-da", "--data-axes", dest="data_axes",
                        type=str,
                        help="Order in which the axes of the image are stored."
                             " The default order is the given by the OME "
                             "standard.",
                        default="XYZCT")
    parser.add_argument("-ldg", "--labels-data-group",
                        dest="labels_data_group",
                        type=str,
                        help="Group within the zarr file where the labels are "
                             "stored. If not set, the normal non-labeled "
                             "dataset is used.",
                        default=None)
    parser.add_argument("-lda", "--labels-data-axes", dest="labels_data_axes",
                        type=str,
                        help="Order in which the axes of the labels are stored"
                             ". The default order is the given by the OME "
                             "standard.",
                        default="XYZCT")
    parser.add_argument("-mdg", "--mask-group", dest="mask_group",
                        type=str,
                        help="Group within the zarr file where the masks are "
                             "stored. If not set, a simplified mask is "
                             "generated to use all the image.",
                        default=None)
    parser.add_argument("-mda", "--mask-data-axes", dest="mask_data_axes",
                        type=str,
                        help="Order in which the axes of the masks are stored"
                             ". The default order is XY for the spatial axes "
                             "of the OME standard.",
                        default="XY")
    parser.add_argument("-ps", "--patch-size", dest="patch_size",
                        type=int,
                        help="Size of the patches extracted from the images.",
                        default=256)
    parser.add_argument("-bs", "--batch-size", dest="batch_size",
                        type=int,
                        help="Size of the mini batches used for training a DL "
                             "model.",
                        default=4)
    parser.add_argument("-nw", "--num-workers", dest="num_workers",
                        type=int,
                        help="Number of workers used to load the dataset.",
                        default=0)
    parser.add_argument("-sam", "--sample-method", dest="sample_method",
                        type=str,
                        help="Patches sampling method.",
                        choices=["grid", "blue-noise", "none"],
                        default="grid")

    args = parser.parse_args()

    torch.manual_seed(777)

    # Parse the input filenames, either they are passed as asolute path/url, a
    # directory, a text file with a list of filenames, or a any combination.
    filenames = []
    for dd in args.data_dir:
        if os.path.isdir(dd):
            filenames += [os.path.join(dd, fn)
                          for fn in sorted(os.listdir(dd)) if 
                          args.source_data in fn.lower()]

        elif dd.endswith(".txt"):
            with open(dd, "r") as fp:
                filenames += [fn.strip("\n ") for fn in fp.readlines()]

        elif ".zarr" in dd:
            filenames.append(dd)

    if 'grid' in args.sample_method:
        patch_sampler = zds.GridPatchSampler(args.patch_size,
                                             min_object_presence=0.1)

    elif 'blue-noise' in args.sample_method:
        patch_sampler = zds.BlueNoisePatchSampler(args.patch_size,
                                                  chunk=10 * args.patch_size)

    else:
        patch_sampler = None

    transform_fn = torchvision.transforms.Compose([
        zds.SelectAxes(source_axes=args.data_axes,
                       axes_selection={"T": 0, "Z": 0},
                       target_axes="CYX"),
        zds.ZarrToArray(np.int32),
    ])

    if args.labels_data_group is not None:
        my_dataset = zds.LabeledZarrDataset(
            filenames,
            transform=transform_fn,
            target_transform=transform_fn,
            patch_sampler=patch_sampler,
            shuffle=True,
            progress_bar=True,
            **args.__dict__)

    else:
        my_dataset = zds.ZarrDataset(
            filenames,
            transform=transform_fn,
            patch_sampler=patch_sampler,
            shuffle=True,
            progress_bar=True,
            **args.__dict__)

    my_dataloader = DataLoader(my_dataset, batch_size=args.batch_size,
                               num_workers=args.num_workers,
                               worker_init_fn=zds.zarrdataset_worker_init,
                               persistent_workers=args.num_workers > 0)

    for i, (x, t) in enumerate(my_dataloader):
        print("Sample %i" % i, x.shape, x.dtype, t.shape, t.dtype)
