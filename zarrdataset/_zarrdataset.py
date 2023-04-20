import math
import random

import numpy as np
import zarr

import dask
import dask.array as da
from tqdm import tqdm

from skimage import transform, measure
from PIL import Image
import boto3
from io import BytesIO

import torch
from torch.utils.data import IterableDataset

from matplotlib.path import Path
from bridson import poisson_disc_samples


def load_image(filename, s3_obj=None):
    if s3_obj is not None:
        # Remove the end-point from the file name
        filename = "/".join(filename.split("/")[4:])
        im_bytes = s3_obj["s3"].get_object(Bucket=s3_obj["bucket_name"],
                                           Key=filename)["Body"].read()
        with Image.open(BytesIO(im_bytes)) as im_s3:
            arr = im_s3.convert("RGB")
    else:
        im = Image.open(filename, mode="r").convert("RGB")
    return im


def connect_s3(filename_sample):
    if (filename_sample.startswith("s3")
       or filename_sample.startswith("http")):
        endpoint = "/".join(filename_sample.split("/")[:3])
        s3_obj = dict(bucket_name=filename_sample.split("/")[3],
                      s3=boto3.client("s3", aws_access_key_id="",
                                      aws_secret_access_key="",
                                      region_name="us-east-2",
                                      endpoint_url=endpoint))

        s3_obj["s3"]._request_signer.sign = (lambda *args, **kwargs: None)
    else:
        s3_obj = None
        
    return s3_obj


def image2dask(arr_src, source_format, data_group, s3_obj=None):
    if (isinstance(arr_src, zarr.Group) or (isinstance(arr_src, str)
       and ".zarr" in source_format)):
        arr = da.from_zarr(arr_src, component=data_group)

    elif isinstance(arr_src, zarr.Array):
        # The array was already open from a zarr file
        arr = da.from_zarr(arr_src)

    elif (isinstance(arr_src, str) and ".zarr" not in source_format):
        # If the input is a path to an image stored in a format
        # supported by PIL, open it and use it as a numpy array.
        im = load_image(arr_src, s3_obj=s3_obj)
        channels = len(im.getbands())
        arr = da.from_delayed(dask.delayed(np.array)(im),
                              shape=(im.size[1], im.size[0], channels),
                              dtype=np.uint8)

    return arr, arr.shape


def parse_roi(filename, source_format):
    """Parse the filename and ROIs from `filename`.

    The filename and ROIs must be separated by a semicolon (;).
    Any number of ROIs are accepted. ROIs are expected to be passed as
    (start_coords:axis_lengths), in the axis order of the input data axes.

    Notes:
    ------
    An example of a ROI structure is the following.

    test_file.zarr;(0, 10, 0, 0, 0):(10, 10, 1, 1, 1)
    Will parse a ROI from \"test_file\" from 0:10 in the first axis, 10:20 in
    the second axis, 0:1 in the third to fifth axes.

    Parameters:
    ----------
    filename : str
        Path to the image.
    source_format : str
        Format of the input file.

    Returns
    -------
    fn : str
    rois : list of tuples
    """
    rois = []
    if isinstance(filename, str):
        split_pos = filename.lower().find(source_format)
        rois_str = filename[split_pos + len(source_format):]
        fn = filename[:split_pos + len(source_format)]
        rois_str = rois_str.split(";")[1:]

        for roi in rois_str:
            start_coords, axis_lengths = roi.split(":")

            start_coords = tuple([int(c.strip("\n\r ()"))
                                  for c in start_coords.split(",")])

            axis_lengths = tuple([int(ln.strip("\n\r ()"))
                                  for ln in axis_lengths.split(",")])

            roi_slices = tuple([slice(c_i, c_i + l_i, None) for c_i, l_i in
                                zip(start_coords, axis_lengths)])

            rois.append(roi_slices)

    return fn, rois


def map_axes_order(src_data_axes, dst_data_axes):
    dst_data_axes = list(filter(lambda sa: sa in src_data_axes, dst_data_axes))
    slices_order = [src_data_axes.index(a) for a in dst_data_axes]
    return slices_order


def get_spatial_axes_order(data_axes, spatial_axes="YX"):
    unused_axes = list(set(data_axes) - set(spatial_axes))
    transpose_order = map_axes_order(data_axes, unused_axes
                                                + list(spatial_axes))
    return transpose_order


def get_valid_mask(filename, shape, rois, data_axes, mask_group=None,
                   mask_data_axes=None):

    spatial_axes = get_spatial_axes_order(data_axes, "YX")
    default_mask_scale = 1 / min(shape[spatial_axes[-1]],
                                 shape[spatial_axes[-2]])

    # If the input file is stored in zarr format, try to retrieve the object
    # mask from the `mask_group`.
    if mask_group is not None and ".zarr" in filename:
        mask = da.from_zarr(filename, component=mask_group)

        tr_ord = get_spatial_axes_order(mask_data_axes, "YX")
        mask = mask.transpose(tr_ord).squeeze()
        mask = mask.compute(scheduler="synchronous")

    else:
        scaled_h = int(math.floor(shape[-2] * default_mask_scale))
        scaled_w = int(math.floor(shape[-1] * default_mask_scale))

        mask = np.ones((scaled_h, scaled_w), dtype=bool)

    scale = mask.shape[-1] / shape[-1]
    roi_mask = np.zeros_like(mask, dtype=bool)
    tr_ord = get_spatial_axes_order(data_axes, "YX")

    for roi in rois:
        if len(roi) >= 2:
            roi = [roi[a] for a in tr_ord]
            scaled_roi = (slice(int(math.ceil(roi[-2].start * scale)),
                                int(math.ceil(roi[-2].stop * scale)),
                                None),
                          slice(int(math.ceil(roi[-1].start * scale)),
                                int(math.ceil(roi[-1].stop * scale)),
                                None))
        else:
            scaled_roi = slice(None)

        roi_mask[scaled_roi] = True

    valid_mask = np.bitwise_and(mask, roi_mask)

    return valid_mask


class PatchSampler(object):
    def __init__(self, patch_size):
        self._patch_size = patch_size

    def _sampling_method(self, mask, shape):
        raise NotImplementedError("This is a virtual class and has to be "
                                  "inherited from other class implementing the"
                                  " actual sampling method.")

    def compute_toplefts(self, z_list):
        toplefts = []
        for im in z_list:
            curr_tls = self._sampling_method(im.mask, im.shape)
            curr_brs = curr_tls + self._patch_size
            curr_tls = np.hstack((curr_tls, curr_brs))
            toplefts.append(curr_tls)

        return np.array(toplefts, dtype=object)


class GridPatchSampler(PatchSampler):
    def __init__(self, patch_size, min_object_presence=0.1):
        super(GridPatchSampler, self).__init__(patch_size)
        self._min_object_presence = min_object_presence

    def _sampling_method(self, mask, shape):
        mask_scale = mask.shape[-1] / shape[-1]
        scaled_ps = self._patch_size * mask_scale

        if scaled_ps < 1:
            mask_scale = 1.0
            scaled_h = round(shape[-2] / self._patch_size)
            scaled_w = round(shape[-1] / self._patch_size)
            dwn_valid_mask = transform.resize(mask, (scaled_h, scaled_w),
                                              order=0,
                                              mode="edge",
                                              anti_aliasing=False)

        else:
            scaled_ps = round(scaled_ps)
            scaled_h = mask.shape[-2] // scaled_ps
            scaled_w = mask.shape[-1] // scaled_ps

            dwn_valid_mask = transform.downscale_local_mean(
                mask, factors=(scaled_ps, scaled_ps))

            dwn_valid_mask = dwn_valid_mask[:scaled_h, :scaled_w]

        valid_mask = dwn_valid_mask > self._min_object_presence

        toplefts = np.nonzero(valid_mask)
        toplefts = np.stack(toplefts, axis=1)
        toplefts = toplefts * self._patch_size

        return toplefts


class BlueNoisePatchSampler(PatchSampler):
    def __init__(self, patch_size, allow_overlap=True):
        super(BlueNoisePatchSampler, self).__init__(patch_size)
        self._overlap = math.sqrt(2) ** (not allow_overlap)

    def _sampling_method(self, mask, shape):
        mask_scale =  mask.shape[-1] / shape[-1]

        rad = self._patch_size * mask_scale
        H, W = mask.shape

        if H <= rad or W <= rad:
            sample_tls = np.array([[0, 0]], dtype=np.float32)

        else:
            sample_tls = np.array(poisson_disc_samples(height=H - rad,
                                                       width=W - rad,
                                                       r=rad * self._overlap,
                                                       k=30),
                                  dtype=np.float32)

        # If there are ROIs in the mask, take the sampling positions that
        # are inside them.
        if np.any(np.bitwise_not(mask)):
            validsample_tls = np.zeros(len(sample_tls), dtype=bool)
            mask_conts = measure.find_contours(np.pad(mask, 1),
                                               fully_connected='low',
                                               level=0.999)

            for cont in mask_conts:
                mask_path = Path(cont[:, (1, 0)] - 1)
                validsample_tls = np.bitwise_or(
                    validsample_tls,
                    mask_path.contains_points(sample_tls + rad / 2, 
                                              radius=rad))

            toplefts = sample_tls[validsample_tls]

        else:
            toplefts = sample_tls

        toplefts = np.round(toplefts / mask_scale).astype(np.int64)

        return toplefts


class DaskToArray(object):
    def __init__(self, use_multithread=False):
        self._scheduler = "threads" if use_multithread else "synchronous"

    def __call__(self, pic):
        return pic.compute(scheduler=self._scheduler)


class ImageLoader(object):
    """Image lazy loader class.

    Opens the zarr file, or any image that can be open by PIL, as a Dask array.
    """
    def __init__(self, filename, data_group, data_axes, mask_group=None,
                 mask_data_axes="YX",
                 source_format=".zarr",
                 s3_obj=None):

        # Separate the filename and any ROI passed as the name of the file
        filename, rois = parse_roi(filename, source_format)

        (self._arr,
         self.shape) = image2dask(filename, source_format, data_group, s3_obj)

        if len(rois) == 0:
            rois = [tuple([slice(0, s, None) for s in self.shape])]

        self.mask = get_valid_mask(filename, shape=self.shape, rois=rois,
                                    data_axes=data_axes,
                                    mask_group=mask_group,
                                    mask_data_axes=mask_data_axes)

    def __getitem__(self, index):
        return self._arr[index]


def preload_files(filenames, source_format, data_group="", data_axes="XYZCT",
                  mask_group=None,
                  mask_data_axes=None,
                  s3_obj=None,
                  progress_bar=False):
    """Open a connection to the zarr file using Dask for lazy loading.

    If the mask group is passed, that group within each zarr is used to
    determine the valid regions that can be sampled. If None is passed, that
    means that the full image can be sampled.
    """
    z_list = []

    if progress_bar:
        q = tqdm(desc="Preloading files as dask arrays", total=len(filenames))

    for fn in filenames:
        z_list.append(ImageLoader(fn, data_group=data_group,
                                    data_axes=data_axes,
                                    mask_group=mask_group,
                                    mask_data_axes=mask_data_axes,
                                    source_format=source_format,
                                    s3_obj=s3_obj))

        if progress_bar:
            q.update()

    if progress_bar:
        q.close()

    return np.array(z_list, dtype=object)


def zarrdataset_worker_init(worker_id):
    worker_info = torch.utils.data.get_worker_info()
    dataset_obj = worker_info.dataset

    # Reset the random number generators in each worker.
    torch_seed = torch.initial_seed()
    random.seed(torch_seed)
    np.random.seed(torch_seed % (2**32 - 1))

    # Open a copy of the dataset on each worker.
    n_files = len(dataset_obj._filenames)
    n_files_per_worker = int(math.ceil(n_files / worker_info.num_workers))
    dataset_obj._filenames = \
        dataset_obj._filenames[slice(n_files_per_worker * worker_id,
                                     n_files_per_worker * (worker_id + 1),
                                     None)]
    dataset_obj._initialize()


class ZarrDataset(IterableDataset):
    """A zarr-based dataset.

    Only two-dimensional (+color channels) data is supported by now.
    """
    def __init__(self, filenames, data_group="",
                 data_axes="XYZCT",
                 mask_group=None,
                 mask_data_axes=None,
                 transform=None,
                 patch_sampler=False,
                 shuffle=False,
                 progress_bar=False,
                 **kwargs):

        self._filenames = filenames
        
        self._transform = transform

        self._data_axes = data_axes
        self._data_group = data_group

        self._mask_group = mask_group
        self._mask_data_axes = mask_data_axes

        self._shuffle = shuffle
        self._progress_bar = progress_bar

        self._arr_list = []
        self._patch_sampler = patch_sampler
        self._initialized = False
        self._dataset_size = 0

    def _preload_files(self):
        # If the zarr files are stored in a S3 bucket, create a connection to
        # that bucket.
        self._s3_obj = connect_s3(self._filenames[0])

        self._arr_list = preload_files(self._filenames, source_format=".zarr",
                                    data_group=self._data_group,
                                    data_axes=self._data_axes,
                                    mask_group=self._mask_group,
                                    mask_data_axes=self._mask_data_axes,
                                    s3_obj=self._s3_obj,
                                    progress_bar=self._progress_bar)

        # If a patch sampler was passed, it is used to determine the top-left
        # and bottom-right coordinates of the valid samples that can be
        # retrieved from images. If it is not passed, the full image is
        # returned.
        if self._patch_sampler is not None:
            self._toplefts = self._patch_sampler.compute_toplefts(
                self._arr_list)
            self._dataset_size = sum(map(len, self._toplefts))

        else:
            self._toplefts = None
            self._dataset_size = len(self._arr_list)

    def _initialize(self):
        if self._initialized:
            return

        self._preload_files()

        self._initialized = True

    def _get_coords(self, tlbr, data_axes):
        if tlbr is None:
            return slice(None)

        tl_y, tl_x, br_y, br_x = tlbr
        coords = []
        for a in data_axes:
            if a == "Y":
                coords.append(slice(tl_y, br_y, None))
            elif a == "X":
                coords.append(slice(tl_x, br_x, None))
            else:
                coords.append(slice(None))

        return tuple(coords)

    def _getitem(self, im_id, tlbr):
        coords = self._get_coords(tlbr, self._data_axes)
        patch = self._arr_list[im_id][coords]

        if self._transform is not None:
            patch = self._transform(patch)

        # Returns anything as label, to prevent an error during training
        return patch, 0

    def __iter__(self):
        # Preload the files and masks associated with them
        self._initialize()

        if self._shuffle:
            im_indices = random.sample(range(len(self._arr_list)),
                                       len(self._arr_list))
        else:
            im_indices = range(len(self._arr_list))

        for im_id in im_indices:
            if self._toplefts is not None:
                if self._shuffle:
                    tlbr_indices = random.sample(
                        range(len(self._toplefts[im_id])),
                        len(self._toplefts[im_id]))
                else:
                    tlbr_indices = range(len(self._toplefts[im_id]))

                for tlbr_id in tlbr_indices:
                    yield self._getitem(im_id, self._toplefts[im_id][tlbr_id])

            else:
                yield self._getitem(im_id, None)

    def __len__(self):
        return self._dataset_size


class LabeledZarrDataset(ZarrDataset):
    """A labeled dataset based on the zarr dataset class.
    The densely labeled targets are extracted from group "labels_data_group".
    """
    def __init__(self, filenames, labels_data_group="labels/0/0",
                 labels_data_axes="XYC",
                 input_target_transform=None,
                 target_transform=None,
                 **kwargs):

        # Open the labels from the labels group
        self._labels_data_group = labels_data_group
        self._labels_data_axes = labels_data_axes

        # This is a transform that affects the geometry of the input, and then
        # it has to be applied to the target as well
        self._input_target_transform = input_target_transform

        # This is a transform that only affects the target
        self._target_transform = target_transform

        self._lab_list = []

        super(LabeledZarrDataset, self).__init__(filenames, **kwargs)

    def _preload_files(self):
        # Preload the input images
        super()._preload_files()

        # Preload the target labels
        self._lab_list = preload_files(self._filenames, source_format=".zarr",
                                       data_group=self._labels_data_group,
                                       data_axes=self._labels_data_axes,
                                       mask_group=None,
                                       mask_data_axes=None,
                                       s3_obj=self._s3_obj,
                                       progress_bar=self._progress_bar)

    def _getitem(self, im_id, tlbr):
        coords = self._get_coords(tlbr, self._data_axes)
        patch = self._arr_list[im_id][coords]

        coords = self._get_coords(tlbr, self._labels_data_axes)
        target = self._lab_list[im_id][coords]

        # Transform the input with non-spatial transforms
        if self._transform is not None:
            patch = self._transform(patch)

        # Transform the input and target with the same spatial transforms
        if self._input_target_transform:
            patch, target = self._input_target_transform((patch, target))

        # Transform the target with the target-only transforms
        if self._target_transform:
            target = self._target_transform(target)

        return patch, target


if __name__ == "__main__":
    import argparse
    import os
    from torch.utils.data import DataLoader, BatchSampler, random_split

    parser = argparse.ArgumentParser("Zarr-based data loader demo")
    parser.add_argument("-dd", "--data-dir", dest="data_dir",
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

    if os.path.isdir(args.data_dir):
        filenames = [os.path.join(args.data_dir, fn)
                    for fn in sorted(os.listdir(args.data_dir))]
    elif args.data_dir.endswith(".txt"):
        with open(args.data_dir, "r") as fp:
            filenames = [fn.strip("\n ") for fn in fp.readlines()]

    if 'grid' in args.sample_method:
        patch_sampler = GridPatchSampler(args.patch_size,
                                         min_object_presence=0.1)

    elif 'blue-noise' in args.sample_method:
        patch_sampler = BlueNoisePatchSampler(args.patch_size)

    else:
        patch_sampler = None

    transform_fn = DaskToArray(True)

    if args.labels_data_group is not None:
        my_dataset = LabeledZarrDataset(
            filenames,
            data_axes=args.data_axes,
            data_group=args.data_group,
            labels_data_group=args.labels_data_group,
            labels_data_axes=args.labels_data_axes,
            mask_group=args.mask_group,
            mask_data_axes=args.mask_data_axes,
            transform=transform_fn,
            target_transform=transform_fn,
            patch_sampler=patch_sampler,
            num_workers=args.num_workers,
            shuffle=True,
            progress_bar=True)

    else:
        my_dataset = ZarrDataset(filenames, data_axes=args.data_axes,
                                 data_group=args.data_group,
                                 mask_group=args.mask_group,
                                 mask_data_axes=args.mask_data_axes,
                                 transform=transform_fn,
                                 patch_sampler=patch_sampler,
                                 num_workers=args.num_workers,
                                 shuffle=True,
                                 progress_bar=True)

    my_dataloader = DataLoader(my_dataset, batch_size=args.batch_size,
                               num_workers=args.num_workers,
                               worker_init_fn=zarrdataset_worker_init,
                               persistent_workers=True)

    for e in range(3):
        for i, (x, t) in enumerate(my_dataloader):
            print("Sample %i" % i, x.shape, x.dtype, t.shape, t.dtype)
