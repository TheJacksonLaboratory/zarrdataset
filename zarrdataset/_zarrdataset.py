import math
import random

import numpy as np

import torch
from torch.utils.data import IterableDataset

from tqdm import tqdm

from ._utils import ImageLoader, connect_s3



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
    def __init__(self, filenames, source_format=".zarr", data_group="",
                 data_axes="XYZCT",
                 mask_group=None,
                 mask_data_axes=None,
                 transform=None,
                 patch_sampler=False,
                 shuffle=False,
                 progress_bar=False,                 
                 **kwargs):

        self._filenames = filenames
        if not source_format.startswith("."):
            source_format = "." + source_format

        self._source_format = source_format

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

    def _preload_files(self, filenames, data_group="", data_axes="XYZCT",
                       mask_group=None,
                       mask_data_axes=None,
                       compute_valid_mask=False):
        """Open a connection to the zarr file using Dask for lazy loading.

        If the mask group is passed, that group within each zarr is used to
        determine the valid regions that can be sampled. If None is passed, that
        means that the full image can be sampled.
        """
        z_list = []
        toplefts = []

        if self._progress_bar:
            q = tqdm(desc="Preloading files as dask arrays",
                     total=len(filenames))

        for fn in filenames:
            curr_img = ImageLoader(fn, data_group=data_group,
                                   data_axes=data_axes,
                                   mask_group=mask_group,
                                   mask_data_axes=mask_data_axes,
                                   source_format=self._source_format,
                                   s3_obj=self._s3_obj,
                                   compute_valid_mask=compute_valid_mask)

            # If a patch sampler was passed, it is used to determine the
            # top-left and bottom-right coordinates of the valid samples that
            # can be drawn from images.
            if compute_valid_mask and self._patch_sampler is not None:
                toplefts.append(self._patch_sampler.compute_toplefts(curr_img))

            else:
                toplefts.append([None])

            z_list.append(curr_img)

            if self._progress_bar:
                q.update()

        if self._progress_bar:
            q.close()

        z_list = np.array(z_list, dtype=object)
        toplefts = np.array(toplefts, dtype=object)
        dataset_size = sum(map(len, toplefts))

        return z_list, toplefts, dataset_size

    def _preload_inputs(self):
        (self._arr_list,
         self._toplefts,
         self._dataset_size) = self._preload_files(
            self._filenames,
            data_group=self._data_group,
            data_axes=self._data_axes,
            mask_group=self._mask_group,
            mask_data_axes=self._mask_data_axes,
            compute_valid_mask=True)

    def _initialize(self):
        if self._initialized:
            return

        # If the zarr files are stored in a S3 bucket, create a connection to
        # that bucket.
        self._s3_obj = connect_s3(self._filenames[0])

        self._preload_inputs()

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

        # Returns anything as label, this is just to return a tuple of input,
        # target that is expected for most of training pipelines.
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
            curr_topleft = self._toplefts[im_id]

            if self._shuffle:
                tlbr_indices = random.sample(range(len(curr_topleft)),
                                             len(curr_topleft))
            else:
                tlbr_indices = range(len(curr_topleft))

            for tlbr_id in tlbr_indices:
                yield self._getitem(im_id, curr_topleft[tlbr_id])

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

    def _preload_inputs(self):
        # Preload the input images
        super()._preload_inputs()

        # Preload the target labels
        self._lab_list, _, _ = self._preload_files(
            self._filenames,
            data_group=self._labels_data_group,
            data_axes=self._labels_data_axes,
            mask_group=None,
            mask_data_axes=None,
            compute_valid_mask=False)

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
