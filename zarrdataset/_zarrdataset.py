from functools import reduce
from itertools import repeat
import math
import random

import numpy as np

import torch
from torch.utils.data import IterableDataset
import dask.array as da

from tqdm import tqdm

from ._utils import ImageLoader, connect_s3


def zarrdataset_worker_init(worker_id):
    worker_info = torch.utils.data.get_worker_info()
    dataset_obj = worker_info.dataset
    dataset_obj._worker_id = worker_id

    # Reset the random number generators in each worker.
    torch_seed = torch.initial_seed()
    random.seed(torch_seed)
    np.random.seed(torch_seed % (2**32 - 1))

    # Open a copy of the dataset on each worker.
    n_files = len(dataset_obj._filenames)
    if n_files == 1:
        # Get the topleft positions and distribute them among the workers
        dataset_obj._initialize()

        n_tls = len(dataset_obj._toplefts["images"][0])
        n_tls_per_worker = int(math.ceil(n_tls / worker_info.num_workers))

        dataset_obj._toplefts["images"] =\
            [dataset_obj._toplefts["images"][0][
                slice(n_tls_per_worker * worker_id,
                      n_tls_per_worker * (worker_id + 1),
                      None)]]

    else:
        n_files_per_worker = int(math.ceil(n_files / worker_info.num_workers))
        dataset_obj._filenames =\
            dataset_obj._filenames[slice(n_files_per_worker * worker_id,
                                         n_files_per_worker * (worker_id + 1),
                                         None)]


def chained_zarrdataset_worker_init(worker_id):
    worker_info = torch.utils.data.get_worker_info()
    dataset_obj = worker_info.dataset

    # Reset the random number generators in each worker.
    torch_seed = torch.initial_seed()
    random.seed(torch_seed)
    np.random.seed(torch_seed % (2**32 - 1))

    # Open a copy of the dataset on each worker.
    n_datasets = len(dataset_obj.datasets)
    n_datasets_per_worker = int(math.ceil(n_datasets
                                          / worker_info.num_workers))

    dataset_obj.datasets = \
        dataset_obj.datasets[slice(n_datasets_per_worker * worker_id,
                                   n_datasets_per_worker * (worker_id + 1),
                                   None)]

    for ds in dataset_obj.datasets:
        ds._worker_id = worker_id


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
                 use_dask=False,
                 return_positions=False,
                 **kwargs):

        self._worker_id = 0

        if not isinstance(filenames, list):
            filenames = [filenames]

        self._filenames = filenames
        if not source_format.startswith("."):
            source_format = "." + source_format

        self._source_format = source_format
        self._use_dask = use_dask
        self._transforms = {("images", ): transform}
        self._transforms_order = [("images", )]
        self._output_order = ["images"]

        self._data_axes = {"images": data_axes}
        self._data_group = {"images": data_group}

        self._mask_group = {"images": mask_group}
        self._mask_data_axes = {"images": mask_data_axes}

        self._compute_valid_mask = {"images": True}
        self._arr_lists = {}
        self._toplefts = {}
        self._cached_chunks = {}

        self._shuffle = shuffle
        self._progress_bar = progress_bar

        self._return_positions = return_positions

        self._patch_sampler = patch_sampler
        self._initialized = False

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
            q = tqdm(desc="Preloading zarr files",
                     total=len(filenames),
                     position=self._worker_id)

        for fn in filenames:
            curr_img = ImageLoader(fn, data_group=data_group,
                                   data_axes=data_axes,
                                   mask_group=mask_group,
                                   mask_data_axes=mask_data_axes,
                                   source_format=self._source_format,
                                   s3_obj=self._s3_obj,
                                   compute_valid_mask=compute_valid_mask,
                                   use_dask=self._use_dask)

            # If a patch sampler was passed, it is used to determine the
            # top-left and bottom-right coordinates of the valid samples that
            # can be drawn from images.
            if compute_valid_mask and self._patch_sampler is not None:
                curr_toplefts = self._patch_sampler.compute_chunks(curr_img)

                toplefts.append(curr_toplefts)

            z_list.append(curr_img)

            if self._progress_bar:
                q.update()

        if self._progress_bar:
            q.close()

        z_list = np.array(z_list, dtype=object)

        if len(toplefts):
            toplefts = np.stack(toplefts)

        return z_list, toplefts

    def _initialize(self):
        if self._initialized:
            return

        # If the zarr files are stored in a S3 bucket, create a connection to
        # that bucket.
        self._s3_obj = connect_s3(self._filenames[0])

        for mode in self._output_order:
            (arr_list,
             toplefts) = self._preload_files(
                self._filenames,
                data_group=self._data_group[mode],
                data_axes=self._data_axes.get(mode, None),
                mask_group=self._mask_group.get(mode, None),
                mask_data_axes=self._mask_data_axes.get(mode, None),
                compute_valid_mask=self._compute_valid_mask.get(mode, False))

            self._arr_lists[mode] = arr_list
            self._toplefts[mode] = toplefts

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

    def _getitem(self, tlbr):
        patches = {}

        for mode in self._output_order:
            coords = self._get_coords(tlbr, self._data_axes[mode])
            patches[mode] = self._cached_chunks[mode][coords]

        for inputs in self._transforms_order:
            if self._transforms[inputs] is not None:
                res = self._transforms[inputs](*tuple(patches[mode]
                                                      for mode in inputs))
                if not isinstance(res, tuple):
                    res = (res, )

                for mode, mode_res in zip(inputs, res):
                    patches[mode] = mode_res

        patches = tuple(patches[mode] for mode in self._output_order)

        if "target" not in self._output_order:
            # Returns anything as label, this is just to return a tuple of
            # input, target that is expected for most of training pipelines.
            patches = (*patches, 0)

        return patches

    def __iter__(self):
        # Preload the files and masks associated with them
        self._initialize()

        im_tlbrs = [list(zip(repeat(im_id), list(range(len(tlbr)))))
                    for im_id, tlbr in enumerate(self._toplefts["images"])]
        im_tlbrs = reduce(lambda l1, l2: l1 + l2, im_tlbrs)

        if self._shuffle:
            random.shuffle(im_tlbrs)

        n_patches = 0

        while n_patches or im_tlbrs:
            if not n_patches:
                im_id, chk_id = im_tlbrs.pop(0)

                # Cache the current chunk
                curr_chk_tlbr = self._toplefts["images"][im_id][chk_id]

                top_lefts = self._patch_sampler.compute_patches(
                    self._arr_lists["images"][im_id],
                    self._toplefts["images"][im_id][chk_id])

                n_patches = len(top_lefts)

                if n_patches:
                    for mode in self._arr_lists.keys():
                        coords = self._get_coords(curr_chk_tlbr,
                                                self._data_axes[mode])

                        self._cached_chunks[mode] = \
                            self._arr_lists[mode][im_id][coords]

            else:
                curr_tlbr = top_lefts[len(top_lefts) - n_patches]
                patches = self._getitem(curr_tlbr)

                if self._return_positions:
                    patches = tuple([curr_tlbr.astype(np.int64) + curr_chk_tlbr]
                                    + list(patches))

                n_patches -= 1

                yield patches


class LabeledZarrDataset(ZarrDataset):
    """A labeled dataset based on the zarr dataset class.
    The densely labeled targets are extracted from group "labels_data_group".
    """
    def __init__(self, filenames, labels_data_group="labels/0/0",
                 labels_data_axes="XYC",
                 input_target_transform=None,
                 target_transform=None,
                 **kwargs):

        super(LabeledZarrDataset, self).__init__(filenames, **kwargs)

        # Open the labels from the labels group
        self._data_group["target"] = labels_data_group
        self._data_axes["target"] = labels_data_axes

        # This is a transform that affects the geometry of the input, and then
        # it has to be applied to the target as well
        self._transform[("images", "target")] = input_target_transform
        self._transforms_order.append(("images", "target"))

        # This is a transform that only affects the target
        self._transform[("target", )] = target_transform
        self._transforms_order.append(("target",))

        self._output_order.append("target")

        self._arr_lists["target"] = []
        self._compute_valid_mask["target"] = False
        self._cached_chunks["target"] = None
