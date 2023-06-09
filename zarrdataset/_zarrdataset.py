import math
import random

import numpy as np

from ._utils import ImageLoader, connect_s3, map_axes_order

try:
    from tqdm import tqdm
except ModuleNotFoundError:
    # This removes the dependency on tqdm when it is not installed
    class tqdm(object):
        def __init__(self, *args, **kwargs):
            pass

        def update(self, *args, **kwargs):
            pass

        def set_description(self, *args, **kwargs):
            pass

        def close(self, *args, **kwargs):
            pass


try:
    import torch
    from torch.utils.data import IterableDataset

    def zarrdataset_worker_init(worker_id):
        worker_info = torch.utils.data.get_worker_info()
        dataset_obj = worker_info.dataset

        # Reset the random number generators in each worker.
        torch_seed = torch.initial_seed()
        random.seed(torch_seed)
        np.random.seed(torch_seed % (2**32 - 1))

        # Open a copy of the dataset on each worker.
        n_files = len(dataset_obj._filenames)
        if n_files == 1:
            # Get the topleft positions and distribute them among the workers
            dataset_obj._initialize(force=True)
            dataset_obj._toplefts["images"] =\
                [dataset_obj._toplefts["images"][0][slice(worker_id, None,
                                                          worker_info.num_workers)]]

        else:
            dataset_obj._filenames =\
                dataset_obj._filenames[worker_id::worker_info.num_workers]

        dataset_obj._worker_id = worker_id

    def chained_zarrdataset_worker_init(worker_id):
        worker_info = torch.utils.data.get_worker_info()
        dataset_obj = worker_info.dataset

        # Reset the random number generators in each worker.
        torch_seed = torch.initial_seed()
        random.seed(torch_seed)
        np.random.seed(torch_seed % (2**32 - 1))

        # Open a copy of the dataset on each worker.
        dataset_obj.datasets = \
            dataset_obj.datasets[worker_id::worker_info.num_workers]

        for ds in dataset_obj.datasets:
            ds._worker_id = worker_id

except ModuleNotFoundError:
    import logging
    logging.warning('PyTorch is not installed, the ZarrDataset class will '
                    'still work as a python iterator')
    IterableDataset = object

    def zarrdataset_worker_init(*args):
        pass

    def chained_zarrdataset_worker_init(*args):
        pass


class ZarrDataset(IterableDataset):
    """A zarr-based dataset.

    Only two-dimensional (+color channels) data is supported by now.
    """
    def __init__(self, filenames, data_group="", data_axes="XYZCT",
                 mask_data_group=None,
                 mask_data_axes=None,
                 transform=None,
                 patch_sampler=None,
                 shuffle=False,
                 progress_bar=False,
                 use_dask=False,
                 return_positions=False,
                 return_any_label=True,
                 draw_same_chunk=False,
                 force_compute_valid_mask=False,
                 **kwargs):

        self._worker_id = 0

        if not isinstance(filenames, list):
            filenames = [filenames]

        self._filenames = filenames
        self._use_dask = use_dask
        self._transforms = {("images", ): transform}
        self._transforms_order = [("images", )]
        self._output_order = ["images"]

        self._data_axes = {"images": data_axes}
        self._data_group = {"images": data_group}

        self._mask_data_group = {"images": mask_data_group}
        self._mask_data_axes = {"images": mask_data_axes}

        self._compute_valid_mask = {
            "images": force_compute_valid_mask or mask_data_group is not None
        }

        self._arr_lists = {}
        self._toplefts = {}
        self._cached_chunks = {}
        self._img_scale = {}

        self._shuffle = shuffle
        self._progress_bar = progress_bar

        self._return_positions = return_positions
        self._return_any_label = return_any_label
        self._draw_same_chunk = draw_same_chunk

        self._patch_sampler = patch_sampler
        self._initialized = False

    def _preload_files(self, filenames, data_group="", data_axes="XYZCT",
                       mask_data_group=None,
                       mask_data_axes=None,
                       compute_valid_mask=False):
        """Open a connection to the zarr file using Dask for lazy loading.

        If the mask group is passed, that group within each zarr is used to
        determine the valid regions that can be sampled. If None is passed,
        that means that the full image can be sampled.
        """
        z_list = []
        toplefts = []

        if self._progress_bar:
            q = tqdm(desc="Preloading zarr files",
                     total=len(filenames),
                     position=self._worker_id)

        for fn in filenames:
            if self._progress_bar:
                q.set_description(f"Preloading image {fn}")

            curr_img = ImageLoader(fn, data_group=data_group,
                                   data_axes=data_axes,
                                   mask_data_group=mask_data_group,
                                   mask_data_axes=mask_data_axes,
                                   s3_obj=self._s3_obj,
                                   compute_valid_mask=compute_valid_mask,
                                   use_dask=self._use_dask)

            # If a patch sampler was passed, it is used to determine the
            # top-left and bottom-right coordinates of the valid samples that
            # can be drawn from images.
            if self._patch_sampler is not None:
                curr_toplefts = self._patch_sampler.compute_chunks(curr_img)
                toplefts.append(curr_toplefts)
            else:
                ax_ref_ord = map_axes_order(curr_img.data_axes, "YX")

                if "Y" in curr_img.data_axes:
                    H = curr_img.shape[ax_ref_ord[-2]]
                else:
                    H = 1

                if "X" in curr_img.data_axes:
                    W = curr_img.shape[ax_ref_ord[-1]]
                else:
                    W = 1

                toplefts.append(np.array([[0, 0, H, W]], dtype=np.int64))

            z_list.append(curr_img)

            if self._progress_bar:
                q.update()

        if self._progress_bar:
            q.close()

        z_list = np.array(z_list, dtype=object)

        if len(toplefts):
            # Use Numpy arrays to store the lists instead of python lists.
            # This is beacuase when lists are too big is easier to handle them
            # as numpy arrays.
            toplefts = np.array(toplefts, dtype=object)

        return z_list, toplefts

    def _initialize(self, force=False):
        if self._initialized and not force:
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
                mask_data_group=self._mask_data_group.get(mode, None),
                mask_data_axes=self._mask_data_axes.get(mode, None),
                compute_valid_mask=self._compute_valid_mask.get(mode, False))

            self._arr_lists[mode] = arr_list
            self._toplefts[mode] = toplefts

        self._initialized = True

    def _get_coords(self, tlbr, data_axes, scale=1):
        if tlbr is None:
            return slice(None)

        tl_y, tl_x, br_y, br_x = tlbr

        coords = []
        for a in data_axes:
            if a == "Y":
                coords.append(slice(int(tl_y * scale),
                                    int(br_y * scale), None))
            elif a == "X":
                coords.append(slice(int(tl_x * scale),
                                    int(br_x * scale), None))
            else:
                coords.append(slice(None))

        return tuple(coords)

    def _getitem(self, tlbr):
        patches = {}

        for mode in self._output_order:
            coords = self._get_coords(tlbr, self._data_axes[mode],
                                      self._img_scale[mode])
            patches[mode] = self._cached_chunks[mode][coords]

        for inputs in self._transforms_order:
            if self._transforms[inputs] is not None:
                res = self._transforms[inputs](*tuple(patches[mode]
                                                      for mode in inputs))
                if not isinstance(res, tuple):
                    res = (res, )

                for mode, mode_res in zip(inputs, res):
                    patches[mode] = mode_res

        patches = [patches[mode] for mode in self._output_order]

        if "target" not in self._output_order and self._return_any_label:
            # Returns anything as label, this is just to return a tuple of
            # input, target that is expected for most of training pipelines.
            patches.append(0)

        return patches

    def _cache_chunk(self, im_id, chunk_tlbr):
        ax_ref = map_axes_order(self._data_axes["images"], "Y")
        H_ax_ref = ax_ref[-1]
        H_ref = self._arr_lists["images"][im_id].shape[H_ax_ref]

        for mode in self._arr_lists.keys():
            if "Y" in self._data_axes[mode]:
                ax_img = map_axes_order(self._data_axes[mode], "Y")
                H_ax_img = ax_img[-1]
                H_img = self._arr_lists[mode][im_id].shape[H_ax_img]

                scale =  H_img / H_ref
            else:
                scale =  1 / H_ref

            coords = self._get_coords(
                chunk_tlbr,
                self._data_axes[mode],
                scale)

            self._img_scale[mode] = scale

            self._cached_chunks[mode] = \
                self._arr_lists[mode][im_id][coords]

    def __iter__(self):
        # Preload the files and masks associated with them
        self._initialize()

        samples = [[im_id, chk_id, None, None]
                   for im_id in range(len(self._arr_lists["images"]))
                   for chk_id in range(len(self._toplefts["images"][im_id]))]

        # When chunks must be depleted before moving to the next chunk, shuffle
        # all before hand.
        if self._shuffle and self._draw_same_chunk:
            random.shuffle(samples)

        prev_im_id = -1
        prev_chk_id = -1
        curr_chk = 0

        while samples:
            # When chunks can be sampled even when these are not depleted,
            # shuffle here.
            if self._shuffle and not self._draw_same_chunk:
                curr_chk = random.randrange(0, len(samples))

            im_id = samples[curr_chk][0]
            chk_id = samples[curr_chk][1]

            chunk_tlbr = self._toplefts["images"][im_id][chk_id]
            chunk_tlbr = chunk_tlbr.astype(np.int64)

            # If this chunk is different from the cached one, change the
            # cached chunk for this one.
            if prev_im_id != im_id or prev_chk_id != chk_id:
                prev_im_id = im_id
                prev_chk_id = chk_id

                if self._patch_sampler is not None:
                    patches_tls = self._patch_sampler.compute_patches(
                        self._arr_lists["images"][im_id],
                        chunk_tlbr
                    )

                else:
                    patches_tls = [chunk_tlbr]

                if not len(patches_tls):
                    samples.pop(curr_chk)
                    continue

                self._cache_chunk(im_id, chunk_tlbr)

            # Initialize the count of top-left positions for patches inside
            # this chunk.
            if samples[curr_chk][2] is None:
                samples[curr_chk][2] = len(patches_tls)
                samples[curr_chk][3] = 0

            if self._shuffle:
                curr_patch = random.randrange(0, samples[curr_chk][2])
            else:
                curr_patch = samples[curr_chk][3]

            samples[curr_chk][3] = samples[curr_chk][3] + 1

            # When all possible patches have been extracted from the current
            # chunk, remove that chunk from the list of samples.
            if samples[curr_chk][3] == samples[curr_chk][2]:
                samples.pop(curr_chk)

            patch_tlbr = patches_tls[curr_patch]
            patches = self._getitem(patch_tlbr)

            if self._return_positions:
                pos = np.copy(patch_tlbr)
                pos += np.broadcast_to(chunk_tlbr[:2], (2, 2)).reshape(4)
                patches = [pos] + patches

            if len(patches) > 1:
                patches = tuple(patches)
            else:
                patches = patches[0]

            yield patches


class LabeledZarrDataset(ZarrDataset):
    """A labeled dataset based on the zarr dataset class.
    The densely labeled targets are extracted from group "labels_data_group".
    """
    def __init__(self, filenames, labels_data_group="labels/0/0",
                 labels_data_axes="CYX",
                 input_target_transform=None,
                 target_transform=None,
                 **kwargs):

        kwargs["return_any_label"] = False
        super(LabeledZarrDataset, self).__init__(filenames, **kwargs)

        # Open the labels from the labels group
        self._data_group["target"] = labels_data_group
        self._data_axes["target"] = labels_data_axes

        # This is a transform that affects the geometry of the input, and then
        # it has to be applied to the target as well
        self._transforms[("images", "target")] = input_target_transform
        self._transforms_order.append(("images", "target"))

        # This is a transform that only affects the target
        self._transforms[("target", )] = target_transform
        self._transforms_order.append(("target",))

        self._output_order.append("target")

        self._arr_lists["target"] = []
        self._compute_valid_mask["target"] = False
        self._cached_chunks["target"] = None
