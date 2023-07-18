import random

import numpy as np

from ._utils import (connect_s3, map_axes_order, parse_metadata, parse_rois,
                     scale_coords)
from ._imageloaders import ImageLoader, MaskLoader

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
        w_sel = slice(worker_id, None, worker_info.num_workers)

        dataset_obj = worker_info.dataset

        # Reset the random number generators in each worker.
        torch_seed = torch.initial_seed()
        random.seed(torch_seed)
        np.random.seed(torch_seed % (2**32 - 1))

        # Open a copy of the dataset on each worker.
        n_files = len(dataset_obj._filenames["images"])
        if n_files == 1:
            # Get the topleft positions and distribute them among the workers
            dataset_obj._initialize(force=True)
            dataset_obj._toplefts[dataset_obj._mode_tls] =\
                [dataset_obj._toplefts[dataset_obj._mode_tls][0][w_sel]]

        else:
            modes = list(dataset_obj._filenames.keys())
            dataset_obj._filenames = dict((k, dataset_obj._filenames[k][w_sel])
                                          for k in modes)

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


def _preload_files(filenames, default_data_group=None,
                   default_source_axes=None,
                   default_axes=None,
                   default_roi=None,
                   image_loader=ImageLoader,
                   image_loader_opts=None,
                   patch_sampler=None,
                   s3_obj=None,
                   worker_id=0,
                   progress_bar=False):
    """Open a connection to the zarr file using Dask for lazy loading.

    If the mask group is passed, that group within each zarr is used to 
    determine the valid regions that can be sampled. If None is passed, that
    means that the full image can be sampled.
    """
    z_list = []
    toplefts = []

    if image_loader_opts is None:
        image_loader_opts = {}

    if default_roi is None:
        default_roi = [slice(None)]

    if progress_bar:
        q = tqdm(desc="Preloading zarr files", total=len(filenames),
                 position=worker_id)

    for fn in filenames:
        if progress_bar:
            q.set_description(f"Preloading image {fn}")

        # Separate the filename and any ROI passed as the name of the file
        (fn,
         parsed_data_group,
         parsed_source_axes,
         parsed_axes,
         parsed_rois) = parse_metadata(fn)

        if parsed_data_group is not None:
            data_group = parsed_data_group
        else:
            data_group = default_data_group

        if parsed_source_axes is not None:
            source_axes = parsed_source_axes
        else:
            source_axes = default_source_axes

        if parsed_axes is not None:
            axes = parsed_axes
        else:
            axes = default_axes

        if len(parsed_rois) > 0:
            rois = parse_rois(parsed_rois)
        else:
            rois = default_roi

        for roi in rois:
            curr_img = image_loader(fn, data_group=data_group,
                                    source_axes=source_axes,
                                    axes=axes,
                                    s3_obj=s3_obj,
                                    roi=roi,
                                    **image_loader_opts)

            # If a patch sampler was passed, it is used to determine the
            # top-left and bottom-right coordinates of the valid samples that
            # can be drawn from images.
            if patch_sampler is not None:
                toplefts.append(patch_sampler.compute_chunks(curr_img))
            else:
                toplefts.append(None)

            z_list.append(curr_img)

        if progress_bar:
            q.update()

    if progress_bar:
        q.close()

    z_list = np.array(z_list, dtype=object)

    if len(toplefts):
        # Use Numpy arrays to store the lists instead of python lists.
        # This is beacuase when lists are too big is easier to handle them as
        # numpy arrays.
        toplefts = np.array(toplefts, dtype=object)

    return z_list, toplefts


class ZarrDataset(IterableDataset):
    """A zarr-based dataset.

    Only two-dimensional (+color channels) data is supported by now.
    """
    def __init__(self, filenames, data_group="", source_axes="XYZCT",
                 axes=None,
                 roi=None,
                 transform=None,
                 patch_sampler=None,
                 shuffle=False,
                 progress_bar=False,
                 return_positions=False,
                 return_any_label=True,
                 draw_same_chunk=False,
                 **kwargs):

        self._worker_id = 0

        if not isinstance(filenames, list):
            filenames = [filenames]

        self._filenames = {"images": filenames}
        self._transforms = {("images", ): transform}
        self._transforms_order = [("images", )]
        self._output_order = ["images"]

        self._source_axes = {"images": source_axes}
        self._axes = {"images": axes}
        self._data_group = {"images": data_group}

        if roi is None:
            roi = ""

        if isinstance(roi, str):
            roi = [roi]

        self._roi = {"images": parse_rois(roi)}

        self._s3_obj = {"images": None}

        self._mode_tls = "images"
        self._arr_lists = {"images": []}
        self._toplefts = {"images": []}
        self._cached_chunks = {"images": None}
        self._img_scale = {"images": None}
        self._image_loader = {"images": ImageLoader}
        self._image_loader_opts = {"images": {}}

        self._shuffle = shuffle
        self._progress_bar = progress_bar

        self._return_positions = return_positions
        self._return_any_label = return_any_label
        self._draw_same_chunk = draw_same_chunk

        self._patch_sampler = patch_sampler

        self._initialized = False

    def _initialize(self, force=False):
        if self._initialized and not force:
            return

        # If the zarr files are stored in a S3 bucket, create a connection to
        # that bucket.

        for mode in self._filenames:
            self._s3_obj[mode] = connect_s3(self._filenames[mode][0])

            if mode == self._mode_tls:
                patch_sampler = self._patch_sampler
            else:
                patch_sampler = None

            (arr_list,
             toplefts) = _preload_files(
                self._filenames[mode],
                default_data_group=self._data_group[mode],
                default_source_axes=self._source_axes[mode],
                default_axes=self._axes[mode],
                default_roi=self._roi[mode],
                image_loader=self._image_loader[mode],
                image_loader_opts=self._image_loader_opts[mode],
                patch_sampler=patch_sampler,
                s3_obj=self._s3_obj[mode],
                worker_id=self._worker_id,
                progress_bar=self._progress_bar)

            self._arr_lists[mode] = arr_list
            self._toplefts[mode] = toplefts

        self._initialized = True

    def __getitem__(self, tlbr):
        patches = {}

        for mode in self._output_order:
            coords = tlbr if tlbr is not None else slice(None)
            coords = self._scale_mode_coords(mode, coords)
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
        for mode in self._output_order:
            coords = self._scale_mode_coords(mode, chunk_tlbr)
            self._cached_chunks[mode] = self._arr_lists[mode][im_id][coords]

    def _scale_mode_coords(self, mode, coords):
        ax_ref = self._arr_lists[self._mode_tls][0].axes
        sh_ref = self._arr_lists[self._mode_tls][0].shape

        ax_curr = self._arr_lists[mode][0].axes
        sh_curr = self._arr_lists[mode][0].shape
        curr_coords = []
        scale = []
        for a, s in zip(ax_curr, sh_curr):
            if a in ax_ref:
                ax_i = ax_ref.index(a)
                curr_coords.append(coords[ax_i])
                scale.append(s / sh_ref[ax_i])
            else:
                curr_coords.append(None)
                scale.append(1.0)

        mode_coords = scale_coords(curr_coords, scale)

        return mode_coords

    def __iter__(self):
        # Preload the files and masks associated with them
        self._initialize()

        samples = [
            [im_id, chk_id, None, None]
            for im_id in range(len(self._arr_lists["images"]))
            for chk_id in range(len(self._toplefts[self._mode_tls][im_id]))
            ]

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

            chunk_tlbr = tuple(self._toplefts[self._mode_tls][im_id][chk_id])

            # If this chunk is different from the cached one, change the
            # cached chunk for this one.
            if prev_im_id != im_id or prev_chk_id != chk_id:
                prev_im_id = im_id
                prev_chk_id = chk_id

                if self._patch_sampler is not None:
                    patches_tls = self._patch_sampler.compute_patches(
                        self._arr_lists[self._mode_tls][im_id],
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
            patch_tlbr = [tlbr for tlbr in patch_tlbr]
            patches = self.__getitem__(patch_tlbr)

            if self._return_positions:
                pos = [[tlbr.start + chk_tl.start, tlbr.stop + chk_tl.start]
                       for tlbr, chk_tl in zip(patch_tlbr, chunk_tlbr)]
                pos = np.array(pos, dtype=np.int64)
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
                 labels_source_axes="CYX",
                 input_target_transform=None,
                 target_transform=None,
                 **kwargs):

        kwargs["return_any_label"] = False
        super(LabeledZarrDataset, self).__init__(filenames, **kwargs)

        # Open the labels from the labels group
        self._data_group["target"] = labels_data_group
        self._source_axes["target"] = labels_source_axes

        # This is a transform that affects the geometry of the input, and then
        # it has to be applied to the target as well
        self._transforms[("images", "target")] = input_target_transform
        self._transforms_order.append(("images", "target"))

        # This is a transform that only affects the target
        self._transforms[("target", )] = target_transform
        self._transforms_order.append(("target",))

        self._output_order.append("target")

        self._arr_lists["target"] = []
        self._cached_chunks["target"] = None


class MaskedZarrDataset(ZarrDataset):
    """A masked dataset based on the zarr dataset class.
    """
    def __init__(self, filenames, mask_filenames=None, mask_data_group=None,
                 mask_source_axes=None,
                 mask_axes=None,
                 mask_roi=None,
                 mask_func=None,
                 mask_func_opts=None,
                 **kwargs):

        super(MaskedZarrDataset, self).__init__(filenames, **kwargs)
        if mask_filenames is not None:
            if not isinstance(mask_filenames, list):
                mask_filenames = [mask_filenames]

            self._filenames["masks"] = mask_filenames
        else:
            self._filenames["masks"] = filenames

        if mask_data_group is None:
            mask_data_group = self._data_group["images"]

        if mask_source_axes is None:
            mask_source_axes = self._source_axes["images"]

        if mask_axes is None:
            mask_axes = self._axes["images"]

        if mask_roi is None:
            parsed_mask_roi = self._roi["images"]
        else:
            if isinstance(mask_roi, str):
                mask_roi = [mask_roi]
            parsed_mask_roi = parse_rois(mask_roi)

        self._roi["masks"] = parsed_mask_roi

        self._data_group["masks"] = mask_data_group
        self._source_axes["masks"] = mask_source_axes
        self._axes["masks"] = mask_axes
        self._img_scale["masks"] = None

        self._cached_chunks["masks"] = None
        self._s3_obj["masks"] = None

        self._mode_tls = "masks"
        self._image_loader["masks"] = MaskLoader
        self._image_loader_opts["masks"] = {"mask_func": mask_func,
                                            "mask_func_opts": mask_func_opts}
