import zarr
from functools import reduce, partial
import operator
import random

import numpy as np

from ._utils import parse_metadata
from ._imageloaders import ImageCollection

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
        n_files = len(dataset_obj._collections["images"])
        if n_files == 1:
            # Get the topleft positions and distribute them among the workers
            dataset_obj._initialize(force=True)
            dataset_obj._toplefts =\
                [dataset_obj._toplefts[0][w_sel]]

        else:
            modes = list(dataset_obj._collections.keys())
            dataset_obj._collections = dict(
                (k, dataset_obj._collections[k][w_sel])
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


class ZarrDatasetBase(IterableDataset):
    """A zarr-based dataset.

    Only two-dimensional (+color channels) data is supported by now.
    """
    def __init__(self, filenames, source_axes, axes=None, data_group=None,
                 roi=None,
                 transform=None,
                 patch_sampler=None,
                 shuffle=False,
                 progress_bar=False,
                 return_positions=False,
                 return_any_label=True,
                 draw_same_chunk=False,
                 zarr_store=zarr.storage.FSStore,
                 **kwargs):

        self._worker_id = 0

        self._transforms = {("images", ): transform}
        self._transforms_order = [("images", )]
        self._output_order = ["images"]

        if not isinstance(filenames, list):
            filenames = [filenames]

        filenames = reduce(
            operator.add,
            map(partial(parse_metadata, default_source_axes=source_axes,
                        default_data_group=data_group,
                        default_axes=axes,
                        default_rois=roi),
                filenames),
            [])

        self._collections = {"images": filenames}
        self._zarr_store = {"images": zarr_store}
        self._image_loader_func = {"images": None}
        self._image_loader_func_args = {"images": None}

        self._shuffle = shuffle
        self._progress_bar = progress_bar
        self._return_positions = return_positions
        self._return_any_label = return_any_label
        self._draw_same_chunk = draw_same_chunk

        self._patch_sampler = patch_sampler

        self._arr_lists = {}
        self._toplefts = {}

        self._initialized = False

    def _initialize(self, force=False):
        if self._initialized and not force:
            return

        arr_lists = []
        toplefts = []

        if self._progress_bar:
            q = tqdm(desc="Preloading zarr files",
                     total=len(self._collections),
                     position=self._worker_id)

        modes = self._collections.keys()

        for collection in zip(*self._collections.values()):
            collection = dict(map(tuple, zip(modes, collection)))
            for mode in collection.keys():
                collection[mode]["zarr_store"] = self._zarr_store[mode]
                collection[mode]["image_func"] = self._image_loader_func[mode]
                collection[mode]["image_func_args"] =\
                    self._image_loader_func_args[mode]

            if self._progress_bar:
                q.set_description(f"Preloading image "
                                  f"{collection['images']['filename']}")

            curr_img = ImageCollection(collection)

            # If a patch sampler was passed, it is used to determine the
            # top-left and bottom-right coordinates of the valid samples that
            # can be drawn from images.
            if self._patch_sampler is not None:
                toplefts.append(self._patch_sampler.compute_chunks(curr_img))
            else:
                toplefts.append([[slice(None)] * len(curr_img.collection["images"].axes)])

            arr_lists.append(curr_img)

            if self._progress_bar:
                q.update()

        if self._progress_bar:
            q.close()

        self._arr_lists = np.array(arr_lists, dtype=object)
        self._toplefts = np.array(toplefts, dtype=object)
        self._initialized = True

    def __getitem__(self, tlbr):
        coords = tlbr if tlbr is not None else slice(None)
        patches = self._curr_collection[coords]

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

    def __iter__(self):
        # Preload the files and masks associated with them
        self._initialize()

        samples = [
            [im_id, chk_id, None, None]
            for im_id in range(len(self._arr_lists))
            for chk_id in range(len(self._toplefts[im_id]))
            ]

        # When chunks must be depleted before moving to the next chunk, shuffle
        # all before hand.
        if self._shuffle and self._draw_same_chunk:
            random.shuffle(samples)

        prev_im_id = -1
        prev_chk_id = -1
        curr_chk = 0
        self._curr_collection = None

        while samples:
            # When chunks can be sampled even when these are not depleted,
            # shuffle here.
            if self._shuffle and not self._draw_same_chunk:
                curr_chk = random.randrange(0, len(samples))

            im_id = samples[curr_chk][0]
            chk_id = samples[curr_chk][1]

            chunk_tlbr = tuple(self._toplefts[im_id][chk_id])

            # If this chunk is different from the cached one, change the
            # cached chunk for this one.
            if prev_im_id != im_id or chk_id != prev_chk_id:
                prev_chk_id = chk_id

                if prev_im_id != im_id:
                    if self._curr_collection is not None:
                        self._curr_collection.free_cache()

                    prev_im_id = im_id
                    self._curr_collection = self._arr_lists[im_id]

                if self._patch_sampler is not None:
                    patches_tls = self._patch_sampler.compute_patches(
                        self._curr_collection,
                        chunk_tlbr
                    )

                else:
                    patches_tls = [chunk_tlbr]

                if not len(patches_tls):
                    samples.pop(curr_chk)
                    continue

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
            patches = self.__getitem__(patch_tlbr)

            if self._return_positions:
                pos = [[tlbr.start if tlbr.start is not None else 0,
                        tlbr.stop if tlbr.stop is not None else -1]
                       for tlbr in patch_tlbr]
                pos = np.array(pos, dtype=np.int64)
                patches = [pos] + patches

            if len(patches) > 1:
                patches = tuple(patches)
            else:
                patches = patches[0]

            yield patches


class LabeledZarrDataset(ZarrDatasetBase):
    """A labeled dataset based on the zarr dataset class.
    The densely labeled targets are extracted from group "labels_data_group".
    """
    def __init__(self, filenames, source_axes, axes=None, data_group=None,
                 roi=None,
                 labels_filenames=None,
                 labels_data_group=None,
                 labels_source_axes=None,
                 labels_axes=None,
                 labels_roi=None,
                 target_transform=None,
                 input_target_transform=None,
                 labels_zarr_store=zarr.storage.FSStore,
                 **kwargs):
        # Override the selection to always return labels with this class
        kwargs["return_any_label"] = False

        if labels_filenames is not None:
            if not isinstance(labels_filenames, list):
                labels_filenames = [labels_filenames]
        else:
            labels_filenames = filenames

        if labels_data_group is None:
            labels_data_group = data_group

        if labels_source_axes is None:
            labels_source_axes = source_axes

        if labels_axes is None:
            labels_axes = labels_source_axes

        if labels_roi is None:
            labels_roi = roi

        super(LabeledZarrDataset, self).__init__(filenames, source_axes,
                                                 axes=axes,
                                                 data_group=data_group,
                                                 roi=roi,
                                                 **kwargs)

        labels_filenames = reduce(
            operator.add,
            map(partial(parse_metadata, default_source_axes=labels_source_axes,
                        default_data_group=labels_data_group,
                        default_axes=labels_axes,
                        default_rois=labels_roi,
                        ignore_rois=True),
                labels_filenames),
            [])

        self._collections["target"] = labels_filenames
        self._image_loader_func["target"] = None
        self._image_loader_func_args["target"] = None
        self._zarr_store["target"] = labels_zarr_store

        # This is a transform that affects the geometry of the input, and then
        # it has to be applied to the target as well.
        self._transforms[("images", "target")] = input_target_transform
        self._transforms_order.append(("images", "target"))

        # This is a transform that only affects the target
        self._transforms[("target", )] = target_transform
        self._transforms_order.append(("target",))

        self._output_order.append("target")


class MaskedZarrDataset(ZarrDatasetBase):
    """A masked dataset based on the zarr dataset class.
    """
    def __init__(self, filenames, source_axes, axes=None, data_group=None,
                 roi=None,
                 mask_filenames=None,
                 mask_data_group=None,
                 mask_source_axes=None,
                 mask_axes=None,
                 mask_roi=None,
                 mask_func=None,
                 mask_func_args=None,
                 mask_zarr_store=zarr.storage.FSStore,
                 **kwargs):

        if mask_filenames is not None:
            if not isinstance(mask_filenames, list):
                mask_filenames = [mask_filenames]
        else:
            mask_filenames = filenames

        if mask_data_group is None:
            mask_data_group = data_group

        if mask_source_axes is None:
            mask_source_axes = source_axes

        if mask_axes is None:
            mask_axes = mask_source_axes

        if mask_roi is None:
            mask_roi = roi

        super(MaskedZarrDataset, self).__init__(filenames, source_axes,
                                                axes=axes,
                                                data_group=data_group,
                                                roi=roi,
                                                **kwargs)

        # Match the passed mask filenames to each filename in the images
        # modality
        mask_filenames = reduce(
            operator.add,
            map(partial(parse_metadata, default_source_axes=mask_source_axes,
                        default_data_group=mask_data_group,
                        default_axes=mask_axes,
                        default_rois=mask_roi,
                        ignore_rois=True),
                mask_filenames),
            [])

        self._collections["masks"] = mask_filenames
        self._image_loader_func["masks"] = mask_func
        self._image_loader_func_args["masks"] = mask_func_args
        self._zarr_store["masks"] = mask_zarr_store


class ZarrDataset(MaskedZarrDataset, LabeledZarrDataset):
    """A dataset based on the zarr dataset class capable of handling labeled
    datasets from masked inputs.
    """
    def __init__(self, filenames, **kwargs):
        super(ZarrDataset, self).__init__(filenames, **kwargs)
