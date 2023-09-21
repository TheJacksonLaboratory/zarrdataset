from collections import OrderedDict
from typing import Iterable, Callable
from functools import reduce, partial
import operator
import random
import zarr
import numpy as np

from ._utils import parse_metadata
from ._imageloaders import ImageCollection
from ._samplers import PatchSampler

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

        dataset_obj._worker_sel = w_sel
        dataset_obj._worker_id = worker_id
        dataset_obj._num_workers = worker_info.num_workers

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
            ds._num_workers = 1

except ModuleNotFoundError:
    import logging
    logging.warning('PyTorch is not installed, the BaseZarrDataset class will '
                    'still work as a python iterator')
    IterableDataset = object

    def zarrdataset_worker_init(*args):
        pass

    def chained_zarrdataset_worker_init(*args):
        pass


class ImageSample():
    def __init__(self, im_id, chk_id, shuffle=False):
        self.im_id = im_id
        self.chk_id = chk_id
        self._shuffle = shuffle
        self._ordering = None

        if self._shuffle:
            self._rng_seed = random.randint(1, 100000)
        else:
            self._rng_seed = None

        self._current_patch_idx = 0
        self.num_patches = None

    def free_sampler(self):
        del self._ordering
        self._ordering = None

    def next_patch(self):
        if self._shuffle and self._ordering is None:
            curr_state = random.getstate()
            random.seed(self._rng_seed)
            self._ordering = list(range(self.num_patches))
            random.shuffle(self._ordering)
            random.setstate(curr_state)

        if self._shuffle:
            curr_patch = self._ordering[self._current_patch_idx]
        else:
            curr_patch = self._current_patch_idx

        self._current_patch_idx += 1
        is_empty = self._current_patch_idx >= self.num_patches

        return curr_patch, is_empty


class DatasetSpecs(object):
    """Data specification guidelines to add a mode to a ZarrDataset.

    Parameters
    ----------
    mode: str
        Specifies the use of this dataset (input image data, labels, masks).
    filenames: (str, Iterable[str], zarr.Group, Iterable[zarr.Group],
                zarr.Array, Iterable[zarr.Array], np.ndarray,
                Iterable[np.ndarray])
        The input source either specified by a path/url to a file or a
        supported array-like object, or list of them.
    source_axes: str
        The orignal array axes ordering.
    axes: (str, None)
        The axes ordering as it being used from the array (may involve
        permuting, dropping unused axes, and creating new axes).
    data_group: (str, int, None)
        The group for zarr images, or key for tiff files
    roi: (str, slice, Iterable[slice])
        Regions of interest from the input array that can be used for data
        sampling.
    image_loader_func: (Callable, None)
        A transformation applied to the input array before sampling. Could be
        used to define a mask generation function. This is not a data
        augmentation transform. To specify a data augmetation transform use
        `transform` instead.
    zarr_store: (zarr.storage.Store, None)
        A specific zarr.storage.Store class to be used to load zarr files.
    transform: (Callable, None)
        A transform applied to the array before returning it after sampling.
        This can be used to specify data augmentation transforms.
    """
    def __init__(self, mode: str,
                 filenames: (str, Iterable[str], zarr.Group,
                             Iterable[zarr.Group],
                             zarr.Array,
                             Iterable[zarr.Array],
                             np.ndarray,
                             Iterable[np.ndarray]),
                 source_axes: str,
                 axes: (str, None) = None,
                 data_group: (str, int, None) = None,
                 roi: (str, slice, Iterable[slice]) = None,
                 image_loader_func: (Callable, None) = None,
                 zarr_store: (zarr.storage.Store, None) = None,
                 transform: (Callable, None) = None):

        self.specs = dict(
            mode=mode,
            filenames=filenames,
            source_axes = source_axes,
            axes=axes,
            data_group=data_group,
            roi=roi,
            image_loader_func=image_loader_func,
            transforms=OrderedDict(),
            zarr_store=zarr_store
        )

        if transform is not None:
            self.specs["transforms"][(mode, )] = transform

    def __getitem__(self, key):
        return self.specs[key]

    def keys(self):
        return self.specs.keys()

    def items(self):
        return self.specs.items()

    def get(self, key, default=None):
        return self.specs.get(key, default)


class ImagesDatasetSpecs(DatasetSpecs):
    """Specification to add `image` data to a ZarrDataset.

    Parameters
    ----------
    filenames: (str, Iterable[str], zarr.Group, Iterable[zarr.Group],
                zarr.Array, Iterable[zarr.Array], np.ndarray,
                Iterable[np.ndarray])
        The input source either specified by a path/url to a file or a
        supported array-like object, or list of them.
    source_axes: str
        The orignal array axes ordering.
    axes: (str, None)
        The axes ordering as it being used from the array (may involve
        permuting, dropping unused axes, and creating new axes).
    data_group: (str, int, None)
        The group for zarr images, or key for tiff files
    roi: (str, slice, Iterable[slice])
        Regions of interest from the input array that can be used for data
        sampling.
    image_loader_func: (Callable, None)
        A transformation applied to the input array before sampling. Could be
        used to define a mask generation function. This is not a data
        augmentation transform. To specify a data augmetation transform use
        `transform` instead.
    zarr_store: (zarr.storage.Store, None)
        A specific zarr.storage.Store class to be used to load zarr files.
    transform: (Callable, None)
        A transform applied to the array before returning it after sampling.
        This can be used to specify data augmentation transforms.
    """
    def __init__(self,
                 filenames: (str, Iterable[str], zarr.Group,
                             Iterable[zarr.Group],
                             zarr.Array,
                             Iterable[zarr.Array],
                             np.ndarray,
                             Iterable[np.ndarray]),
                 source_axes: str,
                 axes: (str, None) = None,
                 data_group: (str, int, None) = None,
                 roi: (str, slice, Iterable[slice]) = None,
                 image_loader_func: (Callable, None) = None,
                 zarr_store: (zarr.storage.Store, None) = None,
                 transform: (Callable, None) = None):

        super(ImagesDatasetSpecs, self).__init__(
            "Images",
            filenames,
            source_axes,
            axes,
            data_group,
            roi,
            image_loader_func,
            zarr_store,
            transform
        )


class LabelsDatasetSpecs(DatasetSpecs):
    """Specification to add `labels` to a ZarrDataset.

    Parameters
    ----------
    filenames: (str, Iterable[str], zarr.Group, Iterable[zarr.Group],
                zarr.Array, Iterable[zarr.Array], np.ndarray,
                Iterable[np.ndarray])
        The input source either specified by a path/url to a file or a
        supported array-like object, or list of them.
    source_axes: str
        The orignal array axes ordering.
    axes: (str, None)
        The axes ordering as it being used from the array (may involve
        permuting, dropping unused axes, and creating new axes).
    data_group: (str, int, None)
        The group for zarr images, or key for tiff files
    roi: (str, slice, Iterable[slice])
        Regions of interest from the input array that can be used for data
        sampling.
    image_loader_func: (Callable, None)
        A transformation applied to the input array before sampling. Could be
        used to define a mask generation function. This is not a data
        augmentation transform. To specify a data augmetation transform use
        `transform` instead.
    zarr_store: (zarr.storage.Store, None)
        A specific zarr.storage.Store class to be used to load zarr files.
    transform: (Callable, None)
        A transform applied to the array before returning it after sampling.
        This can be used to specify data augmentation transforms.
    input_target_transform: (Callable, None)
        A function that is applied to both, input and target images. This can
        be used to specify data augmentation transforms that affect the target.
    """
    def __init__(self,
                 filenames: (str, Iterable[str], zarr.Group,
                             Iterable[zarr.Group],
                             zarr.Array,
                             Iterable[zarr.Array],
                             np.ndarray,
                             Iterable[np.ndarray]),
                 source_axes: str,
                 axes: (str, None) = None,
                 data_group: (str, int, None) = None,
                 roi: (str, slice, Iterable[slice]) = None,
                 image_loader_func: (Callable, None) = None,
                 zarr_store: (zarr.storage.Store, None) = None,
                 transform: (Callable, None) = None,
                 input_target_transform: (Callable, None) = None):

        super(LabelsDatasetSpecs, self).__init__(
            "labels",
            filenames,
            source_axes,
            axes,
            data_group,
            roi,
            image_loader_func,
            zarr_store,
            transform
        )

        if input_target_transform is not None:
            self.specs["transforms"][("labels", )] = transform
            self.specs["transforms"][("images", "labels")] =\
                input_target_transform


class MasksDatasetSpecs(DatasetSpecs):
    """Specification to add `masks` to a ZarrDataset.

    Parameters
    ----------
    filenames: (str, Iterable[str], zarr.Group, Iterable[zarr.Group],
                zarr.Array, Iterable[zarr.Array], np.ndarray,
                Iterable[np.ndarray])
        The input source either specified by a path/url to a file or a
        supported array-like object, or list of them.
    source_axes: str
        The orignal array axes ordering.
    axes: (str, None)
        The axes ordering as it being used from the array (may involve
        permuting, dropping unused axes, and creating new axes).
    data_group: (str, int, None)
        The group for zarr images, or key for tiff files
    roi: (str, slice, Iterable[slice])
        Regions of interest from the input array that can be used for data
        sampling.
    image_loader_func: (Callable, None)
        A transformation applied to the input array before sampling. Could be
        used to define a mask generation function. This is not a data
        augmentation transform. To specify a data augmetation transform use
        `transform` instead.
    zarr_store: (zarr.storage.Store, None)
        A specific zarr.storage.Store class to be used to load zarr files.
    """
    def __init__(self,
                 filenames: (str, Iterable[str], zarr.Group,
                             Iterable[zarr.Group],
                             zarr.Array,
                             Iterable[zarr.Array],
                             np.ndarray,
                             Iterable[np.ndarray]),
                 source_axes: str,
                 axes: (str, None) = None,
                 data_group: (str, int, None) = None,
                 roi: (str, slice, Iterable[slice]) = None,
                 image_loader_func: (Callable, None) = None,
                 zarr_store: (zarr.storage.Store, None) = None):

        super(MasksDatasetSpecs, self).__init__(
            "masks",
            filenames,
            source_axes,
            axes,
            data_group,
            roi,
            image_loader_func,
            zarr_store,
            None
        )


class BaseZarrDataset(IterableDataset):
    """A zarr-based dataset.

    Sampling from  spatial (+color channels) axes is supported by now.
    """
    def __init__(self, patch_sampler=None, shuffle=False, progress_bar=False,
                 return_positions=False,
                 return_any_label=True,
                 return_worker_id=False,
                 draw_same_chunk=False,
                 **kwargs):

        self._worker_sel = slice(None)
        self._worker_id = 0
        self._num_workers = 1

        self._shuffle = shuffle
        self._progress_bar = progress_bar
        self._return_positions = return_positions
        self._return_any_label = return_any_label
        self._return_worker_id = return_worker_id
        self._draw_same_chunk = draw_same_chunk

        self._patch_sampler = patch_sampler

        self._transforms = OrderedDict()
        self._transforms_order = []
        self._output_order = []

        self._collections = {}
        self._zarr_store = {}
        self._image_loader_func = {}

        self._arr_lists = []
        self._toplefts = []

        self._initialized = False

    def _initialize(self, force=False):
        if self._initialized and not force:
            return

        arr_lists = []
        toplefts = []

        if self._progress_bar:
            q = tqdm(desc="Preloading zarr files",
                     total=len(self._collections["images"]),
                     position=self._worker_id)

        modes = self._collections.keys()

        for collection in zip(*self._collections.values()):
            collection = dict(map(tuple, zip(modes, collection)))
            for mode in collection.keys():
                collection[mode]["zarr_store"] = self._zarr_store[mode]
                collection[mode]["image_func"] = self._image_loader_func[mode]

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
                toplefts.append([
                    dict((ax, slice(None))
                         for ax in curr_img.collection["images"].axes)
                    ]
                )

            arr_lists.append(curr_img)

            if self._progress_bar:
                q.update()

        if self._progress_bar:
            q.close()

        if len(arr_lists) == 1:
            self._arr_lists = np.array(arr_lists, dtype=object)
            self._toplefts = np.array([toplefts[0][self._worker_sel]],
                                      dtype=object)

        elif len(arr_lists) < self._num_workers:
            self._arr_lists = np.array(arr_lists, dtype=object)
            self._toplefts = np.array([tls[self._worker_sel]
                                       for tls in toplefts],
                                      dtype=object)

        else:
            self._arr_lists = np.array(arr_lists[self._worker_sel],
                                       dtype=object)
            self._toplefts = np.array(toplefts[self._worker_sel],
                                      dtype=object)

        self._initialized = True

    def __getitem__(self, tlbr):
        coords = tlbr if tlbr is not None else slice(None)
        patches = self._curr_collection[coords]

        for inputs, transform in self._transforms.items():
            res = transform(*tuple(patches[mode] for mode in inputs))

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
            ImageSample(im_id, chk_id, shuffle=self._shuffle)
            for im_id in range(len(self._arr_lists))
            for chk_id in range(len(self._toplefts[im_id]))
        ]

        # Shuffle chunks here if samples will come from the same chunk until
        # they are depleted.
        if self._shuffle and self._draw_same_chunk:
            random.shuffle(samples)

        prev_im_id = -1
        prev_chk_id = -1
        prev_chk = -1
        curr_chk = 0
        self._curr_collection = None

        while samples:
            # Shuffle chunks here if samples can come from different chunks.
            if self._shuffle and not self._draw_same_chunk:
                curr_chk = random.randrange(0, len(samples))

            im_id = samples[curr_chk].im_id
            chk_id = samples[curr_chk].chk_id

            chunk_tlbr = self._toplefts[im_id][chk_id]

            # If this sample is from a different image or chunk, free the
            # previous sample and re-sample the patches from the current chunk.
            if prev_im_id != im_id or chk_id != prev_chk_id:
                if prev_chk >= 0:
                    # Free the patch ordering from the previous chunk to save
                    # memory.
                    samples[prev_chk].free_sampler()

                prev_chk = curr_chk
                prev_chk_id = chk_id

                if prev_im_id != im_id:
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
                    prev_chk = -1
                    continue

            # Initialize the count of top-left positions for patches inside
            # this chunk.
            if samples[curr_chk].num_patches is None:
                samples[curr_chk].num_patches = len(patches_tls)

            curr_patch, is_empty = samples[curr_chk].next_patch()

            # When all possible patches have been extracted from the current
            # chunk, remove that chunk from the list of samples.
            if is_empty:
                samples.pop(curr_chk)
                prev_chk = -1

            patch_tlbr = patches_tls[curr_patch]
            patches = self.__getitem__(patch_tlbr)

            if self._return_positions:
                pos = [
                    [patch_tlbr[ax].start
                     if patch_tlbr[ax].start is not None else 0,
                     patch_tlbr[ax].stop
                     if patch_tlbr[ax].stop is not None else -1
                    ] if ax in patch_tlbr else [0, -1]
                    for ax in self._collections["images"][0]["axes"]
                ]
                patches = [np.array(pos, dtype=np.int64)] + patches

            if self._return_worker_id:
                wid = [np.array(self._worker_id, dtype=np.int64)]
                patches = wid + patches

            if len(patches) > 1:
                patches = tuple(patches)
            else:
                patches = patches[0]

            yield patches


# class LabeledZarrDataset(BaseZarrDataset):
#     """A labeled dataset based on the zarr dataset class.
#     The densely labeled targets are extracted from group "labels_data_group".
#     """
#     def __init__(self, filenames, source_axes, axes=None, data_group=None,
#                  roi=None,
#                  labels_filenames=None,
#                  labels_data_group=None,
#                  labels_source_axes=None,
#                  labels_axes=None,
#                  labels_roi=None,
#                  target_transform=None,
#                  input_target_transform=None,
#                  labels_zarr_store=None,
#                  **kwargs):
#         # Override the selection to always return labels with this class
#         kwargs["return_any_label"] = False

#         if not isinstance(labels_filenames, list):
#             labels_filenames = [labels_filenames]

#         if labels_data_group is None:
#             labels_data_group = data_group

#         if labels_source_axes is None:
#             labels_source_axes = source_axes

#         if labels_axes is None:
#             labels_axes = labels_source_axes

#         if labels_roi is None:
#             if labels_data_group == data_group:
#                 labels_roi = roi
#             else:
#                 labels_roi = [[slice(None)] * len(labels_source_axes)]

#         super(LabeledZarrDataset, self).__init__(filenames, source_axes,
#                                                  axes=axes,
#                                                  data_group=data_group,
#                                                  roi=roi,
#                                                  **kwargs)

#         labels_filenames = reduce(
#             operator.add,
#             map(partial(parse_metadata, default_source_axes=labels_source_axes,
#                         default_data_group=labels_data_group,
#                         default_axes=labels_axes,
#                         default_rois=labels_roi,
#                         override_meta=True),
#                 labels_filenames),
#             [])

#         self._collections["target"] = labels_filenames
#         self._image_loader_func["target"] = None
#         self._zarr_store["target"] = labels_zarr_store

#         # This is a transform that affects the geometry of the input, and then
#         # it has to be applied to the target as well.
#         self._transforms[("images", "target")] = input_target_transform
#         self._transforms_order.append(("images", "target"))

#         # This is a transform that only affects the target
#         self._transforms[("target", )] = target_transform
#         self._transforms_order.append(("target",))

#         self._output_order.append("target")


# class MaskedZarrDataset(BaseZarrDataset):
#     """A masked dataset based on the zarr dataset class.
#     """
#     def __init__(self, filenames, source_axes, axes=None, data_group=None,
#                  roi=None,
#                  mask_filenames=None,
#                  mask_data_group=None,
#                  mask_source_axes=None,
#                  mask_axes=None,
#                  mask_roi=None,
#                  mask_func=None,
#                  mask_zarr_store=None,
#                  **kwargs):

#         if not isinstance(mask_filenames, list):
#             mask_filenames = [mask_filenames]

#         if mask_data_group is None:
#             mask_data_group = data_group

#         if mask_source_axes is None:
#             mask_source_axes = source_axes

#         if mask_axes is None:
#             mask_axes = mask_source_axes

#         if mask_roi is None:
#             if mask_data_group == data_group:
#                 mask_roi = roi
#             else:
#                 mask_roi = [[slice(None)] * len(mask_source_axes)]

#         super(MaskedZarrDataset, self).__init__(filenames, source_axes,
#                                                 axes=axes,
#                                                 data_group=data_group,
#                                                 roi=roi,
#                                                 **kwargs)

#         if not isinstance(mask_filenames, list):
#             mask_filenames = [mask_filenames]

#         mask_filenames = reduce(
#             operator.add,
#             map(partial(parse_metadata, default_source_axes=mask_source_axes,
#                         default_data_group=mask_data_group,
#                         default_axes=mask_axes,
#                         default_rois=mask_roi,
#                         override_meta=True),
#                 mask_filenames),
#             [])

#         self._collections["masks"] = mask_filenames
#         self._image_loader_func["masks"] = mask_func
#         self._zarr_store["masks"] = mask_zarr_store


class ZarrDataset(BaseZarrDataset):
    """A dataset based on the zarr dataset class capable of handling labeled
    datasets from masked inputs.

    Parameters
    ----------
    dataset_specs: OrderedDict,
    patch_sampler: (PatchSampler, None)
    shuffle: bool
    progress_bar: bool
    return_positions: bool
    return_any_label: bool
    return_worker_id: bool
    draw_same_chunk: bool
    """
    def __init__(self,
                 dataset_specs: OrderedDict,
                 patch_sampler: (PatchSampler, None) = None,
                 shuffle: bool = False,
                 progress_bar: bool = False,
                 return_positions: bool = False,
                 return_any_label: bool = True,
                 return_worker_id: bool = False,
                 draw_same_chunk: bool = False):

        super(ZarrDataset, self).__init__(
            patch_sampler=patch_sampler,
            shuffle=shuffle,
            progress_bar=progress_bar,
            return_positions=return_positions,
            return_any_label=return_any_label,
            return_worker_id=return_worker_id,
            draw_same_chunk=draw_same_chunk
        )

        if "images" not in dataset_specs:
            raise ValueError("Data specifications must contain at least the "
                             "information about input images")

        # Iterate the modes in the dataset specifications
        for mode, specs in dataset_specs.items():
            if not isinstance(specs["filenames"], list):
                filenames_src = [specs["filenames"]]
            else:
                filenames_src = specs["filenames"]

            source_axes = specs["source_axes"]
            data_group = specs.get("data_group", None)
            axes = specs.get("axes", None)
            roi = specs.get("roi", None)

            self._collections[mode] = reduce(
                operator.add,
                map(partial(parse_metadata,
                            default_source_axes=source_axes,
                            default_data_group=data_group,
                            default_axes=axes,
                            default_rois=roi),
                    filenames_src),
                []
            )

            if "transforms" in specs.keys():
                for t_ord, t in specs["transforms"].items():
                    self._transforms[t_ord] = t

            if "mask" not in mode:
                self._output_order.append(mode)

            self._zarr_store[mode] = specs.get("zarr_store", None)

            self._image_loader_func[mode] = specs.get("image_loader_func",
                                                      None)

        if "labels" in dataset_specs:
            self._return_any_label = False
