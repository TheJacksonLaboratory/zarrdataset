from collections import OrderedDict
from typing import Iterable, Union, Callable
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

    def zarrdataset_worker_init_fn(worker_id):
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

    def chained_zarrdataset_worker_init_fn(worker_id):
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

    def zarrdataset_worker_init_fn(*args):
        pass

    def chained_zarrdataset_worker_init_fn(*args):
        pass


class ImageSample():
    _current_patch_idx = 0
    _ordering = None
    _rng_seed = None
    num_patches = None

    def __init__(self, im_id: int, chk_id: int, shuffle: bool = False):
        self.im_id = im_id
        self.chk_id = chk_id
        self._shuffle = shuffle

        if self._shuffle:
            self._rng_seed = random.randint(1, 100000)

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
    filenames: Union[str,
                     Iterable[str],
                     zarr.Group,
                     Iterable[zarr.Group],
                     zarr.Array,
                     Iterable[zarr.Array],
                     np.ndarray,
                     Iterable[np.ndarray]]
        The input source either specified by a path/url to a file or a
        supported array-like object, or list of them.
    source_axes: str
        The orignal array axes ordering.
    axes: Union[str, None]
        The axes ordering as it being used from the array (may involve
        permuting, dropping unused axes, and creating new axes).
    data_group: Union[str, int, None]
        The group for zarr images, or key for tiff files
    roi: Union[str, slice, Iterable[slice], None]
        Regions of interest from the input array that can be used for data
        sampling.
    image_loader_func: Union[Callable, None]
        A transformation applied to the input array before sampling. Could be
        used to define a mask generation function. This is not a data
        augmentation transform. To specify a data augmetation transform use
        `transform` instead.
    zarr_store: Union[zarr.storage.Store, None]
        A specific zarr.storage.Store class to be used to load zarr files.
    transform: Union[Callable, None]
        A transform applied to the array before returning it after sampling.
        This can be used to specify data augmentation transforms.
    """
    def __init__(self, mode: str,
                 filenames: Union[str, Iterable[str], zarr.Group,
                                  Iterable[zarr.Group],
                                  zarr.Array,
                                  Iterable[zarr.Array],
                                  np.ndarray,
                                  Iterable[np.ndarray]],
                 source_axes: str,
                 axes: Union[str, None] = None,
                 data_group: Union[str, int, None] = None,
                 roi: Union[str, slice, Iterable[slice], None] = None,
                 image_loader_func: Union[Callable, None] = None,
                 zarr_store: Union[zarr.storage.Store, None] = None,
                 transform: Union[Callable, None] = None):

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
    filenames: Union[str,
                     Iterable[str],
                     zarr.Group,
                     Iterable[zarr.Group],
                     zarr.Array,
                     Iterable[zarr.Array],
                     np.ndarray,
                     Iterable[np.ndarray]]
        The input source either specified by a path/url to a file or a
        supported array-like object, or list of them.
    source_axes: str
        The orignal array axes ordering.
    axes: Union[str, None]
        The axes ordering as it being used from the array (may involve
        permuting, dropping unused axes, and creating new axes).
    data_group: Union[str, int, None]
        The group for zarr images, or key for tiff files
    roi: Union[str, slice, Iterable[slice], None]
        Regions of interest from the input array that can be used for data
        sampling.
    image_loader_func: Union[Callable, None]
        A transformation applied to the input array before sampling. Could be
        used to define a mask generation function. This is not a data
        augmentation transform. To specify a data augmetation transform use
        `transform` instead.
    zarr_store: Union[zarr.storage.Store, None]
        A specific zarr.storage.Store class to be used to load zarr files.
    transform: Union[Callable, None]
        A transform applied to the array before returning it after sampling.
        This can be used to specify data augmentation transforms.
    """
    def __init__(self,
                 filenames: Union[str, Iterable[str], zarr.Group,
                                  Iterable[zarr.Group],
                                  zarr.Array,
                                  Iterable[zarr.Array],
                                  np.ndarray,
                                  Iterable[np.ndarray]],
                 source_axes: str,
                 axes: Union[str, None] = None,
                 data_group: Union[str, int, None] = None,
                 roi: Union[str, slice, Iterable[slice], None] = None,
                 image_loader_func: Union[Callable, None] = None,
                 zarr_store: Union[zarr.storage.Store, None] = None,
                 transform: Union[Callable, None] = None):

        super().__init__(
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
    filenames: Union[str,
                     Iterable[str],
                     zarr.Group,
                     Iterable[zarr.Group],
                     zarr.Array,
                     Iterable[zarr.Array],
                     np.ndarray,
                     Iterable[np.ndarray]]
        The input source either specified by a path/url to a file or a
        supported array-like object, or list of them.
    source_axes: str
        The orignal array axes ordering.
    axes: Union[str, None]
        The axes ordering as it being used from the array (may involve
        permuting, dropping unused axes, and creating new axes).
    data_group: Union[str, int, None]
        The group for zarr images, or key for tiff files
    roi: Union[str, slice, Iterable[slice], None]
        Regions of interest from the input array that can be used for data
        sampling.
    image_loader_func: Union[Callable, None]
        A transformation applied to the input array before sampling. Could be
        used to define a mask generation function. This is not a data
        augmentation transform. To specify a data augmetation transform use
        `transform` instead.
    zarr_store: Union[zarr.storage.Store, None]
        A specific zarr.storage.Store class to be used to load zarr files.
    transform: Union[Callable, None]
        A transform applied to the array before returning it after sampling.
        This can be used to specify data augmentation transforms.
    input_target_transform: Union[Callable, None]
        A function that is applied to both, input and target images. This can
        be used to specify data augmentation transforms that affect the target.
    """
    def __init__(self,
                 filenames: Union[str, Iterable[str], zarr.Group,
                                  Iterable[zarr.Group],
                                  zarr.Array,
                                  Iterable[zarr.Array],
                                  np.ndarray,
                                  Iterable[np.ndarray]],
                 source_axes: str,
                 axes: Union[str, None] = None,
                 data_group: Union[str, int, None] = None,
                 roi: Union[str, slice, Iterable[slice], None] = None,
                 image_loader_func: Union[Callable, None] = None,
                 zarr_store: Union[zarr.storage.Store, None] = None,
                 transform: Union[Callable, None] = None,
                 input_target_transform:Union[Callable, None] = None):

        super().__init__(
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
    filenames: Union[str,
                     Iterable[str],
                     zarr.Group,
                     Iterable[zarr.Group],
                     zarr.Array,
                     Iterable[zarr.Array],
                     np.ndarray,
                     Iterable[np.ndarray]]
        The input source either specified by a path/url to a file or a
        supported array-like object, or list of them.
    source_axes: str
        The orignal array axes ordering.
    axes: Union[str, None]
        The axes ordering as it being used from the array (may involve
        permuting, dropping unused axes, and creating new axes).
    data_group: Union[str, int, None]
        The group for zarr images, or key for tiff files
    roi: Union[str, slice, Iterable[slice], None]
        Regions of interest from the input array that can be used for data
        sampling.
    image_loader_func: Union[Callable, None]
        A transformation applied to the input array before sampling. Could be
        used to define a mask generation function. This is not a data
        augmentation transform. To specify a data augmetation transform use
        `transform` instead.
    zarr_store: Union[zarr.storage.Store, None]
        A specific zarr.storage.Store class to be used to load zarr files.
    """
    def __init__(self,
                 filenames: Union[str, Iterable[str], zarr.Group,
                                  Iterable[zarr.Group],
                                  zarr.Array,
                                  Iterable[zarr.Array],
                                  np.ndarray,
                                  Iterable[np.ndarray]],
                 source_axes: str,
                 axes: Union[str, None] = None,
                 data_group: Union[str, int, None] = None,
                 roi: Union[str, slice, Iterable[slice], None] = None,
                 image_loader_func: Union[Callable, None] = None,
                 zarr_store: Union[zarr.storage.Store, None] = None):

        super().__init__(
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
            collection = dict([(m, c) for m, c in zip(modes, collection)])
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

                samples[curr_chk].num_patches = len(patches_tls)

                if not len(patches_tls):
                    samples.pop(curr_chk)
                    prev_chk = -1
                    continue

            # # Initialize the count of top-left positions for patches inside
            # # this chunk.
            # if samples[curr_chk].num_patches is None:

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


class ZarrDataset(BaseZarrDataset):
    """A dataset based on the zarr dataset class capable of handling labeled
    datasets from masked inputs.

    Parameters
    ----------
    dataset_specs: OrderedDict
        An ordered dictionary containing the specifications of the datasets
        used as inputs. The order of the content of the dictionary determines
        the order patches are extracted, processed and concatenated to generate
        a single output when this dataset is iterated. Normal dictionaries can
        be passed inside `dataset_specs`; however, DataSpecs can be used as
        guideline of the parameters that are expected.
    patch_sampler: Union[PatchSampler, None]
        The patch sampling algorithm used to extract patches from images.
    shuffle: bool
        Whether samples are extracted in order or at random.
    progress_bar: bool
        Display a progress bar to show the status of data initialization.
        Requires `tqdm` to be installed in the environment.
    return_positions: bool
        Return the top-left positions from where the samples where extracted
        along with the set of patches.
    return_any_label: bool
        Return a label `0` along with the samples extracted, when no `labels`
        dataset specifications are passed. This is useful if this dataset is
        used as a generator from which a pair (input, target) is expected.
    return_worker_id: bool
        Return the worker id that extracted the sample.
    draw_same_chunk: bool
        Whether continue extracting samples from the same chunk, until
        depleting the posible patches to extract, before extract samples from
        a different chunk. This can be used to reduce the overhead of
        retrieving different chunks when sampling patches at random locations
        whithin the input image.
    """
    def __init__(self,
                 dataset_specs: OrderedDict,
                 patch_sampler: Union[PatchSampler, None] = None,
                 shuffle: bool = False,
                 progress_bar: bool = False,
                 return_positions: bool = False,
                 return_any_label: bool = True,
                 return_worker_id: bool = False,
                 draw_same_chunk: bool = False):

        super().__init__(
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
                    specs["filenames"]
                    if isinstance(specs["filenames"], list) else
                    [specs["filenames"]]),
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
