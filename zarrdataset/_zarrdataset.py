from collections import OrderedDict
from collections.abc import Iterable
from typing import Union, Callable
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
    TQDM_SUPPORT=True

except ModuleNotFoundError:
    TQDM_SUPPORT=False

    # This removes the dependency on tqdm when it is not installed
    class tqdm(object):
        """Placeholder used to remove dependency in `tqdm`, so ZarrDataset can
        still be used when `tqdm` is not installed.
        """
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
    PYTORCH_SUPPORT = True

except ModuleNotFoundError:
    import logging
    logging.warning('PyTorch is not installed, the BaseZarrDataset class will '
                    'still work as a python iterator')
    IterableDataset = object
    PYTORCH_SUPPORT = False


def zarrdataset_worker_init_fn(worker_id):
    """ZarrDataset multithread workers initialization function.
    """
    if not PYTORCH_SUPPORT:
        return

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
    """ZarrDataset multithread workers initialization function for PyTorch's
    ChainedDatasets.
    """
    if not PYTORCH_SUPPORT:
        return

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


class DatasetSpecs(dict):
    """Data specification guidelines to add image modalities to a ZarrDataset.

    Parameters
    ----------
    modality: str
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
    transform: Union[Callable, Iterable[Callable], None]
        A transform, or list of sequential transforms, to be applied to the
        array before returning it after sampling. This can be used to specify
        data augmentation transforms.
    add_to_output: bool
        Whether add this modality to the output after sampling or not. For
        example, labels would be added to the output along with the input
        image array, while masks might not be needed.
    """
    def __init__(self, modality: str,
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
                 transform: Union[Callable, Iterable[Callable], None] = None,
                 add_to_output: bool = True):

        super().__init__()

        self["modality"] = modality
        self["filenames"] = filenames
        self["source_axes"] = source_axes
        self["axes"] = axes
        self["data_group"] = data_group
        self["roi"] = roi
        self["image_loader_func"] = image_loader_func
        self["transforms"] = []
        self["zarr_store"] = zarr_store
        self["add_to_output"] = add_to_output

        if transform is not None:
            if not isinstance(transform, Iterable):
                transform = [transform]

            self["transforms"].append(((modality, ), transform))


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
    transform: Union[Callable, Iterable[Callable], None]
        A transform, or list of sequential transforms, to be applied to the
        array before returning it after sampling. This can be used to specify
        data augmentation transforms.
    modality: str
        Specifies the use of this dataset (default is `images` for image data).
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
                 transform: Union[Callable, Iterable[Callable], None] = None,
                 modality: str ="images"):

        super().__init__(
            modality,
            filenames,
            source_axes,
            axes,
            data_group,
            roi,
            image_loader_func,
            zarr_store,
            transform,
            add_to_output=True
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
    transform: Union[Callable, Iterable[Callable], None]
        A transform, or list of sequential transforms, to be applied to the
        array before returning it after sampling. This can be used to specify
        data augmentation transforms.
    input_label_transform: Union[Callable, None]
        A transform applied to the array before returning it after sampling.
        This can be used to specify data augmentation transforms.
    modality: str
        Specifies the use of this dataset (default is `labels`).
    """
    def __init__(
      self,
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
      transform: Union[Callable, Iterable[Callable], None] = None,
      input_label_transform: Union[Callable, Iterable[Callable], None] = None,
      input_mode: str = "images",
      modality: str = "labels"):

        super().__init__(
            modality,
            filenames,
            source_axes,
            axes,
            data_group,
            roi,
            image_loader_func,
            zarr_store,
            transform,
            add_to_output=True
        )

        if input_label_transform is not None:
            self["transforms"].append((("labels", ), transform))
            if not isinstance(input_label_transform, Iterable):
                input_label_transform = [input_label_transform]

            self["transforms"].append(
                ((input_mode, "labels"), input_label_transform)
            )


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
    modality: str
        Specifies the use of this dataset (default is `masks`).
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
                 modality: str = "masks"):

        super().__init__(
            modality,
            filenames,
            source_axes,
            axes,
            data_group,
            roi,
            image_loader_func,
            zarr_store,
            None,
            add_to_output=False
        )


class ZarrDataset(IterableDataset):
    """A Zarr-based dataset class capable of handling large volumes of image
    data stored in OME-NGFF Zarr format. This class can match the coordinates
    of the different image modalities to those in the `images` mode, so labels
    and masks are retrieved from these same coordinates. All spatial axes are
    scaled using the `images` mode as reference, therefore labels and masks do
    not need to share the same sizes as the arrays in the `images` mode.

    Parameters
    ----------
    dataset_specs: Union[dict, Iterable[dict], None]
        A list of dictionaries containing the specifications of the datasets
        used as inputs.
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
    return_worker_id: bool
        Return the worker id that extracted the sample.
    draw_same_chunk: bool
        Whether continue extracting samples from the same chunk, until
        depleting the posible patches to extract, before extract samples from
        a different chunk. This can be used to reduce the overhead of
        retrieving different chunks when sampling patches at random locations
        whithin the input image.
    """
    def __init__(self, dataset_specs: Union[Iterable[dict], None] = None,
                 patch_sampler: Union[PatchSampler, None] = None,
                 shuffle: bool = False,
                 progress_bar: bool = False,
                 return_positions: bool = False,
                 return_worker_id: bool = False,
                 draw_same_chunk: bool = False):

        self._worker_sel = slice(None)
        self._worker_id = 0
        self._num_workers = 1

        self._shuffle = shuffle
        self._progress_bar = progress_bar
        self._return_positions = return_positions
        self._return_worker_id = return_worker_id
        self._draw_same_chunk = draw_same_chunk

        self._patch_sampler = patch_sampler

        self._transforms = []
        self._output_order = []

        self._collections = {}
        self._zarr_store = {}
        self._image_loader_func = {}

        self._arr_lists = []
        self._toplefts = []

        self._ref_mod = None

        self._initialized = False

        if dataset_specs is not None:
            # Iterate the modalities in the dataset specifications
            if not isinstance(dataset_specs, list):
                dataset_specs = [dataset_specs]

            for specs in dataset_specs:
                self.add_modality(**specs)

    def _initialize(self, force=False):
        if self._ref_mod is None:
            raise ValueError("No image modalities have been added to the "
                             "dataset. Use `.add_modality` to add at least one"
                             " modality to this dataset prior to "
                             "initialization.")

        if self._initialized and not force:
            return

        arr_lists = []
        toplefts = []

        if self._progress_bar:
            q = tqdm(desc="Preloading zarr files",
                     total=len(self._collections[self._ref_mod]),
                     position=self._worker_id)

        modes = self._collections.keys()

        for collection in zip(*self._collections.values()):
            collection = {m: c for m, c in zip(modes, collection)}
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
                    {ax: slice(None)
                     for ax in curr_img.collection[self._ref_mod].axes}
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

        for inputs, transforms_list in self._transforms:
            res = (patches[mode] for mode in inputs)

            for transform in transforms_list:
                res = transform(*res)

                # In case that the transform returns a single value, wrap it
                # in a tuple to keep the same number of outputs as inputs.
                if not isinstance(res, (tuple, list)):
                    res = (res, )

            for mode, mode_res in zip(inputs, res):
                patches[mode] = mode_res

        patches = [patches[mode] for mode in self._output_order]

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
                    for ax in self._collections[self._ref_mod][0]["axes"]
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

    def add_transform(self, modalities: Union[str, Iterable[str]],
                      transform: Union[Callable, Iterable[Callable]]):
        """Add a pre-processing transform pipeline to the dataset, applied to
        the arrays from modalities specified with `modes`. This will be
        performed after any other pre-processing transforms already registered.

        Parameters
        ----------
        modalities: Union[str, Iterable[str]]
            The modalities on which this transform is applied (e.g., ``images``
            to apply only on image arrays, or (``images``, ``labels``) to apply
            it to both, images and labels arrays)
        transform: Union[Callable, Iterable[Callable]]
            A function, or sequence of functions, that receives the same number
            of inputs as specified in `modalities`, and returns that same
            number of outputs.
        """
        if isinstance(modalities, str):
            modalities = (modalities, )

        if not isinstance(transform, Iterable):
            transform = [transform]

        self._transforms.append((modalities, transform))

    def add_modality(self,
                     modality: str,
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
                     transforms: Union[Iterable[tuple], None] = None,
                     add_to_output: bool = True):
        """Add a new modality to the dataset.

        Parameters
        ----------
        modality: str
            The name of the new modality added (e.g., ``images``, ``labels``,
            ``masks``, etc.).
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
            A transformation applied to the input array before sampling. Could
            be used to define a mask generation function. This is not a data
            augmentation transform. To specify a data augmetation transform use
            `transform` instead.
        zarr_store: Union[zarr.storage.Store, None]
            A specific zarr.storage.Store class to be used to load zarr files.
        transform: Union[Iterable[tuple], None]
            A list of transforms applied to arrays before yielding them. This
            can be used to specify data augmentation transforms, and can be
            applied to this and other existing modalities in this dataset.
            For example, to add a transform affecting images and labels, use
            the tuple ('images', 'labels') as key for that transform, and make
            sure the function associated to that key receives, and returns the
            same number of inputs and ouputs as specified in the key.
        add_to_output: bool
            Whether add this modality to the output after sampling or not. For
            example, labels would be added to the output along with the input
            image array, while masks might not be needed.
        """
        if self._ref_mod is None:
            self._ref_mod = modality

        self._collections[modality] = reduce(
            operator.add,
            map(partial(parse_metadata,
                        default_source_axes=source_axes,
                        default_data_group=data_group,
                        default_axes=axes,
                        default_rois=roi),
                filenames if isinstance(filenames, list) else [filenames]),
            []
        )

        if transforms is not None:
            for t_ord, t in transforms:
                self.add_transform(modalities=t_ord, transform=t)

        if add_to_output:
            self._output_order.append(modality)

        self._zarr_store[modality] = zarr_store

        self._image_loader_func[modality] = image_loader_func

    def __repr__(self) -> str:
        """ZarrDataset string representation.
        """
        transforms_repr_str = "\n"
        for inputs, transforms in self._transforms:
            for t in transforms:
                transforms_repr_str += f"\t{inputs}: {t}\n"

        repr_str = (f"ZarrDataset (PyTorch support:{PYTORCH_SUPPORT}, tqdm "
                    f"support :{TQDM_SUPPORT})"
                    + "\n"
                    + f"Modalities: {','.join(self._collections.keys())}"
                    + "\n"
                    + f"Transforms order: {transforms_repr_str}"
                    + "\n"
                    + f"Using {self._ref_mod} modality as reference.")

        if self._patch_sampler is not None:
            repr_str += ("\n"
                         + f"Using {str(self._patch_sampler)}")

        return repr_str
