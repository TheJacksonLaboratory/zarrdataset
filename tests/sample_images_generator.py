import zarrdataset as zds

from functools import partial
import os
import shutil
from pathlib import Path
import numpy as np
import zarr
from PIL import Image
import tifffile
import torch
from unittest import mock


class ImageTransformTest(zds.MaskGenerator):
    def __init__(self, axes):
        super(ImageTransformTest, self).__init__(axes=axes)

    def _compute_transform(self, image: np.ndarray):
        return image * 2


def input_target_transform(x, t, dtype=np.float32):
    return x.astype(dtype), t.astype(dtype)


def remove_directory(dir=None):
    if dir is not None and os.path.isdir(dir):
        shutil.rmtree(dir)


def generate_collection(source_axes="CYX", shape=(3, 2048, 2048),
                        chunks=(3, 512, 512),
                        dtype=np.uint8,
                        label=0,
                        mask_threshold=127,
                        data_group="0/0",
                        mask_group="labels/masks/0/0",
                        labels_group="labels/segmentation/0/0",
                        classes_group="labels/classes/0/0",
                        **kwargs):

    img_axes = "CYX"

    rel_axes_order = zds.map_axes_order(source_axes=source_axes,
                                        target_axes=img_axes)

    # Generate a test image on memory
    z1, z2 = np.meshgrid(np.arange(shape[rel_axes_order[-1]],
                                   dtype=np.float32),
                         np.arange(shape[rel_axes_order[-2]],
                                   dtype=np.float32))

    img_channels = []
    for _ in range(shape[source_axes.index("C")]):
        img_channels.append(z1 ** round(np.random.rand() * 3)
                            + z2 ** round(np.random.rand() * 3))

    img = np.stack(img_channels)
    img = img / img.max(axis=(1, 2))[:, None, None]
    img = img * 255
    img = img.astype(dtype)

    # Generate a mask from the image
    mask = img.mean(axis=0) < mask_threshold

    # Label the regions of `img`
    labs = np.log2(img.mean(axis=0) + 1).round().astype(np.uint32)

    new_axes = list(set(source_axes) - set(img_axes))
    permute_order = zds.map_axes_order(source_axes=img_axes,
                                       target_axes=source_axes)
    img = img.transpose(permute_order)
    img = np.expand_dims(img, tuple(source_axes.index(a) for a in new_axes))

    store = zarr.MemoryStore()
    main_grp = zarr.open(store)
    main_grp.create_dataset(data_group, data=img, dtype=dtype,
                            chunks=chunks)

    chunk_height = chunks[rel_axes_order[-2]]
    chunk_width = chunks[rel_axes_order[-1]]

    main_grp.create_dataset(mask_group, data=mask, dtype=bool,
                            chunks=(chunk_height, chunk_width))

    main_grp.create_dataset(labels_group, data=labs, dtype=np.uint32,
                            chunks=(chunk_height, chunk_width))

    main_grp.create_dataset(classes_group, data=np.array([[[label]]]),
                            dtype=np.uint32,
                            chunks=(1, 1, 1))

    return main_grp


def generate_zarr_group(dst_dir, img_idx, image_specs):
    z_groups = generate_collection(**image_specs)
    destroy_func = partial(remove_directory, dir=None)
    return z_groups, None, None, None, destroy_func


def generate_zarr_array(dst_dir, img_idx, image_specs):
    z_groups = generate_collection(**image_specs)

    img_src = z_groups[image_specs["data_group"]]

    if "mask_group" in image_specs:
        mask_src = z_groups[image_specs["mask_group"]]
    else:
        mask_src = None

    if "labels_group" in image_specs:
        labels_src = z_groups[image_specs["labels_group"]]
    else:
        labels_src = None

    if "classes_group" in image_specs:
        classes_src = z_groups[image_specs["classes_group"]]
    else:
        classes_src = None

    destroy_func = partial(remove_directory, dir=None)

    return img_src, mask_src, labels_src, classes_src, destroy_func


def generate_ndarray(dst_dir, img_idx, image_specs):
    z_groups = generate_collection(**image_specs)

    img_src = z_groups[image_specs["data_group"]][:]

    if "mask_group" in image_specs:
        mask_src = z_groups[image_specs["mask_group"]][:]
    else:
        mask_src = None

    if "labels_group" in image_specs:
        labels_src = z_groups[image_specs["labels_group"]][:]
    else:
        labels_src = None

    if "classes_group" in image_specs:
        classes_src = z_groups[image_specs["classes_group"]][:]
    else:
        classes_src = None

    destroy_func = partial(remove_directory, dir=None)

    return img_src, mask_src, labels_src, classes_src, destroy_func


def generate_zarr_file(dst_dir, img_idx, image_specs):
    z_groups = generate_collection(**image_specs)

    Path(dst_dir).mkdir(parents=True, exist_ok=True)
    image_filename = f'{dst_dir}/zarr_group_{img_idx}.zarr'
    z_grp_dst = zarr.open(image_filename, mode="w")
    zarr.copy_all(z_groups, z_grp_dst)

    destroy_func = partial(remove_directory, dir=dst_dir)

    return image_filename, None, None, None, destroy_func


def generate_tiffs(dst_dir, img_idx, image_specs):
    z_groups = generate_collection(**image_specs)

    Path(dst_dir).mkdir(parents=True, exist_ok=True)

    image_filename = f'{dst_dir}/img_{img_idx}.ome.tiff'
    with tifffile.TiffWriter(image_filename, bigtiff=True) as tif:
        tif.write(
            z_groups[image_specs["data_group"]][:],
            metadata={"axes": None},
            tile=(128, 128),
            photometric='rgb'
        )

        tif.write(
            z_groups[image_specs["mask_group"]][:],
            metadata={"axes": "YX"},
            tile=(128, 128),
        )
    
        tif.write(
            z_groups[image_specs["labels_group"]][:],
            metadata={"axes": "YX"},
            tile=(128, 128),
        )

        tif.write(
            z_groups[image_specs["classes_group"]][:],
            metadata={"axes": "CYX"},
        )

    destroy_func = partial(remove_directory, dir=dst_dir)

    return image_filename, None, None, None, destroy_func


def generate_pngs(dst_dir, img_idx, image_specs):

    z_groups = generate_collection(**image_specs)

    Path(dst_dir).mkdir(parents=True, exist_ok=True)

    permute_order = zds.map_axes_order(source_axes=image_specs["source_axes"],
                                       target_axes="YXC")
    img_data = z_groups[image_specs["data_group"]][:]
    img_data = img_data.transpose(permute_order)
    img_data = img_data.reshape(-1, img_data.shape[-3], img_data.shape[-2],
                                img_data.shape[-1])
    img_data = img_data[0, ..., :3]

    image = Image.fromarray(img_data)

    image_filename = f"{dst_dir}/img_{img_idx}.png"
    image.save(image_filename,
                quality_opts={'compress_level': 9, 'optimize': False})

    mask_data = z_groups[image_specs["mask_group"]][:]
    mask = Image.fromarray(mask_data)

    mask_filename = f"{dst_dir}/mask_{img_idx}.png"
    mask.save(mask_filename,
                quality_opts={'compress_level': 9, 'optimize': False})

    labels_data = z_groups[image_specs["labels_group"]][:]
    labels = Image.fromarray(labels_data)

    labels_filename = f"{dst_dir}/labels_{img_idx}.png"
    labels.save(labels_filename,
                quality_opts={'compress_level': 9, 'optimize': False})

    destroy_func = partial(remove_directory, dir=dst_dir)

    return image_filename, mask_filename, labels_filename, None, destroy_func


GENERATORS = {
    "zarr_group": generate_zarr_group,
    "zarr_file": generate_zarr_file,
    "zarr_array": generate_zarr_array,
    "ndarray": generate_ndarray,
    "tiff": generate_tiffs,
    "png": generate_pngs,
}


def generate_image_data(image_specs):

    if image_specs["type"] in GENERATORS.keys():
        (img_src,
         mask_src,
         labels_src,
         classes_src,
         destroy_func) = GENERATORS[image_specs["type"]](
            dst_dir=image_specs["source"]["dst_dir"],
            img_idx=image_specs["source"]["img_idx"],
            image_specs=image_specs["specs"])

        shape = image_specs["specs"]["shape"]

    else:
        arr_src, store = zds.image2array(image_specs["source"],
                                         image_specs["specs"]["data_group"])
        shape = arr_src.shape

        if store is not None:
            store.close()

        img_src = image_specs["source"]
        mask_src = None
        labels_src = None
        classes_src = None

        destroy_func = lambda: None

    args = dict(
        filename=img_src,
        data_group=image_specs["specs"]["data_group"],
        source_axes=image_specs["specs"]["source_axes"],
        roi=None,
        mask_filename=mask_src,
        mask_data_group=image_specs["specs"].get("mask_group", None),
        mask_source_axes="YX",
        labels_filename=labels_src,
        labels_data_group=image_specs["specs"].get("labels_group", None),
        labels_source_axes="YX",
        labels_roi=None,
        classes_filename=classes_src,
        classes_data_group=image_specs["specs"].get("classes_group", None),
        classes_source_axes="CYX",
    )

    return args, shape, destroy_func


def randomize_roi(source_axes, image_shape):
    # Generate a ROI from the image at random
    roi = {}

    np.random.seed(478963)
    for ax, s in zip(source_axes, image_shape):
        if np.random.rand() < 0.5 and ax in "ZYX":
            if s > 1:
                ax_start = np.random.randint(0, s - 2)
                ax_length = np.random.randint(1, s - ax_start)
            else:
                ax_start = 0
                ax_length = 1

        else:
            ax_start = 0
            ax_length = -1

        roi[ax] = (ax_start, ax_length)

    return roi


def generate_roi(roi, axes, shape):
    parsable_roi = "("
    parsable_roi += ",".join([
        str(roi[ax][0]) if roi[ax][0] is not None else "0"
        for ax in axes
        ])
    parsable_roi += "):("
    parsable_roi += ",".join([
        str(roi[ax][1]) if roi[ax][1] is not None else "-1"
        for ax in axes
        ])
    parsable_roi += ")"

    new_shape = [
        roi[ax][1] if roi[ax][1] > 0 else s
        for ax, s in zip(axes, shape)
    ]

    return parsable_roi, new_shape


def permute_axes(image_shape, source_axes, axes):
    axes_perm = zds.map_axes_order(source_axes=source_axes,
                                   target_axes=axes)
    drop_axes_count = len(source_axes) - len(axes)
    permuted_shape = [image_shape[a] for a in axes_perm[drop_axes_count:]]
    return permuted_shape


def randomize_axes(source_axes, shape):
    np.random.seed(478963)
    axes = np.random.permutation(list(source_axes))
    axes = "".join(
        ax
        for ax in axes
        if (shape[source_axes.index(ax)] > 1
          or (shape[source_axes.index(ax)] <= 1 and np.random.rand() < 0.5))
    )
    return axes


def generate_sample_image(image_specs, random_roi=False, random_axes=False):
    image_args, shape, destroy_func = generate_image_data(image_specs)

    if random_roi:
        roi = randomize_roi(image_args["source_axes"], shape)

        (image_args["roi"],
         expected_shape) = generate_roi(roi, image_args["source_axes"], shape)
    else:
        expected_shape = shape

    if random_axes:
        image_args["axes"] = randomize_axes(image_args["source_axes"],
                                            expected_shape)
        expected_shape = permute_axes(expected_shape,
                                      image_args["source_axes"],
                                      image_args["axes"])
    else:
        image_args["axes"] = image_args["source_axes"]

    return image_args, expected_shape, destroy_func


def generate_sample_dataset(dataset_specs, random_roi=False,
                            random_axes=False,
                            apply_transform=False,
                            input_dtype=np.float32,
                            target_dtype=np.int64,
                            input_target_dtype=np.float64):
    dataset_shapes = []
    dataset_destroyers = []
    dataset_args = None

    for image_specs in dataset_specs:
        (image_args,
         shape,
         destroy_func) = generate_image_data(image_specs)

        dataset_shapes.append(shape)
        dataset_destroyers.append(destroy_func)

        if dataset_args is None:
            dataset_args = image_args
            dataset_args["filenames"] = [image_args["filename"]]
            if image_args["labels_filename"] is not None:
                dataset_args["labels_filenames"] =\
                    [image_args["labels_filename"]]
            if image_args["mask_filename"] is not None:
                dataset_args["mask_filenames"] =\
                    [image_args["mask_filename"]]
        else:
            dataset_args["filenames"].append(image_args["filename"])

            if image_args["labels_filename"] is not None:
                dataset_args["labels_filenames"].append(
                    image_args["labels_filename"]
                )

            if image_args["mask_filename"] is not None:
                dataset_args["mask_filenames"].append(
                    image_args["mask_filename"]
                )

    min_shape = np.stack(dataset_shapes).min(axis=0)

    if random_roi:
        roi = randomize_roi(dataset_args["source_axes"], min_shape)
        expected_shapes = []
        expected_labels_shapes = []

        for curr_shape in dataset_shapes:
            (dataset_args["roi"],
             curr_expected_shape) = generate_roi(
                 roi, dataset_args["source_axes"], curr_shape)

            curr_lables_shape = [
                curr_shape[dataset_args["source_axes"].index(ax)]
                for ax in dataset_args["labels_source_axes"]
                if ax in dataset_args["source_axes"]
            ]
            (dataset_args["labels_roi"],
             curr_label_expected_shape) = generate_roi(
                 roi, dataset_args["labels_source_axes"], curr_lables_shape)

            expected_shapes.append(curr_expected_shape)
            expected_labels_shapes.append(curr_label_expected_shape)

        _, min_labels_shape = generate_roi(roi,
                                           dataset_args["labels_source_axes"],
                                           min_shape)

        _, min_shape = generate_roi(roi, dataset_args["source_axes"],
                                    min_shape)

    else:
        expected_shapes = dataset_shapes
        expected_labels_shapes = [
            [
                img_shape[dataset_args["source_axes"].index(ax)]
                for ax in dataset_args["labels_source_axes"]
                if ax in dataset_args["source_axes"]
            ]
            for img_shape in dataset_shapes
        ]

        min_labels_shape = [
            min_shape[dataset_args["source_axes"].index(ax)]
            for ax in dataset_args["labels_source_axes"]
            if ax in dataset_args["source_axes"]
        ]

    if random_axes:
        dataset_args["axes"] = randomize_axes(dataset_args["source_axes"],
                                              min_shape)
        dataset_args["labels_axes"] = randomize_axes(
            dataset_args["labels_source_axes"], min_labels_shape)

        expected_shapes = [
            permute_axes(img_shape, dataset_args["source_axes"],
                         dataset_args["axes"])
            for img_shape in expected_shapes
        ]

        expected_labels_shapes = [
            permute_axes(labels_shape, dataset_args["labels_source_axes"],
                         dataset_args["labels_axes"])
            for labels_shape in expected_labels_shapes
        ]

    if apply_transform:
        dataset_args["transform"] = zds.ToDtype(dtype=input_dtype)

        if dataset_args["labels_filename"] is not None:
            dataset_args["target_transform"] = zds.ToDtype(dtype=target_dtype)

            dataset_args["input_target_transform"] = partial(
                input_target_transform, dtype=input_target_dtype)

    return (dataset_args, expected_shapes, expected_labels_shapes,
            dataset_destroyers)


IMAGE_SPECS = [
    {
        "source": "https://uk1s3.embassy.ebi.ac.uk/idr/zarr/v0.1/6001237.zarr",
        "type": "url",
        "specs": {
            "data_group": "0",
            "shape": [1, 4, 39, 1024, 1024],
            "chunks": [1, 1, 1, 1024, 1024],
            "source_axes": "TCZYX",
            "dtype": np.uint16
        }
    },
    {
        "source": "https://uk1s3.embassy.ebi.ac.uk/idr/zarr/v0.4/idr0048A/9846151.zarr/",
        "type": "url",
        "specs": {
            "data_group": "0/0",
            "shape": [1, 3, 1402, 5192, 2947],
            "chunks": [1, 1, 1, 1024, 1024],
            "source_axes": "TCZYX",
            "dtype": np.uint16
        }
    },
    {
        "source": dict(dst_dir="tests/test_zarrs", img_idx=1),
        "type": "zarr_group",
        "specs": {
            "data_group": "0/0",
            "mask_group": "masks/0/0",
            "labels_group": "labels/0/0",
            "classes_group": "classes/0/0",
            "shape": [768, 512, 3],
            "chunks": [768, 512, 3],
            "source_axes": "YXC",
            "dtype": np.uint8
        }
    },
    {
        "source": dict(dst_dir="tests/test_zarrs", img_idx=2),
        "type": "zarr_file",
        "specs": {
            "data_group": "0",
            "mask_group": "masks/0",
            "labels_group": "labels/0",
            "classes_group": "classes/0",
            "shape": [1, 4, 1125, 512],
            "chunks": [1, 4, 256, 128],
            "source_axes": "TCYX",
            "dtype": np.int32
        }
    },
    {
        "source": dict(dst_dir="tests/test_tiffs", img_idx=3),
        "type": "tiff",
        "specs": {
            "data_group": "0",
            "mask_group": "1",
            "labels_group": "2",
            "classes_group": "3",
            "shape": [1125, 512, 3],
            "chunks": [256, 128, 3],
            "source_axes": "YXC",
            "dtype": np.uint16
        }
    },
    {
        "source": dict(dst_dir="tests/test_pngs", img_idx=4),
        "type": "png",
        "specs": {
            "data_group": "0",
            "mask_group": "masks/0",
            "labels_group": "labels/0",
            "classes_group": "classes/0",
            "shape": [1125, 512, 3],
            "chunks": [256, 128, 3],
            "source_axes": "YXC",
            "dtype": np.uint8
        }
    },
    {
        "source": dict(dst_dir=None, img_idx=None),
        "type": "ndarray",
        "specs": {
            "data_group": "0",
            "mask_group": "masks/0",
            "labels_group": "labels/0",
            "classes_group": "classes/0",
            "shape": [1125, 3, 512],
            "chunks": [256, 3, 128],
            "source_axes": "YCX",
            "dtype": np.uint8
        }
    },
    {
        "source": dict(dst_dir=None, img_idx=None),
        "type": "zarr_array",
        "specs": {
            "data_group": "0",
            "mask_group": "masks/0",
            "labels_group": "labels/0",
            "classes_group": "classes/0",
            "shape": [1, 3, 1125, 512, 1],
            "chunks": [1, 3, 256, 128, 1],
            "source_axes": "ZCXYT",
            "dtype": np.float32
        }
    },
    {
        "source": "https://live.staticflickr.com/4908/31072787307_59f7943caa_o.jpg",
        "credit": "Photo: Guilhem Vellut / CC BY 2.0",
        "type": "url",
        "specs": {
            "data_group": "",
            "shape": [5472, 3648, 3],
            "chunks": [5472, 3648, 1],
            "source_axes": "YXC",
            "dtype": np.uint8
        }
    },
]

UNLABELED_DATASET_SPECS = [
    [
        {
            "source": "https://uk1s3.embassy.ebi.ac.uk/idr/zarr/v0.1/6001237.zarr",
            "type": "zarr",
            "specs": {
                "data_group": "2",
                "shape": [1, 4, 39, 256, 256],
                "chunks": [1, 1, 1, 256, 256],
                "source_axes": "TCZYX",
                "dtype": np.uint16
            }
        },
        {
            "source": "https://uk1s3.embassy.ebi.ac.uk/idr/zarr/v0.1/6001237.zarr",
            "type": "zarr",
            "specs": {
                "data_group": "2",
                "shape": [1, 4, 39, 256, 256],
                "chunks": [1, 1, 1, 256, 256],
                "source_axes": "TCZYX",
                "dtype": np.uint16
            }
        },
    ],
]


LABELED_DATASET_SPECS = [
    [
        {
            "source": dict(dst_dir=None, img_idx=None),
            "type": "zarr_array",
            "specs": {
                "data_group": "0",
                "mask_group": "masks/0",
                "labels_group": "labels/0",
                "classes_group": "classes/0",
                "shape": [1, 3, 1125, 512, 1],
                "chunks": [1, 3, 256, 128, 1],
                "source_axes": "ZCXYT",
                "dtype": np.float32
            }
        },
        {
            "source": dict(dst_dir=None, img_idx=None),
            "type": "zarr_array",
            "specs": {
                "data_group": "0",
                "mask_group": "masks/0",
                "labels_group": "labels/0",
                "classes_group": "classes/0",
                "shape": [1, 3, 778, 1015, 1],
                "chunks": [1, 3, 128, 128, 1],
                "source_axes": "ZCXYT",
                "dtype": np.float32
            }
        },
    ],
    [
        {
            "source": dict(dst_dir="tests/test_pngs", img_idx=0),
            "type": "png",
            "specs": {
                "data_group": "0",
                "mask_group": "masks/0",
                "labels_group": "labels/0",
                "classes_group": "classes/0",
                "shape": [1027, 528, 3],
                "chunks": [256, 128, 3],
                "source_axes": "YXC",
                "dtype": np.uint8
            }
        },
    ],
]


UNLABELED_3D_DATASET_SPECS = [
    [
        {
            "source": "https://uk1s3.embassy.ebi.ac.uk/idr/zarr/v0.1/6001240.zarr",
            "type": "url",
            "specs": {
                "data_group": "0",
                "shape": [1, 2, 236, 275, 271],
                "chunks": [1, 1, 1, 275, 271],
                "source_axes": "TCZYX",
                "dtype": np.uint16
            }
        },
        {
            "source": "https://uk1s3.embassy.ebi.ac.uk/idr/zarr/v0.1/6001241.zarr",
            "type": "url",
            "specs": {
                "data_group": "0",
                "shape": [1, 2, 236, 263, 278],
                "chunks": [1, 1, 1, 263, 278],
                "source_axes": "TCZYX",
                "dtype": np.uint16
            }
        },
    ],
]


def base_test_image_loader(image_specs, random_roi, random_axes,
                           apply_transform):

    (image_args,
     expected_shape,
     destroy_func) = generate_sample_image(image_specs, random_roi=random_roi,
                                           random_axes=random_axes)

    if apply_transform:
        transform = ImageTransformTest(image_args["axes"])
    else:
        transform = None

    img = zds.ImageLoader(
        filename=image_args["filename"],
        source_axes=image_args["source_axes"],
        data_group=image_args["data_group"],
        axes=image_args["axes"],
        roi=image_args["roi"],
        image_func=transform,
        zarr_store=None,
        spatial_axes="ZYX",
        mode="r")

    assert isinstance(img, zds.ImageBase), (f"Image loader returned an "
                                            f"incorrect type of object, "
                                            f"expected one based on ImageBase,"
                                            f" got {type(img)}")
    assert all(map(lambda s1, s2: s1 == s2, img.shape, expected_shape)),\
          (f"Expected image of shape {expected_shape}"
           f", got {img.shape}")

    del img

    destroy_func()


def base_test_zarrdataset(dataset_specs, patch_sampler_class, dataset_class,
                          random_roi,
                          random_axes,
                          apply_transform,
                          input_dtype,
                          target_dtype,
                          input_target_dtype):

    dataset_classes = {
        "ZarrDataset": zds.ZarrDataset,
        "LabeledZarrDataset": zds.LabeledZarrDataset,
        "MaskedZarrDataset": zds.MaskedZarrDataset,
        "MaskedLabeledZarrDataset": zds.MaskedLabeledZarrDataset
    }

    patch_sampler_classes = {
        "None": lambda patch_size: None,
        "GridPatchSampler": \
            lambda patch_size: zds.GridPatchSampler(patch_size),
        "BlueNoisePatchSampler": \
            lambda patch_size: zds.BlueNoisePatchSampler(patch_size),
    }

    (dataset_args,
     expected_shapes,
     expected_labels_shapes,
     destroy_funcs) = generate_sample_dataset(
         dataset_specs,
         random_axes=random_axes,
         random_roi=random_roi,
         apply_transform=apply_transform,
         input_dtype=input_dtype,
         target_dtype=target_dtype,
         input_target_dtype=input_target_dtype)

    min_img_shape = np.array(expected_shapes).min(axis=0)

    patch_size = dict(
        (ax, np.random.randint(s // 8, s // 4) if s > 8 else 1)
        for ax, s in zip(dataset_args["axes"], min_img_shape)
        if ax in "ZYX"
    )

    patch_sampler = patch_sampler_classes[patch_sampler_class](patch_size)

    ds = dataset_classes[dataset_class](**dataset_args,
                       patch_sampler=patch_sampler,
                       return_any_label=True)

    if not apply_transform:
        input_dtype = dataset_specs[0]["specs"]["dtype"]
        if dataset_specs[0]["type"] in ["png"]:
            label_dtype = np.uint8
        else:
            label_dtype = np.uint32

    elif dataset_class in ["LabeledZarrDataset", "MaskedLabeledZarrDataset"]:
        input_dtype = input_target_dtype
        label_dtype = target_dtype

    for i, (img_pair, expected_shape, expected_labels_shape) in enumerate(
      zip(ds, expected_shapes, expected_labels_shapes)):
        img, label = img_pair

        if patch_sampler is not None:
            expected_shape = dict(
                (ax, patch_size[ax] if ax in patch_size else s)
                for ax, s in zip(dataset_args["axes"], expected_shape)
            )

            expected_labels_shape = dict(
                (ax, patch_size[ax] if ax in patch_size else s)
                for ax, s in zip(dataset_args["labels_axes"],
                                 expected_labels_shape)
            )

        else:
            expected_shape = dict(
                (ax, s)
                for ax, s in zip(dataset_args["axes"], expected_shape)
            )

            expected_labels_shape = dict(
                (ax, s)
                for ax, s in zip(dataset_args["labels_axes"],
                                 expected_labels_shape)
            )

        assert isinstance(img, np.ndarray),\
            (f"Sample {i}, expected to be a Numpy NDArray, got {type(img)}")

        assert img.dtype == input_dtype,\
            (f"Expected sample {i} be of type {input_dtype}, got {img.dtype}")

        assert all(map(lambda s, ax:
                       s == expected_shape[ax],
                       img.shape,
                       dataset_args["axes"])),\
            (f"Expected sample {i} of shape {expected_shape}, got {img.shape}"
             f" [{dataset_args['axes']}]")

        if dataset_class in ["LabeledZarrDataset", "MaskedLabeledZarrDataset"]:
            assert all(map(lambda s, ax:
                           s == expected_labels_shape[ax],
                           label.shape,
                           dataset_args["labels_axes"])),\
                (f"Expected label {i} of shape {expected_labels_shape}, "
                 f"got {label.shape}")

            assert label.dtype == label_dtype,\
                (f"Expected label {i} be of type {label_dtype}, got"
                 f" {label.dtype}")

    for destroy_func in destroy_funcs:
        destroy_func()


def base_test_zarrdataset_pytorch(dataset_specs, patch_sampler_class,
                                    dataset_class,
                                    random_roi,
                                    random_axes,
                                    apply_transform,
                                    input_dtype,
                                    target_dtype,
                                    input_target_dtype,
                                    num_workers):

    dataset_classes = {
        "ZarrDataset": zds.ZarrDataset,
        "LabeledZarrDataset": zds.LabeledZarrDataset,
        "MaskedZarrDataset": zds.MaskedZarrDataset,
        "MaskedLabeledZarrDataset": zds.MaskedLabeledZarrDataset
    }

    patch_sampler_classes = {
        "None": lambda patch_size: None,
        "GridPatchSampler": \
            lambda patch_size: zds.GridPatchSampler(patch_size),
        "BlueNoisePatchSampler": \
            lambda patch_size: zds.BlueNoisePatchSampler(patch_size),
    }

    (dataset_args,
    expected_shapes,
    expected_labels_shapes,
    destroy_funcs) = generate_sample_dataset(
        dataset_specs,
        random_axes=random_axes,
        random_roi=random_roi,
        apply_transform=apply_transform,
        input_dtype=input_dtype,
        target_dtype=target_dtype,
        input_target_dtype=input_target_dtype)

    min_img_shape = np.array(expected_shapes).min(axis=0)

    patch_size = dict(
        (ax, np.random.randint(s // 8, s // 4) if s > 8 else 1)
        for ax, s in zip(dataset_args["axes"], min_img_shape)
        if ax in "ZYX"
    )

    patch_sampler = patch_sampler_classes[patch_sampler_class](patch_size)

    ds = dataset_classes[dataset_class](**dataset_args,
                        patch_sampler=patch_sampler,
                        return_any_label=True)

    dl = torch.utils.data.DataLoader(
        ds, num_workers=num_workers,
        worker_init_fn=zds.zarrdataset_worker_init,
        batch_size=1)

    if not apply_transform:
        meta = np.empty((0, ), dtype=dataset_specs[0]["specs"]["dtype"])
        meta_pt = torch.from_numpy(meta)
        input_dtype = meta_pt.dtype
        if dataset_specs[0]["type"] in ["png"]:
            label_dtype = torch.uint8
        else:
            label_dtype = torch.int32

    elif dataset_class in ["LabeledZarrDataset", "MaskedLabeledZarrDataset"]:
        meta = np.empty((0, ), dtype=input_target_dtype)
        meta_pt = torch.from_numpy(meta)
        input_dtype = meta_pt.dtype

        meta = np.empty((0, ), dtype=target_dtype)
        meta_pt = torch.from_numpy(meta)
        label_dtype = meta_pt.dtype

    else:
        meta = np.empty((0, ), dtype=input_dtype)
        meta_pt = torch.from_numpy(meta)
        input_dtype = meta_pt.dtype

        label_dtype = None

    for i, (img_pair, expected_shape, expected_labels_shape) in enumerate(
        zip(dl, expected_shapes, expected_labels_shapes)):
        img, label = img_pair

        if patch_sampler is not None:
            expected_shape = dict(
                (ax, patch_size[ax] if ax in patch_size else s)
                for ax, s in zip(dataset_args["axes"], expected_shape)
            )

            expected_labels_shape = dict(
                (ax, patch_size[ax] if ax in patch_size else s)
                for ax, s in zip(dataset_args["labels_axes"],
                                    expected_labels_shape)
            )

        else:
            expected_shape = dict(
                (ax, s)
                for ax, s in zip(dataset_args["axes"], expected_shape)
            )

            expected_labels_shape = dict(
                (ax, s)
                for ax, s in zip(dataset_args["labels_axes"],
                                    expected_labels_shape)
            )

        assert isinstance(img, torch.Tensor),\
            (f"Sample {i}, expected to be a PyTorch Tensor, got "
                f"{type(img)}")

        assert img.dtype == input_dtype,\
            (f"Expected sample {i} be of type {input_dtype}, got "
                f"{img.dtype}")

        assert all(map(lambda s, ax:
                    s == expected_shape[ax],
                    img.shape[1:],
                    dataset_args["axes"])),\
            (f"Expected sample {i} of shape {expected_shape}, got "
                f"{img.shape[1:]} [{dataset_args['axes']}]")

        if dataset_class in ["LabeledZarrDataset", "MaskedLabeledZarrDataset"]:
            assert all(map(lambda s, ax:
                        s == expected_labels_shape[ax],
                        label.shape[1:],
                        dataset_args["labels_axes"])),\
                (f"Expected label {i} of shape {expected_labels_shape}, "
                    f"got {label.shape[1:]}")

            assert label.dtype == label_dtype,\
                (f"Expected label {i} be of type {label_dtype}, got"
                f" {label.dtype}")

    for destroy_func in destroy_funcs:
        destroy_func()


def base_test_zarrdataset_chain_pytorch(dataset_specs, patch_sampler_class,
                                        dataset_class,
                                        random_roi,
                                        random_axes,
                                        apply_transform,
                                        input_dtype,
                                        target_dtype,
                                        input_target_dtype,
                                        num_workers):
    dataset_classes = {
        "ZarrDataset": zds.ZarrDataset,
        "LabeledZarrDataset": zds.LabeledZarrDataset,
        "MaskedZarrDataset": zds.MaskedZarrDataset,
        "MaskedLabeledZarrDataset": zds.MaskedLabeledZarrDataset
    }

    patch_sampler_classes = {
        "None": lambda patch_size: None,
        "GridPatchSampler": \
            lambda patch_size: zds.GridPatchSampler(patch_size),
        "BlueNoisePatchSampler": \
            lambda patch_size: zds.BlueNoisePatchSampler(patch_size),
    }

    (dataset_args,
    expected_shapes,
    expected_labels_shapes,
    destroy_funcs) = generate_sample_dataset(
        dataset_specs,
        random_axes=random_axes,
        random_roi=random_roi,
        apply_transform=apply_transform,
        input_dtype=input_dtype,
        target_dtype=target_dtype,
        input_target_dtype=input_target_dtype)

    min_img_shape = np.array(expected_shapes).min(axis=0)

    patch_size = dict(
        (ax, np.random.randint(s // 8, s // 4) if s > 8 else 1)
        for ax, s in zip(dataset_args["axes"], min_img_shape)
        if ax in "ZYX"
    )

    patch_sampler = patch_sampler_classes[patch_sampler_class](patch_size)

    filenames = list(dataset_args["filenames"])
    del dataset_args["filenames"]

    assert isinstance(str(dataset_args["transform"]), str),\
        (f"Expected description of the pre-processing transform pipeline "
            f"to be a string, got {type(str(dataset_args['transform']))}")

    ds = [dataset_classes[dataset_class](filenames=fn,
                        **dataset_args,
                        patch_sampler=patch_sampler,
                        return_any_label=True)
            for fn in filenames]

    chain_ds = torch.utils.data.ChainDataset(ds)

    dl = torch.utils.data.DataLoader(
        chain_ds, num_workers=num_workers,
        worker_init_fn=zds.chained_zarrdataset_worker_init,
        batch_size=1)

    if not apply_transform:
        meta = np.empty((0, ), dtype=dataset_specs[0]["specs"]["dtype"])
        meta_pt = torch.from_numpy(meta)
        input_dtype = meta_pt.dtype
        if dataset_specs[0]["type"] in ["png"]:
            label_dtype = torch.uint8
        else:
            label_dtype = torch.int32

    elif dataset_class in ["LabeledZarrDataset",
                            "MaskedLabeledZarrDataset"]:
        meta = np.empty((0, ), dtype=input_target_dtype)
        meta_pt = torch.from_numpy(meta)
        input_dtype = meta_pt.dtype

        meta = np.empty((0, ), dtype=target_dtype)
        meta_pt = torch.from_numpy(meta)
        label_dtype = meta_pt.dtype

    else:
        meta = np.empty((0, ), dtype=input_dtype)
        meta_pt = torch.from_numpy(meta)
        input_dtype = meta_pt.dtype

        label_dtype = None

    for i, (img_pair, expected_shape, expected_labels_shape) in enumerate(
        zip(dl, expected_shapes, expected_labels_shapes)):
        img, label = img_pair

        if patch_sampler is not None:
            expected_shape = dict(
                (ax, patch_size[ax] if ax in patch_size else s)
                for ax, s in zip(dataset_args["axes"], expected_shape)
            )

            expected_labels_shape = dict(
                (ax, patch_size[ax] if ax in patch_size else s)
                for ax, s in zip(dataset_args["labels_axes"],
                                    expected_labels_shape)
            )

        else:
            expected_shape = dict(
                (ax, s)
                for ax, s in zip(dataset_args["axes"], expected_shape)
            )

            expected_labels_shape = dict(
                (ax, s)
                for ax, s in zip(dataset_args["labels_axes"],
                                    expected_labels_shape)
            )

        assert isinstance(img, torch.Tensor),\
            (f"Sample {i}, expected to be a PyTorch Tensor, got "
                f"{type(img)}")

        assert img.dtype == input_dtype,\
            (f"Expected sample {i} be of type {input_dtype}, got "
                f"{img.dtype}")

        assert all(map(lambda s, ax:
                    s == expected_shape[ax],
                    img.shape[1:],
                    dataset_args["axes"])),\
            (f"Expected sample {i} of shape {expected_shape}, got "
                f"{img.shape[1:]} [{dataset_args['axes']}]")

        if dataset_class in ["LabeledZarrDataset",
                                "MaskedLabeledZarrDataset"]:
            assert all(map(lambda s, ax:
                        s == expected_labels_shape[ax],
                        label.shape[1:],
                        dataset_args["labels_axes"])),\
                (f"Expected label {i} of shape {expected_labels_shape}, "
                f"got {label.shape[1:]}")

            assert label.dtype == label_dtype,\
                (f"Expected label {i} be of type {label_dtype}, got"
                f" {label.dtype}")

    for destroy_func in destroy_funcs:
        destroy_func()


if __name__ == "__main__":
    from itertools import product

    image_specs_pars = IMAGE_SPECS[4:5]
    random_roi_pars = [False]
    random_axes_pars = [False]
    apply_transform_pars = [False]

    #------------------- Test Image Loader
    for image_specs, random_roi, random_axes, apply_transform in product(
      image_specs_pars,
      random_roi_pars,
      random_axes_pars,
      apply_transform_pars):
        base_test_image_loader(image_specs, random_roi, random_axes,
                               apply_transform)

    # dataset_specs_pars = LABELED_DATASET_SPECS
    # random_roi_pars = [True]
    # random_axes_pars = [True]
    # patch_sampler_class_pars = [
    #     lambda patch_size: None
    # ]
    # dataset_class_pars = [
    #     LabeledZarrDataset,
    #     MaskedLabeledZarrDataset,
    # ]
    # input_dtype_pars = [np.uint16]
    # target_dtype_pars = [np.float32]
    # input_target_dtype_pars = [np.float64]

    # for (dataset_specs, patch_sampler_class, dataset_class,
    #      random_roi,
    #      random_axes,
    #      apply_transform,
    #      input_dtype,
    #      target_dtype,
    #      input_target_dtype) in tqdm.tqdm(
    #          product(dataset_specs_pars, patch_sampler_class_pars,
    #                  dataset_class_pars,
    #                  random_roi_pars,
    #                  random_axes_pars,
    #                  apply_transform_pars,
    #                  input_dtype_pars,
    #                  target_dtype_pars,
    #                  input_target_dtype_pars)):
    #     base_test_zarrdataset(dataset_specs, patch_sampler_class, dataset_class,
    #                           random_roi,
    #                           random_axes,
    #                           apply_transform,
    #                           input_dtype,
    #                           target_dtype,
    #                           input_target_dtype)

    # ---------------------- Test Torch DataLoader compatibility
    # dataset_specs_pars = LABELED_DATASET_SPECS
    # random_roi_pars = [True]
    # random_axes_pars = [True]
    # dataset_class_pars = ["MaskedZarrDataset"]
    # patch_sampler_class_pars = ["GridPatchSampler"]
    # apply_transform_pars = [True]
    # input_dtype_pars = [np.float32]
    # target_dtype_pars = [np.int64]
    # input_target_dtype_pars = [np.float64]
    # num_workers_pars = [0, 1, 2, 3]

    # for (dataset_specs, patch_sampler_class, dataset_class, random_roi, 
    #         random_axes,
    #         apply_transform,
    #         input_dtype,
    #         target_dtype,
    #         input_target_dtype,
    #         num_workers) in product(dataset_specs_pars,
    #                                         patch_sampler_class_pars,
    #                                         dataset_class_pars,
    #                                         random_roi_pars,
    #                                         random_axes_pars,
    #                                         apply_transform_pars,
    #                                         input_dtype_pars,
    #                                         target_dtype_pars,
    #                                         input_target_dtype_pars,
    #                                         num_workers_pars):
    #     base_test_zarrdataset_pytorch(dataset_specs, patch_sampler_class,
    #                                     dataset_class,
    #                                     random_roi,
    #                                     random_axes,
    #                                     apply_transform,
    #                                     input_dtype,
    #                                     target_dtype,
    #                                     input_target_dtype,
    #                                     num_workers)
    import importlib
    with mock.patch.dict('sys.modules', {'torch': None}):
        importlib.reload(zds._zarrdataset)

        dataset = zds._zarrdataset.ZarrDataset(filenames=[], source_axes="")
