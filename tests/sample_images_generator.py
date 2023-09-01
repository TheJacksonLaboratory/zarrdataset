import shutil
from pathlib import Path
import numpy as np
import zarr
from PIL import Image
import tifffile
from zarrdataset import *


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

    rel_axes_order = map_axes_order(source_axes=source_axes,
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
    permute_order = map_axes_order(source_axes=img_axes,
                                   target_axes=source_axes)
    img = img.transpose(permute_order)
    img = np.expand_dims(img, tuple(source_axes.index(a) for a in new_axes))

    store = zarr.MemoryStore()
    main_grp = zarr.open(store)
    main_grp.create_dataset(data_group, data=img, dtype=np.uint8,
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
    destroy_func = lambda: None
    return z_groups, None, None, None, destroy_func


def generate_zarr_array(dst_dir, img_idx, image_specs):
    z_groups = generate_collection(**image_specs)

    img_src = z_groups[image_specs["data_group"]]

    if image_specs.get("mask_group", None) is not None:
        mask_src = z_groups[image_specs["mask_group"]]
    else:
        mask_src = None

    if image_specs.get("labels_group", None) is not None:
        labels_src = z_groups[image_specs["labels_group"]]
    else:
        labels_src = None

    if image_specs.get("classes_group", None) is not None:
        classes_src = z_groups[image_specs["classes_group"]]
    else:
        classes_src = None

    destroy_func = lambda: None

    return img_src, mask_src, labels_src, classes_src, destroy_func


def generate_ndarray(dst_dir, img_idx, image_specs):
    z_groups = generate_collection(**image_specs)

    img_src = z_groups[image_specs["data_group"]][:]

    if image_specs.get("mask_group", None) is not None:
        mask_src = z_groups[image_specs["mask_group"]][:]
    else:
        mask_src = None

    if image_specs.get("labels_group", None) is not None:
        labels_src = z_groups[image_specs["labels_group"]][:]
    else:
        labels_src = None

    if image_specs.get("classes_group", None) is not None:
        classes_src = z_groups[image_specs["classes_group"]][:]
    else:
        classes_src = None

    destroy_func = lambda: None

    return img_src, mask_src, labels_src, classes_src, destroy_func


def generate_zarr_file(dst_dir, img_idx, image_specs):
    z_groups = generate_collection(**image_specs)

    Path(dst_dir).mkdir(parents=True, exist_ok=True)
    image_filename = f'{dst_dir}/zarr_group_{img_idx}.zarr'
    z_grp_dst = zarr.open(image_filename, mode="w")
    zarr.copy_all(z_groups, z_grp_dst)

    destroy_func = lambda: shutil.rmtree(dst_dir)

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
            compression='jpeg',
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

    destroy_func = lambda: shutil.rmtree(dst_dir)

    return image_filename, None, None, None, destroy_func


def generate_pngs(dst_dir, img_idx, image_specs):
    z_groups = generate_collection(**image_specs)

    Path(dst_dir).mkdir(parents=True, exist_ok=True)

    permute_order = map_axes_order(source_axes=image_specs["source_axes"],
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

    destroy_func = lambda: shutil.rmtree(dst_dir)

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
        s3_conn = connect_s3(image_specs["source"])
        arr_src, store = image2array(image_specs["source"],
                                     image_specs["specs"]["data_group"],
                                     s3_obj=s3_conn)
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
                ax_shape = ax_length
            else:
                ax_start = 0
                ax_length = 1
                ax_shape = 1

        else:
            ax_start = 0
            ax_length = -1
            ax_shape = s

        roi[ax] = (ax_start, ax_length, ax_shape)

    return roi


def generate_roi(roi, axes):
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

    new_axes = "".join([ax for ax in axes if roi[ax][2] > 1])
    new_shape = [roi[ax][2] for ax in axes if roi[ax][2] > 1]

    return parsable_roi, new_axes, new_shape


def permute_axes(image_shape, source_axes, axes):
    axes_perm = map_axes_order(source_axes=source_axes,
                               target_axes=axes)
    permuted_shape = [image_shape[a] for a in axes_perm]
    return permuted_shape


def randomize_axes(source_axes):
    np.random.seed(478963)
    axes = np.random.permutation(list(source_axes))
    axes = "".join(axes)
    return axes


def generate_sample_image(image_specs, random_roi=False, random_axes=False):
    image_args, shape, destroy_func = generate_image_data(image_specs)

    if random_roi:
        roi = randomize_roi(image_args["source_axes"], shape)

        (image_args["roi"],
         reduced_axes,
         expected_shape) = generate_roi(roi, image_args["source_axes"])
    else:
        reduced_axes = image_args["source_axes"]
        expected_shape = shape

    if random_axes:
        image_args["axes"] = randomize_axes(reduced_axes)
        expected_shape = permute_axes(expected_shape,
                                      reduced_axes,
                                      image_args["axes"])
    else:
        image_args["axes"] = reduced_axes

    return image_args, expected_shape, destroy_func


def generate_sample_dataset(dataset_specs, random_roi=False,
                            random_axes=False):
    dataset_shapes = []
    dataset_args = None

    for image_specs in dataset_specs:
        (image_args,
         shape,
         destroy_func) = generate_image_data(image_specs)

        dataset_shapes.append(shape)

        if dataset_args is None:
            dataset_args = image_args
            dataset_args["filenames"] = image_args["filename"]
        else:
            dataset_args["filenames"].append(image_args["filename"])

    min_shape = np.stack(dataset_shapes).min(axis=0)

    if random_roi:
        roi = randomize_roi(dataset_args["source_axes"], min_shape)

        (dataset_args["roi"],
         reduced_axes,
         expected_shape) = generate_roi(roi, dataset_args["source_axes"])

        (dataset_args["labels_roi"],
         labels_reduced_axes,
         labels_expected_shape) = generate_roi(
            roi, dataset_args["labels_source_axes"])

        expected_shapes = [expected_shape] * len(dataset_shapes)
        expected_labels_shapes = [labels_expected_shape] * len(dataset_shapes)

    else:
        reduced_axes = dataset_args["source_axes"]
        labels_reduced_axes = dataset_args["labels_source_axes"]

        expected_shapes = dataset_shapes

        expected_labels_shapes = [
            [s
             for ax, s in zip(dataset_args["source_axes"], img_shape)
             if ax in dataset_args["labels_source_axes"]]
            for img_shape in dataset_shapes
        ]

    if random_axes:
        dataset_args["axes"] = randomize_axes(reduced_axes)
        dataset_args["labels_axes"] = randomize_axes(labels_reduced_axes)

        expected_shape = [
            permute_axes(img_shape, reduced_axes, dataset_args["axes"])
            for img_shape in expected_shapes
        ]

        expected_labels_shapes = [
            permute_axes(labels_shape, labels_reduced_axes,
                         dataset_args["labels_axes"])
            for labels_shape in expected_labels_shapes
        ]
    else:
        dataset_args["axes"] = reduced_axes
        dataset_args["labels_axes"] = labels_reduced_axes

    return image_args, expected_shapes, expected_labels_shapes, destroy_func


IMAGE_SPECS = [
    {
        "source": "https://uk1s3.embassy.ebi.ac.uk/idr/zarr/v0.1/6001237.zarr",
        "type": "zarr",
        "specs": {
            "data_group": "0",
            "shape": [1, 4, 39, 1024, 1024],
            "chunks": [1, 1, 1, 1024, 1024],
            "source_axes": "TCZYX",
            "dtype": np.uint16
        }
    },
    {
        "source": "https://uk1s3.embassy.ebi.ac.uk/idr/zarr/v0.1/6001237.zarr",
        "type": "zarr",
        "specs": {
            "data_group": "1",
            "shape": [1, 4, 39, 512, 512],
            "chunks": [1, 1, 1, 512, 512],
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
    {
        "source": "https://r0k.us/graphics/kodak/kodak/kodim01.png",
        "type": "image",
        "specs": {
            "data_group": "",
            "shape": [512, 768, 3],
            "chunks": [512, 768, 3],
            "source_axes": "YXC",
            "dtype": np.uint8
        }
    },
    {
        "source": "https://r0k.us/graphics/kodak/kodak/kodim02.png",
        "type": "image",
        "specs": {
            "data_group": "",
            "shape": [512, 768, 3],
            "chunks": [512, 768, 3],
            "source_axes": "YXC",
            "dtype": np.uint8
        }
    },
    {
        "source": "https://r0k.us/graphics/kodak/kodak/kodim03.png",
        "type": "image",
        "specs": {
            "data_group": "",
            "shape": [512, 768, 3],
            "chunks": [512, 768, 3],
            "source_axes": "YXC",
            "dtype": np.uint8
        }
    },
    {
        "source": "https://r0k.us/graphics/kodak/kodak/kodim04.png",
        "type": "image",
        "specs": {
            "data_group": "",
            "shape": [768, 512, 3],
            "chunks": [768, 512, 3],
            "source_axes": "YXC",
            "dtype": np.uint8
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
]


if __name__ == "__main__":
    np.random.seed(478963)

    (image_args,
     expected_shape,
     destroy_func) = generate_sample_image(IMAGE_SPECS[-1],
                                           random_roi=False,
                                           random_axes=True)

    img = ImageLoader(
        filename=image_args["filename"],
        source_axes=image_args["source_axes"],
        data_group=image_args["data_group"],
        axes=image_args["axes"],
        roi=image_args["roi"],
        image_func=None,
        zarr_store=None,
        spatial_axes="ZYX",
        mode="r")

    assert isinstance(img, ImageBase), (f"Image loader returned an incorrect"
                                        f" type of object, expected one based"
                                        f" in ImageBase, got {type(img)}")
    assert img.shape == expected_shape,\
          (f"Expected image of shape {expected_shape}, got {img.shape}")

    del img

    destroy_func()