import zarrdataset as zds

from pathlib import Path
import numpy as np
import zarr
from PIL import Image
import tifffile


class ImageTransformTest(zds.MaskGenerator):
    def __init__(self, axes):
        super(ImageTransformTest, self).__init__(axes=axes)

    def _compute_transform(self, image: np.ndarray):
        return image * 2


def input_target_transform(x, t, dtype=np.float32):
    return x.astype(dtype), t.astype(dtype)


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
    mask = mask[::2, ::2]

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


def generate_zarr_group(dst_dir, image_specs):
    z_groups = generate_collection(**image_specs)

    return z_groups, None, None, None


def generate_zarr_array(dst_dir, image_specs):
    if dst_dir is None:
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

        return img_src, mask_src, labels_src, classes_src

    else:
        image_specs["data_group"] = "0"
        z_groups = generate_collection(**image_specs)
        image_specs["data_group"] = None

        img_idx = np.random.randint(10000)

        image_filename = f'{dst_dir}/zarr_array_{img_idx}.zarr'

        z_arr = zarr.open(image_filename, mode='w', shape=z_groups["0"].shape,
                          chunks=z_groups["0"].chunks)

        z_arr[:] = z_groups["0"][:]

        return image_filename, None, None, None


def generate_ndarray(dst_dir, image_specs):
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

    return img_src, mask_src, labels_src, classes_src


def generate_zarr_file(dst_dir, image_specs):
    z_groups = generate_collection(**image_specs)

    img_idx = np.random.randint(10000)

    image_filename = f'{dst_dir}/zarr_group_{img_idx}.zarr'
    z_grp_dst = zarr.open(image_filename, mode="w")
    zarr.copy_all(z_groups, z_grp_dst)

    return image_filename, None, None, None


def generate_tiffs(dst_dir, image_specs):
    src_data_group = image_specs["data_group"]

    if src_data_group is None:
        image_specs["data_group"] = "0"

    z_groups = generate_collection(**image_specs)

    img_idx = np.random.randint(10000)

    image_filename = f'{dst_dir}/img_{img_idx}.ome.tiff'
    with tifffile.TiffWriter(image_filename, bigtiff=True) as tif:
        tif.write(
            z_groups[image_specs["data_group"]][:],
            metadata={"axes": None},
            tile=(128, 128),
            photometric='rgb'
        )

        if "mask_group" in image_specs:
            tif.write(
                z_groups[image_specs["mask_group"]][:],
                metadata={"axes": "YX"},
                tile=(128, 128),
            )

        if "labels_group" in image_specs:
            tif.write(
                z_groups[image_specs["labels_group"]][:],
                metadata={"axes": "YX"},
                tile=(128, 128),
            )

        if "classes_group" in image_specs:
            tif.write(
                z_groups[image_specs["classes_group"]][:],
                metadata={"axes": "CYX"},
            )

    image_specs["data_group"] = src_data_group

    return image_filename, None, None, None


def generate_pngs(dst_dir, image_specs):
    z_groups = generate_collection(**image_specs)

    permute_order = zds.map_axes_order(source_axes=image_specs["source_axes"],
                                       target_axes="YXC")
    img_data = z_groups[image_specs["data_group"]][:]
    img_data = img_data.transpose(permute_order)
    img_data = img_data.reshape(-1, img_data.shape[-3], img_data.shape[-2],
                                img_data.shape[-1])
    img_data = img_data[0, ..., :3]

    image = Image.fromarray(img_data)

    img_idx = np.random.randint(10000)

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

    return image_filename, mask_filename, labels_filename, None


def generate_empty_file(dst_dir, image_specs):
    filename = Path(dst_dir) / image_specs["filename"]
    fp = open(filename, "w")
    fp.close()

    return filename, None, None, None


IMAGE_SPECS = [
    {
        "dst_dir": None,
        "source": "https://uk1s3.embassy.ebi.ac.uk/idr/zarr/v0.1/6001237.zarr",
        "credit": ("© 2016-2023 University of Dundee & Open Microscopy "
                   "Environment. Creative Commons Attribution 4.0 "
                   "International License"),
        "specs": {
            "data_group": "0",
            "shape": [1, 4, 39, 1024, 1024],
            "chunks": [1, 1, 1, 1024, 1024],
            "source_axes": "TCZYX",
            "dtype": np.uint16
        }
    },
    {
        "dst_dir": None,
        "source": ("https://uk1s3.embassy.ebi.ac.uk/idr/zarr/v0.4/idr0048A/"
                   "9846151.zarr/"),
        "credit": ("© 2016-2023 University of Dundee & Open Microscopy "
                   "Environment. Creative Commons Attribution 4.0 "
                   "International License"),
        "specs": {
            "data_group": "0/0",
            "shape": [1, 3, 1402, 5192, 2947],
            "chunks": [1, 1, 1, 1024, 1024],
            "source_axes": "TCZYX",
            "dtype": np.uint16
        }
    },
    {
        "dst_dir": "tests/test_zarrs",
        "source": generate_zarr_group,
        "credit": None,
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
        "dst_dir": "tests/test_zarrs",
        "source": generate_zarr_file,
        "credit": None,
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
        "dst_dir": "tests/test_tiffs",
        "source": generate_tiffs,
        "credit": None,
        "specs": {
            "data_group": "0",
            "mask_group": "1",
            "labels_group": "2",
            "classes_group": "3",
            "shape": [1125, 512, 3],
            "chunks": [128, 128, 3],
            "source_axes": "YXC",
            "dtype": np.uint16
        }
    },
    {
        "dst_dir": "tests/test_tiffs",
        "source": generate_tiffs,
        "credit": None,
        "specs": {
            "data_group": "0/0",
            "mask_group": "0/1",
            "labels_group": "0/2",
            "classes_group": "0/3",
            "shape": [1125, 512, 3],
            "chunks": [128, 128, 3],
            "source_axes": "YXC",
            "dtype": np.uint16
        }
    },
    {
        "dst_dir": "tests/test_tiffs",
        "source": generate_tiffs,
        "credit": None,
        "specs": {
            "data_group": None,
            "shape": [1125, 512, 3],
            "chunks": [128, 128, 3],
            "source_axes": "YXC",
            "dtype": np.uint16
        }
    },
    {
        "dst_dir": "tests/test_tiffs",
        "source": generate_tiffs,
        "credit": None,
        "specs": {
            "data_group": 0,
            "shape": [1125, 512, 3],
            "chunks": [128, 128, 3],
            "source_axes": "YXC",
            "dtype": np.uint16
        }
    },
    {
        "dst_dir": "tests/test_pngs",
        "source": generate_pngs,
        "credit": None,
        "specs": {
            "data_group": "0",
            "mask_group": "masks/0",
            "labels_group": "labels/0",
            "classes_group": "classes/0",
            "shape": [1125, 512, 3],
            "chunks": [1125, 512, 3],
            "source_axes": "YXC",
            "dtype": np.uint8
        }
    },
    {
        "dst_dir": None,
        "source": generate_ndarray,
        "credit": None,
        "specs": {
            "data_group": "0",
            "mask_group": "masks/0",
            "labels_group": "labels/0",
            "classes_group": "classes/0",
            "shape": [1125, 3, 512],
            "chunks": [1125, 3, 512],
            "source_axes": "YCX",
            "dtype": np.uint8
        }
    },
    {
        "dst_dir": None,
        "source": generate_zarr_array,
        "credit": None,
        "specs": {
            "data_group": "0",
            "mask_group": "masks/0",
            "labels_group": "labels/0",
            "classes_group": "classes/0",
            "shape": [1, 3, 1024, 512, 1],
            "chunks": [1, 3, 256, 128, 1],
            "source_axes": "ZCXYT",
            "dtype": np.float32
        }
    },
    {
        "dst_dir": "tests/test_zarrs",
        "source": generate_zarr_array,
        "credit": None,
        "specs": {
            "data_group": None,
            "shape": [1, 3, 1024, 1024, 1],
            "chunks": [1, 3, 256, 256, 1],
            "source_axes": "ZCXYT",
            "dtype": np.float32
        }
    },
    {
        "dst_dir": None,
        "source": ("https://cellpainting-gallery.s3.amazonaws.com/"
                   "cpg0000-jump-pilot/source_4/images/2020_11_04_CPJUMP1/"
                   "images/BR00116991__2020-11-05T19_51_35-Measurement1/"
                   "Images/r01c01f01p01-ch1sk1fk1fl1.tiff"),
        "credit": ("From the dataset cpg0000 (Chandrasekaran et al., 2022), "
                   "available from the Cell Painting Gallery (Weisbart et al.,"
                   " 2024) on the Registry of Open Data on AWS "
                   "(https://registry.opendata.aws/cellpainting-gallery/)"),
        "specs": {
            "data_group": None,
            "shape": [1080, 1080],
            "chunks": [1080, 1080],
            "source_axes": "YX",
            "dtype": np.uint8
        }
    },
]

MASKABLE_IMAGE_SPECS = [
    {
        "dst_dir": None,
        "source": ("https://uk1s3.embassy.ebi.ac.uk/idr/zarr/v0.4/idr0073A/"
                   "9798462.zarr"),
        "credit": ("© 2016-2023 University of Dundee & Open Microscopy "
                   "Environment. Creative Commons Attribution 4.0 "
                   "International License"),
        "specs": {
            "data_group": "0",
            "shape": [1, 3, 1, 16433, 21115],
            "chunks": [1, 1, 1, 1024, 1024],
            "source_axes": "TCZYX",
            "axes": "YXC",
            "roi": "(0,0,0,0,0):(1,-1,1,-1,-1)",
            "dtype": np.uint16
        }
    },
]


UNSUPPORTED_IMAGE_SPECS = [
    {
        "dst_dir": "tests/test_unsupported",
        "source": generate_empty_file,
        "credit": None,
        "specs": {
            "filename": "image.unknown",
            "wrong_data_group": None,
        }
    },
    {
        "dst_dir": "tests/test_unsupported",
        "source": generate_empty_file,
        "credit": None,
        "specs": {
            "filename": "image.txt",
            "wrong_data_group": None,
        }
    },
    {
        "dst_dir": "tests/test_unsupported",
        "source": generate_empty_file,
        "credit": None,
        "specs": {
            "filename": "image.zip",
            "wrong_data_group": None,
        }
    },
    {
        "dst_dir": "tests/test_zarrs",
        "source": generate_zarr_group,
        "credit": None,
        "specs": {
            "data_group": "0",
            "shape": [1, 2, 2],
            "chunks": [1, 2, 2],
            "source_axes": "CYX",
            "dtype": np.uint8,
            "wrong_data_group": None,
        }
    },
    {
        "dst_dir": "tests/test_zarrs",
        "source": generate_zarr_file,
        "credit": None,
        "specs": {
            "data_group": "0",
            "shape": [1, 2, 2],
            "chunks": [1, 2, 2],
            "source_axes": "CYX",
            "dtype": np.uint8,
            "wrong_data_group": None,
        }
    },
    {
        "dst_dir": "tests/test_tiffs",
        "source": generate_tiffs,
        "credit": None,
        "specs": {
            "data_group": "0",
            "shape": [2, 2, 3],
            "chunks": [2, 2, 3],
            "source_axes": "YXC",
            "dtype": np.uint8,
            "wrong_data_group": dict(data_group="0"),
        }
    },
]
