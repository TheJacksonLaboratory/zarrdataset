import numpy as np
import zarr
from PIL import Image
import tifffile


EXPECTED_SHAPES = [
    dict(Y=1097, X=1015, C=3, T=1, Z=1),
    dict(Y=1020, X=1047, C=3, T=1, Z=1),
    dict(Y=1148, X=997, C=3, T=1, Z=1),
]


def generate_group(height=2048, width=2048, chunk_height=512, chunk_width=512,
                   coeff_1=2,
                   coeff_2=2,
                   label=0,
                   mask_threshold=127):
    # Generate a test image on memory
    z1, z2 = np.meshgrid(np.arange(width, dtype=np.float32),
                         np.arange(height, dtype=np.float32))
    z3 = z1 ** coeff_1 + z2 ** coeff_2

    img = np.stack((z1, z2, z3))
    img = img / img.max(axis=(1, 2))[:, None, None]
    img = img * 255
    img = img.astype(np.uint8)

    # Mask a subsampled version of `img`
    mask = img[2, ::4, ::4] < mask_threshold

    # Label the regions of `img`
    labs = (1 * (img[2, ...] > 63)
            + 2 * (img[2, ...] > 127)
            + 3 * (img[2, ...] > 191))

    store = zarr.MemoryStore()
    grp = zarr.open(store)
    data_grp = grp.create_group("0")
    data_grp.create_dataset("0", data=img[None, :, None, ...],
                            dtype=np.uint8,
                            chunks=(1, 3, 1, chunk_height, chunk_width))
    data_grp.create_dataset("1", data=img[None, :, None, ::2, ::2],
                            dtype=np.uint8,
                            chunks=(1, 3, 1, chunk_height // 2,
                                    chunk_width // 2))
    data_grp.create_dataset("2", data=img[None, :, None, ::4, ::4],
                            dtype=np.uint8,
                            chunks=(1, 3, 1, chunk_height // 4,
                                    chunk_width // 4))

    mask_grp = grp.create_group("masks/0")
    mask_grp.create_dataset("0", data=mask, chunks=(chunk_height // 4,
                                                    chunk_width // 4))

    labs_grp = grp.create_group("labels/0")
    labs_grp.create_dataset("0", data=labs, chunks=(chunk_height, chunk_width))

    labs_grp = grp.create_group("classes/0")
    labs_grp.create_dataset("0", data=np.array([label]), dtype=np.uint32,
                            chunks=(1,))

    return grp


def generate_zarrs(z_groups):
    for g, z_grp in enumerate(z_groups):
        z_grp_dst = zarr.open(f"tests/test_zarrs/zarr_group_{g}.zarr",
                              mode="w")

        zarr.copy_all(z_grp, z_grp_dst)


def generate_tiffs(z_groups):
    for g, z_grp in enumerate(z_groups):
        tifffile.imwrite(f'tests/test_tiffs/img_{g}.ome.tif',
                         data=z_grp["0/0"][:].transpose(0, 2, 3, 4, 1),
                         dtype=np.uint8,
                         photometric='rgb')


def generate_images(z_groups):
    for g, z_grp in enumerate(z_groups):    
        data = np.moveaxis(z_grp[f"0/0"][0, :, 0], 0, -1)
        image = Image.fromarray(data)
        image.save(f'tests/test_images/img_{g}.png',
                   quality_opts={'compress_level': 9, 'optimize': False})

        mask_data = z_grp[f"masks/0/0"][:]
        mask = Image.fromarray(mask_data)
        mask.save(f'tests/test_images/mask_{g}.png',
                   quality_opts={'compress_level': 9, 'optimize': False})

        labels_data = z_grp[f"labels/0/0"][:]
        labels = Image.fromarray(labels_data)
        labels.save(f'tests/test_images/labels_{g}.png',
                    quality_opts={'compress_level': 9, 'optimize': False})


if __name__ == "__main__":
    images_specs = [
        dict(height=e_s["Y"], width=e_s["X"], chunk_height=512,
             chunk_width=512,
             coeff_1=2,
             coeff_2=2,
             label=0,
             mask_threshold=195)
        for e_s in EXPECTED_SHAPES]

    grps = [generate_group(**specs) for specs in images_specs]

    generate_zarrs(grps)
    generate_tiffs(grps)
    generate_images(grps)
