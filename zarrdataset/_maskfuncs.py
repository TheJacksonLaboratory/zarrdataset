import numpy as np

import zarr
import dask
import dask.array as da

from skimage import morphology, color, filters, transform


def compute_tissue_chunk(chunk, mask_scale=1/16, min_size=16,
                         area_threshold=128,
                         offset=1):
    img = np.copy(chunk)
    gray = color.rgb2gray(img)
    scaled_gray = transform.rescale(gray, scale=mask_scale, order=0,
                                    preserve_range=True)

    thresh = filters.threshold_otsu(scaled_gray)
    chunk_mask = scaled_gray > thresh

    chunk_mask = morphology.remove_small_objects(chunk_mask == 0,
                                                 min_size=min_size ** 2,
                                                 connectivity=2)
    chunk_mask = morphology.remove_small_holes(
        chunk_mask, area_threshold=area_threshold ** 2)
    chunk_mask = morphology.binary_dilation(chunk_mask,
                                            morphology.disk(min_size))

    h, w = chunk_mask.shape
    return chunk_mask[offset:h-offset, offset:w-offset]


def compute_tissue_mask(img, mask_scale=1/16, min_size=16,
                        area_threshold=128):
    if isinstance(img, da.core.Array):
        org_img = img
    else:
        if isinstance(img, np.ndarray):
            org_img = da.from_array(img)
        elif isinstance(img, zarr.Array):
            org_img = da.from_zarr(img)
        else:
            raise ValueError(f"Cannot compute mask from input of type "
                             f"{type(img)}, only dask.array.core.Array, "
                             f"numpy.ndarray, or zarr.Array are supported")

    chk = org_img.chunksize[:2]

    H, W, _ = org_img.shape
    padded_H = (chk[0] - H) % chk[0]
    padded_W = (chk[1] - W) % chk[1]

    if padded_H or padded_W:
        base_wsi = da.pad(org_img, ((0, padded_H), (0, padded_W), (0, 0)))
    else:
        base_wsi = org_img

    base_wsi = base_wsi.rechunk((chk[0], chk[1], 3))

    mask = base_wsi.map_overlap(compute_tissue_chunk, mask_scale=mask_scale,
                                min_size=min_size,
                                area_threshold=area_threshold,
                                offset=1,
                                chunks=(int(chk[0] * mask_scale),
                                        int(chk[1] * mask_scale)),
                                drop_axis=(2,),
                                dtype=bool,
                                depth=(int(1 / mask_scale),
                                       int(1 / mask_scale),
                                       0),
                                boundary=255,
                                trim=False,
                                meta=np.empty((0, 0), dtype=bool))

    mask = mask[:int(H * mask_scale), :int(W * mask_scale)]

    mask = mask.persist(scheduler="synchronous")

    mask_axes = "YX"

    return mask, mask_axes
