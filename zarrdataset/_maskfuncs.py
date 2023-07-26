from skimage import morphology, color, filters, transform


def compute_tissue_mask(img, mask_scale=1/16, min_size=16, area_threshold=128,
                        thresh=None):
    gray = color.rgb2gray(img)
    scaled_gray = transform.rescale(gray, scale=mask_scale, order=0,
                                    preserve_range=True)

    if thresh is None:
        thresh = filters.threshold_otsu(scaled_gray)

    chunk_mask = scaled_gray > thresh

    chunk_mask = morphology.remove_small_objects(chunk_mask == 0,
                                                 min_size=min_size ** 2,
                                                 connectivity=2)
    mask = morphology.remove_small_holes(chunk_mask,
                                         area_threshold=area_threshold ** 2)
    mask = morphology.binary_dilation(chunk_mask,
                                      morphology.disk(min_size))

    return mask
