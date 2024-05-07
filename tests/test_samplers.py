import pytest
import os
import shutil
from pathlib import Path
import operator

from skimage import transform
from tests.utils import IMAGE_SPECS, MASKABLE_IMAGE_SPECS

import zarrdataset as zds
import math
import numpy as np


@pytest.fixture(scope="function")
def image_collection(request):
    dst_dir = request.param["dst_dir"]

    if dst_dir is not None:
        dst_dir = Path(request.param["dst_dir"])
        dst_dir.mkdir(parents=True, exist_ok=True)

    mask_args = None
    if not isinstance(request.param["source"], str):
        (img_src,
         mask_src,
         labels_src,
         classes_src) = request.param["source"](request.param["dst_dir"],
                                                request.param["specs"])

        if mask_src is not None:
            mask_args = dict(
                filename=mask_src,
                source_axes="YX",
                data_group=request.param["specs"]["mask_group"],
            )

    else:
        img_src = request.param["source"]

    collection_args = dict(
        images=dict(
            filename=img_src,
            source_axes=request.param["specs"]["source_axes"],
            axes=request.param["specs"].get("axes", None),
            data_group=request.param["specs"].get("data_group", None),
            roi=request.param["specs"].get("roi", None),
        ),
    )

    if mask_args is not None:
        collection_args["masks"] = mask_args

    image_collection = zds.ImageCollection(collection_args=collection_args)

    yield image_collection

    if dst_dir is not None and os.path.isdir(dst_dir):
        shutil.rmtree(dst_dir)


@pytest.fixture(scope="function")
def image_collection_mask_not2scale(request):
    dst_dir = request.param["dst_dir"]

    if dst_dir is not None:
        dst_dir = Path(request.param["dst_dir"])
        dst_dir.mkdir(parents=True, exist_ok=True)

    (img_src,
     mask_src,
     labels_src,
     classes_src) = request.param["source"](request.param["dst_dir"],
                                            request.param["specs"])

    collection_args = dict(
        images=dict(
            filename=img_src,
            source_axes=request.param["specs"]["source_axes"],
            data_group=request.param["specs"]["data_group"],
        ),
        masks=dict(
            filename=np.ones((3, 5), dtype=bool),
            source_axes="YX",
            data_group=None,
        )
    )

    image_collection = zds.ImageCollection(collection_args=collection_args)

    yield image_collection

    if dst_dir is not None and os.path.isdir(dst_dir):
        shutil.rmtree(dst_dir)


@pytest.mark.parametrize("patch_size", [
    512,
])
def test_string_PatchSampler(patch_size):
    patch_sampler = zds.PatchSampler(patch_size=patch_size)
    ps_str_repr = str(patch_sampler)

    assert f"patches of size {patch_sampler._patch_size}" in ps_str_repr,\
        (f"Expected string representation of PatchSampler to show the patch"
         f" size, got `{ps_str_repr}` instead.")

    patch_sampler = zds.BlueNoisePatchSampler(patch_size=patch_size)
    ps_str_repr = str(patch_sampler)

    assert f"patches of size {patch_sampler._patch_size}" in ps_str_repr,\
        (f"Expected string representation of BlueNoisePatchSampler to show the"
         f" patch size, got `{ps_str_repr}` instead.")


@pytest.mark.parametrize("patch_size, spatial_axes, expected_patch_size", [
    (512, "X", dict(X=512)),
    ((128, 64), "XY", dict(X=128, Y=64)),
])
def test_PatchSampler_correct_patch_size(patch_size, spatial_axes,
                                         expected_patch_size):
    patch_sampler = zds.PatchSampler(patch_size=patch_size,
                                     spatial_axes=spatial_axes)

    assert patch_sampler._patch_size == expected_patch_size, \
        (f"Expected `patch_size` to be a dictionary as {expected_patch_size}, "
         f"got {patch_sampler._patch_size} instead.")


@pytest.mark.parametrize("stride, spatial_axes, expected_stride", [
    (512, "X", dict(X=512)),
    ((128, 64), "XY", dict(X=128, Y=64)),
])
def test_PatchSampler_correct_stride(stride, spatial_axes, expected_stride):
    patch_sampler = zds.PatchSampler(patch_size=512, stride=stride,
                                     spatial_axes=spatial_axes)

    assert patch_sampler._stride == expected_stride, \
        (f"Expected `stride` to be a dictionary as {expected_stride}, "
         f"got {patch_sampler._stride} instead.")


@pytest.mark.parametrize("pad, spatial_axes, expected_pad", [
    (512, "X", dict(X=512)),
    ((128, 64), "XY", dict(X=128, Y=64)),
])
def test_PatchSampler_correct_pad(pad, spatial_axes, expected_pad):
    patch_sampler = zds.PatchSampler(patch_size=512, pad=pad,
                                     spatial_axes=spatial_axes)

    assert patch_sampler._pad == expected_pad, \
        (f"Expected `pad` to be a dictionary as {expected_pad}, "
         f"got {patch_sampler._pad} instead.")


@pytest.mark.parametrize("patch_size, spatial_axes", [
    ((512, 128), "X"),
    ((128, ), "XY"),
    ("patch_size", "ZYX"),
])
def test_PatchSampler_incorrect_patch_size(patch_size, spatial_axes):
    with pytest.raises(ValueError):
        patch_sampler = zds.PatchSampler(patch_size=patch_size,
                                         spatial_axes=spatial_axes)


@pytest.mark.parametrize("stride, spatial_axes", [
    ((512, 128), "X"),
    ((128, ), "XY"),
    ("stride", "ZYX"),
])
def test_PatchSampler_incorrect_stride(stride, spatial_axes):
    with pytest.raises(ValueError):
        patch_sampler = zds.PatchSampler(patch_size=512,
                                         stride=stride,
                                         spatial_axes=spatial_axes)


@pytest.mark.parametrize("pad, spatial_axes", [
    ((512, 128), "X"),
    ((128, ), "XY"),
    ("pad", "ZYX"),
])
def test_PatchSampler_incorrect_pad(pad, spatial_axes):
    with pytest.raises(ValueError):
        patch_sampler = zds.PatchSampler(patch_size=512,
                                         pad=pad,
                                         spatial_axes=spatial_axes)


@pytest.mark.parametrize("patch_size, image_collection", [
    (32, IMAGE_SPECS[10])
], indirect=["image_collection"])
def test_PatchSampler_chunk_generation(patch_size, image_collection):
    patch_sampler = zds.PatchSampler(patch_size)

    chunks_toplefts = patch_sampler.compute_chunks(image_collection)

    chunk_size = {
        ax: cs
        for ax, cs in zip(image_collection.collection["images"].axes,
                          image_collection.collection["images"].chunk_size)
    }

    scaled_chunk_size = {
        ax: int(cs * image_collection.collection["masks"].scale[ax])
        for ax, cs in zip(image_collection.collection["images"].axes,
                          image_collection.collection["images"].chunk_size)
        if ax in image_collection.collection["masks"].axes
    }

    scaled_mask = transform.downscale_local_mean(
        image_collection.collection["masks"][:],
        factors=(scaled_chunk_size["Y"], scaled_chunk_size["X"])
    )
    expected_chunks_toplefts = np.nonzero(scaled_mask)

    expected_chunks_toplefts = [
        dict(
            [("Z", slice(0, 1, None))]
            + [
               (ax, slice(tl * chunk_size[ax], (tl + 1) * chunk_size[ax]))
               for ax, tl in zip("YX", tls)
            ]
        )
        for tls in zip(*expected_chunks_toplefts)
    ]

    assert all(map(operator.eq, chunks_toplefts, expected_chunks_toplefts)), \
        (f"Expected chunks to be {expected_chunks_toplefts[:3]}, got "
         f"{chunks_toplefts[:3]} instead.")


@pytest.mark.parametrize("patch_size, image_collection", [
    (32, IMAGE_SPECS[10])
], indirect=["image_collection"])
def test_PatchSampler(patch_size, image_collection):
    patch_sampler = zds.PatchSampler(patch_size)

    chunks_toplefts = patch_sampler.compute_chunks(image_collection)

    patches_toplefts = patch_sampler.compute_patches(
        image_collection,
        chunk_tlbr=chunks_toplefts[0]
    )

    scaled_patch_size = {
        ax: int(patch_size * scl)
        for ax, scl in image_collection.collection["masks"].scale.items()
    }

    scaled_mask = transform.downscale_local_mean(
        image_collection.collection["masks"][chunks_toplefts[0]],
        factors=(scaled_patch_size["Y"], scaled_patch_size["X"])
    )
    expected_patches_toplefts = np.nonzero(scaled_mask)

    expected_patches_toplefts = [
        dict(
            [("Z", slice(0, 1, None))]
            + [
               (ax, slice(tl * patch_size, (tl + 1) * patch_size))
               for ax, tl in zip("YX", tls)
            ]
        )
        for tls in zip(*expected_patches_toplefts)
    ]

    assert all(map(operator.eq, patches_toplefts, expected_patches_toplefts)),\
        (f"Expected patches to be {expected_patches_toplefts[:3]}, got "
         f"{patches_toplefts[:3]} instead.")


@pytest.mark.parametrize("patch_size, stride, image_collection", [
    (32, 32, IMAGE_SPECS[10]),
    (32, 16, IMAGE_SPECS[10]),
    (32, 64, IMAGE_SPECS[10]),
], indirect=["image_collection"])
def test_PatchSampler_stride(patch_size, stride, image_collection):
    patch_sampler = zds.PatchSampler(patch_size, stride=stride)

    chunks_toplefts = patch_sampler.compute_chunks(image_collection)

    patches_toplefts = patch_sampler.compute_patches(
        image_collection,
        chunk_tlbr=chunks_toplefts[0]
    )

    scaled_patch_size = {
        ax: int(stride * scl)
        for ax, scl in image_collection.collection["masks"].scale.items()
    }

    scaled_mask = transform.downscale_local_mean(
        image_collection.collection["masks"][chunks_toplefts[0]],
        factors=(scaled_patch_size["Y"], scaled_patch_size["X"])
    )
    expected_patches_toplefts = np.nonzero(scaled_mask)

    expected_patches_toplefts = [
        dict(
            [("Z", slice(0, 1, None))]
            + [
               (ax, slice(tl * stride, tl * stride + patch_size))
               for ax, tl in zip("YX", tls)
            ]
        )
        for tls in zip(*expected_patches_toplefts)
    ]
    assert all(map(operator.eq, patches_toplefts, expected_patches_toplefts)),\
        (f"Expected patches to be {expected_patches_toplefts[:3]}, got "
         f"{patches_toplefts[:3]} instead.")


@pytest.mark.parametrize("patch_size, pad, image_collection", [
    (32, 0, IMAGE_SPECS[10]),
    (32, 2, IMAGE_SPECS[10]),
], indirect=["image_collection"])
def test_PatchSampler_pad(patch_size, pad, image_collection):
    patch_sampler = zds.PatchSampler(patch_size, pad=pad)

    chunks_toplefts = patch_sampler.compute_chunks(image_collection)

    patches_toplefts = patch_sampler.compute_patches(
        image_collection,
        chunk_tlbr=chunks_toplefts[0]
    )

    scaled_patch_size = {
        ax: int(patch_size * scl)
        for ax, scl in image_collection.collection["masks"].scale.items()
    }

    scaled_mask = transform.downscale_local_mean(
        image_collection.collection["masks"][chunks_toplefts[0]],
        factors=(scaled_patch_size["Y"], scaled_patch_size["X"])
    )
    expected_patches_toplefts = np.nonzero(scaled_mask)

    # TODO: Change expected patches toplefts for strided ones
    expected_patches_toplefts = [
        dict(
            [("Z", slice(0, 1, None))]
            + [
               (ax, slice(tl * patch_size - pad, (tl + 1) * patch_size + pad))
               for ax, tl in zip("YX", tls)
            ]
        )
        for tls in zip(*expected_patches_toplefts)
    ]

    assert all(map(operator.eq, patches_toplefts, expected_patches_toplefts)),\
        (f"Expected patches to be {expected_patches_toplefts[:3]}, got "
         f"{patches_toplefts[:3]} instead.")


@pytest.mark.parametrize("patch_size, allow_incomplete_patches,"
                         "image_collection", [
    (1024, True, IMAGE_SPECS[10]),
    (1024, False, IMAGE_SPECS[10]),
], indirect=["image_collection"])
def test_PatchSampler_incomplete_patches(patch_size, allow_incomplete_patches,
                                         image_collection):
    patch_sampler = zds.PatchSampler(
        patch_size,
        allow_incomplete_patches=allow_incomplete_patches
    )

    chunks_toplefts = patch_sampler.compute_chunks(image_collection)

    patches_toplefts = patch_sampler.compute_patches(
        image_collection,
        chunk_tlbr=chunks_toplefts[0]
    )

    expected_num_patches = 1 if allow_incomplete_patches else 0

    assert len(patches_toplefts) == expected_num_patches,\
        (f"Expected to have {expected_num_patches}, when "
         f"`allow_incomplete_patches` is {allow_incomplete_patches} "
         f"got {len(patches_toplefts)} instead.")


@pytest.mark.parametrize("patch_size, axes, resample, allow_overlap,"
                         "image_collection", [
    (dict(X=32, Y=32, Z=1), "XYZ", True, True, IMAGE_SPECS[10]),
    (dict(X=32, Y=32), "YX", False, False, IMAGE_SPECS[10]),
], indirect=["image_collection"])
def test_BlueNoisePatchSampler(patch_size, axes, resample, allow_overlap,
                               image_collection):
    np.random.seed(447788)

    patch_sampler = zds.BlueNoisePatchSampler(patch_size,
                                              resample_positions=resample,
                                              allow_overlap=allow_overlap,
                                              spatial_axes=axes)

    chunks_toplefts = patch_sampler.compute_chunks(image_collection)

    patches_toplefts = patch_sampler.compute_patches(
        image_collection,
        chunk_tlbr=chunks_toplefts[0]
    )

    assert len(patches_toplefts) == len(patch_sampler._base_chunk_tls), \
        (f"Expected {len(patch_sampler._base_chunk_tls)} patches, got "
         f"{len(patches_toplefts)} instead.")


@pytest.mark.parametrize("image_collection_mask_not2scale", [
    IMAGE_SPECS[10]
], indirect=["image_collection_mask_not2scale"])
def test_BlueNoisePatchSampler_mask_not2scale(image_collection_mask_not2scale):
    np.random.seed(447788)

    patch_size = dict(X=1024, Y=1024)

    patch_sampler = zds.BlueNoisePatchSampler(patch_size)

    chunks_toplefts = patch_sampler.compute_chunks(
        image_collection_mask_not2scale
    )

    patches_toplefts = patch_sampler.compute_patches(
        image_collection_mask_not2scale,
        chunk_tlbr=chunks_toplefts[0]
    )

    # Samples can be retrieved from chunks that are not multiple of the patch
    # size. The ZarrDataset class should handle these cases, either by droping
    # these patches, or by adding padding when allowed by the user.
    assert len(patches_toplefts) == 1, \
        (f"Expected 0 patches, got {len(patches_toplefts)} instead.")


@pytest.mark.parametrize("patch_size, stride, image_collection, specs", [
    (512, 512, MASKABLE_IMAGE_SPECS[0], MASKABLE_IMAGE_SPECS[0]),
    (512, 256, MASKABLE_IMAGE_SPECS[0], MASKABLE_IMAGE_SPECS[0])
], indirect=["image_collection"])
def test_unique_sampling_PatchSampler(patch_size, stride, image_collection,
                                      specs):
    from skimage import color, filters, morphology
    import zarr

    z_img = zarr.open(specs["source"], mode="r")

    im_gray = color.rgb2gray(z_img["4"][0, :, 0], channel_axis=0)
    thresh = filters.threshold_otsu(im_gray)

    mask = im_gray > thresh
    mask = morphology.remove_small_objects(mask == 0, min_size=16 ** 2,
                                           connectivity=2)
    mask = morphology.remove_small_holes(mask, area_threshold=128)
    mask = morphology.binary_erosion(mask, morphology.disk(8))
    mask = morphology.binary_dilation(mask, morphology.disk(8))

    image_collection.collection["masks"] = zds.ImageLoader(filename=mask,
                                                           source_axes="YX",
                                                           mode="masks")
    image_collection.reset_scales()

    patch_sampler = zds.PatchSampler(patch_size, stride=stride,
                                     min_area=1/16 ** 2)

    chunks_toplefts = patch_sampler.compute_chunks(image_collection)

    all_patches_tls = []
    all_chunks_tls = []

    for ctl in chunks_toplefts:
        patches_toplefts = patch_sampler.compute_patches(image_collection,
                                                         chunk_tlbr=ctl)

        for ptl in patches_toplefts:
            assert ptl not in all_patches_tls,\
                (f"Expected no repetitions in patch sampling, got {ptl} "
                 f"twice instead at chunk {ctl}. Possible duplicate of patch "
                 f"{all_patches_tls.index(ptl)}, from chunk "
                 f"{all_chunks_tls[all_patches_tls.index(ptl)]}")
            all_patches_tls.append(ptl)
            all_chunks_tls.append(ctl)


@pytest.mark.parametrize("patch_size, image_collection, specs", [
    (512, MASKABLE_IMAGE_SPECS[0], MASKABLE_IMAGE_SPECS[0])
], indirect=["image_collection"])
def test_unique_sampling_BlueNoisePatchSampler(patch_size, image_collection,
                                               specs):
    from skimage import color, filters, morphology
    import zarr

    z_img = zarr.open(specs["source"], mode="r")

    im_gray = color.rgb2gray(z_img["4"][0, :, 0], channel_axis=0)
    thresh = filters.threshold_otsu(im_gray)

    mask = im_gray > thresh
    mask = morphology.remove_small_objects(mask == 0, min_size=16 ** 2,
                                           connectivity=2)
    mask = morphology.remove_small_holes(mask, area_threshold=128)
    mask = morphology.binary_erosion(mask, morphology.disk(8))
    mask = morphology.binary_dilation(mask, morphology.disk(8))

    image_collection.collection["masks"] = zds.ImageLoader(filename=mask,
                                                           source_axes="YX",
                                                           mode="masks")
    image_collection.reset_scales()

    patch_sampler = zds.BlueNoisePatchSampler(patch_size)

    chunks_toplefts = patch_sampler.compute_chunks(image_collection)

    all_patches_tls = []
    all_chunks_tls = []

    for ctl in chunks_toplefts:
        patches_toplefts = patch_sampler.compute_patches(image_collection,
                                                         chunk_tlbr=ctl)

        for ptl in patches_toplefts:
            assert ptl not in all_patches_tls,\
                (f"Expected no repetitions in patch sampling, got {ptl} "
                 f"twice instead at chunk {ctl}. Possible duplicate of patch "
                 f"{all_patches_tls.index(ptl)}, from chunk "
                 f"{all_chunks_tls[all_patches_tls.index(ptl)]}")
            all_patches_tls.append(ptl)
            all_chunks_tls.append(ctl)


@pytest.mark.parametrize("min_area, patch_size, "
                         "image_collection_mask_not2scale", [
    (0.5, 32, IMAGE_SPECS[11]),
    (4096, 512, IMAGE_SPECS[11]),
    (0.5, 1024, IMAGE_SPECS[11])
], indirect=["image_collection_mask_not2scale"])
def test_min_area_sampling_PatchSampler(min_area, patch_size,
                                        image_collection_mask_not2scale):
    patch_sampler = zds.PatchSampler(patch_size, min_area=min_area)

    chunks_toplefts = patch_sampler.compute_chunks(
        image_collection_mask_not2scale
    )

    all_patches_tls = []
    all_chunks_tls = []

    for ctl in chunks_toplefts:
        patches_toplefts = patch_sampler.compute_patches(
            image_collection_mask_not2scale,
            chunk_tlbr=ctl
        )

        for ptl in patches_toplefts:
            assert ptl not in all_patches_tls,\
                (f"Expected no repetitions in patch sampling, got {ptl} "
                 f"twice instead at chunk {ctl}. Possible duplicate of patch "
                 f"{all_patches_tls.index(ptl)}, from chunk "
                 f"{all_chunks_tls[all_patches_tls.index(ptl)]}")
            all_patches_tls.append(ptl)
            all_chunks_tls.append(ctl)
