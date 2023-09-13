from unittest import mock
import zarrdataset as zds
import importlib
import torch
import tifffile


def test_compatibility_no_pytroch():
    with mock.patch.dict('sys.modules', {'torch': None}):
        importlib.reload(zds._zarrdataset)

        dataset = zds._zarrdataset.ZarrDataset(filenames=[], source_axes="")

        assert isinstance(object, type(dataset).__bases__), \
            (f"When pytorch is not installed, ZarrDataset should be inherited"
             f" from object, not {type(dataset).__bases__}")

    with mock.patch.dict('sys.modules', {'torch': torch}):
        importlib.reload(zds._zarrdataset)
        importlib.reload(zds)


def test_compatibility_no_tifffile():
    with mock.patch.dict('sys.modules', {'tifffile': None}):
        importlib.reload(zds._imageloaders)

        assert not zds._imageloaders.TIFFFILE_SUPPORT,\
            (f"If TiffFile is not installed, image loading functionalities "
             f"depending on it should be disabled, but TIFFFILE_SUPPORT is "
             f"{zds._imageloaders.TIFFFILE_SUPPORT}")

    with mock.patch.dict('sys.modules', {'tifffile': tifffile}):
        importlib.reload(zds._imageloaders)
        importlib.reload(zds)
