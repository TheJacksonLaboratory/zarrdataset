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
