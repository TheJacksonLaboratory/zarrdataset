import pytest
from unittest import mock
from collections import OrderedDict
import zarrdataset as zds
import importlib
import torch


def test_compatibility_no_pytroch():
    with mock.patch.dict('sys.modules', {'torch': None}):
        importlib.reload(zds._zarrdataset)

        dataset = zds._zarrdataset.ZarrDataset(
            dataset_specs=OrderedDict(
                images=zds.ImagesDatasetSpecs(
                    filenames=None
                )
            )
        )

        assert isinstance(object, type(dataset).__bases__), \
            (f"When pytorch is not installed, ZarrDataset should be inherited"
             f" from object, not {type(dataset).__bases__}")

    with mock.patch.dict('sys.modules', {'torch': torch}):
        importlib.reload(zds._zarrdataset)
        importlib.reload(zds)



@pytest.mark.parametrize("images_dataset_specs", [None])
def test_ZarrDataset(images_dataset_specs):
    zds.ZarrDataset(
        dataset_specs=OrderedDict(
            images=images_dataset_specs
        )
    )
