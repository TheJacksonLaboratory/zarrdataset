import zarrdataset as zds
import numpy as np


def test_ToDtype():
    transform = zds.ToDtype(dtype=np.float32)

    a = np.empty((0, ), dtype=np.int16)

    b = transform(a)

    assert b.dtype == np.float32, \
        (f"Expected array after transformation be of type np.float32, got "
         f"{b.dtype} instead.")

    assert "dtype=<class 'numpy.float32'>" in str(transform), \
        (f"Expected string generated from the transform to include "
         f"'dtype=np.float32', got {str(transform)} instead.")
