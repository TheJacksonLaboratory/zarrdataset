import numpy as np


class ZarrToArray(object):
    """Converted from a zarr array to numpy ndarray.

    Parameters
    ----------
    dtype : numpy.dtype or None
        The data type to cast the array before computing from the dask graph.
        If None, use the same dtype from the original array.
    """
    def __init__(self, dtype=None):
        self._dtype = dtype

    def __call__(self, pic) -> np.ndarray:
        """Transform a dask array into a numpy ndarray.

        Parameters
        ----------
        pic : zarr.Array
            A zarr array

        Returns
        -------
        arr : numpy.ndarray
            The computed array resulting from the dask graph in the specified
            datatype.
        """
        if self._dtype is not None:
            pic = pic.astype(self._dtype)

        return pic[:]

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(dtype={self._dtype})"
