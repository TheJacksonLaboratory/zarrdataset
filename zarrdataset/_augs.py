import numpy as np
from ._utils import map_axes_order, select_axes


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


class DaskToArray(object):
    """Converted from dask array to numpy ndarray.

    This allows to compute the dask array at the very last moment possible to
    prevent loading large arrays into memory.

    Parameters
    ----------
    dtype : numpy.dtype or None
        The data type to cast the array before computing from the dask graph.
        If None, use the same dtype from the original array.
    use_multithread : bool
        Whether use multithreading to compute the dask graph or not.
    """
    def __init__(self, dtype=None, use_multithread=False):
        self._scheduler = "threads" if use_multithread else "synchronous"
        self._dtype = dtype

    def __call__(self, pic) -> np.ndarray:
        """Transform a dask array into a numpy ndarray.

        Parameters
        ----------
        pic : dask.array.core.Array
            A dask array 

        Returns
        -------
        arr : numpy.ndarray
            The computed array resulting from the dask graph in the specified
            datatype.
        """
        if self._dtype is not None:
            pic = pic.astype(self._dtype)

        arr = pic.compute(scheduler=self._scheduler)
        return arr

    def __repr__(self) -> str:
        return (f"{self.__class__.__name__}(scheduler={self._scheduler}, "
                f"dtype={self._dtype})")
