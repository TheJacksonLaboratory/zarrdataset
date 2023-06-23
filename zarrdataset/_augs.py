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


class SelectAxes(object):
    """Pick the axes from an array and permute them into the desired order.

    Parameters
    ----------
    source_axes : str
        Set of axes in the input array.
    target_axes : str
        Set of axes of in the desired order.
    axes_selection : dict or None
        Dictionary defining the indices to take from each axes of the array.
        Unspecified axes, or indices passed as None will be picked completelly.
        If None is passed, take the complete array instead.

    Notes
    -----
    The axes must be passed as a string in format `XYC`, and cannot be appear
    more than once.
    """
    def __init__(self, source_axes, axes_selection=None, target_axes="CYX"):
        if axes_selection is None:
            axes_selection = dict((ax, None) for ax in source_axes)

        self._source_axes = source_axes
        self._target_axes = target_axes
        self._axes_selection = axes_selection

        (sel_slices,
         source_axes) = select_axes(self._source_axes, self._axes_selection)

        new_axes = "".join(set(target_axes) - set(self._source_axes))
        source_axes = new_axes + source_axes

        self._permute_order = map_axes_order(source_axes, target_axes)
        self._sel_slices = tuple([None] * len(new_axes) + list(sel_slices))

    def __call__(self, pic) -> np.ndarray:
        """Permute the axes of pic to match `target_axes`.

        Parameters
        ----------
        pic : dask.array.core.Array, numpy.ndarray
            A dask or numpy array

        Returns
        -------
        perm_pic : dask.array.core.Array, numpy.ndarray
            The array with the selected axes in the order specified in
            `target_axes`.
        """
        sel_pic = pic[self._sel_slices]
        perm_pic = sel_pic.transpose(self._permute_order)

        return perm_pic

    def __repr__(self) -> str:
        return (f"{self.__class__.__name__}(source_axes={self._source_axes}, "
                f"target_axes={self._target_axes}, "
                f"axes_selection={self._axes_selection})")
