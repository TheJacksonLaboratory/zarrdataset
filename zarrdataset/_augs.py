import numpy as np
from ._utils import map_axes_order

class DaskToArray(object):
    """Converted from dask array to numpy ndarray.

    This allows to compute the dask array at the very last moment possible to
    prevent loading large arrays into memory.

    Parameters
    ----------
    use_multithread : bool
        Whether use multithreading to compute the dask graph or not.
    """
    def __init__(self, use_multithread=False):
        self._scheduler = "threads" if use_multithread else "synchronous"

    def __call__(self, pic) -> np.ndarray:
        """Transform a dask array into a numpy ndarray.

        Parameters
        ----------
        pic : dask.array.core.Array
            A dask array 

        Returns
        -------
        arr : numpy.ndarray
            The computed array resulting from the dask graph.
        """
        arr = pic.compute(scheduler=self._scheduler)
        return arr

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(scheduler={self._scheduler})"


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

        sel_slices = []
        source_axes = list(source_axes)
        for ax in self._source_axes:
            idx = axes_selection.get(ax, None)

            if idx is None:
                sel_slices.append(slice(None))

            else:
                # If a single index in the current axis is selected, remove it
                # from the set of axes that have to be permuted to match
                # `target_axes`.
                sel_slices.append(idx)
                source_axes.remove(ax)

        source_axes = "".join(source_axes)
        new_axes = "".join(set(target_axes) - set(self._source_axes))

        source_axes = new_axes + source_axes
        self._permute_order = map_axes_order(source_axes, target_axes)

        self._sel_slices = tuple([None] * len(new_axes) + sel_slices)

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
