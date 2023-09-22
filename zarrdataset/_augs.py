import numpy as np


class ToDtype(object):
    """Converted a numpy NDArray to the specified data type.

    Parameters
    ----------
    dtype : numpy.dtype
        The data type to cast the input array.
    """
    def __init__(self, dtype: np.dtype) -> None:
        self._dtype = dtype

    def __call__(self, image:np.ndarray) -> np.ndarray:
        """Casts the type of `image` to the data type specified with `dtype`.

        Parameters
        ----------
        image : np.ndarray
            A numpy NDArray

        Returns
        -------
        casted_image : numpy.ndarray
            The sampe input image
        """
        casted_image = image.astype(self._dtype)
        return casted_image

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(dtype={self._dtype})"
