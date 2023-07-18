import numpy as np
import numpy.typing as npt
from numpy.core._exceptions import UFuncTypeError
from numpy.exceptions import DTypePromotionError


def np_general_all_close(arr_a: npt.NDArray, arr_b: npt.NDArray, rtol: float = 1e-05, atol: float = 1e-08) -> bool:
    """
    Data type agnostic function to define if two numpy array have elements that are close

    Parameters
    ----------
    arr_a: npt.NDArray
    arr_b: npt.NDArray
    rtol: float
        See np.allclose
    atol: float
        See np.allclose

    Returns
    -------
    Bool
    """
    try:
        return np.allclose(arr_a, arr_b, rtol=rtol, atol=atol, equal_nan=True)
    except UFuncTypeError:
        return np.allclose(arr_a.astype(np.float64), arr_b.astype(np.float64), rtol=rtol, atol=atol, equal_nan=True)
    except DTypePromotionError:
        return bool(np.all(arr_a == arr_b))
