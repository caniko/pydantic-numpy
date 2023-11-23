from functools import cache
from typing import Optional

import numpy as np
import numpy.typing as npt
from hypothesis.extra.numpy import arrays
from hypothesis.strategies import floats
from pydantic import BaseModel


@cache
def cached_calculation(array_type_hint) -> type[BaseModel]:
    class ModelForTesting(BaseModel):
        array_field: array_type_hint

    return ModelForTesting


@cache
def cached_hyp_array(numpy_dtype: npt.DTypeLike, dimensions: Optional[int] = None, *, _axis_length: int = 1):
    if np.issubdtype(numpy_dtype, np.floating):
        if numpy_dtype == np.float16:
            width = 16
        elif numpy_dtype == np.float32:
            width = 32
        elif numpy_dtype in (np.longdouble, np.float64):
            width = 64
        else:
            raise RuntimeError

        element_strategy = floats(allow_infinity=False, allow_nan=False, width=width)
    else:
        element_strategy = None

    return arrays(numpy_dtype, tuple(_axis_length for _ in range(dimensions or 1)), elements=element_strategy)
