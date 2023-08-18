from functools import cache
from typing import Optional

import numpy.typing as npt
from hypothesis.extra.numpy import arrays
from pydantic import BaseModel


@cache
def cached_calculation(array_type_hint) -> type[BaseModel]:
    class ModelForTesting(BaseModel):
        array_field: array_type_hint

    return ModelForTesting


@cache
def cached_hyp_array(numpy_dtype: npt.DTypeLike, dimensions: Optional[int] = None, *, _axis_length: int = 1):
    return arrays(numpy_dtype, tuple(_axis_length for _ in range(dimensions or 1)))
