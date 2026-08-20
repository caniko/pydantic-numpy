import numpy as np
from typing_extensions import NotRequired, TypedDict

SupportedDTypes = type[np.generic]


class NumpyArrayTypeData(TypedDict):
    data_type: str
    data: list
    shape: NotRequired[list[int]]
