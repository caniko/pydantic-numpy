import numpy as np
from typing_extensions import TypedDict

SupportedDTypes = type[np.generic]


class NumpyDataDict(TypedDict):
    data_type: str
    data: list
