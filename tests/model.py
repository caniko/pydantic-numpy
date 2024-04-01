from pydantic_numpy import NpNDArray
from pydantic_numpy.model import NumpyModel
from pydantic_numpy.typing import Np1DArray


class NpNDArrayModel(NumpyModel):
    array: NpNDArray


class N1DArrayModel(NumpyModel):
    array: Np1DArray


class NpNDArrayModelWithNonArray(NpNDArrayModel):
    non_array: int


class NpNDArrayModelWithNonArrayWithArbitrary(NpNDArrayModelWithNonArray, arbitrary_types_allowed=True):
    my_arbitrary_slice: slice
