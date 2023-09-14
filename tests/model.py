from pydantic_numpy import NpNDArray
from pydantic_numpy.model import NumpyModel


class NpNDArrayModel(NumpyModel):
    array: NpNDArray


class NpNDArrayModelWithNonArray(NpNDArrayModel):
    non_array: int


class NpNDArrayModelWithNonArrayWithArbitrary(NpNDArrayModelWithNonArray, arbitrary_types_allowed=True):
    my_arbitrary_slice: slice
