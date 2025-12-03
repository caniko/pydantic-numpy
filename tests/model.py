from pydantic import BaseModel, ConfigDict

from pydantic_numpy.model import NumpyModel
from pydantic_numpy.typing import Np1DArray, NpNDArray

from typing import Generic, TypeVar


T = TypeVar("T")


class GenericForTesting(BaseModel, Generic[T]):
    array_field: T


class NpNDArrayModel(NumpyModel):
    array: NpNDArray


class N1DArrayModel(NumpyModel):
    array: Np1DArray


class NpNDArrayModelWithNonArray(NpNDArrayModel):
    non_array: int


class NpNDArrayModelWithNonArrayWithArbitrary(NpNDArrayModelWithNonArray):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    my_arbitrary_slice: slice
