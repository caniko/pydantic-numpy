from typing import Annotated, Any, Union

import numpy as np
from pydantic import FilePath

from pydantic_numpy.helper.annotation import NpArrayPydanticAnnotation
from pydantic_numpy.model import MultiArrayNumpyFile

Np1DArray = Annotated[
    Union[np.ndarray[tuple[int], np.dtype[Any]], FilePath, MultiArrayNumpyFile],
    NpArrayPydanticAnnotation.factory(data_type=None, dimensions=1, strict_data_typing=False),
]

Np1DArrayInt64 = Annotated[
    Union[np.ndarray[tuple[int], np.dtype[np.int64]], FilePath, MultiArrayNumpyFile],
    NpArrayPydanticAnnotation.factory(data_type=np.int64, dimensions=1, strict_data_typing=False),
]

Np1DArrayInt32 = Annotated[
    Union[np.ndarray[tuple[int], np.dtype[np.int32]], FilePath, MultiArrayNumpyFile],
    NpArrayPydanticAnnotation.factory(data_type=np.int32, dimensions=1, strict_data_typing=False),
]

Np1DArrayInt16 = Annotated[
    Union[np.ndarray[tuple[int], np.dtype[np.int16]], FilePath, MultiArrayNumpyFile],
    NpArrayPydanticAnnotation.factory(data_type=np.int16, dimensions=1, strict_data_typing=False),
]

Np1DArrayInt8 = Annotated[
    Union[np.ndarray[tuple[int], np.dtype[np.int8]], FilePath, MultiArrayNumpyFile],
    NpArrayPydanticAnnotation.factory(data_type=np.int8, dimensions=1, strict_data_typing=False),
]

Np1DArrayUint64 = Annotated[
    Union[np.ndarray[tuple[int], np.dtype[np.uint64]], FilePath, MultiArrayNumpyFile],
    NpArrayPydanticAnnotation.factory(data_type=np.uint64, dimensions=1, strict_data_typing=False),
]

Np1DArrayUint32 = Annotated[
    Union[np.ndarray[tuple[int], np.dtype[np.uint32]], FilePath, MultiArrayNumpyFile],
    NpArrayPydanticAnnotation.factory(data_type=np.uint32, dimensions=1, strict_data_typing=False),
]

Np1DArrayUint16 = Annotated[
    Union[np.ndarray[tuple[int], np.dtype[np.uint16]], FilePath, MultiArrayNumpyFile],
    NpArrayPydanticAnnotation.factory(data_type=np.uint16, dimensions=1, strict_data_typing=False),
]

Np1DArrayUint8 = Annotated[
    Union[np.ndarray[tuple[int], np.dtype[np.uint8]], FilePath, MultiArrayNumpyFile],
    NpArrayPydanticAnnotation.factory(data_type=np.uint8, dimensions=1, strict_data_typing=False),
]

Np1DArrayFpLongDouble = Annotated[
    Union[np.ndarray[tuple[int], np.dtype[np.longdouble]], FilePath, MultiArrayNumpyFile],
    NpArrayPydanticAnnotation.factory(data_type=np.longdouble, dimensions=1, strict_data_typing=False),
]

Np1DArrayFp64 = Annotated[
    Union[np.ndarray[tuple[int], np.dtype[np.float64]], FilePath, MultiArrayNumpyFile],
    NpArrayPydanticAnnotation.factory(data_type=np.float64, dimensions=1, strict_data_typing=False),
]

Np1DArrayFp32 = Annotated[
    Union[np.ndarray[tuple[int], np.dtype[np.float32]], FilePath, MultiArrayNumpyFile],
    NpArrayPydanticAnnotation.factory(data_type=np.float32, dimensions=1, strict_data_typing=False),
]

Np1DArrayFp16 = Annotated[
    Union[np.ndarray[tuple[int], np.dtype[np.float16]], FilePath, MultiArrayNumpyFile],
    NpArrayPydanticAnnotation.factory(data_type=np.float16, dimensions=1, strict_data_typing=False),
]

Np1DArrayComplexLongDouble = Annotated[
    Union[np.ndarray[tuple[int], np.dtype[np.clongdouble]], FilePath, MultiArrayNumpyFile],
    NpArrayPydanticAnnotation.factory(data_type=np.clongdouble, dimensions=1, strict_data_typing=False),
]

Np1DArrayComplex128 = Annotated[
    Union[np.ndarray[tuple[int], np.dtype[np.complex128]], FilePath, MultiArrayNumpyFile],
    NpArrayPydanticAnnotation.factory(data_type=np.complex128, dimensions=1, strict_data_typing=False),
]

Np1DArrayComplex64 = Annotated[
    Union[np.ndarray[tuple[int], np.dtype[np.complex64]], FilePath, MultiArrayNumpyFile],
    NpArrayPydanticAnnotation.factory(data_type=np.complex64, dimensions=1, strict_data_typing=False),
]

Np1DArrayBool = Annotated[
    Union[np.ndarray[tuple[int], np.dtype[np.bool_]], FilePath, MultiArrayNumpyFile],
    NpArrayPydanticAnnotation.factory(data_type=np.bool_, dimensions=1, strict_data_typing=False),
]

Np1DArrayDatetime64 = Annotated[
    Union[np.ndarray[tuple[int], np.dtype[np.datetime64]], FilePath, MultiArrayNumpyFile],
    NpArrayPydanticAnnotation.factory(data_type=np.datetime64, dimensions=1, strict_data_typing=False),
]

Np1DArrayTimedelta64 = Annotated[
    Union[np.ndarray[tuple[int], np.dtype[np.timedelta64]], FilePath, MultiArrayNumpyFile],
    NpArrayPydanticAnnotation.factory(data_type=np.timedelta64, dimensions=1, strict_data_typing=False),
]

__all__ = [
    "Np1DArray",
    "Np1DArrayInt64",
    "Np1DArrayInt32",
    "Np1DArrayInt16",
    "Np1DArrayInt8",
    "Np1DArrayUint64",
    "Np1DArrayUint32",
    "Np1DArrayUint16",
    "Np1DArrayUint8",
    "Np1DArrayFpLongDouble",
    "Np1DArrayFp64",
    "Np1DArrayFp32",
    "Np1DArrayFp16",
    "Np1DArrayComplexLongDouble",
    "Np1DArrayComplex128",
    "Np1DArrayComplex64",
    "Np1DArrayBool",
    "Np1DArrayDatetime64",
    "Np1DArrayTimedelta64",
]
