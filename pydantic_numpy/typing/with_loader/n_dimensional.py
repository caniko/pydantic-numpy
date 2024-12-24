from typing import Annotated, Any, Union

import numpy as np
from pydantic import FilePath

from pydantic_numpy.helper.annotation import NpArrayPydanticAnnotation
from pydantic_numpy.model import MultiArrayNumpyFile

type NpLoadingNDArray = Annotated[
    Union[np.ndarray[Any, np.dtype[Any]], FilePath, MultiArrayNumpyFile],
    NpArrayPydanticAnnotation.factory(data_type=None, dimensions=None, strict_data_typing=False),
]

type NpLoadingNDArrayInt64 = Annotated[
    Union[np.ndarray[Any, np.dtype[np.int64]], FilePath, MultiArrayNumpyFile],
    NpArrayPydanticAnnotation.factory(data_type=np.int64, dimensions=None, strict_data_typing=False),
]

type NpLoadingNDArrayInt32 = Annotated[
    Union[np.ndarray[Any, np.dtype[np.int32]], FilePath, MultiArrayNumpyFile],
    NpArrayPydanticAnnotation.factory(data_type=np.int32, dimensions=None, strict_data_typing=False),
]

type NpLoadingNDArrayInt16 = Annotated[
    Union[np.ndarray[Any, np.dtype[np.int16]], FilePath, MultiArrayNumpyFile],
    NpArrayPydanticAnnotation.factory(data_type=np.int16, dimensions=None, strict_data_typing=False),
]

type NpLoadingNDArrayInt8 = Annotated[
    Union[np.ndarray[Any, np.dtype[np.int8]], FilePath, MultiArrayNumpyFile],
    NpArrayPydanticAnnotation.factory(data_type=np.int8, dimensions=None, strict_data_typing=False),
]

type NpLoadingNDArrayUint64 = Annotated[
    Union[np.ndarray[Any, np.dtype[np.uint64]], FilePath, MultiArrayNumpyFile],
    NpArrayPydanticAnnotation.factory(data_type=np.uint64, dimensions=None, strict_data_typing=False),
]

type NpLoadingNDArrayUint32 = Annotated[
    Union[np.ndarray[Any, np.dtype[np.uint32]], FilePath, MultiArrayNumpyFile],
    NpArrayPydanticAnnotation.factory(data_type=np.uint32, dimensions=None, strict_data_typing=False),
]

type NpLoadingNDArrayUint16 = Annotated[
    Union[np.ndarray[Any, np.dtype[np.uint16]], FilePath, MultiArrayNumpyFile],
    NpArrayPydanticAnnotation.factory(data_type=np.uint16, dimensions=None, strict_data_typing=False),
]

type NpLoadingNDArrayUint8 = Annotated[
    Union[np.ndarray[Any, np.dtype[np.uint8]], FilePath, MultiArrayNumpyFile],
    NpArrayPydanticAnnotation.factory(data_type=np.uint8, dimensions=None, strict_data_typing=False),
]

type NpLoadingNDArrayFpLongDouble = Annotated[
    Union[np.ndarray[Any, np.dtype[np.longdouble]], FilePath, MultiArrayNumpyFile],
    NpArrayPydanticAnnotation.factory(data_type=np.longdouble, dimensions=None, strict_data_typing=False),
]

type NpLoadingNDArrayFp64 = Annotated[
    Union[np.ndarray[Any, np.dtype[np.float64]], FilePath, MultiArrayNumpyFile],
    NpArrayPydanticAnnotation.factory(data_type=np.float64, dimensions=None, strict_data_typing=False),
]

type NpLoadingNDArrayFp32 = Annotated[
    Union[np.ndarray[Any, np.dtype[np.float32]], FilePath, MultiArrayNumpyFile],
    NpArrayPydanticAnnotation.factory(data_type=np.float32, dimensions=None, strict_data_typing=False),
]

type NpLoadingNDArrayFp16 = Annotated[
    Union[np.ndarray[Any, np.dtype[np.float16]], FilePath, MultiArrayNumpyFile],
    NpArrayPydanticAnnotation.factory(data_type=np.float16, dimensions=None, strict_data_typing=False),
]

type NpLoadingNDArrayComplexLongDouble = Annotated[
    Union[np.ndarray[Any, np.dtype[np.clongdouble]], FilePath, MultiArrayNumpyFile],
    NpArrayPydanticAnnotation.factory(data_type=np.clongdouble, dimensions=None, strict_data_typing=False),
]

type NpLoadingNDArrayComplex128 = Annotated[
    Union[np.ndarray[Any, np.dtype[np.complex128]], FilePath, MultiArrayNumpyFile],
    NpArrayPydanticAnnotation.factory(data_type=np.complex128, dimensions=None, strict_data_typing=False),
]

type NpLoadingNDArrayComplex64 = Annotated[
    Union[np.ndarray[Any, np.dtype[np.complex64]], FilePath, MultiArrayNumpyFile],
    NpArrayPydanticAnnotation.factory(data_type=np.complex64, dimensions=None, strict_data_typing=False),
]

type NpLoadingNDArrayBool = Annotated[
    Union[np.ndarray[Any, np.dtype[np.bool_]], FilePath, MultiArrayNumpyFile],
    NpArrayPydanticAnnotation.factory(data_type=np.bool_, dimensions=None, strict_data_typing=False),
]

type NpLoadingNDArrayDatetime64 = Annotated[
    Union[np.ndarray[Any, np.dtype[np.datetime64]], FilePath, MultiArrayNumpyFile],
    NpArrayPydanticAnnotation.factory(data_type=np.datetime64, dimensions=None, strict_data_typing=False),
]

type NpLoadingNDArrayTimedelta64 = Annotated[
    Union[np.ndarray[Any, np.dtype[np.timedelta64]], FilePath, MultiArrayNumpyFile],
    NpArrayPydanticAnnotation.factory(data_type=np.timedelta64, dimensions=None, strict_data_typing=False),
]

__all__ = [
    "NpLoadingNDArray",
    "NpLoadingNDArrayInt64",
    "NpLoadingNDArrayInt32",
    "NpLoadingNDArrayInt16",
    "NpLoadingNDArrayInt8",
    "NpLoadingNDArrayUint64",
    "NpLoadingNDArrayUint32",
    "NpLoadingNDArrayUint16",
    "NpLoadingNDArrayUint8",
    "NpLoadingNDArrayFpLongDouble",
    "NpLoadingNDArrayFp64",
    "NpLoadingNDArrayFp32",
    "NpLoadingNDArrayFp16",
    "NpLoadingNDArrayComplexLongDouble",
    "NpLoadingNDArrayComplex128",
    "NpLoadingNDArrayComplex64",
    "NpLoadingNDArrayBool",
    "NpLoadingNDArrayDatetime64",
    "NpLoadingNDArrayTimedelta64",
]
