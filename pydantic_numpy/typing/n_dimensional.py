from typing import Annotated, Any, Union

import numpy as np
from pydantic import FilePath

from pydantic_numpy.helper.annotation import NpArrayPydanticAnnotation
from pydantic_numpy.model import MultiArrayNumpyFile

NpNDArray = Annotated[
    Union[np.ndarray[Any, np.dtype[Any]], FilePath, MultiArrayNumpyFile],
    NpArrayPydanticAnnotation.factory(data_type=None, dimensions=None, strict_data_typing=False),
]

NpNDArrayInt64 = Annotated[
    Union[np.ndarray[Any, np.dtype[np.int64]], FilePath, MultiArrayNumpyFile],
    NpArrayPydanticAnnotation.factory(data_type=np.int64, dimensions=None, strict_data_typing=False),
]

NpNDArrayInt32 = Annotated[
    Union[np.ndarray[Any, np.dtype[np.int32]], FilePath, MultiArrayNumpyFile],
    NpArrayPydanticAnnotation.factory(data_type=np.int32, dimensions=None, strict_data_typing=False),
]

NpNDArrayInt16 = Annotated[
    Union[np.ndarray[Any, np.dtype[np.int16]], FilePath, MultiArrayNumpyFile],
    NpArrayPydanticAnnotation.factory(data_type=np.int16, dimensions=None, strict_data_typing=False),
]

NpNDArrayInt8 = Annotated[
    Union[np.ndarray[Any, np.dtype[np.int8]], FilePath, MultiArrayNumpyFile],
    NpArrayPydanticAnnotation.factory(data_type=np.int8, dimensions=None, strict_data_typing=False),
]

NpNDArrayUint64 = Annotated[
    Union[np.ndarray[Any, np.dtype[np.uint64]], FilePath, MultiArrayNumpyFile],
    NpArrayPydanticAnnotation.factory(data_type=np.uint64, dimensions=None, strict_data_typing=False),
]

NpNDArrayUint32 = Annotated[
    Union[np.ndarray[Any, np.dtype[np.uint32]], FilePath, MultiArrayNumpyFile],
    NpArrayPydanticAnnotation.factory(data_type=np.uint32, dimensions=None, strict_data_typing=False),
]

NpNDArrayUint16 = Annotated[
    Union[np.ndarray[Any, np.dtype[np.uint16]], FilePath, MultiArrayNumpyFile],
    NpArrayPydanticAnnotation.factory(data_type=np.uint16, dimensions=None, strict_data_typing=False),
]

NpNDArrayUint8 = Annotated[
    Union[np.ndarray[Any, np.dtype[np.uint8]], FilePath, MultiArrayNumpyFile],
    NpArrayPydanticAnnotation.factory(data_type=np.uint8, dimensions=None, strict_data_typing=False),
]

NpNDArrayFpLongDouble = Annotated[
    Union[np.ndarray[Any, np.dtype[np.longdouble]], FilePath, MultiArrayNumpyFile],
    NpArrayPydanticAnnotation.factory(data_type=np.longdouble, dimensions=None, strict_data_typing=False),
]

NpNDArrayFp64 = Annotated[
    Union[np.ndarray[Any, np.dtype[np.float64]], FilePath, MultiArrayNumpyFile],
    NpArrayPydanticAnnotation.factory(data_type=np.float64, dimensions=None, strict_data_typing=False),
]

NpNDArrayFp32 = Annotated[
    Union[np.ndarray[Any, np.dtype[np.float32]], FilePath, MultiArrayNumpyFile],
    NpArrayPydanticAnnotation.factory(data_type=np.float32, dimensions=None, strict_data_typing=False),
]

NpNDArrayFp16 = Annotated[
    Union[np.ndarray[Any, np.dtype[np.float16]], FilePath, MultiArrayNumpyFile],
    NpArrayPydanticAnnotation.factory(data_type=np.float16, dimensions=None, strict_data_typing=False),
]

NpNDArrayComplexLongDouble = Annotated[
    Union[np.ndarray[Any, np.dtype[np.clongdouble]], FilePath, MultiArrayNumpyFile],
    NpArrayPydanticAnnotation.factory(data_type=np.clongdouble, dimensions=None, strict_data_typing=False),
]

NpNDArrayComplex128 = Annotated[
    Union[np.ndarray[Any, np.dtype[np.complex128]], FilePath, MultiArrayNumpyFile],
    NpArrayPydanticAnnotation.factory(data_type=np.complex128, dimensions=None, strict_data_typing=False),
]

NpNDArrayComplex64 = Annotated[
    Union[np.ndarray[Any, np.dtype[np.complex64]], FilePath, MultiArrayNumpyFile],
    NpArrayPydanticAnnotation.factory(data_type=np.complex64, dimensions=None, strict_data_typing=False),
]

NpNDArrayBool = Annotated[
    Union[np.ndarray[Any, np.dtype[np.bool_]], FilePath, MultiArrayNumpyFile],
    NpArrayPydanticAnnotation.factory(data_type=np.bool_, dimensions=None, strict_data_typing=False),
]

NpNDArrayDatetime64 = Annotated[
    Union[np.ndarray[Any, np.dtype[np.datetime64]], FilePath, MultiArrayNumpyFile],
    NpArrayPydanticAnnotation.factory(data_type=np.datetime64, dimensions=None, strict_data_typing=False),
]

NpNDArrayTimedelta64 = Annotated[
    Union[np.ndarray[Any, np.dtype[np.timedelta64]], FilePath, MultiArrayNumpyFile],
    NpArrayPydanticAnnotation.factory(data_type=np.timedelta64, dimensions=None, strict_data_typing=False),
]

__all__ = [
    "NpNDArray",
    "NpNDArrayInt64",
    "NpNDArrayInt32",
    "NpNDArrayInt16",
    "NpNDArrayInt8",
    "NpNDArrayUint64",
    "NpNDArrayUint32",
    "NpNDArrayUint16",
    "NpNDArrayUint8",
    "NpNDArrayFpLongDouble",
    "NpNDArrayFp64",
    "NpNDArrayFp32",
    "NpNDArrayFp16",
    "NpNDArrayComplexLongDouble",
    "NpNDArrayComplex128",
    "NpNDArrayComplex64",
    "NpNDArrayBool",
    "NpNDArrayDatetime64",
    "NpNDArrayTimedelta64",
]
