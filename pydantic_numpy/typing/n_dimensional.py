from typing import Annotated, Any

import numpy as np

from pydantic_numpy.helper.annotation import NpArrayPydanticAnnotation

NpNDArray = Annotated[
    np.ndarray[Any, np.dtype[Any]],
    NpArrayPydanticAnnotation.factory(data_type=None, dimensions=None, strict_data_typing=False),
]

NpNDArrayInt64 = Annotated[
    np.ndarray[Any, np.dtype[np.int64]],
    NpArrayPydanticAnnotation.factory(data_type=np.int64, dimensions=None, strict_data_typing=False),
]

NpNDArrayInt32 = Annotated[
    np.ndarray[Any, np.dtype[np.int32]],
    NpArrayPydanticAnnotation.factory(data_type=np.int32, dimensions=None, strict_data_typing=False),
]

NpNDArrayInt16 = Annotated[
    np.ndarray[Any, np.dtype[np.int16]],
    NpArrayPydanticAnnotation.factory(data_type=np.int16, dimensions=None, strict_data_typing=False),
]

NpNDArrayInt8 = Annotated[
    np.ndarray[Any, np.dtype[np.int8]],
    NpArrayPydanticAnnotation.factory(data_type=np.int8, dimensions=None, strict_data_typing=False),
]

NpNDArrayUint64 = Annotated[
    np.ndarray[Any, np.dtype[np.uint64]],
    NpArrayPydanticAnnotation.factory(data_type=np.uint64, dimensions=None, strict_data_typing=False),
]

NpNDArrayUint32 = Annotated[
    np.ndarray[Any, np.dtype[np.uint32]],
    NpArrayPydanticAnnotation.factory(data_type=np.uint32, dimensions=None, strict_data_typing=False),
]

NpNDArrayUint16 = Annotated[
    np.ndarray[Any, np.dtype[np.uint16]],
    NpArrayPydanticAnnotation.factory(data_type=np.uint16, dimensions=None, strict_data_typing=False),
]

NpNDArrayUint8 = Annotated[
    np.ndarray[Any, np.dtype[np.uint8]],
    NpArrayPydanticAnnotation.factory(data_type=np.uint8, dimensions=None, strict_data_typing=False),
]

NpNDArrayFpLongDouble = Annotated[
    np.ndarray[Any, np.dtype[np.longdouble]],
    NpArrayPydanticAnnotation.factory(data_type=np.longdouble, dimensions=None, strict_data_typing=False),
]

NpNDArrayFp64 = Annotated[
    np.ndarray[Any, np.dtype[np.float64]],
    NpArrayPydanticAnnotation.factory(data_type=np.float64, dimensions=None, strict_data_typing=False),
]

NpNDArrayFp32 = Annotated[
    np.ndarray[Any, np.dtype[np.float32]],
    NpArrayPydanticAnnotation.factory(data_type=np.float32, dimensions=None, strict_data_typing=False),
]

NpNDArrayFp16 = Annotated[
    np.ndarray[Any, np.dtype[np.float16]],
    NpArrayPydanticAnnotation.factory(data_type=np.float16, dimensions=None, strict_data_typing=False),
]

NpNDArrayComplexLongDouble = Annotated[
    np.ndarray[Any, np.dtype[np.clongdouble]],
    NpArrayPydanticAnnotation.factory(data_type=np.clongdouble, dimensions=None, strict_data_typing=False),
]

NpNDArrayComplex128 = Annotated[
    np.ndarray[Any, np.dtype[np.complex128]],
    NpArrayPydanticAnnotation.factory(data_type=np.complex128, dimensions=None, strict_data_typing=False),
]

NpNDArrayComplex64 = Annotated[
    np.ndarray[Any, np.dtype[np.complex64]],
    NpArrayPydanticAnnotation.factory(data_type=np.complex64, dimensions=None, strict_data_typing=False),
]

NpNDArrayBool = Annotated[
    np.ndarray[Any, np.dtype[np.bool_]],
    NpArrayPydanticAnnotation.factory(data_type=np.bool_, dimensions=None, strict_data_typing=False),
]

NpNDArrayDatetime64 = Annotated[
    np.ndarray[Any, np.dtype[np.datetime64]],
    NpArrayPydanticAnnotation.factory(data_type=np.datetime64, dimensions=None, strict_data_typing=False),
]

NpNDArrayTimedelta64 = Annotated[
    np.ndarray[Any, np.dtype[np.timedelta64]],
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
