from typing import Annotated, Any

import numpy as np

from pydantic_numpy.helper.annotation import NpArrayPydanticAnnotation

NpStrictNDArrayInt64 = Annotated[
    np.ndarray[Any, np.dtype[np.int64]],
    NpArrayPydanticAnnotation.factory(data_type=np.int64, dimensions=None, strict_data_typing=True),
]

NpStrictNDArrayInt32 = Annotated[
    np.ndarray[Any, np.dtype[np.int32]],
    NpArrayPydanticAnnotation.factory(data_type=np.int32, dimensions=None, strict_data_typing=True),
]

NpStrictNDArrayInt16 = Annotated[
    np.ndarray[Any, np.dtype[np.int16]],
    NpArrayPydanticAnnotation.factory(data_type=np.int16, dimensions=None, strict_data_typing=True),
]

NpStrictNDArrayInt8 = Annotated[
    np.ndarray[Any, np.dtype[np.int8]],
    NpArrayPydanticAnnotation.factory(data_type=np.int8, dimensions=None, strict_data_typing=True),
]

NpStrictNDArrayUint64 = Annotated[
    np.ndarray[Any, np.dtype[np.uint64]],
    NpArrayPydanticAnnotation.factory(data_type=np.uint64, dimensions=None, strict_data_typing=True),
]

NpStrictNDArrayUint32 = Annotated[
    np.ndarray[Any, np.dtype[np.uint32]],
    NpArrayPydanticAnnotation.factory(data_type=np.uint32, dimensions=None, strict_data_typing=True),
]

NpStrictNDArrayUint16 = Annotated[
    np.ndarray[Any, np.dtype[np.uint16]],
    NpArrayPydanticAnnotation.factory(data_type=np.uint16, dimensions=None, strict_data_typing=True),
]

NpStrictNDArrayUint8 = Annotated[
    np.ndarray[Any, np.dtype[np.uint8]],
    NpArrayPydanticAnnotation.factory(data_type=np.uint8, dimensions=None, strict_data_typing=True),
]

NpStrictNDArrayFpLongDouble = Annotated[
    np.ndarray[Any, np.dtype[np.longdouble]],
    NpArrayPydanticAnnotation.factory(data_type=np.longdouble, dimensions=None, strict_data_typing=True),
]

NpStrictNDArrayFp64 = Annotated[
    np.ndarray[Any, np.dtype[np.float64]],
    NpArrayPydanticAnnotation.factory(data_type=np.float64, dimensions=None, strict_data_typing=True),
]

NpStrictNDArrayFp32 = Annotated[
    np.ndarray[Any, np.dtype[np.float32]],
    NpArrayPydanticAnnotation.factory(data_type=np.float32, dimensions=None, strict_data_typing=True),
]

NpStrictNDArrayFp16 = Annotated[
    np.ndarray[Any, np.dtype[np.float16]],
    NpArrayPydanticAnnotation.factory(data_type=np.float16, dimensions=None, strict_data_typing=True),
]

NpStrictNDArrayComplexLongDouble = Annotated[
    np.ndarray[Any, np.dtype[np.clongdouble]],
    NpArrayPydanticAnnotation.factory(data_type=np.clongdouble, dimensions=None, strict_data_typing=True),
]

NpStrictNDArrayComplex128 = Annotated[
    np.ndarray[Any, np.dtype[np.complex128]],
    NpArrayPydanticAnnotation.factory(data_type=np.complex128, dimensions=None, strict_data_typing=True),
]

NpStrictNDArrayComplex64 = Annotated[
    np.ndarray[Any, np.dtype[np.complex64]],
    NpArrayPydanticAnnotation.factory(data_type=np.complex64, dimensions=None, strict_data_typing=True),
]

NpStrictNDArrayBool = Annotated[
    np.ndarray[Any, np.dtype[np.bool_]],
    NpArrayPydanticAnnotation.factory(data_type=np.bool_, dimensions=None, strict_data_typing=True),
]

NpStrictNDArrayDatetime64 = Annotated[
    np.ndarray[Any, np.dtype[np.datetime64]],
    NpArrayPydanticAnnotation.factory(data_type=np.datetime64, dimensions=None, strict_data_typing=True),
]

NpStrictNDArrayTimedelta64 = Annotated[
    np.ndarray[Any, np.dtype[np.timedelta64]],
    NpArrayPydanticAnnotation.factory(data_type=np.timedelta64, dimensions=None, strict_data_typing=True),
]

__all__ = [
    "NpStrictNDArrayInt64",
    "NpStrictNDArrayInt32",
    "NpStrictNDArrayInt16",
    "NpStrictNDArrayInt8",
    "NpStrictNDArrayUint64",
    "NpStrictNDArrayUint32",
    "NpStrictNDArrayUint16",
    "NpStrictNDArrayUint8",
    "NpStrictNDArrayFpLongDouble",
    "NpStrictNDArrayFp64",
    "NpStrictNDArrayFp32",
    "NpStrictNDArrayFp16",
    "NpStrictNDArrayComplexLongDouble",
    "NpStrictNDArrayComplex128",
    "NpStrictNDArrayComplex64",
    "NpStrictNDArrayBool",
    "NpStrictNDArrayDatetime64",
    "NpStrictNDArrayTimedelta64",
]
