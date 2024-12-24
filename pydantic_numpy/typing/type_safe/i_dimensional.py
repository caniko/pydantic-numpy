from typing import Annotated, Any

import numpy as np

from pydantic_numpy.helper.annotation import NpArrayPydanticAnnotation

type Np1DArray = Annotated[
    np.ndarray[tuple[int], np.dtype[Any]],
    NpArrayPydanticAnnotation.factory(data_type=None, dimensions=1, strict_data_typing=False),
]

type Np1DArrayInt64 = Annotated[
    np.ndarray[tuple[int], np.dtype[np.int64]],
    NpArrayPydanticAnnotation.factory(data_type=np.int64, dimensions=1, strict_data_typing=True),
]

type Np1DArrayInt32 = Annotated[
    np.ndarray[tuple[int], np.dtype[np.int32]],
    NpArrayPydanticAnnotation.factory(data_type=np.int32, dimensions=1, strict_data_typing=True),
]

type Np1DArrayInt16 = Annotated[
    np.ndarray[tuple[int], np.dtype[np.int16]],
    NpArrayPydanticAnnotation.factory(data_type=np.int16, dimensions=1, strict_data_typing=True),
]

type Np1DArrayInt8 = Annotated[
    np.ndarray[tuple[int], np.dtype[np.int8]],
    NpArrayPydanticAnnotation.factory(data_type=np.int8, dimensions=1, strict_data_typing=True),
]

type Np1DArrayUint64 = Annotated[
    np.ndarray[tuple[int], np.dtype[np.uint64]],
    NpArrayPydanticAnnotation.factory(data_type=np.uint64, dimensions=1, strict_data_typing=True),
]

type Np1DArrayUint32 = Annotated[
    np.ndarray[tuple[int], np.dtype[np.uint32]],
    NpArrayPydanticAnnotation.factory(data_type=np.uint32, dimensions=1, strict_data_typing=True),
]

type Np1DArrayUint16 = Annotated[
    np.ndarray[tuple[int], np.dtype[np.uint16]],
    NpArrayPydanticAnnotation.factory(data_type=np.uint16, dimensions=1, strict_data_typing=True),
]

type Np1DArrayUint8 = Annotated[
    np.ndarray[tuple[int], np.dtype[np.uint8]],
    NpArrayPydanticAnnotation.factory(data_type=np.uint8, dimensions=1, strict_data_typing=True),
]

type Np1DArrayFpLongDouble = Annotated[
    np.ndarray[tuple[int], np.dtype[np.longdouble]],
    NpArrayPydanticAnnotation.factory(data_type=np.longdouble, dimensions=1, strict_data_typing=True),
]

type Np1DArrayFp64 = Annotated[
    np.ndarray[tuple[int], np.dtype[np.float64]],
    NpArrayPydanticAnnotation.factory(data_type=np.float64, dimensions=1, strict_data_typing=True),
]

type Np1DArrayFp32 = Annotated[
    np.ndarray[tuple[int], np.dtype[np.float32]],
    NpArrayPydanticAnnotation.factory(data_type=np.float32, dimensions=1, strict_data_typing=True),
]

type Np1DArrayFp16 = Annotated[
    np.ndarray[tuple[int], np.dtype[np.float16]],
    NpArrayPydanticAnnotation.factory(data_type=np.float16, dimensions=1, strict_data_typing=True),
]

type Np1DArrayComplexLongDouble = Annotated[
    np.ndarray[tuple[int], np.dtype[np.clongdouble]],
    NpArrayPydanticAnnotation.factory(data_type=np.clongdouble, dimensions=1, strict_data_typing=True),
]

type Np1DArrayComplex128 = Annotated[
    np.ndarray[tuple[int], np.dtype[np.complex128]],
    NpArrayPydanticAnnotation.factory(data_type=np.complex128, dimensions=1, strict_data_typing=True),
]

type Np1DArrayComplex64 = Annotated[
    np.ndarray[tuple[int], np.dtype[np.complex64]],
    NpArrayPydanticAnnotation.factory(data_type=np.complex64, dimensions=1, strict_data_typing=True),
]

type Np1DArrayBool = Annotated[
    np.ndarray[tuple[int], np.dtype[np.bool_]],
    NpArrayPydanticAnnotation.factory(data_type=np.bool_, dimensions=1, strict_data_typing=True),
]

type Np1DArrayDatetime64 = Annotated[
    np.ndarray[tuple[int], np.dtype[np.datetime64]],
    NpArrayPydanticAnnotation.factory(data_type=np.datetime64, dimensions=1, strict_data_typing=True),
]

type Np1DArrayTimedelta64 = Annotated[
    np.ndarray[tuple[int], np.dtype[np.timedelta64]],
    NpArrayPydanticAnnotation.factory(data_type=np.timedelta64, dimensions=1, strict_data_typing=True),
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
