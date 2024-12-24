from typing import Annotated, Any

import numpy as np

from pydantic_numpy.helper.annotation import NpArrayPydanticAnnotation

type Np3DArray = Annotated[
    np.ndarray[tuple[int, int, int], np.dtype[Any]],
    NpArrayPydanticAnnotation.factory(data_type=None, dimensions=3, strict_data_typing=False),
]

type Np3DArrayInt64 = Annotated[
    np.ndarray[tuple[int, int, int], np.dtype[np.int64]],
    NpArrayPydanticAnnotation.factory(data_type=np.int64, dimensions=3, strict_data_typing=True),
]

type Np3DArrayInt32 = Annotated[
    np.ndarray[tuple[int, int, int], np.dtype[np.int32]],
    NpArrayPydanticAnnotation.factory(data_type=np.int32, dimensions=3, strict_data_typing=True),
]

type Np3DArrayInt16 = Annotated[
    np.ndarray[tuple[int, int, int], np.dtype[np.int16]],
    NpArrayPydanticAnnotation.factory(data_type=np.int16, dimensions=3, strict_data_typing=True),
]

type Np3DArrayInt8 = Annotated[
    np.ndarray[tuple[int, int, int], np.dtype[np.int8]],
    NpArrayPydanticAnnotation.factory(data_type=np.int8, dimensions=3, strict_data_typing=True),
]

type Np3DArrayUint64 = Annotated[
    np.ndarray[tuple[int, int, int], np.dtype[np.uint64]],
    NpArrayPydanticAnnotation.factory(data_type=np.uint64, dimensions=3, strict_data_typing=True),
]

type Np3DArrayUint32 = Annotated[
    np.ndarray[tuple[int, int, int], np.dtype[np.uint32]],
    NpArrayPydanticAnnotation.factory(data_type=np.uint32, dimensions=3, strict_data_typing=True),
]

type Np3DArrayUint16 = Annotated[
    np.ndarray[tuple[int, int, int], np.dtype[np.uint16]],
    NpArrayPydanticAnnotation.factory(data_type=np.uint16, dimensions=3, strict_data_typing=True),
]

type Np3DArrayUint8 = Annotated[
    np.ndarray[tuple[int, int, int], np.dtype[np.uint8]],
    NpArrayPydanticAnnotation.factory(data_type=np.uint8, dimensions=3, strict_data_typing=True),
]

type Np3DArrayFpLongDouble = Annotated[
    np.ndarray[tuple[int, int, int], np.dtype[np.longdouble]],
    NpArrayPydanticAnnotation.factory(data_type=np.longdouble, dimensions=3, strict_data_typing=True),
]

type Np3DArrayFp64 = Annotated[
    np.ndarray[tuple[int, int, int], np.dtype[np.float64]],
    NpArrayPydanticAnnotation.factory(data_type=np.float64, dimensions=3, strict_data_typing=True),
]

type Np3DArrayFp32 = Annotated[
    np.ndarray[tuple[int, int, int], np.dtype[np.float32]],
    NpArrayPydanticAnnotation.factory(data_type=np.float32, dimensions=3, strict_data_typing=True),
]

type Np3DArrayFp16 = Annotated[
    np.ndarray[tuple[int, int, int], np.dtype[np.float16]],
    NpArrayPydanticAnnotation.factory(data_type=np.float16, dimensions=3, strict_data_typing=True),
]

type Np3DArrayComplexLongDouble = Annotated[
    np.ndarray[tuple[int, int, int], np.dtype[np.clongdouble]],
    NpArrayPydanticAnnotation.factory(data_type=np.clongdouble, dimensions=3, strict_data_typing=True),
]

type Np3DArrayComplex128 = Annotated[
    np.ndarray[tuple[int, int, int], np.dtype[np.complex128]],
    NpArrayPydanticAnnotation.factory(data_type=np.complex128, dimensions=3, strict_data_typing=True),
]

type Np3DArrayComplex64 = Annotated[
    np.ndarray[tuple[int, int, int], np.dtype[np.complex64]],
    NpArrayPydanticAnnotation.factory(data_type=np.complex64, dimensions=3, strict_data_typing=True),
]

type Np3DArrayBool = Annotated[
    np.ndarray[tuple[int, int, int], np.dtype[np.bool_]],
    NpArrayPydanticAnnotation.factory(data_type=np.bool_, dimensions=3, strict_data_typing=True),
]

type Np3DArrayDatetime64 = Annotated[
    np.ndarray[tuple[int, int, int], np.dtype[np.datetime64]],
    NpArrayPydanticAnnotation.factory(data_type=np.datetime64, dimensions=3, strict_data_typing=True),
]

type Np3DArrayTimedelta64 = Annotated[
    np.ndarray[tuple[int, int, int], np.dtype[np.timedelta64]],
    NpArrayPydanticAnnotation.factory(data_type=np.timedelta64, dimensions=3, strict_data_typing=True),
]

__all__ = [
    "Np3DArray",
    "Np3DArrayInt64",
    "Np3DArrayInt32",
    "Np3DArrayInt16",
    "Np3DArrayInt8",
    "Np3DArrayUint64",
    "Np3DArrayUint32",
    "Np3DArrayUint16",
    "Np3DArrayUint8",
    "Np3DArrayFpLongDouble",
    "Np3DArrayFp64",
    "Np3DArrayFp32",
    "Np3DArrayFp16",
    "Np3DArrayComplexLongDouble",
    "Np3DArrayComplex128",
    "Np3DArrayComplex64",
    "Np3DArrayBool",
    "Np3DArrayDatetime64",
    "Np3DArrayTimedelta64",
]
