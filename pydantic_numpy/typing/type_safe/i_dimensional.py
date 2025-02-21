from typing import Annotated, Any, TypeAlias

import numpy as np

from pydantic_numpy.helper.annotation import NpArrayPydanticAnnotation

Np1DArray: TypeAlias = Annotated[
    np.ndarray[tuple[int, ...], np.dtype[Any]],
    NpArrayPydanticAnnotation.factory(data_type=None, dimensions=1, strict_data_typing=False),
]

Np1DArrayInt64: TypeAlias = Annotated[
    np.ndarray[tuple[int, ...], np.dtype[np.int64]],
    NpArrayPydanticAnnotation.factory(data_type=np.int64, dimensions=1, strict_data_typing=True),
]

Np1DArrayInt32: TypeAlias = Annotated[
    np.ndarray[tuple[int, ...], np.dtype[np.int32]],
    NpArrayPydanticAnnotation.factory(data_type=np.int32, dimensions=1, strict_data_typing=True),
]

Np1DArrayInt16: TypeAlias = Annotated[
    np.ndarray[tuple[int, ...], np.dtype[np.int16]],
    NpArrayPydanticAnnotation.factory(data_type=np.int16, dimensions=1, strict_data_typing=True),
]

Np1DArrayInt8: TypeAlias = Annotated[
    np.ndarray[tuple[int, ...], np.dtype[np.int8]],
    NpArrayPydanticAnnotation.factory(data_type=np.int8, dimensions=1, strict_data_typing=True),
]

Np1DArrayUint64: TypeAlias = Annotated[
    np.ndarray[tuple[int, ...], np.dtype[np.uint64]],
    NpArrayPydanticAnnotation.factory(data_type=np.uint64, dimensions=1, strict_data_typing=True),
]

Np1DArrayUint32: TypeAlias = Annotated[
    np.ndarray[tuple[int, ...], np.dtype[np.uint32]],
    NpArrayPydanticAnnotation.factory(data_type=np.uint32, dimensions=1, strict_data_typing=True),
]

Np1DArrayUint16: TypeAlias = Annotated[
    np.ndarray[tuple[int, ...], np.dtype[np.uint16]],
    NpArrayPydanticAnnotation.factory(data_type=np.uint16, dimensions=1, strict_data_typing=True),
]

Np1DArrayUint8: TypeAlias = Annotated[
    np.ndarray[tuple[int, ...], np.dtype[np.uint8]],
    NpArrayPydanticAnnotation.factory(data_type=np.uint8, dimensions=1, strict_data_typing=True),
]

Np1DArrayFpLongDouble: TypeAlias = Annotated[
    np.ndarray[tuple[int, ...], np.dtype[np.longdouble]],
    NpArrayPydanticAnnotation.factory(data_type=np.longdouble, dimensions=1, strict_data_typing=True),
]

Np1DArrayFp64: TypeAlias = Annotated[
    np.ndarray[tuple[int, ...], np.dtype[np.float64]],
    NpArrayPydanticAnnotation.factory(data_type=np.float64, dimensions=1, strict_data_typing=True),
]

Np1DArrayFp32: TypeAlias = Annotated[
    np.ndarray[tuple[int, ...], np.dtype[np.float32]],
    NpArrayPydanticAnnotation.factory(data_type=np.float32, dimensions=1, strict_data_typing=True),
]

Np1DArrayFp16: TypeAlias = Annotated[
    np.ndarray[tuple[int, ...], np.dtype[np.float16]],
    NpArrayPydanticAnnotation.factory(data_type=np.float16, dimensions=1, strict_data_typing=True),
]

Np1DArrayComplexLongDouble: TypeAlias = Annotated[
    np.ndarray[tuple[int, ...], np.dtype[np.clongdouble]],
    NpArrayPydanticAnnotation.factory(data_type=np.clongdouble, dimensions=1, strict_data_typing=True),
]

Np1DArrayComplex128: TypeAlias = Annotated[
    np.ndarray[tuple[int, ...], np.dtype[np.complex128]],
    NpArrayPydanticAnnotation.factory(data_type=np.complex128, dimensions=1, strict_data_typing=True),
]

Np1DArrayComplex64: TypeAlias = Annotated[
    np.ndarray[tuple[int, ...], np.dtype[np.complex64]],
    NpArrayPydanticAnnotation.factory(data_type=np.complex64, dimensions=1, strict_data_typing=True),
]

Np1DArrayBool: TypeAlias = Annotated[
    np.ndarray[tuple[int, ...], np.dtype[np.bool_]],
    NpArrayPydanticAnnotation.factory(data_type=np.bool_, dimensions=1, strict_data_typing=True),
]

Np1DArrayDatetime64: TypeAlias = Annotated[
    np.ndarray[tuple[int, ...], np.dtype[np.datetime64]],
    NpArrayPydanticAnnotation.factory(data_type=np.datetime64, dimensions=1, strict_data_typing=True),
]

Np1DArrayTimedelta64: TypeAlias = Annotated[
    np.ndarray[tuple[int, ...], np.dtype[np.timedelta64]],
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
