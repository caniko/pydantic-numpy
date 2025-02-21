from typing import Annotated, Any, TypeAlias

import numpy as np

from pydantic_numpy.helper.annotation import NpArrayPydanticAnnotation

Np2DArray: TypeAlias = Annotated[
    np.ndarray[tuple[int, ...], np.dtype[Any]],
    NpArrayPydanticAnnotation.factory(data_type=None, dimensions=2, strict_data_typing=False),
]

Np2DArrayInt64: TypeAlias = Annotated[
    np.ndarray[tuple[int, ...], np.dtype[np.int64]],
    NpArrayPydanticAnnotation.factory(data_type=np.int64, dimensions=2, strict_data_typing=True),
]

Np2DArrayInt32: TypeAlias = Annotated[
    np.ndarray[tuple[int, ...], np.dtype[np.int32]],
    NpArrayPydanticAnnotation.factory(data_type=np.int32, dimensions=2, strict_data_typing=True),
]

Np2DArrayInt16: TypeAlias = Annotated[
    np.ndarray[tuple[int, ...], np.dtype[np.int16]],
    NpArrayPydanticAnnotation.factory(data_type=np.int16, dimensions=2, strict_data_typing=True),
]

Np2DArrayInt8: TypeAlias = Annotated[
    np.ndarray[tuple[int, ...], np.dtype[np.int8]],
    NpArrayPydanticAnnotation.factory(data_type=np.int8, dimensions=2, strict_data_typing=True),
]

Np2DArrayUint64: TypeAlias = Annotated[
    np.ndarray[tuple[int, ...], np.dtype[np.uint64]],
    NpArrayPydanticAnnotation.factory(data_type=np.uint64, dimensions=2, strict_data_typing=True),
]

Np2DArrayUint32: TypeAlias = Annotated[
    np.ndarray[tuple[int, ...], np.dtype[np.uint32]],
    NpArrayPydanticAnnotation.factory(data_type=np.uint32, dimensions=2, strict_data_typing=True),
]

Np2DArrayUint16: TypeAlias = Annotated[
    np.ndarray[tuple[int, ...], np.dtype[np.uint16]],
    NpArrayPydanticAnnotation.factory(data_type=np.uint16, dimensions=2, strict_data_typing=True),
]

Np2DArrayUint8: TypeAlias = Annotated[
    np.ndarray[tuple[int, ...], np.dtype[np.uint8]],
    NpArrayPydanticAnnotation.factory(data_type=np.uint8, dimensions=2, strict_data_typing=True),
]

Np2DArrayFpLongDouble: TypeAlias = Annotated[
    np.ndarray[tuple[int, ...], np.dtype[np.longdouble]],
    NpArrayPydanticAnnotation.factory(data_type=np.longdouble, dimensions=2, strict_data_typing=True),
]

Np2DArrayFp64: TypeAlias = Annotated[
    np.ndarray[tuple[int, ...], np.dtype[np.float64]],
    NpArrayPydanticAnnotation.factory(data_type=np.float64, dimensions=2, strict_data_typing=True),
]

Np2DArrayFp32: TypeAlias = Annotated[
    np.ndarray[tuple[int, ...], np.dtype[np.float32]],
    NpArrayPydanticAnnotation.factory(data_type=np.float32, dimensions=2, strict_data_typing=True),
]

Np2DArrayFp16: TypeAlias = Annotated[
    np.ndarray[tuple[int, ...], np.dtype[np.float16]],
    NpArrayPydanticAnnotation.factory(data_type=np.float16, dimensions=2, strict_data_typing=True),
]

Np2DArrayComplexLongDouble: TypeAlias = Annotated[
    np.ndarray[tuple[int, ...], np.dtype[np.clongdouble]],
    NpArrayPydanticAnnotation.factory(data_type=np.clongdouble, dimensions=2, strict_data_typing=True),
]

Np2DArrayComplex128: TypeAlias = Annotated[
    np.ndarray[tuple[int, ...], np.dtype[np.complex128]],
    NpArrayPydanticAnnotation.factory(data_type=np.complex128, dimensions=2, strict_data_typing=True),
]

Np2DArrayComplex64: TypeAlias = Annotated[
    np.ndarray[tuple[int, ...], np.dtype[np.complex64]],
    NpArrayPydanticAnnotation.factory(data_type=np.complex64, dimensions=2, strict_data_typing=True),
]

Np2DArrayBool: TypeAlias = Annotated[
    np.ndarray[tuple[int, ...], np.dtype[np.bool_]],
    NpArrayPydanticAnnotation.factory(data_type=np.bool_, dimensions=2, strict_data_typing=True),
]

Np2DArrayDatetime64: TypeAlias = Annotated[
    np.ndarray[tuple[int, ...], np.dtype[np.datetime64]],
    NpArrayPydanticAnnotation.factory(data_type=np.datetime64, dimensions=2, strict_data_typing=True),
]

Np2DArrayTimedelta64: TypeAlias = Annotated[
    np.ndarray[tuple[int, ...], np.dtype[np.timedelta64]],
    NpArrayPydanticAnnotation.factory(data_type=np.timedelta64, dimensions=2, strict_data_typing=True),
]

__all__ = [
    "Np2DArray",
    "Np2DArrayInt64",
    "Np2DArrayInt32",
    "Np2DArrayInt16",
    "Np2DArrayInt8",
    "Np2DArrayUint64",
    "Np2DArrayUint32",
    "Np2DArrayUint16",
    "Np2DArrayUint8",
    "Np2DArrayFpLongDouble",
    "Np2DArrayFp64",
    "Np2DArrayFp32",
    "Np2DArrayFp16",
    "Np2DArrayComplexLongDouble",
    "Np2DArrayComplex128",
    "Np2DArrayComplex64",
    "Np2DArrayBool",
    "Np2DArrayDatetime64",
    "Np2DArrayTimedelta64",
]
