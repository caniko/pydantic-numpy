from typing import Annotated, Any, TypeAlias

import numpy as np

from pydantic_numpy.helper.annotation import NpArrayPydanticAnnotation

Np3DArray: TypeAlias = Annotated[
    np.ndarray[tuple[int, ...], np.dtype[Any]],
    NpArrayPydanticAnnotation.factory(data_type=None, dimensions=3, strict_data_typing=False),
]

Np3DArrayInt64: TypeAlias = Annotated[
    np.ndarray[tuple[int, ...], np.dtype[np.int64]],
    NpArrayPydanticAnnotation.factory(data_type=np.int64, dimensions=3, strict_data_typing=True),
]

Np3DArrayInt32: TypeAlias = Annotated[
    np.ndarray[tuple[int, ...], np.dtype[np.int32]],
    NpArrayPydanticAnnotation.factory(data_type=np.int32, dimensions=3, strict_data_typing=True),
]

Np3DArrayInt16: TypeAlias = Annotated[
    np.ndarray[tuple[int, ...], np.dtype[np.int16]],
    NpArrayPydanticAnnotation.factory(data_type=np.int16, dimensions=3, strict_data_typing=True),
]

Np3DArrayInt8: TypeAlias = Annotated[
    np.ndarray[tuple[int, ...], np.dtype[np.int8]],
    NpArrayPydanticAnnotation.factory(data_type=np.int8, dimensions=3, strict_data_typing=True),
]

Np3DArrayUint64: TypeAlias = Annotated[
    np.ndarray[tuple[int, ...], np.dtype[np.uint64]],
    NpArrayPydanticAnnotation.factory(data_type=np.uint64, dimensions=3, strict_data_typing=True),
]

Np3DArrayUint32: TypeAlias = Annotated[
    np.ndarray[tuple[int, ...], np.dtype[np.uint32]],
    NpArrayPydanticAnnotation.factory(data_type=np.uint32, dimensions=3, strict_data_typing=True),
]

Np3DArrayUint16: TypeAlias = Annotated[
    np.ndarray[tuple[int, ...], np.dtype[np.uint16]],
    NpArrayPydanticAnnotation.factory(data_type=np.uint16, dimensions=3, strict_data_typing=True),
]

Np3DArrayUint8: TypeAlias = Annotated[
    np.ndarray[tuple[int, ...], np.dtype[np.uint8]],
    NpArrayPydanticAnnotation.factory(data_type=np.uint8, dimensions=3, strict_data_typing=True),
]

Np3DArrayFpLongDouble: TypeAlias = Annotated[
    np.ndarray[tuple[int, ...], np.dtype[np.longdouble]],
    NpArrayPydanticAnnotation.factory(data_type=np.longdouble, dimensions=3, strict_data_typing=True),
]

Np3DArrayFp64: TypeAlias = Annotated[
    np.ndarray[tuple[int, ...], np.dtype[np.float64]],
    NpArrayPydanticAnnotation.factory(data_type=np.float64, dimensions=3, strict_data_typing=True),
]

Np3DArrayFp32: TypeAlias = Annotated[
    np.ndarray[tuple[int, ...], np.dtype[np.float32]],
    NpArrayPydanticAnnotation.factory(data_type=np.float32, dimensions=3, strict_data_typing=True),
]

Np3DArrayFp16: TypeAlias = Annotated[
    np.ndarray[tuple[int, ...], np.dtype[np.float16]],
    NpArrayPydanticAnnotation.factory(data_type=np.float16, dimensions=3, strict_data_typing=True),
]

Np3DArrayComplexLongDouble: TypeAlias = Annotated[
    np.ndarray[tuple[int, ...], np.dtype[np.clongdouble]],
    NpArrayPydanticAnnotation.factory(data_type=np.clongdouble, dimensions=3, strict_data_typing=True),
]

Np3DArrayComplex128: TypeAlias = Annotated[
    np.ndarray[tuple[int, ...], np.dtype[np.complex128]],
    NpArrayPydanticAnnotation.factory(data_type=np.complex128, dimensions=3, strict_data_typing=True),
]

Np3DArrayComplex64: TypeAlias = Annotated[
    np.ndarray[tuple[int, ...], np.dtype[np.complex64]],
    NpArrayPydanticAnnotation.factory(data_type=np.complex64, dimensions=3, strict_data_typing=True),
]

Np3DArrayBool: TypeAlias = Annotated[
    np.ndarray[tuple[int, ...], np.dtype[np.bool_]],
    NpArrayPydanticAnnotation.factory(data_type=np.bool_, dimensions=3, strict_data_typing=True),
]

Np3DArrayDatetime64: TypeAlias = Annotated[
    np.ndarray[tuple[int, ...], np.dtype[np.datetime64]],
    NpArrayPydanticAnnotation.factory(data_type=np.datetime64, dimensions=3, strict_data_typing=True),
]

Np3DArrayTimedelta64: TypeAlias = Annotated[
    np.ndarray[tuple[int, ...], np.dtype[np.timedelta64]],
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
