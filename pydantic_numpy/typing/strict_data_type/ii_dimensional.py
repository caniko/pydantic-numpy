from typing import Annotated

import numpy as np

from pydantic_numpy.helper.annotation import NpArrayPydanticAnnotation

NpStrict2DArrayInt64 = Annotated[
    np.ndarray[tuple[int, int], np.dtype[np.int64]],
    NpArrayPydanticAnnotation.factory(data_type=np.int64, dimensions=2, strict_data_typing=True),
]

NpStrict2DArrayInt32 = Annotated[
    np.ndarray[tuple[int, int], np.dtype[np.int32]],
    NpArrayPydanticAnnotation.factory(data_type=np.int32, dimensions=2, strict_data_typing=True),
]

NpStrict2DArrayInt16 = Annotated[
    np.ndarray[tuple[int, int], np.dtype[np.int16]],
    NpArrayPydanticAnnotation.factory(data_type=np.int16, dimensions=2, strict_data_typing=True),
]

NpStrict2DArrayInt8 = Annotated[
    np.ndarray[tuple[int, int], np.dtype[np.int8]],
    NpArrayPydanticAnnotation.factory(data_type=np.int8, dimensions=2, strict_data_typing=True),
]

NpStrict2DArrayUint64 = Annotated[
    np.ndarray[tuple[int, int], np.dtype[np.uint64]],
    NpArrayPydanticAnnotation.factory(data_type=np.uint64, dimensions=2, strict_data_typing=True),
]

NpStrict2DArrayUint32 = Annotated[
    np.ndarray[tuple[int, int], np.dtype[np.uint32]],
    NpArrayPydanticAnnotation.factory(data_type=np.uint32, dimensions=2, strict_data_typing=True),
]

NpStrict2DArrayUint16 = Annotated[
    np.ndarray[tuple[int, int], np.dtype[np.uint16]],
    NpArrayPydanticAnnotation.factory(data_type=np.uint16, dimensions=2, strict_data_typing=True),
]

NpStrict2DArrayUint8 = Annotated[
    np.ndarray[tuple[int, int], np.dtype[np.uint8]],
    NpArrayPydanticAnnotation.factory(data_type=np.uint8, dimensions=2, strict_data_typing=True),
]

NpStrict2DArrayFpLongDouble = Annotated[
    np.ndarray[tuple[int, int], np.dtype[np.longdouble]],
    NpArrayPydanticAnnotation.factory(data_type=np.longdouble, dimensions=2, strict_data_typing=True),
]

NpStrict2DArrayFp64 = Annotated[
    np.ndarray[tuple[int, int], np.dtype[np.float64]],
    NpArrayPydanticAnnotation.factory(data_type=np.float64, dimensions=2, strict_data_typing=True),
]

NpStrict2DArrayFp32 = Annotated[
    np.ndarray[tuple[int, int], np.dtype[np.float32]],
    NpArrayPydanticAnnotation.factory(data_type=np.float32, dimensions=2, strict_data_typing=True),
]

NpStrict2DArrayFp16 = Annotated[
    np.ndarray[tuple[int, int], np.dtype[np.float16]],
    NpArrayPydanticAnnotation.factory(data_type=np.float16, dimensions=2, strict_data_typing=True),
]

NpStrict2DArrayComplexLongDouble = Annotated[
    np.ndarray[tuple[int, int], np.dtype[np.clongdouble]],
    NpArrayPydanticAnnotation.factory(data_type=np.clongdouble, dimensions=2, strict_data_typing=True),
]

NpStrict2DArrayComplex128 = Annotated[
    np.ndarray[tuple[int, int], np.dtype[np.complex128]],
    NpArrayPydanticAnnotation.factory(data_type=np.complex128, dimensions=2, strict_data_typing=True),
]

NpStrict2DArrayComplex64 = Annotated[
    np.ndarray[tuple[int, int], np.dtype[np.complex64]],
    NpArrayPydanticAnnotation.factory(data_type=np.complex64, dimensions=2, strict_data_typing=True),
]

NpStrict2DArrayBool = Annotated[
    np.ndarray[tuple[int, int], np.dtype[np.bool_]],
    NpArrayPydanticAnnotation.factory(data_type=np.bool_, dimensions=2, strict_data_typing=True),
]

NpStrict2DArrayDatetime64 = Annotated[
    np.ndarray[tuple[int, int], np.dtype[np.datetime64]],
    NpArrayPydanticAnnotation.factory(data_type=np.datetime64, dimensions=2, strict_data_typing=True),
]

NpStrict2DArrayTimedelta64 = Annotated[
    np.ndarray[tuple[int, int], np.dtype[np.timedelta64]],
    NpArrayPydanticAnnotation.factory(data_type=np.timedelta64, dimensions=2, strict_data_typing=True),
]

__all__ = [
    "NpStrict2DArrayInt64",
    "NpStrict2DArrayInt32",
    "NpStrict2DArrayInt16",
    "NpStrict2DArrayInt8",
    "NpStrict2DArrayUint64",
    "NpStrict2DArrayUint32",
    "NpStrict2DArrayUint16",
    "NpStrict2DArrayUint8",
    "NpStrict2DArrayFpLongDouble",
    "NpStrict2DArrayFp64",
    "NpStrict2DArrayFp32",
    "NpStrict2DArrayFp16",
    "NpStrict2DArrayComplexLongDouble",
    "NpStrict2DArrayComplex128",
    "NpStrict2DArrayComplex64",
    "NpStrict2DArrayBool",
    "NpStrict2DArrayDatetime64",
    "NpStrict2DArrayTimedelta64",
]
