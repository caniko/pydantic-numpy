from typing import Annotated

import numpy as np

from pydantic_numpy.helper.annotation import NpArrayPydanticAnnotation

NpStrict3DArrayInt64 = Annotated[
    np.ndarray[tuple[int, int, int], np.dtype[np.int64]],
    NpArrayPydanticAnnotation.factory(data_type=np.int64, dimensions=3, strict_data_typing=True),
]

NpStrict3DArrayInt32 = Annotated[
    np.ndarray[tuple[int, int, int], np.dtype[np.int32]],
    NpArrayPydanticAnnotation.factory(data_type=np.int32, dimensions=3, strict_data_typing=True),
]

NpStrict3DArrayInt16 = Annotated[
    np.ndarray[tuple[int, int, int], np.dtype[np.int16]],
    NpArrayPydanticAnnotation.factory(data_type=np.int16, dimensions=3, strict_data_typing=True),
]

NpStrict3DArrayInt8 = Annotated[
    np.ndarray[tuple[int, int, int], np.dtype[np.int8]],
    NpArrayPydanticAnnotation.factory(data_type=np.int8, dimensions=3, strict_data_typing=True),
]

NpStrict3DArrayUint64 = Annotated[
    np.ndarray[tuple[int, int, int], np.dtype[np.uint64]],
    NpArrayPydanticAnnotation.factory(data_type=np.uint64, dimensions=3, strict_data_typing=True),
]

NpStrict3DArrayUint32 = Annotated[
    np.ndarray[tuple[int, int, int], np.dtype[np.uint32]],
    NpArrayPydanticAnnotation.factory(data_type=np.uint32, dimensions=3, strict_data_typing=True),
]

NpStrict3DArrayUint16 = Annotated[
    np.ndarray[tuple[int, int, int], np.dtype[np.uint16]],
    NpArrayPydanticAnnotation.factory(data_type=np.uint16, dimensions=3, strict_data_typing=True),
]

NpStrict3DArrayUint8 = Annotated[
    np.ndarray[tuple[int, int, int], np.dtype[np.uint8]],
    NpArrayPydanticAnnotation.factory(data_type=np.uint8, dimensions=3, strict_data_typing=True),
]

NpStrict3DArrayFpLongDouble = Annotated[
    np.ndarray[tuple[int, int, int], np.dtype[np.longdouble]],
    NpArrayPydanticAnnotation.factory(data_type=np.longdouble, dimensions=3, strict_data_typing=True),
]

NpStrict3DArrayFp64 = Annotated[
    np.ndarray[tuple[int, int, int], np.dtype[np.float64]],
    NpArrayPydanticAnnotation.factory(data_type=np.float64, dimensions=3, strict_data_typing=True),
]

NpStrict3DArrayFp32 = Annotated[
    np.ndarray[tuple[int, int, int], np.dtype[np.float32]],
    NpArrayPydanticAnnotation.factory(data_type=np.float32, dimensions=3, strict_data_typing=True),
]

NpStrict3DArrayFp16 = Annotated[
    np.ndarray[tuple[int, int, int], np.dtype[np.float16]],
    NpArrayPydanticAnnotation.factory(data_type=np.float16, dimensions=3, strict_data_typing=True),
]

NpStrict3DArrayComplexLongDouble = Annotated[
    np.ndarray[tuple[int, int, int], np.dtype[np.clongdouble]],
    NpArrayPydanticAnnotation.factory(data_type=np.clongdouble, dimensions=3, strict_data_typing=True),
]

NpStrict3DArrayComplex128 = Annotated[
    np.ndarray[tuple[int, int, int], np.dtype[np.complex128]],
    NpArrayPydanticAnnotation.factory(data_type=np.complex128, dimensions=3, strict_data_typing=True),
]

NpStrict3DArrayComplex64 = Annotated[
    np.ndarray[tuple[int, int, int], np.dtype[np.complex64]],
    NpArrayPydanticAnnotation.factory(data_type=np.complex64, dimensions=3, strict_data_typing=True),
]

NpStrict3DArrayBool = Annotated[
    np.ndarray[tuple[int, int, int], np.dtype[np.bool_]],
    NpArrayPydanticAnnotation.factory(data_type=np.bool_, dimensions=3, strict_data_typing=True),
]

NpStrict3DArrayDatetime64 = Annotated[
    np.ndarray[tuple[int, int, int], np.dtype[np.datetime64]],
    NpArrayPydanticAnnotation.factory(data_type=np.datetime64, dimensions=3, strict_data_typing=True),
]

NpStrict3DArrayTimedelta64 = Annotated[
    np.ndarray[tuple[int, int, int], np.dtype[np.timedelta64]],
    NpArrayPydanticAnnotation.factory(data_type=np.timedelta64, dimensions=3, strict_data_typing=True),
]

__all__ = [
    "NpStrict3DArrayInt64",
    "NpStrict3DArrayInt32",
    "NpStrict3DArrayInt16",
    "NpStrict3DArrayInt8",
    "NpStrict3DArrayUint64",
    "NpStrict3DArrayUint32",
    "NpStrict3DArrayUint16",
    "NpStrict3DArrayUint8",
    "NpStrict3DArrayFpLongDouble",
    "NpStrict3DArrayFp64",
    "NpStrict3DArrayFp32",
    "NpStrict3DArrayFp16",
    "NpStrict3DArrayComplexLongDouble",
    "NpStrict3DArrayComplex128",
    "NpStrict3DArrayComplex64",
    "NpStrict3DArrayBool",
    "NpStrict3DArrayDatetime64",
    "NpStrict3DArrayTimedelta64",
]
