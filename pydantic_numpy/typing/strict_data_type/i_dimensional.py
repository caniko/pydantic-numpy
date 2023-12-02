from typing import Annotated

import numpy as np

from pydantic_numpy.helper.annotation import NpArrayPydanticAnnotation

NpStrict1DArrayInt64 = Annotated[
    np.ndarray[tuple[int], np.dtype[np.int64]],
    NpArrayPydanticAnnotation.factory(data_type=np.int64, dimensions=1, strict_data_typing=True),
]

NpStrict1DArrayInt32 = Annotated[
    np.ndarray[tuple[int], np.dtype[np.int32]],
    NpArrayPydanticAnnotation.factory(data_type=np.int32, dimensions=1, strict_data_typing=True),
]

NpStrict1DArrayInt16 = Annotated[
    np.ndarray[tuple[int], np.dtype[np.int16]],
    NpArrayPydanticAnnotation.factory(data_type=np.int16, dimensions=1, strict_data_typing=True),
]

NpStrict1DArrayInt8 = Annotated[
    np.ndarray[tuple[int], np.dtype[np.int8]],
    NpArrayPydanticAnnotation.factory(data_type=np.int8, dimensions=1, strict_data_typing=True),
]

NpStrict1DArrayUint64 = Annotated[
    np.ndarray[tuple[int], np.dtype[np.uint64]],
    NpArrayPydanticAnnotation.factory(data_type=np.uint64, dimensions=1, strict_data_typing=True),
]

NpStrict1DArrayUint32 = Annotated[
    np.ndarray[tuple[int], np.dtype[np.uint32]],
    NpArrayPydanticAnnotation.factory(data_type=np.uint32, dimensions=1, strict_data_typing=True),
]

NpStrict1DArrayUint16 = Annotated[
    np.ndarray[tuple[int], np.dtype[np.uint16]],
    NpArrayPydanticAnnotation.factory(data_type=np.uint16, dimensions=1, strict_data_typing=True),
]

NpStrict1DArrayUint8 = Annotated[
    np.ndarray[tuple[int], np.dtype[np.uint8]],
    NpArrayPydanticAnnotation.factory(data_type=np.uint8, dimensions=1, strict_data_typing=True),
]

NpStrict1DArrayFpLongDouble = Annotated[
    np.ndarray[tuple[int], np.dtype[np.longdouble]],
    NpArrayPydanticAnnotation.factory(data_type=np.longdouble, dimensions=1, strict_data_typing=True),
]

NpStrict1DArrayFp64 = Annotated[
    np.ndarray[tuple[int], np.dtype[np.float64]],
    NpArrayPydanticAnnotation.factory(data_type=np.float64, dimensions=1, strict_data_typing=True),
]

NpStrict1DArrayFp32 = Annotated[
    np.ndarray[tuple[int], np.dtype[np.float32]],
    NpArrayPydanticAnnotation.factory(data_type=np.float32, dimensions=1, strict_data_typing=True),
]

NpStrict1DArrayFp16 = Annotated[
    np.ndarray[tuple[int], np.dtype[np.float16]],
    NpArrayPydanticAnnotation.factory(data_type=np.float16, dimensions=1, strict_data_typing=True),
]

NpStrict1DArrayComplexLongDouble = Annotated[
    np.ndarray[tuple[int], np.dtype[np.clongdouble]],
    NpArrayPydanticAnnotation.factory(data_type=np.clongdouble, dimensions=1, strict_data_typing=True),
]

NpStrict1DArrayComplex128 = Annotated[
    np.ndarray[tuple[int], np.dtype[np.complex128]],
    NpArrayPydanticAnnotation.factory(data_type=np.complex128, dimensions=1, strict_data_typing=True),
]

NpStrict1DArrayComplex64 = Annotated[
    np.ndarray[tuple[int], np.dtype[np.complex64]],
    NpArrayPydanticAnnotation.factory(data_type=np.complex64, dimensions=1, strict_data_typing=True),
]

NpStrict1DArrayBool = Annotated[
    np.ndarray[tuple[int], np.dtype[np.bool_]],
    NpArrayPydanticAnnotation.factory(data_type=np.bool_, dimensions=1, strict_data_typing=True),
]

NpStrict1DArrayDatetime64 = Annotated[
    np.ndarray[tuple[int], np.dtype[np.datetime64]],
    NpArrayPydanticAnnotation.factory(data_type=np.datetime64, dimensions=1, strict_data_typing=True),
]

NpStrict1DArrayTimedelta64 = Annotated[
    np.ndarray[tuple[int], np.dtype[np.timedelta64]],
    NpArrayPydanticAnnotation.factory(data_type=np.timedelta64, dimensions=1, strict_data_typing=True),
]

__all__ = [
    "NpStrict1DArrayInt64",
    "NpStrict1DArrayInt32",
    "NpStrict1DArrayInt16",
    "NpStrict1DArrayInt8",
    "NpStrict1DArrayUint64",
    "NpStrict1DArrayUint32",
    "NpStrict1DArrayUint16",
    "NpStrict1DArrayUint8",
    "NpStrict1DArrayFpLongDouble",
    "NpStrict1DArrayFp64",
    "NpStrict1DArrayFp32",
    "NpStrict1DArrayFp16",
    "NpStrict1DArrayComplexLongDouble",
    "NpStrict1DArrayComplex128",
    "NpStrict1DArrayComplex64",
    "NpStrict1DArrayBool",
    "NpStrict1DArrayDatetime64",
    "NpStrict1DArrayTimedelta64",
]
