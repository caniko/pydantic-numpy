from typing import Annotated, Any, Union

import numpy as np
from pydantic import FilePath

from pydantic_numpy.helper.annotation import NpArrayPydanticAnnotation
from pydantic_numpy.model import MultiArrayNumpyFile

Np2DArray = Annotated[
    Union[np.ndarray[tuple[int, int], np.dtype[Any]], FilePath, MultiArrayNumpyFile],
    NpArrayPydanticAnnotation.factory(data_type=None, dimensions=2, strict_data_typing=False),
]

Np2DArrayInt64 = Annotated[
    Union[np.ndarray[tuple[int, int], np.dtype[np.int64]], FilePath, MultiArrayNumpyFile],
    NpArrayPydanticAnnotation.factory(data_type=np.int64, dimensions=2, strict_data_typing=False),
]

Np2DArrayInt32 = Annotated[
    Union[np.ndarray[tuple[int, int], np.dtype[np.int32]], FilePath, MultiArrayNumpyFile],
    NpArrayPydanticAnnotation.factory(data_type=np.int32, dimensions=2, strict_data_typing=False),
]

Np2DArrayInt16 = Annotated[
    Union[np.ndarray[tuple[int, int], np.dtype[np.int16]], FilePath, MultiArrayNumpyFile],
    NpArrayPydanticAnnotation.factory(data_type=np.int16, dimensions=2, strict_data_typing=False),
]

Np2DArrayInt8 = Annotated[
    Union[np.ndarray[tuple[int, int], np.dtype[np.int8]], FilePath, MultiArrayNumpyFile],
    NpArrayPydanticAnnotation.factory(data_type=np.int8, dimensions=2, strict_data_typing=False),
]

Np2DArrayUint64 = Annotated[
    Union[np.ndarray[tuple[int, int], np.dtype[np.uint64]], FilePath, MultiArrayNumpyFile],
    NpArrayPydanticAnnotation.factory(data_type=np.uint64, dimensions=2, strict_data_typing=False),
]

Np2DArrayUint32 = Annotated[
    Union[np.ndarray[tuple[int, int], np.dtype[np.uint32]], FilePath, MultiArrayNumpyFile],
    NpArrayPydanticAnnotation.factory(data_type=np.uint32, dimensions=2, strict_data_typing=False),
]

Np2DArrayUint16 = Annotated[
    Union[np.ndarray[tuple[int, int], np.dtype[np.uint16]], FilePath, MultiArrayNumpyFile],
    NpArrayPydanticAnnotation.factory(data_type=np.uint16, dimensions=2, strict_data_typing=False),
]

Np2DArrayUint8 = Annotated[
    Union[np.ndarray[tuple[int, int], np.dtype[np.uint8]], FilePath, MultiArrayNumpyFile],
    NpArrayPydanticAnnotation.factory(data_type=np.uint8, dimensions=2, strict_data_typing=False),
]

Np2DArrayFpLongDouble = Annotated[
    Union[np.ndarray[tuple[int, int], np.dtype[np.longdouble]], FilePath, MultiArrayNumpyFile],
    NpArrayPydanticAnnotation.factory(data_type=np.longdouble, dimensions=2, strict_data_typing=False),
]

Np2DArrayFp64 = Annotated[
    Union[np.ndarray[tuple[int, int], np.dtype[np.float64]], FilePath, MultiArrayNumpyFile],
    NpArrayPydanticAnnotation.factory(data_type=np.float64, dimensions=2, strict_data_typing=False),
]

Np2DArrayFp32 = Annotated[
    Union[np.ndarray[tuple[int, int], np.dtype[np.float32]], FilePath, MultiArrayNumpyFile],
    NpArrayPydanticAnnotation.factory(data_type=np.float32, dimensions=2, strict_data_typing=False),
]

Np2DArrayFp16 = Annotated[
    Union[np.ndarray[tuple[int, int], np.dtype[np.float16]], FilePath, MultiArrayNumpyFile],
    NpArrayPydanticAnnotation.factory(data_type=np.float16, dimensions=2, strict_data_typing=False),
]

Np2DArrayComplexLongDouble = Annotated[
    Union[np.ndarray[tuple[int, int], np.dtype[np.clongdouble]], FilePath, MultiArrayNumpyFile],
    NpArrayPydanticAnnotation.factory(data_type=np.clongdouble, dimensions=2, strict_data_typing=False),
]

Np2DArrayComplex128 = Annotated[
    Union[np.ndarray[tuple[int, int], np.dtype[np.complex128]], FilePath, MultiArrayNumpyFile],
    NpArrayPydanticAnnotation.factory(data_type=np.complex128, dimensions=2, strict_data_typing=False),
]

Np2DArrayComplex64 = Annotated[
    Union[np.ndarray[tuple[int, int], np.dtype[np.complex64]], FilePath, MultiArrayNumpyFile],
    NpArrayPydanticAnnotation.factory(data_type=np.complex64, dimensions=2, strict_data_typing=False),
]

Np2DArrayBool = Annotated[
    Union[np.ndarray[tuple[int, int], np.dtype[np.bool_]], FilePath, MultiArrayNumpyFile],
    NpArrayPydanticAnnotation.factory(data_type=np.bool_, dimensions=2, strict_data_typing=False),
]

Np2DArrayDatetime64 = Annotated[
    Union[np.ndarray[tuple[int, int], np.dtype[np.datetime64]], FilePath, MultiArrayNumpyFile],
    NpArrayPydanticAnnotation.factory(data_type=np.datetime64, dimensions=2, strict_data_typing=False),
]

Np2DArrayTimedelta64 = Annotated[
    Union[np.ndarray[tuple[int, int], np.dtype[np.timedelta64]], FilePath, MultiArrayNumpyFile],
    NpArrayPydanticAnnotation.factory(data_type=np.timedelta64, dimensions=2, strict_data_typing=False),
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
