from typing import Annotated, Any, Union

import numpy as np
from pydantic import FilePath

from pydantic_numpy.helper.annotation import NpArrayPydanticAnnotation
from pydantic_numpy.model import MultiArrayNumpyFile

Np3DArray = Annotated[
    Union[np.ndarray[tuple[int, int, int], np.dtype[Any]], FilePath, MultiArrayNumpyFile],
    NpArrayPydanticAnnotation.factory(data_type=None, dimensions=3, strict_data_typing=False),
]

Np3DArrayInt64 = Annotated[
    Union[np.ndarray[tuple[int, int, int], np.dtype[np.int64]], FilePath, MultiArrayNumpyFile],
    NpArrayPydanticAnnotation.factory(data_type=np.int64, dimensions=3, strict_data_typing=False),
]

Np3DArrayInt32 = Annotated[
    Union[np.ndarray[tuple[int, int, int], np.dtype[np.int32]], FilePath, MultiArrayNumpyFile],
    NpArrayPydanticAnnotation.factory(data_type=np.int32, dimensions=3, strict_data_typing=False),
]

Np3DArrayInt16 = Annotated[
    Union[np.ndarray[tuple[int, int, int], np.dtype[np.int16]], FilePath, MultiArrayNumpyFile],
    NpArrayPydanticAnnotation.factory(data_type=np.int16, dimensions=3, strict_data_typing=False),
]

Np3DArrayInt8 = Annotated[
    Union[np.ndarray[tuple[int, int, int], np.dtype[np.int8]], FilePath, MultiArrayNumpyFile],
    NpArrayPydanticAnnotation.factory(data_type=np.int8, dimensions=3, strict_data_typing=False),
]

Np3DArrayUint64 = Annotated[
    Union[np.ndarray[tuple[int, int, int], np.dtype[np.uint64]], FilePath, MultiArrayNumpyFile],
    NpArrayPydanticAnnotation.factory(data_type=np.uint64, dimensions=3, strict_data_typing=False),
]

Np3DArrayUint32 = Annotated[
    Union[np.ndarray[tuple[int, int, int], np.dtype[np.uint32]], FilePath, MultiArrayNumpyFile],
    NpArrayPydanticAnnotation.factory(data_type=np.uint32, dimensions=3, strict_data_typing=False),
]

Np3DArrayUint16 = Annotated[
    Union[np.ndarray[tuple[int, int, int], np.dtype[np.uint16]], FilePath, MultiArrayNumpyFile],
    NpArrayPydanticAnnotation.factory(data_type=np.uint16, dimensions=3, strict_data_typing=False),
]

Np3DArrayUint8 = Annotated[
    Union[np.ndarray[tuple[int, int, int], np.dtype[np.uint8]], FilePath, MultiArrayNumpyFile],
    NpArrayPydanticAnnotation.factory(data_type=np.uint8, dimensions=3, strict_data_typing=False),
]

Np3DArrayFpLongDouble = Annotated[
    Union[np.ndarray[tuple[int, int, int], np.dtype[np.longdouble]], FilePath, MultiArrayNumpyFile],
    NpArrayPydanticAnnotation.factory(data_type=np.longdouble, dimensions=3, strict_data_typing=False),
]

Np3DArrayFp64 = Annotated[
    Union[np.ndarray[tuple[int, int, int], np.dtype[np.float64]], FilePath, MultiArrayNumpyFile],
    NpArrayPydanticAnnotation.factory(data_type=np.float64, dimensions=3, strict_data_typing=False),
]

Np3DArrayFp32 = Annotated[
    Union[np.ndarray[tuple[int, int, int], np.dtype[np.float32]], FilePath, MultiArrayNumpyFile],
    NpArrayPydanticAnnotation.factory(data_type=np.float32, dimensions=3, strict_data_typing=False),
]

Np3DArrayFp16 = Annotated[
    Union[np.ndarray[tuple[int, int, int], np.dtype[np.float16]], FilePath, MultiArrayNumpyFile],
    NpArrayPydanticAnnotation.factory(data_type=np.float16, dimensions=3, strict_data_typing=False),
]

Np3DArrayComplexLongDouble = Annotated[
    Union[np.ndarray[tuple[int, int, int], np.dtype[np.clongdouble]], FilePath, MultiArrayNumpyFile],
    NpArrayPydanticAnnotation.factory(data_type=np.clongdouble, dimensions=3, strict_data_typing=False),
]

Np3DArrayComplex128 = Annotated[
    Union[np.ndarray[tuple[int, int, int], np.dtype[np.complex128]], FilePath, MultiArrayNumpyFile],
    NpArrayPydanticAnnotation.factory(data_type=np.complex128, dimensions=3, strict_data_typing=False),
]

Np3DArrayComplex64 = Annotated[
    Union[np.ndarray[tuple[int, int, int], np.dtype[np.complex64]], FilePath, MultiArrayNumpyFile],
    NpArrayPydanticAnnotation.factory(data_type=np.complex64, dimensions=3, strict_data_typing=False),
]

Np3DArrayBool = Annotated[
    Union[np.ndarray[tuple[int, int, int], np.dtype[np.bool_]], FilePath, MultiArrayNumpyFile],
    NpArrayPydanticAnnotation.factory(data_type=np.bool_, dimensions=3, strict_data_typing=False),
]

Np3DArrayDatetime64 = Annotated[
    Union[np.ndarray[tuple[int, int, int], np.dtype[np.datetime64]], FilePath, MultiArrayNumpyFile],
    NpArrayPydanticAnnotation.factory(data_type=np.datetime64, dimensions=3, strict_data_typing=False),
]

Np3DArrayTimedelta64 = Annotated[
    Union[np.ndarray[tuple[int, int, int], np.dtype[np.timedelta64]], FilePath, MultiArrayNumpyFile],
    NpArrayPydanticAnnotation.factory(data_type=np.timedelta64, dimensions=3, strict_data_typing=False),
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
