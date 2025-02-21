from typing import Annotated, Any, TypeAlias, Union

import numpy as np
from pydantic import FilePath

from pydantic_numpy.helper.annotation import NpArrayPydanticAnnotation
from pydantic_numpy.model import MultiArrayNumpyFile

NpLoading2DArray: TypeAlias = Annotated[
    Union[np.ndarray[tuple[int, ...], np.dtype[Any]], FilePath, MultiArrayNumpyFile],
    NpArrayPydanticAnnotation.factory(data_type=None, dimensions=2, strict_data_typing=False),
]

NpLoading2DArrayInt64: TypeAlias = Annotated[
    Union[np.ndarray[tuple[int, ...], np.dtype[np.int64]], FilePath, MultiArrayNumpyFile],
    NpArrayPydanticAnnotation.factory(data_type=np.int64, dimensions=2, strict_data_typing=False),
]

NpLoading2DArrayInt32: TypeAlias = Annotated[
    Union[np.ndarray[tuple[int, ...], np.dtype[np.int32]], FilePath, MultiArrayNumpyFile],
    NpArrayPydanticAnnotation.factory(data_type=np.int32, dimensions=2, strict_data_typing=False),
]

NpLoading2DArrayInt16: TypeAlias = Annotated[
    Union[np.ndarray[tuple[int, ...], np.dtype[np.int16]], FilePath, MultiArrayNumpyFile],
    NpArrayPydanticAnnotation.factory(data_type=np.int16, dimensions=2, strict_data_typing=False),
]

NpLoading2DArrayInt8: TypeAlias = Annotated[
    Union[np.ndarray[tuple[int, ...], np.dtype[np.int8]], FilePath, MultiArrayNumpyFile],
    NpArrayPydanticAnnotation.factory(data_type=np.int8, dimensions=2, strict_data_typing=False),
]

NpLoading2DArrayUint64: TypeAlias = Annotated[
    Union[np.ndarray[tuple[int, ...], np.dtype[np.uint64]], FilePath, MultiArrayNumpyFile],
    NpArrayPydanticAnnotation.factory(data_type=np.uint64, dimensions=2, strict_data_typing=False),
]

NpLoading2DArrayUint32: TypeAlias = Annotated[
    Union[np.ndarray[tuple[int, ...], np.dtype[np.uint32]], FilePath, MultiArrayNumpyFile],
    NpArrayPydanticAnnotation.factory(data_type=np.uint32, dimensions=2, strict_data_typing=False),
]

NpLoading2DArrayUint16: TypeAlias = Annotated[
    Union[np.ndarray[tuple[int, ...], np.dtype[np.uint16]], FilePath, MultiArrayNumpyFile],
    NpArrayPydanticAnnotation.factory(data_type=np.uint16, dimensions=2, strict_data_typing=False),
]

NpLoading2DArrayUint8: TypeAlias = Annotated[
    Union[np.ndarray[tuple[int, ...], np.dtype[np.uint8]], FilePath, MultiArrayNumpyFile],
    NpArrayPydanticAnnotation.factory(data_type=np.uint8, dimensions=2, strict_data_typing=False),
]

NpLoading2DArrayFpLongDouble: TypeAlias = Annotated[
    Union[np.ndarray[tuple[int, ...], np.dtype[np.longdouble]], FilePath, MultiArrayNumpyFile],
    NpArrayPydanticAnnotation.factory(data_type=np.longdouble, dimensions=2, strict_data_typing=False),
]

NpLoading2DArrayFp64: TypeAlias = Annotated[
    Union[np.ndarray[tuple[int, ...], np.dtype[np.float64]], FilePath, MultiArrayNumpyFile],
    NpArrayPydanticAnnotation.factory(data_type=np.float64, dimensions=2, strict_data_typing=False),
]

NpLoading2DArrayFp32: TypeAlias = Annotated[
    Union[np.ndarray[tuple[int, ...], np.dtype[np.float32]], FilePath, MultiArrayNumpyFile],
    NpArrayPydanticAnnotation.factory(data_type=np.float32, dimensions=2, strict_data_typing=False),
]

NpLoading2DArrayFp16: TypeAlias = Annotated[
    Union[np.ndarray[tuple[int, ...], np.dtype[np.float16]], FilePath, MultiArrayNumpyFile],
    NpArrayPydanticAnnotation.factory(data_type=np.float16, dimensions=2, strict_data_typing=False),
]

NpLoading2DArrayComplexLongDouble: TypeAlias = Annotated[
    Union[np.ndarray[tuple[int, ...], np.dtype[np.clongdouble]], FilePath, MultiArrayNumpyFile],
    NpArrayPydanticAnnotation.factory(data_type=np.clongdouble, dimensions=2, strict_data_typing=False),
]

NpLoading2DArrayComplex128: TypeAlias = Annotated[
    Union[np.ndarray[tuple[int, ...], np.dtype[np.complex128]], FilePath, MultiArrayNumpyFile],
    NpArrayPydanticAnnotation.factory(data_type=np.complex128, dimensions=2, strict_data_typing=False),
]

NpLoading2DArrayComplex64: TypeAlias = Annotated[
    Union[np.ndarray[tuple[int, ...], np.dtype[np.complex64]], FilePath, MultiArrayNumpyFile],
    NpArrayPydanticAnnotation.factory(data_type=np.complex64, dimensions=2, strict_data_typing=False),
]

NpLoading2DArrayBool: TypeAlias = Annotated[
    Union[np.ndarray[tuple[int, ...], np.dtype[np.bool_]], FilePath, MultiArrayNumpyFile],
    NpArrayPydanticAnnotation.factory(data_type=np.bool_, dimensions=2, strict_data_typing=False),
]

NpLoading2DArrayDatetime64: TypeAlias = Annotated[
    Union[np.ndarray[tuple[int, ...], np.dtype[np.datetime64]], FilePath, MultiArrayNumpyFile],
    NpArrayPydanticAnnotation.factory(data_type=np.datetime64, dimensions=2, strict_data_typing=False),
]

NpLoading2DArrayTimedelta64: TypeAlias = Annotated[
    Union[np.ndarray[tuple[int, ...], np.dtype[np.timedelta64]], FilePath, MultiArrayNumpyFile],
    NpArrayPydanticAnnotation.factory(data_type=np.timedelta64, dimensions=2, strict_data_typing=False),
]

__all__ = [
    "NpLoading2DArray",
    "NpLoading2DArrayInt64",
    "NpLoading2DArrayInt32",
    "NpLoading2DArrayInt16",
    "NpLoading2DArrayInt8",
    "NpLoading2DArrayUint64",
    "NpLoading2DArrayUint32",
    "NpLoading2DArrayUint16",
    "NpLoading2DArrayUint8",
    "NpLoading2DArrayFpLongDouble",
    "NpLoading2DArrayFp64",
    "NpLoading2DArrayFp32",
    "NpLoading2DArrayFp16",
    "NpLoading2DArrayComplexLongDouble",
    "NpLoading2DArrayComplex128",
    "NpLoading2DArrayComplex64",
    "NpLoading2DArrayBool",
    "NpLoading2DArrayDatetime64",
    "NpLoading2DArrayTimedelta64",
]
