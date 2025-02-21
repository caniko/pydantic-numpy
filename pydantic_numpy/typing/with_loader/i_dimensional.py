from typing import Annotated, Any, TypeAlias, Union

import numpy as np
from pydantic import FilePath

from pydantic_numpy.helper.annotation import NpArrayPydanticAnnotation
from pydantic_numpy.model import MultiArrayNumpyFile

NpLoading1DArray: TypeAlias = Annotated[
    Union[np.ndarray[tuple[int, ...], np.dtype[Any]], FilePath, MultiArrayNumpyFile],
    NpArrayPydanticAnnotation.factory(data_type=None, dimensions=1, strict_data_typing=False),
]

NpLoading1DArrayInt64: TypeAlias = Annotated[
    Union[np.ndarray[tuple[int, ...], np.dtype[np.int64]], FilePath, MultiArrayNumpyFile],
    NpArrayPydanticAnnotation.factory(data_type=np.int64, dimensions=1, strict_data_typing=False),
]

NpLoading1DArrayInt32: TypeAlias = Annotated[
    Union[np.ndarray[tuple[int, ...], np.dtype[np.int32]], FilePath, MultiArrayNumpyFile],
    NpArrayPydanticAnnotation.factory(data_type=np.int32, dimensions=1, strict_data_typing=False),
]

NpLoading1DArrayInt16: TypeAlias = Annotated[
    Union[np.ndarray[tuple[int, ...], np.dtype[np.int16]], FilePath, MultiArrayNumpyFile],
    NpArrayPydanticAnnotation.factory(data_type=np.int16, dimensions=1, strict_data_typing=False),
]

NpLoading1DArrayInt8: TypeAlias = Annotated[
    Union[np.ndarray[tuple[int, ...], np.dtype[np.int8]], FilePath, MultiArrayNumpyFile],
    NpArrayPydanticAnnotation.factory(data_type=np.int8, dimensions=1, strict_data_typing=False),
]

NpLoading1DArrayUint64: TypeAlias = Annotated[
    Union[np.ndarray[tuple[int, ...], np.dtype[np.uint64]], FilePath, MultiArrayNumpyFile],
    NpArrayPydanticAnnotation.factory(data_type=np.uint64, dimensions=1, strict_data_typing=False),
]

NpLoading1DArrayUint32: TypeAlias = Annotated[
    Union[np.ndarray[tuple[int, ...], np.dtype[np.uint32]], FilePath, MultiArrayNumpyFile],
    NpArrayPydanticAnnotation.factory(data_type=np.uint32, dimensions=1, strict_data_typing=False),
]

NpLoading1DArrayUint16: TypeAlias = Annotated[
    Union[np.ndarray[tuple[int, ...], np.dtype[np.uint16]], FilePath, MultiArrayNumpyFile],
    NpArrayPydanticAnnotation.factory(data_type=np.uint16, dimensions=1, strict_data_typing=False),
]

NpLoading1DArrayUint8: TypeAlias = Annotated[
    Union[np.ndarray[tuple[int, ...], np.dtype[np.uint8]], FilePath, MultiArrayNumpyFile],
    NpArrayPydanticAnnotation.factory(data_type=np.uint8, dimensions=1, strict_data_typing=False),
]

NpLoading1DArrayFpLongDouble: TypeAlias = Annotated[
    Union[np.ndarray[tuple[int, ...], np.dtype[np.longdouble]], FilePath, MultiArrayNumpyFile],
    NpArrayPydanticAnnotation.factory(data_type=np.longdouble, dimensions=1, strict_data_typing=False),
]

NpLoading1DArrayFp64: TypeAlias = Annotated[
    Union[np.ndarray[tuple[int, ...], np.dtype[np.float64]], FilePath, MultiArrayNumpyFile],
    NpArrayPydanticAnnotation.factory(data_type=np.float64, dimensions=1, strict_data_typing=False),
]

NpLoading1DArrayFp32: TypeAlias = Annotated[
    Union[np.ndarray[tuple[int, ...], np.dtype[np.float32]], FilePath, MultiArrayNumpyFile],
    NpArrayPydanticAnnotation.factory(data_type=np.float32, dimensions=1, strict_data_typing=False),
]

NpLoading1DArrayFp16: TypeAlias = Annotated[
    Union[np.ndarray[tuple[int, ...], np.dtype[np.float16]], FilePath, MultiArrayNumpyFile],
    NpArrayPydanticAnnotation.factory(data_type=np.float16, dimensions=1, strict_data_typing=False),
]

NpLoading1DArrayComplexLongDouble: TypeAlias = Annotated[
    Union[np.ndarray[tuple[int, ...], np.dtype[np.clongdouble]], FilePath, MultiArrayNumpyFile],
    NpArrayPydanticAnnotation.factory(data_type=np.clongdouble, dimensions=1, strict_data_typing=False),
]

NpLoading1DArrayComplex128: TypeAlias = Annotated[
    Union[np.ndarray[tuple[int, ...], np.dtype[np.complex128]], FilePath, MultiArrayNumpyFile],
    NpArrayPydanticAnnotation.factory(data_type=np.complex128, dimensions=1, strict_data_typing=False),
]

NpLoading1DArrayComplex64: TypeAlias = Annotated[
    Union[np.ndarray[tuple[int, ...], np.dtype[np.complex64]], FilePath, MultiArrayNumpyFile],
    NpArrayPydanticAnnotation.factory(data_type=np.complex64, dimensions=1, strict_data_typing=False),
]

NpLoading1DArrayBool: TypeAlias = Annotated[
    Union[np.ndarray[tuple[int, ...], np.dtype[np.bool_]], FilePath, MultiArrayNumpyFile],
    NpArrayPydanticAnnotation.factory(data_type=np.bool_, dimensions=1, strict_data_typing=False),
]

NpLoading1DArrayDatetime64: TypeAlias = Annotated[
    Union[np.ndarray[tuple[int, ...], np.dtype[np.datetime64]], FilePath, MultiArrayNumpyFile],
    NpArrayPydanticAnnotation.factory(data_type=np.datetime64, dimensions=1, strict_data_typing=False),
]

NpLoading1DArrayTimedelta64: TypeAlias = Annotated[
    Union[np.ndarray[tuple[int, ...], np.dtype[np.timedelta64]], FilePath, MultiArrayNumpyFile],
    NpArrayPydanticAnnotation.factory(data_type=np.timedelta64, dimensions=1, strict_data_typing=False),
]

__all__ = [
    "NpLoading1DArray",
    "NpLoading1DArrayInt64",
    "NpLoading1DArrayInt32",
    "NpLoading1DArrayInt16",
    "NpLoading1DArrayInt8",
    "NpLoading1DArrayUint64",
    "NpLoading1DArrayUint32",
    "NpLoading1DArrayUint16",
    "NpLoading1DArrayUint8",
    "NpLoading1DArrayFpLongDouble",
    "NpLoading1DArrayFp64",
    "NpLoading1DArrayFp32",
    "NpLoading1DArrayFp16",
    "NpLoading1DArrayComplexLongDouble",
    "NpLoading1DArrayComplex128",
    "NpLoading1DArrayComplex64",
    "NpLoading1DArrayBool",
    "NpLoading1DArrayDatetime64",
    "NpLoading1DArrayTimedelta64",
]
