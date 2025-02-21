from typing import Annotated, Any, TypeAlias, Union

import numpy as np
from pydantic import FilePath

from pydantic_numpy.helper.annotation import NpArrayPydanticAnnotation
from pydantic_numpy.model import MultiArrayNumpyFile

NpLoading3DArray: TypeAlias = Annotated[
    Union[np.ndarray[tuple[int, ...], np.dtype[Any]], FilePath, MultiArrayNumpyFile],
    NpArrayPydanticAnnotation.factory(data_type=None, dimensions=3, strict_data_typing=False),
]

NpLoading3DArrayInt64: TypeAlias = Annotated[
    Union[np.ndarray[tuple[int, ...], np.dtype[np.int64]], FilePath, MultiArrayNumpyFile],
    NpArrayPydanticAnnotation.factory(data_type=np.int64, dimensions=3, strict_data_typing=False),
]

NpLoading3DArrayInt32: TypeAlias = Annotated[
    Union[np.ndarray[tuple[int, ...], np.dtype[np.int32]], FilePath, MultiArrayNumpyFile],
    NpArrayPydanticAnnotation.factory(data_type=np.int32, dimensions=3, strict_data_typing=False),
]

NpLoading3DArrayInt16: TypeAlias = Annotated[
    Union[np.ndarray[tuple[int, ...], np.dtype[np.int16]], FilePath, MultiArrayNumpyFile],
    NpArrayPydanticAnnotation.factory(data_type=np.int16, dimensions=3, strict_data_typing=False),
]

NpLoading3DArrayInt8: TypeAlias = Annotated[
    Union[np.ndarray[tuple[int, ...], np.dtype[np.int8]], FilePath, MultiArrayNumpyFile],
    NpArrayPydanticAnnotation.factory(data_type=np.int8, dimensions=3, strict_data_typing=False),
]

NpLoading3DArrayUint64: TypeAlias = Annotated[
    Union[np.ndarray[tuple[int, ...], np.dtype[np.uint64]], FilePath, MultiArrayNumpyFile],
    NpArrayPydanticAnnotation.factory(data_type=np.uint64, dimensions=3, strict_data_typing=False),
]

NpLoading3DArrayUint32: TypeAlias = Annotated[
    Union[np.ndarray[tuple[int, ...], np.dtype[np.uint32]], FilePath, MultiArrayNumpyFile],
    NpArrayPydanticAnnotation.factory(data_type=np.uint32, dimensions=3, strict_data_typing=False),
]

NpLoading3DArrayUint16: TypeAlias = Annotated[
    Union[np.ndarray[tuple[int, ...], np.dtype[np.uint16]], FilePath, MultiArrayNumpyFile],
    NpArrayPydanticAnnotation.factory(data_type=np.uint16, dimensions=3, strict_data_typing=False),
]

NpLoading3DArrayUint8: TypeAlias = Annotated[
    Union[np.ndarray[tuple[int, ...], np.dtype[np.uint8]], FilePath, MultiArrayNumpyFile],
    NpArrayPydanticAnnotation.factory(data_type=np.uint8, dimensions=3, strict_data_typing=False),
]

NpLoading3DArrayFpLongDouble: TypeAlias = Annotated[
    Union[np.ndarray[tuple[int, ...], np.dtype[np.longdouble]], FilePath, MultiArrayNumpyFile],
    NpArrayPydanticAnnotation.factory(data_type=np.longdouble, dimensions=3, strict_data_typing=False),
]

NpLoading3DArrayFp64: TypeAlias = Annotated[
    Union[np.ndarray[tuple[int, ...], np.dtype[np.float64]], FilePath, MultiArrayNumpyFile],
    NpArrayPydanticAnnotation.factory(data_type=np.float64, dimensions=3, strict_data_typing=False),
]

NpLoading3DArrayFp32: TypeAlias = Annotated[
    Union[np.ndarray[tuple[int, ...], np.dtype[np.float32]], FilePath, MultiArrayNumpyFile],
    NpArrayPydanticAnnotation.factory(data_type=np.float32, dimensions=3, strict_data_typing=False),
]

NpLoading3DArrayFp16: TypeAlias = Annotated[
    Union[np.ndarray[tuple[int, ...], np.dtype[np.float16]], FilePath, MultiArrayNumpyFile],
    NpArrayPydanticAnnotation.factory(data_type=np.float16, dimensions=3, strict_data_typing=False),
]

NpLoading3DArrayComplexLongDouble: TypeAlias = Annotated[
    Union[np.ndarray[tuple[int, ...], np.dtype[np.clongdouble]], FilePath, MultiArrayNumpyFile],
    NpArrayPydanticAnnotation.factory(data_type=np.clongdouble, dimensions=3, strict_data_typing=False),
]

NpLoading3DArrayComplex128: TypeAlias = Annotated[
    Union[np.ndarray[tuple[int, ...], np.dtype[np.complex128]], FilePath, MultiArrayNumpyFile],
    NpArrayPydanticAnnotation.factory(data_type=np.complex128, dimensions=3, strict_data_typing=False),
]

NpLoading3DArrayComplex64: TypeAlias = Annotated[
    Union[np.ndarray[tuple[int, ...], np.dtype[np.complex64]], FilePath, MultiArrayNumpyFile],
    NpArrayPydanticAnnotation.factory(data_type=np.complex64, dimensions=3, strict_data_typing=False),
]

NpLoading3DArrayBool: TypeAlias = Annotated[
    Union[np.ndarray[tuple[int, ...], np.dtype[np.bool_]], FilePath, MultiArrayNumpyFile],
    NpArrayPydanticAnnotation.factory(data_type=np.bool_, dimensions=3, strict_data_typing=False),
]

NpLoading3DArrayDatetime64: TypeAlias = Annotated[
    Union[np.ndarray[tuple[int, ...], np.dtype[np.datetime64]], FilePath, MultiArrayNumpyFile],
    NpArrayPydanticAnnotation.factory(data_type=np.datetime64, dimensions=3, strict_data_typing=False),
]

NpLoading3DArrayTimedelta64: TypeAlias = Annotated[
    Union[np.ndarray[tuple[int, ...], np.dtype[np.timedelta64]], FilePath, MultiArrayNumpyFile],
    NpArrayPydanticAnnotation.factory(data_type=np.timedelta64, dimensions=3, strict_data_typing=False),
]

__all__ = [
    "NpLoading3DArray",
    "NpLoading3DArrayInt64",
    "NpLoading3DArrayInt32",
    "NpLoading3DArrayInt16",
    "NpLoading3DArrayInt8",
    "NpLoading3DArrayUint64",
    "NpLoading3DArrayUint32",
    "NpLoading3DArrayUint16",
    "NpLoading3DArrayUint8",
    "NpLoading3DArrayFpLongDouble",
    "NpLoading3DArrayFp64",
    "NpLoading3DArrayFp32",
    "NpLoading3DArrayFp16",
    "NpLoading3DArrayComplexLongDouble",
    "NpLoading3DArrayComplex128",
    "NpLoading3DArrayComplex64",
    "NpLoading3DArrayBool",
    "NpLoading3DArrayDatetime64",
    "NpLoading3DArrayTimedelta64",
]
