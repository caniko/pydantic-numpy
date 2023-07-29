import numpy as np

from pydantic_numpy.helper.annotation import np_array_pydantic_annotated_typing

NpNDArray = np_array_pydantic_annotated_typing(data_type=None, dimensions=None, strict_data_typing=False)

NpNDArrayInt64 = np_array_pydantic_annotated_typing(data_type=np.int64)
NpNDArrayInt32 = np_array_pydantic_annotated_typing(data_type=np.int32)
NpNDArrayInt16 = np_array_pydantic_annotated_typing(data_type=np.int16)
NpNDArrayInt8 = np_array_pydantic_annotated_typing(data_type=np.int8)

NpNDArrayUint64 = np_array_pydantic_annotated_typing(data_type=np.uint64)
NpNDArrayUint32 = np_array_pydantic_annotated_typing(data_type=np.uint32)
NpNDArrayUint16 = np_array_pydantic_annotated_typing(data_type=np.uint16)
NpNDArrayUint8 = np_array_pydantic_annotated_typing(data_type=np.uint8)

NpNDArrayFpLongDouble = np_array_pydantic_annotated_typing(data_type=np.longdouble)
NpNDArrayFp64 = np_array_pydantic_annotated_typing(data_type=np.float64)
NpNDArrayFp32 = np_array_pydantic_annotated_typing(data_type=np.float32)
NpNDArrayFp16 = np_array_pydantic_annotated_typing(data_type=np.float16)

NpNDArrayComplexLongDouble = np_array_pydantic_annotated_typing(data_type=np.clongdouble)
NpNDArrayComplex128 = np_array_pydantic_annotated_typing(data_type=np.complex128)
NpNDArrayComplex64 = np_array_pydantic_annotated_typing(data_type=np.complex64)

NpNDArrayBool = np_array_pydantic_annotated_typing(data_type=bool)


# Non-number types
NpNDArrayDatetime64 = np_array_pydantic_annotated_typing(data_type=np.datetime64)
NpNDArrayTimedelta64 = np_array_pydantic_annotated_typing(data_type=np.timedelta64)


__all__ = [
    "NpNDArray",
    "NpNDArrayInt64",
    "NpNDArrayInt32",
    "NpNDArrayInt16",
    "NpNDArrayInt8",
    "NpNDArrayUint64",
    "NpNDArrayUint32",
    "NpNDArrayUint16",
    "NpNDArrayUint8",
    "NpNDArrayFpLongDouble",
    "NpNDArrayFp64",
    "NpNDArrayFp32",
    "NpNDArrayFp16",
    "NpNDArrayComplexLongDouble",
    "NpNDArrayComplex128",
    "NpNDArrayComplex64",
    "NpNDArrayBool",
    "NpNDArrayDatetime64",
    "NpNDArrayTimedelta64",
]
