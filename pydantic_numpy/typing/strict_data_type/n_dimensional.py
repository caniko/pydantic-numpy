import numpy as np

from pydantic_numpy.helper.annotation import np_array_pydantic_annotated_typing

NpStrictNDArrayInt64 = np_array_pydantic_annotated_typing(data_type=np.int64, strict_data_typing=True)
NpStrictNDArrayInt32 = np_array_pydantic_annotated_typing(data_type=np.int32, strict_data_typing=True)
NpStrictNDArrayInt16 = np_array_pydantic_annotated_typing(data_type=np.int16, strict_data_typing=True)
NpStrictNDArrayInt8 = np_array_pydantic_annotated_typing(data_type=np.int8, strict_data_typing=True)

NpStrictNDArrayUint64 = np_array_pydantic_annotated_typing(data_type=np.uint64, strict_data_typing=True)
NpStrictNDArrayUint32 = np_array_pydantic_annotated_typing(data_type=np.uint32, strict_data_typing=True)
NpStrictNDArrayUint16 = np_array_pydantic_annotated_typing(data_type=np.uint16, strict_data_typing=True)
NpStrictNDArrayUint8 = np_array_pydantic_annotated_typing(data_type=np.uint8, strict_data_typing=True)

NpStrictNDArrayFpLongDouble = np_array_pydantic_annotated_typing(data_type=np.longdouble, strict_data_typing=True)
NpStrictNDArrayFp64 = np_array_pydantic_annotated_typing(data_type=np.float64, strict_data_typing=True)
NpStrictNDArrayFp32 = np_array_pydantic_annotated_typing(data_type=np.float32, strict_data_typing=True)
NpStrictNDArrayFp16 = np_array_pydantic_annotated_typing(data_type=np.float16, strict_data_typing=True)

NpStrictNDArrayComplexLongDouble = np_array_pydantic_annotated_typing(data_type=np.clongdouble, strict_data_typing=True)
NpStrictNDArrayComplex128 = np_array_pydantic_annotated_typing(data_type=np.complex128, strict_data_typing=True)
NpStrictNDArrayComplex64 = np_array_pydantic_annotated_typing(data_type=np.complex64, strict_data_typing=True)

NpStrictNDArrayBool = np_array_pydantic_annotated_typing(data_type=bool, strict_data_typing=True)


# Non-number types
NpStrictNDArrayDatetime64 = np_array_pydantic_annotated_typing(data_type=np.datetime64, strict_data_typing=True)
NpStrictNDArrayTimedelta64 = np_array_pydantic_annotated_typing(data_type=np.timedelta64, strict_data_typing=True)


__all__ = [
    "NpStrictNDArrayInt64",
    "NpStrictNDArrayInt32",
    "NpStrictNDArrayInt16",
    "NpStrictNDArrayInt8",
    "NpStrictNDArrayUint64",
    "NpStrictNDArrayUint32",
    "NpStrictNDArrayUint16",
    "NpStrictNDArrayUint8",
    "NpStrictNDArrayFpLongDouble",
    "NpStrictNDArrayFp64",
    "NpStrictNDArrayFp32",
    "NpStrictNDArrayFp16",
    "NpStrictNDArrayComplexLongDouble",
    "NpStrictNDArrayComplex128",
    "NpStrictNDArrayComplex64",
    "NpStrictNDArrayBool",
    "NpStrictNDArrayDatetime64",
    "NpStrictNDArrayTimedelta64",
]
