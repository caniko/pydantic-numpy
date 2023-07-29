import numpy as np

from pydantic_numpy.helper.annotation import np_array_pydantic_annotated_typing

NpStrict2DArrayInt64 = np_array_pydantic_annotated_typing(data_type=np.int64, dimensions=2, strict_data_typing=True)
NpStrict2DArrayInt32 = np_array_pydantic_annotated_typing(data_type=np.int32, dimensions=2, strict_data_typing=True)
NpStrict2DArrayInt16 = np_array_pydantic_annotated_typing(data_type=np.int16, dimensions=2, strict_data_typing=True)
NpStrict2DArrayInt8 = np_array_pydantic_annotated_typing(data_type=np.int8, dimensions=2, strict_data_typing=True)

NpStrict2DArrayUint64 = np_array_pydantic_annotated_typing(data_type=np.uint64, dimensions=2, strict_data_typing=True)
NpStrict2DArrayUint32 = np_array_pydantic_annotated_typing(data_type=np.uint32, dimensions=2, strict_data_typing=True)
NpStrict2DArrayUint16 = np_array_pydantic_annotated_typing(data_type=np.uint16, dimensions=2, strict_data_typing=True)
NpStrict2DArrayUint8 = np_array_pydantic_annotated_typing(data_type=np.uint8, dimensions=2, strict_data_typing=True)

NpStrict2DArrayFpLongDouble = np_array_pydantic_annotated_typing(
    data_type=np.longdouble, dimensions=2, strict_data_typing=True
)
NpStrict2DArrayFp64 = np_array_pydantic_annotated_typing(data_type=np.float64, dimensions=2, strict_data_typing=True)
NpStrict2DArrayFp32 = np_array_pydantic_annotated_typing(data_type=np.float32, dimensions=2, strict_data_typing=True)
NpStrict2DArrayFp16 = np_array_pydantic_annotated_typing(data_type=np.float16, dimensions=2, strict_data_typing=True)

NpStrict2DArrayComplexLongDouble = np_array_pydantic_annotated_typing(
    data_type=np.clongdouble, dimensions=2, strict_data_typing=True
)
NpStrict2DArrayComplex128 = np_array_pydantic_annotated_typing(
    data_type=np.complex128, dimensions=2, strict_data_typing=True
)
NpStrict2DArrayComplex64 = np_array_pydantic_annotated_typing(
    data_type=np.complex64, dimensions=2, strict_data_typing=True
)

NpStrict2DArrayBool = np_array_pydantic_annotated_typing(data_type=bool, dimensions=2, strict_data_typing=True)


# Non-number types
NpStrict2DArrayDatetime64 = np_array_pydantic_annotated_typing(
    data_type=np.datetime64, dimensions=2, strict_data_typing=True
)
NpStrict2DArrayTimedelta64 = np_array_pydantic_annotated_typing(
    data_type=np.timedelta64, dimensions=2, strict_data_typing=True
)


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
