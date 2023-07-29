import numpy as np

from pydantic_numpy.helper.annotation import np_array_pydantic_annotated_typing

NpStrict3DArrayInt64 = np_array_pydantic_annotated_typing(data_type=np.int64, dimensions=3, strict_data_typing=True)
NpStrict3DArrayInt32 = np_array_pydantic_annotated_typing(data_type=np.int32, dimensions=3, strict_data_typing=True)
NpStrict3DArrayInt16 = np_array_pydantic_annotated_typing(data_type=np.int16, dimensions=3, strict_data_typing=True)
NpStrict3DArrayInt8 = np_array_pydantic_annotated_typing(data_type=np.int8, dimensions=3, strict_data_typing=True)

NpStrict3DArrayUint64 = np_array_pydantic_annotated_typing(data_type=np.uint64, dimensions=3, strict_data_typing=True)
NpStrict3DArrayUint32 = np_array_pydantic_annotated_typing(data_type=np.uint32, dimensions=3, strict_data_typing=True)
NpStrict3DArrayUint16 = np_array_pydantic_annotated_typing(data_type=np.uint16, dimensions=3, strict_data_typing=True)
NpStrict3DArrayUint8 = np_array_pydantic_annotated_typing(data_type=np.uint8, dimensions=3, strict_data_typing=True)

NpStrict3DArrayFpLongDouble = np_array_pydantic_annotated_typing(
    data_type=np.longdouble, dimensions=3, strict_data_typing=True
)
NpStrict3DArrayFp64 = np_array_pydantic_annotated_typing(data_type=np.float64, dimensions=3, strict_data_typing=True)
NpStrict3DArrayFp32 = np_array_pydantic_annotated_typing(data_type=np.float32, dimensions=3, strict_data_typing=True)
NpStrict3DArrayFp16 = np_array_pydantic_annotated_typing(data_type=np.float16, dimensions=3, strict_data_typing=True)

NpStrict3DArrayComplexLongDouble = np_array_pydantic_annotated_typing(
    data_type=np.clongdouble, dimensions=3, strict_data_typing=True
)
NpStrict3DArrayComplex128 = np_array_pydantic_annotated_typing(
    data_type=np.complex128, dimensions=3, strict_data_typing=True
)
NpStrict3DArrayComplex64 = np_array_pydantic_annotated_typing(
    data_type=np.complex64, dimensions=3, strict_data_typing=True
)

NpStrict3DArrayBool = np_array_pydantic_annotated_typing(data_type=bool, dimensions=3, strict_data_typing=True)


# Non-number types
NpStrict3DArrayDatetime64 = np_array_pydantic_annotated_typing(
    data_type=np.datetime64, dimensions=3, strict_data_typing=True
)
NpStrict3DArrayTimedelta64 = np_array_pydantic_annotated_typing(
    data_type=np.timedelta64, dimensions=3, strict_data_typing=True
)

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
