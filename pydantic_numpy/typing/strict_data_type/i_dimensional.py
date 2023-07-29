import numpy as np

from pydantic_numpy.helper.annotation import np_array_pydantic_annotated_typing

NpStrict1DArrayInt64 = np_array_pydantic_annotated_typing(data_type=np.int64, dimensions=1, strict_data_typing=True)
NpStrict1DArrayInt32 = np_array_pydantic_annotated_typing(data_type=np.int32, dimensions=1, strict_data_typing=True)
NpStrict1DArrayInt16 = np_array_pydantic_annotated_typing(data_type=np.int16, dimensions=1, strict_data_typing=True)
NpStrict1DArrayInt8 = np_array_pydantic_annotated_typing(data_type=np.int8, dimensions=1, strict_data_typing=True)

NpStrict1DArrayUint64 = np_array_pydantic_annotated_typing(data_type=np.uint64, dimensions=1, strict_data_typing=True)
NpStrict1DArrayUint32 = np_array_pydantic_annotated_typing(data_type=np.uint32, dimensions=1, strict_data_typing=True)
NpStrict1DArrayUint16 = np_array_pydantic_annotated_typing(data_type=np.uint16, dimensions=1, strict_data_typing=True)
NpStrict1DArrayUint8 = np_array_pydantic_annotated_typing(data_type=np.uint8, dimensions=1, strict_data_typing=True)

NpStrict1DArrayFpLongDouble = np_array_pydantic_annotated_typing(
    data_type=np.longdouble, dimensions=1, strict_data_typing=True
)
NpStrict1DArrayFp64 = np_array_pydantic_annotated_typing(data_type=np.float64, dimensions=1, strict_data_typing=True)
NpStrict1DArrayFp32 = np_array_pydantic_annotated_typing(data_type=np.float32, dimensions=1, strict_data_typing=True)
NpStrict1DArrayFp16 = np_array_pydantic_annotated_typing(data_type=np.float16, dimensions=1, strict_data_typing=True)

NpStrict1DArrayComplexLongDouble = np_array_pydantic_annotated_typing(
    data_type=np.clongdouble, dimensions=1, strict_data_typing=True
)
NpStrict1DArrayComplex128 = np_array_pydantic_annotated_typing(
    data_type=np.complex128, dimensions=1, strict_data_typing=True
)
NpStrict1DArrayComplex64 = np_array_pydantic_annotated_typing(
    data_type=np.complex64, dimensions=1, strict_data_typing=True
)

NpStrict1DArrayBool = np_array_pydantic_annotated_typing(data_type=bool, dimensions=1, strict_data_typing=True)


# Non-number types
NpStrict1DArrayDatetime64 = np_array_pydantic_annotated_typing(
    data_type=np.datetime64, dimensions=1, strict_data_typing=True
)
NpStrict1DArrayTimedelta64 = np_array_pydantic_annotated_typing(
    data_type=np.timedelta64, dimensions=1, strict_data_typing=True
)


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
