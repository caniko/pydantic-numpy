import numpy as np

from pydantic_numpy.helper.annotation import np_array_pydantic_annotated_typing

Np1DArray = np_array_pydantic_annotated_typing(data_type=None, dimensions=1, strict_data_typing=False)

Np1DArrayInt64 = np_array_pydantic_annotated_typing(data_type=np.int64, dimensions=1)
Np1DArrayInt32 = np_array_pydantic_annotated_typing(data_type=np.int32, dimensions=1)
Np1DArrayInt16 = np_array_pydantic_annotated_typing(data_type=np.int16, dimensions=1)
Np1DArrayInt8 = np_array_pydantic_annotated_typing(data_type=np.int8, dimensions=1)

Np1DArrayUint64 = np_array_pydantic_annotated_typing(data_type=np.uint64, dimensions=1)
Np1DArrayUint32 = np_array_pydantic_annotated_typing(data_type=np.uint32, dimensions=1)
Np1DArrayUint16 = np_array_pydantic_annotated_typing(data_type=np.uint16, dimensions=1)
Np1DArrayUint8 = np_array_pydantic_annotated_typing(data_type=np.uint8, dimensions=1)

Np1DArrayFpLongDouble = np_array_pydantic_annotated_typing(data_type=np.longdouble, dimensions=1)
Np1DArrayFp64 = np_array_pydantic_annotated_typing(data_type=np.float64, dimensions=1)
Np1DArrayFp32 = np_array_pydantic_annotated_typing(data_type=np.float32, dimensions=1)
Np1DArrayFp16 = np_array_pydantic_annotated_typing(data_type=np.float16, dimensions=1)

Np1DArrayComplexLongDouble = np_array_pydantic_annotated_typing(data_type=np.clongdouble, dimensions=1)
Np1DArrayComplex128 = np_array_pydantic_annotated_typing(data_type=np.complex128, dimensions=1)
Np1DArrayComplex64 = np_array_pydantic_annotated_typing(data_type=np.complex64, dimensions=1)

Np1DArrayBool = np_array_pydantic_annotated_typing(data_type=bool, dimensions=1)


# Non-number types
Np1DArrayDatetime64 = np_array_pydantic_annotated_typing(data_type=np.datetime64, dimensions=1)
Np1DArrayTimedelta64 = np_array_pydantic_annotated_typing(data_type=np.timedelta64, dimensions=1)


__all__ = [
    "Np1DArray",
    "Np1DArrayInt64",
    "Np1DArrayInt32",
    "Np1DArrayInt16",
    "Np1DArrayInt8",
    "Np1DArrayUint64",
    "Np1DArrayUint32",
    "Np1DArrayUint16",
    "Np1DArrayUint8",
    "Np1DArrayFpLongDouble",
    "Np1DArrayFp64",
    "Np1DArrayFp32",
    "Np1DArrayFp16",
    "Np1DArrayComplexLongDouble",
    "Np1DArrayComplex128",
    "Np1DArrayComplex64",
    "Np1DArrayBool",
    "Np1DArrayDatetime64",
    "Np1DArrayTimedelta64",
]
