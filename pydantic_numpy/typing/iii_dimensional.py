import numpy as np

from pydantic_numpy.helper.annotation import np_array_pydantic_annotated_typing

Np3DArray = np_array_pydantic_annotated_typing(data_type=None, dimensions=3, strict_data_typing=False)

Np3DArrayInt64 = np_array_pydantic_annotated_typing(data_type=np.int64, dimensions=3)
Np3DArrayInt32 = np_array_pydantic_annotated_typing(data_type=np.int32, dimensions=3)
Np3DArrayInt16 = np_array_pydantic_annotated_typing(data_type=np.int16, dimensions=3)
Np3DArrayInt8 = np_array_pydantic_annotated_typing(data_type=np.int8, dimensions=3)

Np3DArrayUint64 = np_array_pydantic_annotated_typing(data_type=np.uint64, dimensions=3)
Np3DArrayUint32 = np_array_pydantic_annotated_typing(data_type=np.uint32, dimensions=3)
Np3DArrayUint16 = np_array_pydantic_annotated_typing(data_type=np.uint16, dimensions=3)
Np3DArrayUint8 = np_array_pydantic_annotated_typing(data_type=np.uint8, dimensions=3)

Np3DArrayFpLongDouble = np_array_pydantic_annotated_typing(data_type=np.longdouble, dimensions=3)
Np3DArrayFp64 = np_array_pydantic_annotated_typing(data_type=np.float64, dimensions=3)
Np3DArrayFp32 = np_array_pydantic_annotated_typing(data_type=np.float32, dimensions=3)
Np3DArrayFp16 = np_array_pydantic_annotated_typing(data_type=np.float16, dimensions=3)

Np3DArrayComplexLongDouble = np_array_pydantic_annotated_typing(data_type=np.clongdouble, dimensions=3)
Np3DArrayComplex128 = np_array_pydantic_annotated_typing(data_type=np.complex128, dimensions=3)
Np3DArrayComplex64 = np_array_pydantic_annotated_typing(data_type=np.complex64, dimensions=3)

Np3DArrayBool = np_array_pydantic_annotated_typing(data_type=bool, dimensions=3)


# Non-number types
Np3DArrayDatetime64 = np_array_pydantic_annotated_typing(data_type=np.datetime64, dimensions=3)
Np3DArrayTimedelta64 = np_array_pydantic_annotated_typing(data_type=np.timedelta64, dimensions=3)

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
