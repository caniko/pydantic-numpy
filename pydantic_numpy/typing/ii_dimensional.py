import numpy as np

from pydantic_numpy.helper.annotation import np_array_pydantic_annotated_typing

Np2DArray = np_array_pydantic_annotated_typing(data_type=None, dimensions=2, strict_data_typing=False)

Np2DArrayInt64 = np_array_pydantic_annotated_typing(data_type=np.int64, dimensions=2)
Np2DArrayInt32 = np_array_pydantic_annotated_typing(data_type=np.int32, dimensions=2)
Np2DArrayInt16 = np_array_pydantic_annotated_typing(data_type=np.int16, dimensions=2)
Np2DArrayInt8 = np_array_pydantic_annotated_typing(data_type=np.int8, dimensions=2)

Np2DArrayUint64 = np_array_pydantic_annotated_typing(data_type=np.uint64, dimensions=2)
Np2DArrayUint32 = np_array_pydantic_annotated_typing(data_type=np.uint32, dimensions=2)
Np2DArrayUint16 = np_array_pydantic_annotated_typing(data_type=np.uint16, dimensions=2)
Np2DArrayUint8 = np_array_pydantic_annotated_typing(data_type=np.uint8, dimensions=2)

Np2DArrayFpLongDouble = np_array_pydantic_annotated_typing(data_type=np.longdouble, dimensions=2)
Np2DArrayFp64 = np_array_pydantic_annotated_typing(data_type=np.float64, dimensions=2)
Np2DArrayFp32 = np_array_pydantic_annotated_typing(data_type=np.float32, dimensions=2)
Np2DArrayFp16 = np_array_pydantic_annotated_typing(data_type=np.float16, dimensions=2)

Np2DArrayComplexLongDouble = np_array_pydantic_annotated_typing(data_type=np.clongdouble, dimensions=2)
Np2DArrayComplex128 = np_array_pydantic_annotated_typing(data_type=np.complex128, dimensions=2)
Np2DArrayComplex64 = np_array_pydantic_annotated_typing(data_type=np.complex64, dimensions=2)

Np2DArrayBool = np_array_pydantic_annotated_typing(data_type=bool, dimensions=2)


# Non-number types
Np2DArrayDatetime64 = np_array_pydantic_annotated_typing(data_type=np.datetime64, dimensions=2)
Np2DArrayTimedelta64 = np_array_pydantic_annotated_typing(data_type=np.timedelta64, dimensions=2)


__all__ = [
    "Np2DArray",
    "Np2DArrayInt64",
    "Np2DArrayInt32",
    "Np2DArrayInt16",
    "Np2DArrayInt8",
    "Np2DArrayUint64",
    "Np2DArrayUint32",
    "Np2DArrayUint16",
    "Np2DArrayUint8",
    "Np2DArrayFpLongDouble",
    "Np2DArrayFp64",
    "Np2DArrayFp32",
    "Np2DArrayFp16",
    "Np2DArrayComplexLongDouble",
    "Np2DArrayComplex128",
    "Np2DArrayComplex64",
    "Np2DArrayBool",
    "Np2DArrayDatetime64",
    "Np2DArrayTimedelta64",
]
