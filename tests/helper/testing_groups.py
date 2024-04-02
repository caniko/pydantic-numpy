import os

import numpy as np

from pydantic_numpy.typing import *

supported_data_types = (
    # (np.array([0]), np.int64),    # Windows confuses int64 with int32
    (np.array([0], dtype=np.int32), np.int32),
    (np.array([0], dtype=np.int16), np.int16),
    (np.array([0], dtype=np.int8), np.int8),
    (np.array([0], dtype=np.uint64), np.uint64),
    (np.array([0], dtype=np.uint32), np.uint32),
    (np.array([0], dtype=np.uint16), np.uint16),
    (np.array([0], dtype=np.uint8), np.uint8),
    (np.array([0.0]), np.float64),
    (np.array([0.0], dtype=np.float32), np.float32),
    (np.array([0.0], dtype=np.float16), np.float16),
    (np.array([0.0 + 0.0j]), np.complex128),
    (np.array([0.0 + 0.0j], dtype=np.complex64), np.complex64),
    (np.array([False]), np.bool_),
    (np.array([0], dtype=np.timedelta64), np.timedelta64),
)

data_type_1d_array_typing_dimensions = [
    (np.array([0]), np.int64, Np1DArrayInt64, 1),
    (np.array([0], dtype=np.int32), np.int32, Np1DArrayInt32, 1),
    (np.array([0], dtype=np.int16), np.int16, Np1DArrayInt16, 1),
    (np.array([0], dtype=np.int8), np.int8, Np1DArrayInt8, 1),
    (np.array([0], dtype=np.uint64), np.uint64, Np1DArrayUint64, 1),
    (np.array([0], dtype=np.uint32), np.uint32, Np1DArrayUint32, 1),
    (np.array([0], dtype=np.uint16), np.uint16, Np1DArrayUint16, 1),
    (np.array([0], dtype=np.uint8), np.uint8, Np1DArrayUint8, 1),
    (np.array([0.0]), np.float64, Np1DArrayFp64, 1),
    (np.array([0.0], dtype=np.float32), np.float32, Np1DArrayFp32, 1),
    (np.array([0.0], dtype=np.float16), np.float16, Np1DArrayFp16, 1),
    (np.array([0.0 + 0.0j]), np.complex128, Np1DArrayComplex128, 1),
    (np.array([0.0 + 0.0j], dtype=np.complex64), np.complex64, Np1DArrayComplex64, 1),
    (np.array([False]), np.bool_, Np1DArrayBool, 1),
    (np.array([0], dtype=np.timedelta64), np.timedelta64, Np1DArrayTimedelta64, 1),
]
data_type_2d_array_typing_dimensions = [
    (np.array([[0]]), np.int64, Np2DArrayInt64, 2),
    (np.array([[0]], dtype=np.int32), np.int32, Np2DArrayInt32, 2),
    (np.array([[0]], dtype=np.int16), np.int16, Np2DArrayInt16, 2),
    (np.array([[0]], dtype=np.int8), np.int8, Np2DArrayInt8, 2),
    (np.array([[0]], dtype=np.uint64), np.uint64, Np2DArrayUint64, 2),
    (np.array([[0]], dtype=np.uint32), np.uint32, Np2DArrayUint32, 2),
    (np.array([[0]], dtype=np.uint16), np.uint16, Np2DArrayUint16, 2),
    (np.array([[0]], dtype=np.uint8), np.uint8, Np2DArrayUint8, 2),
    (np.array([[0.0]]), np.float64, Np2DArrayFp64, 2),
    (np.array([[0.0]], dtype=np.float32), np.float32, Np2DArrayFp32, 2),
    (np.array([[0.0]], dtype=np.float16), np.float16, Np2DArrayFp16, 2),
    (np.array([[0.0 + 0.0j]]), np.complex128, Np2DArrayComplex128, 2),
    (np.array([[0.0 + 0.0j]], dtype=np.complex64), np.complex64, Np2DArrayComplex64, 2),
    (np.array([[False]]), np.bool_, Np2DArrayBool, 2),
    (np.array([[0]], dtype=np.timedelta64), np.timedelta64, Np2DArrayTimedelta64, 2),
]
data_type_3d_array_typing_dimensions = [
    (np.array([[[0]]]), np.int64, Np3DArrayInt64, 3),
    (np.array([[[0]]], dtype=np.int32), np.int32, Np3DArrayInt32, 3),
    (np.array([[[0]]], dtype=np.int16), np.int16, Np3DArrayInt16, 3),
    (np.array([[[0]]], dtype=np.int8), np.int8, Np3DArrayInt8, 3),
    (np.array([[[0]]], dtype=np.uint64), np.uint64, Np3DArrayUint64, 3),
    (np.array([[[0]]], dtype=np.uint32), np.uint32, Np3DArrayUint32, 3),
    (np.array([[[0]]], dtype=np.uint16), np.uint16, Np3DArrayUint16, 3),
    (np.array([[[0]]], dtype=np.uint8), np.uint8, Np3DArrayUint8, 3),
    (np.array([[[0.0]]]), np.float64, Np3DArrayFp64, 3),
    (np.array([[[0.0]]], dtype=np.float32), np.float32, Np3DArrayFp32, 3),
    (np.array([[[0.0]]], dtype=np.float16), np.float16, Np3DArrayFp16, 3),
    (np.array([[[0.0 + 0.0j]]]), np.complex128, Np3DArrayComplex128, 3),
    (np.array([[[0.0 + 0.0j]]], dtype=np.complex64), np.complex64, Np3DArrayComplex64, 3),
    (np.array([[[False]]]), np.bool_, Np3DArrayBool, 3),
    (np.array([[[0]]], dtype=np.timedelta64), np.timedelta64, Np3DArrayTimedelta64, 3),
]
data_type_nd_array_typing_dimensions_without_complex = [
    (np.array([0]), np.int64, NpNDArrayInt64, None),
    (np.array([0], dtype=np.int32), np.int32, NpNDArrayInt32, None),
    (np.array([0], dtype=np.int16), np.int16, NpNDArrayInt16, None),
    (np.array([0], dtype=np.int8), np.int8, NpNDArrayInt8, None),
    (np.array([0], dtype=np.uint64), np.uint64, NpNDArrayUint64, None),
    (np.array([0], dtype=np.uint32), np.uint32, NpNDArrayUint32, None),
    (np.array([0], dtype=np.uint16), np.uint16, NpNDArrayUint16, None),
    (np.array([0], dtype=np.uint8), np.uint8, NpNDArrayUint8, None),
    (np.array([0.0]), np.float64, NpNDArrayFp64, None),
    (np.array([0.0], dtype=np.float32), np.float32, NpNDArrayFp32, None),
    (np.array([0.0], dtype=np.float16), np.float16, NpNDArrayFp16, None),
    (np.array([False]), np.bool_, NpNDArrayBool, None),
    (np.array([0], dtype=np.timedelta64), np.timedelta64, NpNDArrayTimedelta64, None),
]

data_type_nd_array_typing_dimensions = [
    *data_type_nd_array_typing_dimensions_without_complex,
    (np.array([0.0 + 0.0j]), np.complex128, NpNDArrayComplex128, None),
    (np.array([0.0 + 0.0j], dtype=np.complex64), np.complex64, NpNDArrayComplex64, None),
]

data_type_array_typing_dimensions = [
    *data_type_1d_array_typing_dimensions,
    *data_type_2d_array_typing_dimensions,
    *data_type_3d_array_typing_dimensions,
    *data_type_nd_array_typing_dimensions,
]

# Data type strict
strict_data_type_1d_array_typing_dimensions = [
    (np.array([0]), np.int64, NpStrict1DArrayInt64, 1),
    (np.array([0], dtype=np.int32), np.int32, NpStrict1DArrayInt32, 1),
    (np.array([0], dtype=np.int16), np.int16, NpStrict1DArrayInt16, 1),
    (np.array([0], dtype=np.int8), np.int8, NpStrict1DArrayInt8, 1),
    (np.array([0], dtype=np.uint64), np.uint64, NpStrict1DArrayUint64, 1),
    (np.array([0], dtype=np.uint32), np.uint32, NpStrict1DArrayUint32, 1),
    (np.array([0], dtype=np.uint16), np.uint16, NpStrict1DArrayUint16, 1),
    (np.array([0], dtype=np.uint8), np.uint8, NpStrict1DArrayUint8, 1),
    (np.array([0.0]), np.float64, NpStrict1DArrayFp64, 1),
    (np.array([0.0], dtype=np.float32), np.float32, NpStrict1DArrayFp32, 1),
    (np.array([0.0], dtype=np.float16), np.float16, NpStrict1DArrayFp16, 1),
    (np.array([0.0 + 0.0j]), np.complex128, NpStrict1DArrayComplex128, 1),
    (np.array([0.0 + 0.0j], dtype=np.complex64), np.complex64, NpStrict1DArrayComplex64, 1),
    (np.array([False]), np.bool_, NpStrict1DArrayBool, 1),
    (np.array([0], dtype=np.timedelta64), np.timedelta64, NpStrict1DArrayTimedelta64, 1),
]
strict_data_type_2d_array_typing_dimensions = [
    (np.array([[0]]), np.int64, NpStrict2DArrayInt64, 2),
    (np.array([[0]], dtype=np.int32), np.int32, NpStrict2DArrayInt32, 2),
    (np.array([[0]], dtype=np.int16), np.int16, NpStrict2DArrayInt16, 2),
    (np.array([[0]], dtype=np.int8), np.int8, NpStrict2DArrayInt8, 2),
    (np.array([[0]], dtype=np.uint64), np.uint64, NpStrict2DArrayUint64, 2),
    (np.array([[0]], dtype=np.uint32), np.uint32, NpStrict2DArrayUint32, 2),
    (np.array([[0]], dtype=np.uint16), np.uint16, NpStrict2DArrayUint16, 2),
    (np.array([[0]], dtype=np.uint8), np.uint8, NpStrict2DArrayUint8, 2),
    (np.array([[0.0]]), np.float64, NpStrict2DArrayFp64, 2),
    (np.array([[0.0]], dtype=np.float32), np.float32, NpStrict2DArrayFp32, 2),
    (np.array([[0.0]], dtype=np.float16), np.float16, NpStrict2DArrayFp16, 2),
    (np.array([[0.0 + 0.0j]]), np.complex128, NpStrict2DArrayComplex128, 2),
    (np.array([[0.0 + 0.0j]], dtype=np.complex64), np.complex64, NpStrict2DArrayComplex64, 2),
    (np.array([[False]]), np.bool_, NpStrict2DArrayBool, 2),
    (np.array([[0]], dtype=np.timedelta64), np.timedelta64, NpStrict2DArrayTimedelta64, 2),
]
strict_data_type_3d_array_typing_dimensions = [
    (np.array([[[0]]]), np.int64, NpStrict3DArrayInt64, 3),
    (np.array([[[0]]], dtype=np.int32), np.int32, NpStrict3DArrayInt32, 3),
    (np.array([[[0]]], dtype=np.int16), np.int16, NpStrict3DArrayInt16, 3),
    (np.array([[[0]]], dtype=np.int8), np.int8, NpStrict3DArrayInt8, 3),
    (np.array([[[0]]], dtype=np.uint64), np.uint64, NpStrict3DArrayUint64, 3),
    (np.array([[[0]]], dtype=np.uint32), np.uint32, NpStrict3DArrayUint32, 3),
    (np.array([[[0]]], dtype=np.uint16), np.uint16, NpStrict3DArrayUint16, 3),
    (np.array([[[0]]], dtype=np.uint8), np.uint8, NpStrict3DArrayUint8, 3),
    (np.array([[[0.0]]]), np.float64, NpStrict3DArrayFp64, 3),
    (np.array([[[0.0]]], dtype=np.float32), np.float32, NpStrict3DArrayFp32, 3),
    (np.array([[[0.0]]], dtype=np.float16), np.float16, NpStrict3DArrayFp16, 3),
    (np.array([[[0.0 + 0.0j]]]), np.complex128, NpStrict3DArrayComplex128, 3),
    (np.array([[[0.0 + 0.0j]]], dtype=np.complex64), np.complex64, NpStrict3DArrayComplex64, 3),
    (np.array([[[False]]]), np.bool_, NpStrict3DArrayBool, 3),
    (np.array([[[0]]], dtype=np.timedelta64), np.timedelta64, NpStrict3DArrayTimedelta64, 3),
]
strict_data_type_nd_array_typing_dimensions = [
    (np.array([0]), np.int64, NpStrictNDArrayInt64, None),
    (np.array([0], dtype=np.int32), np.int32, NpStrictNDArrayInt32, None),
    (np.array([0], dtype=np.int16), np.int16, NpStrictNDArrayInt16, None),
    (np.array([0], dtype=np.int8), np.int8, NpStrictNDArrayInt8, None),
    (np.array([0], dtype=np.uint64), np.uint64, NpStrictNDArrayUint64, None),
    (np.array([0], dtype=np.uint32), np.uint32, NpStrictNDArrayUint32, None),
    (np.array([0], dtype=np.uint16), np.uint16, NpStrictNDArrayUint16, None),
    (np.array([0], dtype=np.uint8), np.uint8, NpStrictNDArrayUint8, None),
    (np.array([0.0]), np.float64, NpStrictNDArrayFp64, None),
    (np.array([0.0], dtype=np.float32), np.float32, NpStrictNDArrayFp32, None),
    (np.array([0.0], dtype=np.float16), np.float16, NpStrictNDArrayFp16, None),
    (np.array([0.0 + 0.0j]), np.complex128, NpStrictNDArrayComplex128, None),
    (np.array([0.0 + 0.0j], dtype=np.complex64), np.complex64, NpStrictNDArrayComplex64, None),
    (np.array([False]), np.bool_, NpStrictNDArrayBool, None),
    (np.array([0], dtype=np.timedelta64), np.timedelta64, NpStrictNDArrayTimedelta64, None),
]

strict_data_type_array_typing_dimensions = [
    *strict_data_type_1d_array_typing_dimensions,
    *strict_data_type_2d_array_typing_dimensions,
    *strict_data_type_3d_array_typing_dimensions,
    *strict_data_type_nd_array_typing_dimensions,
]

dimension_testing_group = [
    (np.array([0]), np.int64, Np1DArrayInt64, 1),
    (np.array([[0]]), np.int64, Np2DArrayInt64, 2),
    (np.array([[[0]]]), np.int64, Np3DArrayInt64, 3),
]

if os.name != "nt":

    def get_strict_data_type_nd_array_typing_dimensions_128_bit():
        return [
            (np.array([0.0], dtype=np.float128), np.float128, NpStrictNDArrayFpLongDouble, None),
            (np.array([0.0 + 0.0j], dtype=np.complex256), np.complex256, NpStrictNDArrayComplexLongDouble, None),
        ]
