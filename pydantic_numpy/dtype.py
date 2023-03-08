from __future__ import annotations

from typing import TYPE_CHECKING, Any, TypeVar

import numpy as np
from pydantic import ValidationError
from pydantic.fields import ModelField

from pydantic_numpy.ndarray import NDArray

if TYPE_CHECKING:
    from pydantic.typing import CallableGenerator


class _BaseDType:
    @classmethod
    def __modify_schema__(cls, field_schema: dict[str, Any]) -> None:
        field_schema.update({"type": cls.__name__})

    @classmethod
    def __get_validators__(cls) -> CallableGenerator:
        yield cls.validate

    @classmethod
    def validate(cls, val: Any, field: ModelField) -> "PdNpDtype":
        if field.sub_fields:
            msg = f"{cls.__name__} has no subfields"
            raise ValidationError(msg)
        if not isinstance(val, cls):
            return cls(val)
        return val


PdNpDtype = TypeVar("PdNpDtype", bound=_BaseDType)


class longdouble(np.longdouble, _BaseDType):
    pass


float128 = longdouble


class double(np.double, _BaseDType):
    pass


float64 = double


class single(np.single, _BaseDType):
    pass


float32 = single


class half(np.half, _BaseDType):
    pass


float16 = half


class int_(np.int_, _BaseDType):
    pass


int64 = int_


class intc(np.intc, _BaseDType):
    pass


int32 = intc


class short(np.short, _BaseDType):
    pass


int16 = short


class byte(np.byte, _BaseDType):
    pass


int8 = byte


class uint(np.uint, _BaseDType):
    pass


uint64 = uint


class uintc(np.uintc, _BaseDType):
    pass


uint32 = uintc


class ushort(np.ushort, _BaseDType):
    pass


uint16 = ushort


class ubyte(np.ubyte, _BaseDType):
    pass


uint8 = ubyte


class clongdouble(np.clongdouble, _BaseDType):
    pass


complex256 = clongdouble


class cdouble(np.cdouble, _BaseDType):
    pass


complex128 = cdouble


class csingle(np.csingle, _BaseDType):
    pass


complex64 = csingle


class npbool(np.bool_, _BaseDType):
    pass


# NDArray typings

NDArrayInt64 = NDArray[int, int64]
NDArrayInt32 = NDArray[int, int32]
NDArrayInt16 = NDArray[int, int16]
NDArrayInt8 = NDArray[int, int8]

NDArrayUint64 = NDArray[int, uint64]
NDArrayUint32 = NDArray[int, uint32]
NDArrayUint16 = NDArray[int, uint16]
NDArrayUint8 = NDArray[int, uint8]

NDArrayFp128 = NDArray[float, float128]
NDArrayFp64 = NDArray[float, float64]
NDArrayFp32 = NDArray[float, float32]
NDArrayFp16 = NDArray[float, float16]

NDArrayComplex256 = NDArray[float, complex256]
NDArrayComplex128 = NDArray[float, complex128]
NDArrayComplex64 = NDArray[float, complex64]

NDArrayBool = NDArray[bool, npbool]
