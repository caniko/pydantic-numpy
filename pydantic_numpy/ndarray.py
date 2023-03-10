from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, Any, Generic, Mapping, Optional, TypeVar

import numpy as np
from pydantic import BaseModel, FilePath, validator
from pydantic.fields import ModelField

if TYPE_CHECKING:
    from pydantic.typing import CallableGenerator


class NPFileDesc(BaseModel):
    path: FilePath = ...
    key: Optional[str]

    @validator("path", allow_reuse=True)
    def check_absolute(cls, value: Path) -> Path:
        return value.resolve().absolute()


T = TypeVar("T")
ScalarType = TypeVar("ScalarType", bound=np.generic, covariant=True)


class BaseNDArrayType(Generic[T, ScalarType], np.ndarray[T, np.dtype[ScalarType]], ABC):
    @classmethod
    def __get_validators__(cls) -> "CallableGenerator":
        yield cls.validate

    @classmethod
    def __modify_schema__(cls, field_schema: dict[str, Any], field: Optional[ModelField]) -> None:
        if field and field.sub_fields:
            type_with_potential_subtype = f"np.ndarray[{field.sub_fields[0]}]"
        else:
            type_with_potential_subtype = "np.ndarray"
        field_schema.update({"type": type_with_potential_subtype})

    @classmethod
    @abstractmethod
    def validate(cls, val: Any, field: ModelField) -> np.ndarray[T, np.dtype[ScalarType]]:
        ...

    @staticmethod
    def field_validation(val: Any, field: ModelField) -> np.ndarray[T, np.dtype[ScalarType]]:
        if isinstance(val, Mapping):
            val = NPFileDesc(**val)

        if isinstance(val, NPFileDesc):
            val: NPFileDesc

            if val.path.suffix.lower() not in [".npz", ".npy"]:
                raise ValueError("Expected npz or npy file.")

            if not val.path.is_file():
                raise ValueError(f"Path does not exist {val.path}")

            try:
                content = np.load(str(val.path))
            except FileNotFoundError:
                raise ValueError(f"Failed to load numpy data from file {val.path}")

            if val.path.suffix.lower() == ".npz":
                key = val.key or content.files[0]
                try:
                    data = content[key]
                except KeyError:
                    raise ValueError(f"Key {key} not found in npz.")
            else:
                data = content
        else:
            data = val

        return np.asarray(data, dtype=field.sub_fields[1].type_) if field.sub_fields else np.asarray(data)


class PydanticNDArray(Generic[T, ScalarType], BaseNDArrayType[T, ScalarType]):
    @classmethod
    def validate(cls, val: Any, field: ModelField) -> np.ndarray[T, np.dtype[ScalarType]]:
        return cls.field_validation(val, field)


class PydanticPotentialNDArray(Generic[T, ScalarType], BaseNDArrayType[T, ScalarType]):
    """Like NDArray, but validation errors result in None."""

    @classmethod
    def validate(cls, val: Any, field: ModelField) -> Optional[np.ndarray[T, np.dtype[ScalarType]]]:
        try:
            return cls.field_validation(val, field)
        except ValueError:
            return None


NDArray = PydanticNDArray
PotentialNDArray = PydanticPotentialNDArray
