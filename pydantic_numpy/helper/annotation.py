from collections.abc import Sequence
from pathlib import Path
from typing import Any, Callable, ClassVar, Optional, Union

import numpy as np
from numpy.typing import DTypeLike
from pydantic import FilePath, GetJsonSchemaHandler, PositiveInt, validate_call
from pydantic.json_schema import JsonSchemaValue
from pydantic_core import core_schema
from typing_extensions import Annotated, TypedDict

from pydantic_numpy.helper.validation import (
    create_array_validator,
    validate_multi_array_numpy_file,
    validate_numpy_array_file,
)
from pydantic_numpy.model import MultiArrayNumpyFile


class NumpyDataDict(TypedDict):
    data_type: str
    data: list


class NpArrayPydanticAnnotation:
    dimensions: ClassVar[Optional[PositiveInt]]

    data_type: ClassVar[DTypeLike]
    strict_data_typing: ClassVar[bool]

    @classmethod
    def factory(
        cls, *, data_type: DTypeLike, dimensions: Optional[int] = None, strict_data_typing: bool = False
    ) -> type:
        """
        Create an instance NpArrayPydanticAnnotation that is configured for a specific dimension and dtype.

        The signature of the function is data_type, dimension and not dimension, data_type to reduce amount of
        code for all the types.

        Parameters
        ----------
        data_type: DTypeLike
        dimensions: Optional[int]
            Number of dimensions determine the depth of the numpy array.
        strict_data_typing: bool
            If True, the dtype of the numpy array must be identical to the data_type. No conversion attempts.

        Returns
        -------
        NpArrayPydanticAnnotation
        """
        if strict_data_typing and not data_type:
            msg = "Strict data typing requires data_type (DTypeLike) definition"
            raise ValueError(msg)

        return type(
            (
                f"Np{'Strict' if strict_data_typing else ''}{dimensions or 'N'}DArray"
                f"{data_type.__name__.capitalize() if data_type else ''}PydanticAnnotation"
            ),
            (cls,),
            {"dimensions": dimensions, "data_type": data_type, "strict_data_typing": strict_data_typing},
        )

    @classmethod
    def __get_pydantic_core_schema__(
        cls,
        _source_type: Any,
        _handler: Callable[[Any], core_schema.CoreSchema],
    ) -> core_schema.CoreSchema:
        np_array_validator = create_array_validator(cls.dimensions, cls.data_type, cls.strict_data_typing)
        np_array_schema = core_schema.no_info_plain_validator_function(np_array_validator)

        return core_schema.json_or_python_schema(
            python_schema=core_schema.chain_schema([_common_numpy_array_validator, np_array_schema]),
            json_schema=np_array_schema,
            serialization=core_schema.plain_serializer_function_ser_schema(
                _serialize_numpy_array_to_data_dict, when_used="json"
            ),
        )

    @classmethod
    def __get_pydantic_json_schema__(
        cls, _core_schema: core_schema.CoreSchema, _handler: GetJsonSchemaHandler
    ) -> JsonSchemaValue:
        return dict(
            type=(
                f"np.ndarray[{_int_to_dim_type[cls.dimensions] if cls.dimensions else 'Any'}, "
                f"np.dtype[{cls.data_type.__name__ if _data_type_resolver(cls.data_type) else 'Any'}]"
            ),
            strict_data_typing=cls.strict_data_typing,
        )


def np_array_pydantic_annotated_typing(
    data_type: DTypeLike = None, dimensions: Optional[int] = None, strict_data_typing: bool = False
):
    """
    Generates typing and pydantic annotation of a np.ndarray parametrized with given constraints

    Parameters
    ----------
    data_type: DTypeLike
    dimensions: Optional[int]
        Number of dimensions determine the depth of the numpy array.
    strict_data_typing: bool
        If True, the dtype of the numpy array must be identical to the data_type. No conversion attempts.

    Returns
    -------
    type-hint for np.ndarray with Pydantic support
    """
    return Annotated[
        Union[
            FilePath,
            MultiArrayNumpyFile,
            np.ndarray[  # type: ignore[misc]
                _int_to_dim_type[dimensions] if dimensions else Any,
                np.dtype[data_type] if _data_type_resolver(data_type) else data_type,
            ],
        ],
        NpArrayPydanticAnnotation.factory(
            data_type=data_type, dimensions=dimensions, strict_data_typing=strict_data_typing
        ),
    ]


def _data_type_resolver(data_type: DTypeLike):
    return data_type is not None and issubclass(data_type, np.generic)


def _serialize_numpy_array_to_data_dict(array: np.ndarray) -> NumpyDataDict:
    if issubclass(array.dtype.type, np.timedelta64) or issubclass(array.dtype.type, np.datetime64):
        return dict(data_type=str(array.dtype), data=array.astype(int).tolist())

    return dict(data_type=str(array.dtype), data=array.astype(float).tolist())


@validate_call
def _deserialize_numpy_array_from_data_dict(data_dict: NumpyDataDict) -> np.ndarray:
    return np.array(data_dict["data"]).astype(data_dict["data_type"])


_int_to_dim_type = {1: tuple[int], 2: tuple[int, int], 3: tuple[int, int, int]}
_common_numpy_array_validator = core_schema.union_schema(
    [
        core_schema.chain_schema(
            [
                core_schema.is_instance_schema(Path),
                core_schema.no_info_plain_validator_function(validate_numpy_array_file),
            ]
        ),
        core_schema.chain_schema(
            [
                core_schema.is_instance_schema(MultiArrayNumpyFile),
                core_schema.no_info_plain_validator_function(validate_multi_array_numpy_file),
            ]
        ),
        core_schema.is_instance_schema(np.ndarray),
        core_schema.chain_schema(
            [
                core_schema.is_instance_schema(Sequence),
                core_schema.no_info_plain_validator_function(lambda v: np.asarray(v)),
            ]
        ),
        core_schema.chain_schema(
            [
                core_schema.is_instance_schema(dict),
                core_schema.no_info_plain_validator_function(_deserialize_numpy_array_from_data_dict),
            ]
        ),
    ]
)
