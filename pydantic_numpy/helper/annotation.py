from collections.abc import Sequence
from pathlib import Path
from typing import Any, Callable, ClassVar, Iterable, Optional, Union

import numpy as np
import numpy.typing as npt
from pydantic import FilePath, GetJsonSchemaHandler, PositiveInt, validate_call
from pydantic.json_schema import JsonSchemaValue
from pydantic_core import core_schema
from typing_extensions import Annotated

from pydantic_numpy.helper.typing import NumpyDataDict, SupportedDTypes
from pydantic_numpy.helper.validation import (
    create_array_validator,
    validate_multi_array_numpy_file,
    validate_numpy_array_file,
)
from pydantic_numpy.model import MultiArrayNumpyFile


def serialize_numpy_array_to_data_dict(array_like: npt.ArrayLike) -> NumpyDataDict:
    """
    Serialize a NumPy array into a data dictionary format suitable for frontend display or processing.

    This function converts a given NumPy array into a dictionary format, which includes the data type
    and the data itself. If the array contains datetime or timedelta objects, it converts them into integer
    representations. Otherwise, the array is converted to a floating-point representation. This is particularly
    useful for preparing NumPy array data for JSON serialization or similar use cases where NumPy's native
    data types are not directly compatible.

    Note
    ----
    This function is intended for internal use within a package for handling specific serialization needs
    of NumPy arrays for frontend applications or similar use cases. It should not be used as a general-purpose
    serialization tool.

    Parameters
    ----------
    array_like: np.ndarray
                  The NumPy array to be serialized. This can be a standard numerical array or an array
                  of datetime/timedelta objects.

    Returns
    -------
    NumpyDataDict
                   A dictionary with two keys: 'data_type', a string representing the data type of the array,
                   and 'data', a list of values converted from the array. The conversion is to integer if the
                   original data type is datetime or timedelta, and to float for other data types.

    Example
    -------
    >>> my_array = np.array([1, 2, 3])
    >>> serialize_numpy_array_to_data_dict(my_array)
    {'data_type': 'int64', 'data': [1.0, 2.0, 3.0]}
    """
    array = np.array(array_like)

    if issubclass(array.dtype.type, np.timedelta64) or issubclass(array.dtype.type, np.datetime64):
        return NumpyDataDict(data_type=str(array.dtype), data=array.astype(int).tolist())

    return NumpyDataDict(data_type=str(array.dtype), data=array.astype(float).tolist())


class NpArrayPydanticAnnotation:
    dimensions: ClassVar[Optional[PositiveInt]]

    data_type: ClassVar[SupportedDTypes]
    strict_data_typing: ClassVar[bool]
    serialize_numpy_array_to_json: ClassVar[Callable[[npt.ArrayLike], Iterable]]

    @classmethod
    def factory(
        cls,
        *,
        data_type: Optional[SupportedDTypes] = None,
        dimensions: Optional[int] = None,
        strict_data_typing: bool = False,
        serialize_numpy_array_to_json: Callable[[npt.ArrayLike], Iterable] = serialize_numpy_array_to_data_dict,
    ) -> type:
        """
        Create an instance NpArrayPydanticAnnotation that is configured for a specific dimension and dtype.

        The signature of the function is data_type, dimension and not dimension, data_type to reduce amount of
        code for all the types.

        Parameters
        ----------
        data_type: SupportedDTypes
        dimensions: Optional[int]
            Number of dimensions determine the depth of the numpy array.
        strict_data_typing: bool
            If True, the dtype of the numpy array must be identical to the data_type. No conversion attempts.
        serialize_numpy_array_to_json: Callable[[npt.ArrayLike], Iterable]
            Json serialization function to use. Defaults to NumpyDataDict serializer.

        Returns
        -------
        NpArrayPydanticAnnotation
        """
        if strict_data_typing and not data_type:
            msg = "Strict data typing requires data_type (SupportedDTypes) definition"
            raise ValueError(msg)

        return type(
            (
                f"Np{'Strict' if strict_data_typing else ''}{dimensions or 'N'}DArray"
                f"{data_type.__name__.capitalize() if data_type else ''}PydanticAnnotation"
            ),
            (cls,),
            {
                "dimensions": dimensions,
                "data_type": data_type,
                "strict_data_typing": strict_data_typing,
                "serialize_numpy_array_to_json": serialize_numpy_array_to_json,
            },
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
                cls.serialize_numpy_array_to_json, when_used="json-unless-none"
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
    data_type: Optional[SupportedDTypes] = None,
    dimensions: Optional[int] = None,
    strict_data_typing: bool = False,
    serialize_numpy_array_to_json: Callable[[npt.ArrayLike], Iterable] = serialize_numpy_array_to_data_dict,
):
    """
    Generates typing and pydantic annotation of a np.ndarray parametrized with given constraints

    Parameters
    ----------
    data_type: SupportedDTypes
    dimensions: Optional[int]
        Number of dimensions determine the depth of the numpy array.
    strict_data_typing: bool
        If True, the dtype of the numpy array must be identical to the data_type. No conversion attempts.
    serialize_numpy_array_to_json: Callable[[npt.ArrayLike], Iterable]
        Json serialization function to use. Defaults to NumpyDataDict serializer.

    Returns
    -------
    type-hint for np.ndarray with Pydantic support
    """
    return Annotated[
        Union[
            FilePath,
            MultiArrayNumpyFile,
            np.ndarray[  # type: ignore[misc]
                _int_to_dim_type[dimensions] if dimensions else Any,  # pyright: ignore
                np.dtype[data_type] if _data_type_resolver(data_type) else data_type,  # type: ignore[valid-type]
            ],
        ],
        NpArrayPydanticAnnotation.factory(
            data_type=data_type,
            dimensions=dimensions,
            strict_data_typing=strict_data_typing,
            serialize_numpy_array_to_json=serialize_numpy_array_to_json,
        ),
    ]


def _data_type_resolver(data_type: Optional[SupportedDTypes]) -> bool:
    return data_type is not None and issubclass(data_type, np.generic)


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
