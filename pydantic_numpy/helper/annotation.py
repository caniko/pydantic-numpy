from collections.abc import Sequence
from pathlib import Path
from typing import Any, Callable, ClassVar, Iterable, Optional, Union, cast

import numpy as np
import numpy.typing as npt
from pydantic import FilePath, GetJsonSchemaHandler, PositiveInt, validate_call
from pydantic.json_schema import JsonSchemaValue
from pydantic_core import core_schema
from typing_extensions import Annotated, Final

from pydantic_numpy.helper.typing import NumpyArrayTypeData, SupportedDTypes
from pydantic_numpy.helper.validation import (
    create_array_validator,
    validate_multi_array_numpy_file,
    validate_numpy_array_file,
)
from pydantic_numpy.model import MultiArrayNumpyFile


def pd_np_native_numpy_array_to_data_dict_serializer(array_like: npt.ArrayLike) -> NumpyArrayTypeData:
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
    NumpyArrayTypeData
                   A dictionary with two keys: 'data_type', a string representing the data type of the array,
                   and 'data', a list of values converted from the array. The conversion is to integer if the
                   original data type is datetime or timedelta, and to float for other data types.

    Example
    -------
    >>> my_array = np.array([1, 2, 3])
    >>> pd_np_native_numpy_array_to_data_dict_serializer(my_array)
    {'data_type': 'int64', 'data': [1.0, 2.0, 3.0]}
    """
    array = np.array(array_like)

    data = array.astype(
        int if issubclass(array.dtype.type, np.timedelta64) or issubclass(array.dtype.type, np.datetime64) else float
    ).tolist()
    cast_data = cast(list, data)

    return NumpyArrayTypeData(data_type=str(array.dtype), data=cast_data)


def pd_np_native_numpy_array_json_schema_from_type_data(
    _field_core_schema: core_schema.CoreSchema,
    _handler: GetJsonSchemaHandler,
    dimensions: Optional[PositiveInt] = None,
    data_type: Optional[SupportedDTypes] = None,
) -> JsonSchemaValue:
    """
    Generates a JSON schema for a NumPy array field within a Pydantic model.

    This function constructs a JSON schema definition compatible with Pydantic models
    that are intended to validate NumPy array inputs. It supports specifying the data type
    and dimensions of the NumPy array, which are used to construct a schema that ensures
    input data matches the expected structure and type.

    Parameters
    ----------
    _field_core_schema : core_schema.CoreSchema
        The core schema component of the Pydantic model, used for building basic schema structures.
    _handler : GetJsonSchemaHandler
        A handler function or object responsible for converting Python types to JSON schema components.
    dimensions : Optional[PositiveInt], optional
        The dimensions (shape) of the NumPy array. If specified, the schema will enforce that the
        input array matches this dimensionality. If `None`, no dimensionality constraint is applied,
        by default None.
    data_type : Optional[SupportedDTypes], optional
        The expected data type of the NumPy array elements. If specified, the schema will enforce
        that the input array's data type is compatible with this. If `None`, any data type is allowed,
        by default None.

    Returns
    -------
    JsonSchemaValue
        A dictionary representing the JSON schema for a NumPy array field within a Pydantic model.
        This schema includes details about the expected array dimensions and data type.
    """
    array_shape = _dimensions_to_shape_type[dimensions] if dimensions else "Any"

    if data_type and _data_type_resolver(data_type):
        array_data_type = data_type.__name__
        item_schema = core_schema.list_schema(
            items_schema=core_schema.any_schema(
                metadata=dict(typing=f"Must be compatible with numpy.dtype: {array_data_type}")
            )
        )
    else:
        array_data_type = "Any"
        item_schema = core_schema.list_schema(items_schema=core_schema.any_schema())

    if dimensions:
        data_schema = core_schema.list_schema(items_schema=item_schema, min_length=dimensions, max_length=dimensions)
    else:
        data_schema = item_schema

    return dict(
        title="Numpy Array",
        type=f"np.ndarray[{array_shape}, np.dtype[{array_data_type}]]",
        required=["data_type", "data"],
        properties=dict(
            data_type={"title": "dtype", "default": array_data_type, "type": "string"},
            data=data_schema,
        ),
    )


class NpArrayPydanticAnnotation:
    dimensions: ClassVar[Optional[PositiveInt]]
    data_type: ClassVar[SupportedDTypes]

    strict_data_typing: ClassVar[bool]

    serialize_numpy_array_to_json: ClassVar[Callable[[npt.ArrayLike], Iterable]]
    json_schema_from_type_data: ClassVar[
        Callable[
            [core_schema.CoreSchema, GetJsonSchemaHandler, Optional[PositiveInt], Optional[SupportedDTypes]],
            JsonSchemaValue,
        ]
    ]

    @classmethod
    def factory(
        cls,
        *,
        data_type: Optional[SupportedDTypes] = None,
        dimensions: Optional[PositiveInt] = None,
        strict_data_typing: bool = False,
        serialize_numpy_array_to_json: Callable[
            [npt.ArrayLike], Iterable
        ] = pd_np_native_numpy_array_to_data_dict_serializer,
        json_schema_from_type_data: Callable[
            [core_schema.CoreSchema, GetJsonSchemaHandler, Optional[PositiveInt], Optional[SupportedDTypes]],
            JsonSchemaValue,
        ] = pd_np_native_numpy_array_json_schema_from_type_data,
    ) -> type:
        """
        Create an instance NpArrayPydanticAnnotation that is configured for a specific dimension and dtype.

        The signature of the function is data_type, dimension and not dimension, data_type to reduce amount of
        code for all the types.

        Parameters
        ----------
        data_type: SupportedDTypes
        dimensions: Optional[PositiveInt]
            If defined, the number of dimensions determine the depth of the numpy array. Defaults to None,
            e.g. any number of dimensions
        strict_data_typing: bool
            If True, the dtype of the numpy array must be identical to the data_type. No conversion attempts.
        serialize_numpy_array_to_json: Callable[[npt.ArrayLike], Iterable]
            Json serialization function to use. Defaults to NumpyArrayTypeData serializer.
        json_schema_from_type_data: Callable
            Json schema generation function to use. Defaults to NumpyArrayTypeData schema generator.

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
                "json_schema_from_type_data": json_schema_from_type_data,
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
                cls.serialize_numpy_array_to_json,
                is_field_serializer=False,
                when_used="json-unless-none",
            ),
        )

    @classmethod
    def __get_pydantic_json_schema__(
        cls, field_core_schema: core_schema.CoreSchema, handler: GetJsonSchemaHandler
    ) -> JsonSchemaValue:
        return cls.json_schema_from_type_data(field_core_schema, handler, cls.dimensions, cls.data_type)


def np_array_pydantic_annotated_typing(
    data_type: Optional[SupportedDTypes] = None,
    dimensions: Optional[int] = None,
    strict_data_typing: bool = False,
    serialize_numpy_array_to_json: Callable[
        [npt.ArrayLike], Iterable
    ] = pd_np_native_numpy_array_to_data_dict_serializer,
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
        Json serialization function to use. Defaults to NumpyArrayTypeData serializer.

    Returns
    -------
    type-hint for np.ndarray with Pydantic support

    Note
    ----
    The function generates the type hints dynamically, and will not work with static type checkers such as mypy
    or pyright. For that you need to create your types manually.
    """
    return Annotated[
        Union[
            FilePath,
            MultiArrayNumpyFile,
            np.ndarray[  # type: ignore[misc]
                _dimensions_to_shape_type[dimensions]  # pyright: ignore[reportGeneralTypeIssues]
                if dimensions
                else Any,
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
def _deserialize_numpy_array_from_data_dict(data_dict: NumpyArrayTypeData) -> np.ndarray:
    return np.array(data_dict["data"]).astype(data_dict["data_type"])


# IN_THE_FUTURE: Only works with 3.11 and above
# @validate_call
# def _dimension_type_from_depth(depth: PositiveInt) -> type[tuple[int, ...]]:
#     return tuple[*[int] * depth]  # type: ignore


_dimensions_to_shape_type: Final[dict[PositiveInt, type[tuple[int, ...]]]] = {
    1: tuple[int],  # type: ignore[dict-item]
    2: tuple[int, int],  # type: ignore[dict-item]
    3: tuple[int, int, int],  # type: ignore[dict-item]
    4: tuple[int, int, int, int],  # type: ignore[dict-item]
    5: tuple[int, int, int, int, int],  # type: ignore[dict-item]
    6: tuple[int, int, int, int, int, int],  # type: ignore[dict-item]
    7: tuple[int, int, int, int, int, int, int],  # type: ignore[dict-item]
}


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
