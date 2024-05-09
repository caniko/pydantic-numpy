import platform
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np
import numpy.typing as npt
import orjson
import pytest
from numpy.testing import assert_almost_equal
from pydantic import ValidationError

from pydantic_numpy.helper.validation import PydanticNumpyMultiArrayNumpyFileOnFilePath
from pydantic_numpy.model import MultiArrayNumpyFile
from pydantic_numpy.typing import Np1DArrayInt64
from pydantic_numpy.util import np_general_all_close
from tests.helper.cache import get_numpy_type_model
from tests.helper.testing_groups import (
    data_type_array_typing_dimensions,
    data_type_nd_array_typing_dimensions_without_complex,
    strict_data_type_nd_array_typing_dimensions,
    supported_data_types,
)


@pytest.mark.parametrize("numpy_array, numpy_dtype, pydantic_typing, dimensions", data_type_array_typing_dimensions)
def test_correct_type(
    numpy_array: npt.ArrayLike, numpy_dtype: npt.DTypeLike, pydantic_typing, dimensions: Optional[int]
):
    assert get_numpy_type_model(pydantic_typing)(array_field=numpy_array)


@pytest.mark.parametrize(
    "numpy_array, numpy_dtype, pydantic_typing, dimensions", strict_data_type_nd_array_typing_dimensions
)
@pytest.mark.parametrize("bad_numpy_array, wrong_numpy_type", supported_data_types)
def test_wrong_dtype_type(
    numpy_array: npt.ArrayLike,
    numpy_dtype: npt.DTypeLike,
    pydantic_typing,
    dimensions: Optional[int],
    bad_numpy_array: npt.ArrayLike,
    wrong_numpy_type: npt.DTypeLike,
):
    if wrong_numpy_type == numpy_dtype:
        return

    with pytest.raises(ValidationError):
        get_numpy_type_model(pydantic_typing)(array_field=bad_numpy_array)


def test_wrong_dimension():
    with pytest.raises(ValueError):
        get_numpy_type_model(Np1DArrayInt64)(array_field=np.array([[0]]))


if platform.system() == "Linux":
    from tests.helper.testing_groups import (
        get_strict_data_type_nd_array_typing_dimensions_128_bit,
    )

    @pytest.mark.parametrize(
        "numpy_array, numpy_dtype, pydantic_typing, dimensions", data_type_nd_array_typing_dimensions_without_complex
    )
    def test_json_serialize_deserialize(
        numpy_array: npt.ArrayLike, numpy_dtype: npt.DTypeLike, pydantic_typing, dimensions: Optional[int]
    ):
        numpy_model = get_numpy_type_model(pydantic_typing)
        dumped_model_json_loaded = orjson.loads(numpy_model(array_field=numpy_array).model_dump_json())

        round_trip_result = numpy_model(array_field=dumped_model_json_loaded["array_field"]).array_field

        if issubclass(numpy_dtype, np.timedelta64) or issubclass(numpy_dtype, np.datetime64):
            assert np.all(numpy_array == round_trip_result)
        else:
            assert_almost_equal(numpy_array, round_trip_result)

    @pytest.mark.parametrize("numpy_array, numpy_dtype, pydantic_typing, dimensions", data_type_array_typing_dimensions)
    def test_file_path_passing_validation(
        numpy_array: npt.ArrayLike, numpy_dtype: npt.DTypeLike, pydantic_typing, dimensions: Optional[int]
    ):
        with tempfile.NamedTemporaryFile(mode="w+", delete=True, suffix=".npz") as tf:
            np.savez_compressed(tf.name, my_array=numpy_array)
            numpy_model = get_numpy_type_model(pydantic_typing)(array_field=Path(tf.name))

            assert np_general_all_close(numpy_model.array_field, numpy_array)

    @pytest.mark.parametrize("numpy_array, numpy_dtype, pydantic_typing, dimensions", data_type_array_typing_dimensions)
    def test_file_path_error_on_reading_single_array_file(
        numpy_array: npt.ArrayLike, numpy_dtype: npt.DTypeLike, pydantic_typing, dimensions: Optional[int]
    ):
        with tempfile.NamedTemporaryFile(mode="w+", delete=True, suffix=".npz") as tf:
            np.savez_compressed(tf.name, my_array=numpy_array, my_identical_array=numpy_array)
            model = get_numpy_type_model(pydantic_typing)

            with pytest.raises(PydanticNumpyMultiArrayNumpyFileOnFilePath):
                model(array_field=Path(tf.name))

    @pytest.mark.parametrize("numpy_array, numpy_dtype, pydantic_typing, dimensions", data_type_array_typing_dimensions)
    def test_multi_array_numpy_passing_validation(
        numpy_array: npt.ArrayLike, numpy_dtype: npt.DTypeLike, pydantic_typing, dimensions: Optional[int]
    ):
        with tempfile.NamedTemporaryFile(mode="w+", delete=True, suffix=".npz") as tf:
            np.savez_compressed(tf.name, my_array=numpy_array)
            numpy_model = get_numpy_type_model(pydantic_typing)(
                array_field=MultiArrayNumpyFile(path=Path(tf.name), key="my_array")
            )
            assert np_general_all_close(numpy_model.array_field, numpy_array)

    @pytest.mark.parametrize("numpy_array, numpy_dtype, pydantic_typing, dimensions", data_type_array_typing_dimensions)
    def test_multi_array_numpy_error_on_reading_single_array_file(
        numpy_array: npt.ArrayLike, numpy_dtype: npt.DTypeLike, pydantic_typing, dimensions: Optional[int]
    ):
        with tempfile.NamedTemporaryFile(mode="w+", delete=True, suffix=".npy") as tf:
            np.save(tf.name, numpy_array)
            model = get_numpy_type_model(pydantic_typing)

            with pytest.raises(AttributeError):
                model(array_field=MultiArrayNumpyFile(path=Path(tf.name), key="my_array"))

    @pytest.mark.parametrize(
        "numpy_array, numpy_dtype, pydantic_typing, dimensions",
        get_strict_data_type_nd_array_typing_dimensions_128_bit(),
    )
    def test_correct_128_bit_type(
        numpy_array: npt.ArrayLike, numpy_dtype: npt.DTypeLike, pydantic_typing, dimensions: Optional[int]
    ):
        assert get_numpy_type_model(pydantic_typing)(array_field=numpy_array)

    @pytest.mark.parametrize(
        "numpy_array, numpy_dtype, pydantic_typing, dimensions",
        get_strict_data_type_nd_array_typing_dimensions_128_bit(),
    )
    @pytest.mark.parametrize("bad_numpy_array, wrong_numpy_type", supported_data_types)
    def test_wrong_dtype_128_bit_type(
        numpy_array: npt.ArrayLike,
        numpy_dtype: npt.DTypeLike,
        pydantic_typing,
        dimensions: Optional[int],
        bad_numpy_array: npt.ArrayLike,
        wrong_numpy_type: npt.DTypeLike,
    ):
        if wrong_numpy_type == numpy_dtype:
            return

        with pytest.raises(ValidationError):
            get_numpy_type_model(pydantic_typing)(array_field=bad_numpy_array)
