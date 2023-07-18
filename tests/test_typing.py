import tempfile
from pathlib import Path
from typing import Optional

import numpy as np
import numpy.typing as npt
import pytest
from hypothesis.extra.numpy import arrays
from pydantic import ValidationError

from pydantic_numpy.helper.validation import PydanticNumpyMultiArrayNumpyFileOnFilePath
from pydantic_numpy.model import MultiArrayNumpyFile
from pydantic_numpy.util import np_general_all_close
from tests.helper.cache import cached_calculation
from tests.helper.groups import (
    data_type_array_typing_dimensions,
    dimension_testing_group,
    strict_data_type_nd_array_typing_dimensions,
    supported_data_types,
)

AXIS_LENGTH = 1


@pytest.mark.parametrize("numpy_dtype,pydantic_typing,dimensions", data_type_array_typing_dimensions)
def test_correct_type(numpy_dtype: npt.DTypeLike, pydantic_typing, dimensions: Optional[int]):
    assert cached_calculation(pydantic_typing)(
        array_field=arrays(numpy_dtype, tuple(AXIS_LENGTH for _ in range(dimensions or 1))).example()
    )


@pytest.mark.parametrize("numpy_dtype,pydantic_typing,dimensions", strict_data_type_nd_array_typing_dimensions)
@pytest.mark.parametrize("wrong_numpy_type", supported_data_types)
def test_wrong_dtype_type(numpy_dtype: npt.DTypeLike, pydantic_typing, dimensions: Optional[int], wrong_numpy_type):
    if wrong_numpy_type == numpy_dtype:
        return True

    bad_array = arrays(wrong_numpy_type, tuple(AXIS_LENGTH for _ in range(dimensions or 5))).example()
    with pytest.raises(ValidationError):
        cached_calculation(pydantic_typing)(array_field=bad_array)


@pytest.mark.parametrize("numpy_dtype,pydantic_typing,dimensions", dimension_testing_group)
def test_wrong_dimension(numpy_dtype: npt.DTypeLike, pydantic_typing, dimensions: Optional[int]):
    wrong_dimension = dimensions + 1

    bad_array = arrays(numpy_dtype, tuple(AXIS_LENGTH for _ in range(wrong_dimension or 5))).example()
    with pytest.raises(ValueError):
        cached_calculation(pydantic_typing)(array_field=bad_array)


@pytest.mark.parametrize("numpy_dtype,pydantic_typing,dimensions", data_type_array_typing_dimensions)
def test_file_path_passing_validation(numpy_dtype: npt.DTypeLike, pydantic_typing, dimensions: Optional[int]):
    hyp_array = arrays(numpy_dtype, tuple(AXIS_LENGTH for _ in range(dimensions or 1))).example()
    with tempfile.NamedTemporaryFile(mode="w+", delete=True, suffix=".npz") as tf:
        np.savez_compressed(tf.name, my_array=hyp_array)
        numpy_model = cached_calculation(pydantic_typing)(array_field=Path(tf.name))

        assert np_general_all_close(numpy_model.array_field, hyp_array)


@pytest.mark.parametrize("numpy_dtype,pydantic_typing,dimensions", data_type_array_typing_dimensions)
def test_file_path_error_on_reading_single_array_file(
    numpy_dtype: npt.DTypeLike, pydantic_typing, dimensions: Optional[int]
):
    hyp_array = arrays(numpy_dtype, tuple(AXIS_LENGTH for _ in range(dimensions or 1))).example()
    with tempfile.NamedTemporaryFile(mode="w+", delete=True, suffix=".npz") as tf:
        np.savez_compressed(tf.name, my_array=hyp_array, my_identical_array=hyp_array)
        model = cached_calculation(pydantic_typing)

        with pytest.raises(PydanticNumpyMultiArrayNumpyFileOnFilePath):
            model(array_field=Path(tf.name))


@pytest.mark.parametrize("numpy_dtype,pydantic_typing,dimensions", data_type_array_typing_dimensions)
def test_multi_array_numpy_passing_validation(numpy_dtype: npt.DTypeLike, pydantic_typing, dimensions: Optional[int]):
    hyp_array = arrays(numpy_dtype, tuple(AXIS_LENGTH for _ in range(dimensions or 1))).example()
    with tempfile.NamedTemporaryFile(mode="w+", delete=True, suffix=".npz") as tf:
        np.savez_compressed(tf.name, my_array=hyp_array)
        numpy_model = cached_calculation(pydantic_typing)(
            array_field=MultiArrayNumpyFile(path=Path(tf.name), key="my_array")
        )

        assert np_general_all_close(numpy_model.array_field, hyp_array)


@pytest.mark.parametrize("numpy_dtype,pydantic_typing,dimensions", data_type_array_typing_dimensions)
def test_multi_array_numpy_error_on_reading_single_array_file(
    numpy_dtype: npt.DTypeLike, pydantic_typing, dimensions: Optional[int]
):
    hyp_array = arrays(numpy_dtype, tuple(AXIS_LENGTH for _ in range(dimensions or 1))).example()
    with tempfile.NamedTemporaryFile(mode="w+", delete=True, suffix=".npy") as tf:
        np.save(tf.name, hyp_array)
        model = cached_calculation(pydantic_typing)

        with pytest.raises(AttributeError):
            model(array_field=MultiArrayNumpyFile(path=Path(tf.name), key="my_array"))
