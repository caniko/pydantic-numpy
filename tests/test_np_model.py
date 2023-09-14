import os
import tempfile
from pathlib import Path

import numpy as np
import pytest

from pydantic_numpy.model import model_agnostic_load
from pydantic_numpy.typing import NpNDArray
from tests.helper.cache import cached_hyp_array
from tests.model import (
    NpNDArrayModelWithNonArray,
    NpNDArrayModelWithNonArrayWithArbitrary,
)

TEST_MODEL_OBJECT_ID = "test"
OTHER_TEST_MODEL_OBJECT_ID = "other_test"
NON_ARRAY_VALUE = 5


def _create_example_array():
    return cached_hyp_array(np.float64, 1).example()


def _numpy_model():
    return NpNDArrayModelWithNonArray(array=_create_example_array(), non_array=NON_ARRAY_VALUE)


@pytest.fixture
def numpy_model():
    return _numpy_model()


@pytest.fixture(
    params=[
        _numpy_model(),
        NpNDArrayModelWithNonArrayWithArbitrary(
            array=_create_example_array(), non_array=NON_ARRAY_VALUE, my_arbitrary_slice=slice(0, 10)
        ),
    ]
)
def numpy_model_with_arbitrary(request):
    return request.param


if os.name != "nt":

    def test_io_yaml(numpy_model: NpNDArrayModelWithNonArray) -> None:
        with tempfile.TemporaryDirectory() as tmp_dirname:
            numpy_model.dump(tmp_dirname, TEST_MODEL_OBJECT_ID)
            assert numpy_model.load(tmp_dirname, TEST_MODEL_OBJECT_ID) == numpy_model

    def test_io_compressed_pickle(numpy_model_with_arbitrary: NpNDArrayModelWithNonArray) -> None:
        with tempfile.TemporaryDirectory() as tmp_dirname:
            numpy_model_with_arbitrary.dump(tmp_dirname, TEST_MODEL_OBJECT_ID, pickle=True)
            assert numpy_model_with_arbitrary.load(tmp_dirname, TEST_MODEL_OBJECT_ID) == numpy_model_with_arbitrary

    def test_io_pickle(numpy_model_with_arbitrary: NpNDArrayModelWithNonArray) -> None:
        with tempfile.TemporaryDirectory() as tmp_dirname:
            numpy_model_with_arbitrary.dump(tmp_dirname, TEST_MODEL_OBJECT_ID, pickle=True, compress=False)
            assert numpy_model_with_arbitrary.load(tmp_dirname, TEST_MODEL_OBJECT_ID) == numpy_model_with_arbitrary

    def test_typing_json_dump(numpy_model: NpNDArrayModelWithNonArray):
        assert numpy_model.model_dump_json() == '{"array":"%s","non_array":%s}' % (
            np.array2string(numpy_model.array),
            NON_ARRAY_VALUE,
        ), ""

    def test_model_agnostic_load():
        class NumpyModelAForTest(NpNDArrayModelWithNonArray):
            array: NpNDArray
            non_array: int

        class NumpyModelBForTest(NpNDArrayModelWithNonArray):
            array: NpNDArray
            non_array: int

        model_a = NumpyModelAForTest(array=_create_example_array(), non_array=NON_ARRAY_VALUE)
        model_b = NumpyModelBForTest(array=_create_example_array(), non_array=NON_ARRAY_VALUE)

        with tempfile.TemporaryDirectory() as tmp_dirname:
            tmp_dir_path = Path(tmp_dirname)

            model_a.dump(tmp_dir_path, TEST_MODEL_OBJECT_ID)
            model_b.dump(tmp_dir_path, OTHER_TEST_MODEL_OBJECT_ID)

            models = [model_a, model_b]
            assert model_a == model_agnostic_load(tmp_dir_path, TEST_MODEL_OBJECT_ID, models=models)
            assert model_b == model_agnostic_load(tmp_dir_path, OTHER_TEST_MODEL_OBJECT_ID, models=models)
