import shutil
from pathlib import Path
from typing import Union

import numpy as np
import pytest

from pydantic_numpy import NDArray, NDArrayBool
from pydantic_numpy.model import NumpyModel

TEST_DUMP_PATH: Path = Path("../delete_me_test_dump").resolve()
TEST_MODEL_OBJECT_ID = "test"


class NumpyModelForTest(NumpyModel):
    array: NDArray
    non_array: int


class TestWithArbitraryForTest(NumpyModelForTest):
    my_arbitrary_slice: slice

    class Config:
        arbitrary_types_allowed = True


vanilla_numpy_model = NumpyModelForTest.model_directory_path(TEST_DUMP_PATH, TEST_MODEL_OBJECT_ID)
arbitrary_numpy_model = TestWithArbitraryForTest.model_directory_path(TEST_DUMP_PATH, TEST_MODEL_OBJECT_ID)
numpy_bool_array: NDArrayBool = np.array([True, True, True, True, True], dtype=bool)


def _numpy_model():
    return NumpyModelForTest(array=numpy_bool_array, non_array=5)


@pytest.fixture
def numpy_model():
    return _numpy_model()


@pytest.fixture(
    params=[
        _numpy_model(),
        TestWithArbitraryForTest(array=numpy_bool_array, non_array=5, my_arbitrary_slice=slice(0, 10)),
    ]
)
def numpy_model_with_arbitrary(request):
    return request.param


def test_io_yaml(numpy_model):
    try:
        numpy_model.dump(TEST_DUMP_PATH)
        _test_loaded_numpy_model(numpy_model.load(TEST_DUMP_PATH))
    finally:
        _delete_leftovers()


def test_io_compressed_pickle(numpy_model_with_arbitrary):
    try:
        numpy_model_with_arbitrary.dump(TEST_DUMP_PATH, pickle=True)
        _test_loaded_numpy_model(numpy_model_with_arbitrary.load(TEST_DUMP_PATH))

    finally:
        _delete_leftovers()


def test_io_pickle(numpy_model_with_arbitrary):
    try:
        numpy_model_with_arbitrary.dump(TEST_DUMP_PATH, pickle=True, compress=False)
        _test_loaded_numpy_model(numpy_model_with_arbitrary.load(TEST_DUMP_PATH))
    finally:
        _delete_leftovers()


def _test_loaded_numpy_model(model: Union[NumpyModelForTest, TestWithArbitraryForTest]):
    assert np.all(model.array) and len(model.array) == 5
    if isinstance(model, TestWithArbitraryForTest):
        assert isinstance(model.my_arbitrary_slice, slice)


def _delete_leftovers():
    if vanilla_numpy_model.exists():
        shutil.rmtree(vanilla_numpy_model)

    if arbitrary_numpy_model.exists():
        shutil.rmtree(arbitrary_numpy_model)
