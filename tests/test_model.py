import shutil
from pathlib import Path

import numpy as np
import pytest

from pydantic_numpy import NDArray
from pydantic_numpy.model import NumpyModel

TEST_DUMP_PATH: Path = Path("./test_dump").resolve()


class TestNumpyModel(NumpyModel):
    array: NDArray
    non_array: int


@pytest.fixture
def test_numpy_model():
    return TestNumpyModel(array=np.array([True, False, True, True, True], dtype=bool), non_array=5)


def test_io_yaml(test_numpy_model):
    try:
        test_numpy_model.dump(TEST_DUMP_PATH)
        test_numpy_model.load(TEST_DUMP_PATH)
    finally:
        _delete_leftovers()


def test_io_compressed_pickle(test_numpy_model):
    try:
        test_numpy_model.dump(TEST_DUMP_PATH, pickle=True)
        test_numpy_model.load(TEST_DUMP_PATH)
    finally:
        _delete_leftovers()


def test_io_pickle(test_numpy_model):
    try:
        test_numpy_model.dump(TEST_DUMP_PATH, pickle=True, compress=False)
        test_numpy_model.load(TEST_DUMP_PATH)
    finally:
        _delete_leftovers()


def _delete_leftovers():
    dump_path = TestNumpyModel.model_directory_path(TEST_DUMP_PATH)
    if dump_path.exists():
        shutil.rmtree(dump_path)
