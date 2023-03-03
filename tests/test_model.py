import shutil
import sys
from pathlib import Path

import numpy as np
import pytest

from pydantic_numpy import NDArray

TEST_DUMP_PATH: Path = Path("./test_dump").resolve()


if sys.version_info >= (3, 9):
    from pydantic_numpy.model import NumpyModel

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
            if TEST_DUMP_PATH.exists():
                shutil.rmtree(TEST_DUMP_PATH)

    def test_io_compressed_pickle(test_numpy_model):
        try:
            test_numpy_model.dump(TEST_DUMP_PATH, pickle=True)
            test_numpy_model.load(TEST_DUMP_PATH)
        finally:
            if TEST_DUMP_PATH.exists():
                shutil.rmtree(TEST_DUMP_PATH)

    def test_io_pickle(test_numpy_model):
        try:
            test_numpy_model.dump(TEST_DUMP_PATH, pickle=True, compress=False)
            test_numpy_model.load(TEST_DUMP_PATH)
        finally:
            if TEST_DUMP_PATH.exists():
                shutil.rmtree(TEST_DUMP_PATH)
