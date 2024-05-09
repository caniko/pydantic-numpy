import platform
import tempfile
from pathlib import Path

import numpy as np
import pytest

from pydantic_numpy.model import NumpyModel, model_agnostic_load
from pydantic_numpy.typing import NpNDArray
from tests.model import (
    NpNDArrayModelWithNonArray,
    NpNDArrayModelWithNonArrayWithArbitrary,
)

TEST_MODEL_OBJECT_ID = "test"
OTHER_TEST_MODEL_OBJECT_ID = "other_test"
NON_ARRAY_VALUE = 5


@pytest.fixture
def numpy_model() -> NpNDArrayModelWithNonArray:
    return NpNDArrayModelWithNonArray(array=np.array([0.0]), non_array=NON_ARRAY_VALUE)


@pytest.fixture
def numpy_model_with_arbitrary() -> NpNDArrayModelWithNonArrayWithArbitrary:
    return NpNDArrayModelWithNonArrayWithArbitrary(
        array=np.array([0.0]), non_array=NON_ARRAY_VALUE, my_arbitrary_slice=slice(0, 1)
    )


if platform.system() != "Windows":

    def test_io_yaml(numpy_model: NpNDArrayModelWithNonArray) -> None:
        with tempfile.TemporaryDirectory() as tmp_dirname:
            numpy_model.dump(Path(tmp_dirname), TEST_MODEL_OBJECT_ID)
            assert numpy_model.load(Path(tmp_dirname), TEST_MODEL_OBJECT_ID) == numpy_model

    def test_io_compressed_pickle(numpy_model_with_arbitrary: NpNDArrayModelWithNonArray) -> None:
        with tempfile.TemporaryDirectory() as tmp_dirname:
            numpy_model_with_arbitrary.dump(Path(tmp_dirname), TEST_MODEL_OBJECT_ID, pickle=True)
            assert (
                numpy_model_with_arbitrary.load(Path(tmp_dirname), TEST_MODEL_OBJECT_ID) == numpy_model_with_arbitrary
            )

    def test_io_pickle(numpy_model_with_arbitrary: NpNDArrayModelWithNonArray) -> None:
        with tempfile.TemporaryDirectory() as tmp_dirname:
            numpy_model_with_arbitrary.dump(Path(tmp_dirname), TEST_MODEL_OBJECT_ID, pickle=True, compress=False)
            assert (
                numpy_model_with_arbitrary.load(Path(tmp_dirname), TEST_MODEL_OBJECT_ID) == numpy_model_with_arbitrary
            )

    def test_model_agnostic_load():
        class NumpyModelAForTest(NpNDArrayModelWithNonArray):
            array: NpNDArray
            non_array: int

        class NumpyModelBForTest(NpNDArrayModelWithNonArray):
            array: NpNDArray
            non_array: int

        model_a = NumpyModelAForTest(array=np.array([0.0]), non_array=NON_ARRAY_VALUE)
        model_b = NumpyModelBForTest(array=np.array([0.0]), non_array=NON_ARRAY_VALUE)

        with tempfile.TemporaryDirectory() as tmp_dirname:
            tmp_dir_path = Path(tmp_dirname)

            model_a.dump(tmp_dir_path, TEST_MODEL_OBJECT_ID)
            model_b.dump(tmp_dir_path, OTHER_TEST_MODEL_OBJECT_ID)

            models = [NumpyModelAForTest, NumpyModelBForTest]
            assert model_a == model_agnostic_load(tmp_dir_path, TEST_MODEL_OBJECT_ID, models=models)
            assert model_b == model_agnostic_load(tmp_dir_path, OTHER_TEST_MODEL_OBJECT_ID, models=models)

    def test_simple_eq(numpy_model: NpNDArrayModelWithNonArray) -> None:
        assert numpy_model == numpy_model

    def test_not_eq_different_fields(numpy_model, numpy_model_with_arbitrary) -> None:
        assert numpy_model != numpy_model_with_arbitrary

        class AnotherModel(NumpyModel):
            yarra: NpNDArray

        assert numpy_model != AnotherModel(yarra=np.array([0.0]))

    def test_not_eq_different_inner(numpy_model: NpNDArrayModelWithNonArray) -> None:
        assert numpy_model != NpNDArrayModelWithNonArray(array=np.array([1.0]), non_array=NON_ARRAY_VALUE)

    def test_not_eq_different_shape(numpy_model: NpNDArrayModelWithNonArray) -> None:
        assert numpy_model != NpNDArrayModelWithNonArray(array=np.array([0.0, 1.0]), non_array=NON_ARRAY_VALUE)

    def test_random_not_eq(numpy_model: NpNDArrayModelWithNonArray) -> None:
        for r in (0, 5, 1.0, "1"):
            assert numpy_model != r

    def test_serde_eq(numpy_model: NpNDArrayModelWithNonArray) -> None:
        ser = numpy_model.model_dump_json()
        reread_data = numpy_model.model_validate_json(ser)

        assert numpy_model == reread_data
