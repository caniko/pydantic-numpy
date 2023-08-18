from pydantic_numpy.model import NumpyModel
from pydantic_numpy.typing import NpNDArray


class TestModel(NumpyModel):
    array: NpNDArray


def test_model_json_schema():
    assert TestModel.model_json_schema()
