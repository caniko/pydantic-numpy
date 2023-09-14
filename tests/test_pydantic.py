from tests.model import NpNDArrayModel


def test_model_json_schema():
    assert NpNDArrayModel.model_json_schema()
