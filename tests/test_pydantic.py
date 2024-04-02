import unittest

import numpy as np

from tests.model import N1DArrayModel, NpNDArrayModel, NpNDArrayModelWithNonArray

test_model_instance = NpNDArrayModelWithNonArray(array=np.zeros(10), non_array=2)


class TestModelValidation(unittest.TestCase):
    def test_model_json_schema_np_nd_array_model(self):
        schema = NpNDArrayModel.model_json_schema()
        expected = {
            "properties": {
                "array": {
                    "properties": {
                        "data_type": {"default": "Any", "title": "dtype", "type": "string"},
                        "data": {"items_schema": {"type": "any"}, "type": "list"},
                    },
                    "required": ["data_type", "data"],
                    "title": "Numpy Array",
                    "type": "np.ndarray[Any, np.dtype[Any]]",
                }
            },
            "required": ["array"],
            "title": "NpNDArrayModel",
            "type": "object",
        }
        self.assertEqual(schema, expected)

    def test_model_json_schema_np_1d_array_model(self):
        schema = N1DArrayModel.model_json_schema()
        expected = {
            "properties": {
                "array": {
                    "properties": {
                        "data_type": {"default": "Any", "title": "dtype", "type": "string"},
                        "data": {
                            "items_schema": {"items_schema": {"type": "any"}, "type": "list"},
                            "max_length": 1,
                            "min_length": 1,
                            "type": "list",
                        },
                    },
                    "required": ["data_type", "data"],
                    "title": "Numpy Array",
                    "type": "np.ndarray[tuple[int], np.dtype[Any]]",
                }
            },
            "required": ["array"],
            "title": "N1DArrayModel",
            "type": "object",
        }
        self.assertEqual(schema, expected)

    def test_validate_json(self):
        json_str = test_model_instance.model_dump_json()
        self.assertTrue(test_model_instance.model_validate_json(json_str))
