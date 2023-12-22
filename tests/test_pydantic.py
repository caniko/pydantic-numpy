import unittest

import numpy as np

from tests.model import NpNDArrayModel, NpNDArrayModelWithNonArray

test_model_instance = NpNDArrayModelWithNonArray(array=np.zeros(10), non_array=2)


class TestModelValidation(unittest.TestCase):
    def test_model_json_schema(self):
        self.assertTrue(NpNDArrayModel.model_json_schema())

    def test_validate_json(self):
        json_str = test_model_instance.model_dump_json()
        self.assertTrue(test_model_instance.model_validate_json(json_str))
