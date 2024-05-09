import unittest

import numpy as np

from pydantic_numpy.typing import NpNDArrayFp64, NpNDArrayInt64    
from pydantic_numpy.model import NumpyModel

class TestSerDeser(unittest.TestCase):
    def test_can_ser_and_deser_basic_numpy_to_json_and_compare(self):

        # Given
        class NumpyData(NumpyModel):
            array_float: NpNDArrayFp64 = np.linspace(0.0, 1.0, 20)
            array_int: NpNDArrayInt64 = np.arange(100)

        data = NumpyData()
        
        # When
        ser = data.model_dump_json()
        data_read_correct = NumpyData.model_validate_json(ser)
        data_read_incorrect = NumpyData.model_validate_json(ser)
        data_read_incorrect.array_float[0] = 1.5
        
        # Then
        assert data == data_read_correct
        assert not data == data_read_incorrect
        

        