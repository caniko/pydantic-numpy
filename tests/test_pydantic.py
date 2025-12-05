import unittest

import numpy as np
from jsonschema import Draft202012Validator

import pydantic_numpy.typing as pnd
from pydantic import BaseModel
from tests.model import N1DArrayModel, NpNDArrayModel, NpNDArrayModelWithNonArray

test_model_instance = NpNDArrayModelWithNonArray(array=np.zeros(10), non_array=2)


class TestModelValidation(unittest.TestCase):
    def test_model_json_schema_np_nd_array_model(self):
        schema = NpNDArrayModel.model_json_schema()
        expected = {
            "properties": {
                "array": {
                    "description": "NumPy ndarray with shape Any and dtype Any",
                    "properties": {
                        "data_type": {
                            "default": "Any",
                            "title": "dtype",
                            "type": "string",
                        },
                        "data": {"items": {}, "type": "array"},
                    },
                    "required": ["data_type", "data"],
                    "title": "Numpy Array",
                    "type": "object",
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
                    "description": "NumPy ndarray with shape tuple[int] and dtype Any",
                    "properties": {
                        "data_type": {
                            "default": "Any",
                            "title": "dtype",
                            "type": "string",
                        },
                        "data": {
                            "items": {},
                            "type": "array",
                        },
                    },
                    "required": ["data_type", "data"],
                    "title": "Numpy Array",
                    "type": "object",
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


class TestJsonSchemaValidity(unittest.TestCase):
    """Test that generated JSON schemas are valid according to JSON Schema spec."""

    def test_schema_validity_float_types(self):
        """Test JSON schema validity for floating point dtypes."""

        class Model(BaseModel):
            fp16: pnd.Np1DArrayFp16
            fp32: pnd.Np1DArrayFp32
            fp64: pnd.Np1DArrayFp64

        schema = Model.model_json_schema()
        Draft202012Validator.check_schema(schema)

        # Verify float types map to "number"
        self.assertEqual(
            schema["properties"]["fp64"]["properties"]["data"]["items"],
            {"type": "number"},
        )

    def test_schema_validity_integer_types(self):
        """Test JSON schema validity for integer dtypes."""

        class Model(BaseModel):
            int8: pnd.Np1DArrayInt8
            int16: pnd.Np1DArrayInt16
            int32: pnd.Np1DArrayInt32
            int64: pnd.Np1DArrayInt64
            uint8: pnd.Np1DArrayUint8
            uint16: pnd.Np1DArrayUint16
            uint32: pnd.Np1DArrayUint32
            uint64: pnd.Np1DArrayUint64

        schema = Model.model_json_schema()
        Draft202012Validator.check_schema(schema)

        # Verify integer types map to "integer"
        self.assertEqual(
            schema["properties"]["int64"]["properties"]["data"]["items"],
            {"type": "integer"},
        )
        self.assertEqual(
            schema["properties"]["uint32"]["properties"]["data"]["items"],
            {"type": "integer"},
        )

    def test_schema_validity_complex_types(self):
        """Test JSON schema validity for complex dtypes."""

        class Model(BaseModel):
            c64: pnd.Np1DArrayComplex64
            c128: pnd.Np1DArrayComplex128

        schema = Model.model_json_schema()
        Draft202012Validator.check_schema(schema)

        # Verify complex types map to "number"
        self.assertEqual(
            schema["properties"]["c128"]["properties"]["data"]["items"],
            {"type": "number"},
        )

    def test_schema_validity_bool_type(self):
        """Test JSON schema validity for boolean dtype."""

        class Model(BaseModel):
            arr: pnd.Np1DArrayBool

        schema = Model.model_json_schema()
        Draft202012Validator.check_schema(schema)

        # Verify bool type maps to "boolean"
        self.assertEqual(
            schema["properties"]["arr"]["properties"]["data"]["items"],
            {"type": "boolean"},
        )

    def test_schema_validity_multidimensional(self):
        """Test JSON schema validity for multi-dimensional arrays."""

        class Model(BaseModel):
            arr_1d: pnd.Np1DArrayFp64
            arr_2d: pnd.Np2DArrayFp64
            arr_3d: pnd.Np3DArrayFp64
            arr_nd: pnd.NpNDArrayFp64

        schema = Model.model_json_schema()
        Draft202012Validator.check_schema(schema)

        # Verify 1D array structure
        data_1d = schema["properties"]["arr_1d"]["properties"]["data"]
        self.assertEqual(data_1d["type"], "array")
        self.assertEqual(data_1d["items"], {"type": "number"})

        # Verify 2D array structure (nested)
        data_2d = schema["properties"]["arr_2d"]["properties"]["data"]
        self.assertEqual(data_2d["type"], "array")
        self.assertEqual(data_2d["items"]["type"], "array")
        self.assertEqual(data_2d["items"]["items"], {"type": "number"})

        # Verify 3D array structure (double nested)
        data_3d = schema["properties"]["arr_3d"]["properties"]["data"]
        self.assertEqual(data_3d["type"], "array")
        self.assertEqual(data_3d["items"]["type"], "array")
        self.assertEqual(data_3d["items"]["items"]["type"], "array")
        self.assertEqual(data_3d["items"]["items"]["items"], {"type": "number"})

    def test_schema_validity_any_dtype(self):
        """Test JSON schema validity for arrays with Any dtype."""

        class Model(BaseModel):
            arr: pnd.NpNDArray

        schema = Model.model_json_schema()
        Draft202012Validator.check_schema(schema)

        # Verify Any type uses empty schema
        self.assertEqual(
            schema["properties"]["arr"]["properties"]["data"]["items"],
            {},
        )

    def test_schema_has_valid_type_field(self):
        """Test that type field is a valid JSON Schema type, not numpy type."""

        class Model(BaseModel):
            arr: pnd.Np1DArrayFp64

        schema = Model.model_json_schema()

        # Verify the array property has type "object", not "np.ndarray[...]"
        self.assertEqual(schema["properties"]["arr"]["type"], "object")
        self.assertNotIn("np.ndarray", schema["properties"]["arr"]["type"])
