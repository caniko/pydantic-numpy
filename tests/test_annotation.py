from typing import Annotated, Any

import numpy as np
import orjson
from pydantic import BaseModel
from typing_extensions import TypeAlias

from pydantic_numpy.helper.annotation import NpArrayPydanticAnnotation


def _custom_serializer(array: np.ndarray) -> list[float]:
    return array.astype(float).tolist()


_Np1DArray: TypeAlias = Annotated[
    np.ndarray[tuple[int], np.dtype[Any]],
    NpArrayPydanticAnnotation.factory(
        data_type=None, dimensions=1, strict_data_typing=False, serialize_numpy_array_to_json=_custom_serializer
    ),
]


def test_custom_serializer():
    class FooModel(BaseModel):
        arr: _Np1DArray

    foo_model = FooModel(arr=np.zeros(42))

    model_dict = orjson.loads(foo_model.model_dump_json())

    assert "arr" in model_dict
    assert isinstance(model_dict["arr"], list)
    assert len(model_dict["arr"]) == 42
