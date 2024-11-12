# pydantic-numpy

![Python 3.10-3.13](https://img.shields.io/badge/python-3.9--3.13-blue.svg)
[![Packaged with Poetry](https://img.shields.io/badge/packaging-poetry-cyan.svg)](https://python-poetry.org/)
![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)
![Imports: isort](https://img.shields.io/badge/%20imports-isort-%231674b1?style=flat&labelColor=ef8336)
![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)


## Usage

Package that integrates NumPy Arrays into Pydantic!

- `pydantic_numpy.typing` provides many typings such as `NpNDArrayFp64`, `Np3DArrayFp64` (float64 that must be 3D)! Works with both `pydantic.BaseModel` and `pydantic.dataclass`
- `NumpyModel` (derived from `pydantic.BaseModel`) make it possible to dump and load `np.ndarray` within model fields alongside other fields that are not instances of `np.ndarray`!

See the [`test.helper.testing_groups`](https://github.com/caniko/pydantic-numpy/blob/trunk/tests/helper/testing_groups.py) to see types that are defined explicitly.

### Examples

For more examples see [test_ndarray.py](./tests/test_typing.py)

```python
import numpy as np
from pydantic import BaseModel

import pydantic_numpy.typing as pnd
from pydantic_numpy import np_array_pydantic_annotated_typing
from pydantic_numpy.model import NumpyModel, MultiArrayNumpyFile


class MyBaseModelDerivedModel(BaseModel):
    any_array_dtype_and_dimension: pnd.NpNDArray

    # Must be numpy float32 as dtype
    k: np_array_pydantic_annotated_typing(data_type=np.float32)
    shorthand_for_k: pnd.NpNDArrayFp32

    must_be_1d_np_array: np_array_pydantic_annotated_typing(dimensions=1)


class MyDemoNumpyModel(NumpyModel):
    k: np_array_pydantic_annotated_typing(data_type=np.float32)


# Instantiate from array
cfg = MyDemoModel(k=[1, 2])
# Instantiate from numpy file
cfg = MyDemoModel(k="path_to/array.npy")
# Instantiate from npz file with key
cfg = MyDemoModel(k=MultiArrayNumpyFile(path="path_to/array.npz", key="k"))

cfg.k   # np.ndarray[np.float32]

cfg.dump("path_to_dump_dir", "object_id")
cfg.load("path_to_dump_dir", "object_id")
```

`NumpyModel.load` requires the original model:
```python
MyNumpyModel.load(<path>)
```
Use `model_agnostic_load` when you have several models that may be the correct model:

```python
from pydantic_numpy.model import model_agnostic_load

cfg.dump("path_to_dump_dir", "object_id")
equals_cfg = model_agnostic_load("path_to_dump_dir", "object_id", models=[MyNumpyModel, MyDemoModel])
```

### Custom type
There are two ways to define. Function derived types with `pydantic_numpy.helper.annotation.np_array_pydantic_annotated_typing`.

Function derived types don't work with static type checkers like Pyright and MyPy. In case you need the support,
just create the types yourself:
    
```python
NpStrict1DArrayInt64 = Annotated[
    np.ndarray[tuple[int], np.dtype[np.int64]],
    NpArrayPydanticAnnotation.factory(data_type=np.int64, dimensions=1, strict_data_typing=True),
]
```

#### Custom serialization

If the default serialization of NumpyDataDict, as outlined in [typing.py](https://github.com/caniko/pydantic-numpy/blob/trunk/pydantic_numpy/helper/typing.py), doesn't meet your requirements, you have the option to define a custom type with its own serializer. This can be achieved using the NpArrayPydanticAnnotation.factory method, which accepts a custom serialization function through its serialize_numpy_array_to_json parameter. This parameter expects a function of the form `Callable[[npt.ArrayLike], Iterable]`, allowing you to tailor the serialization process to your specific needs.

Example below illustrates definition of 1d-array of `float32` type that serializes to flat Python list (without nested dict as in default `NumpyDataDict` case):

```python
def _serialize_numpy_array_to_float_list(array_like: npt.ArrayLike) -> Iterable:
    return np.array(array_like).astype(float).tolist()


Np1DArrayFp32 = Annotated[
    np.ndarray[tuple[int], np.dtype[np.float32]],
    NpArrayPydanticAnnotation.factory(
        data_type=np.float32,
        dimensions=1,
        strict_data_typing=False,
        serialize_numpy_array_to_json=_serialize_numpy_array_to_float_list,
    ),
]
```

### Install
```shell
pip install pydantic-numpy
```

### History
The original idea originates from [this discussion](https://gist.github.com/danielhfrank/00e6b8556eed73fb4053450e602d2434), and forked from [cheind's](https://github.com/cheind/pydantic-numpy) repository.
