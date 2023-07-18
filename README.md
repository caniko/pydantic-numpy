# pydantic-numpy

Package that integrates NumPy Arrays into Pydantic!

- `NumpyModel` make it possible to dump and load `np.ndarray` within model fields alongside other fields that are not instances of `np.ndarray`!
- `pydantic_numpy.typing` provides many typings such as `NpNDArrayFp64`, `Np3DArrayFp64` (float64 that must be 3D)!

## Usage

For more examples see [test_ndarray.py](./tests/test_typing.py)

```python
import numpy as np

import pydantic_numpy.typing as pnd
from pydantic_numpy import np_array_pydantic_annotated_typing
from pydantic_numpy.model import NumpyModel, MultiArrayNumpyFile


class MyNumpyModel(NumpyModel):
    any_array_dtype_and_dimension: pnd.NpNDArray

    # Must be numpy float32 as dtype
    k: np_array_pydantic_annotated_typing(data_type=np.float32)
    shorthand_for_k: pnd.NpNDArrayFp32

    must_be_1d_np_array: np_array_pydantic_annotated_typing(dimensions=1)


class MyDemoModel(NumpyModel):
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

`NumpyModel.load` requires the original mode, use `model_agnostic_load` when you have several models that may be the right model:
```python
from pydantic_numpy.model.np_model import model_agnostic_load

cfg.dump("path_to_dump_dir", "object_id")
equals_cfg = model_agnostic_load("path_to_dump_dir", "object_id", models=[MyNumpyModel, MyDemoModel])
```

### Data type (dtype) support!

This package also comes with `pydantic_numpy.dtype`, which adds subtyping support such as `NpNDArray[float, pnd.float32]`. All subfields must be from this package as numpy dtypes have no Pydantic support, which is implemented in this package through the [generic class workflow](https://pydantic-docs.helpmanual.io/usage/types/#generic-classes-as-types).

### Install
```shell
pip install pydantic-numpy
```

## Considerations
You can install from [cheind's](https://github.com/cheind/pydantic-numpy) repository if you want Python `3.8` support, but this version only support Pydantic V1 and will not work with V2.

### Licensing notice
As of version `3.0.0` the license has moved over to BSD-4. The versions prior are under the MIT license.

### History
The original idea originates from [this discussion](https://gist.github.com/danielhfrank/00e6b8556eed73fb4053450e602d2434), and forked from [cheind's](https://github.com/cheind/pydantic-numpy) repository.