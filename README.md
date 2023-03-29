# pydantic-numpy

Integrate NumPy into Pydantic, and provide tooling! `NumpyModel` make it possible to dump and load `np.ndarray` within model fields!

### Install
```shell
pip install pydantic-numpy
```

## Usage

For more examples see [test_ndarray.py](./tests/test_ndarray.py)

```python
import pydantic_numpy.dtype as pnd
from pydantic_numpy import NDArray, NDArrayFp32, NumpyModel


class MyPydanticNumpyModel(NumpyModel):
    K: NDArray[float, pnd.float32]
    C: NDArrayFp32  # <- Shorthand for same type as K


# Instantiate from array
cfg = MyPydanticNumpyModel(K=[1, 2])
# Instantiate from numpy file
cfg = MyPydanticNumpyModel(K={"path": "path_to/array.npy"})
# Instantiate from npz file with key
cfg = MyPydanticNumpyModel(K={"path": "path_to/array.npz", "key": "K"})

cfg.K
# np.ndarray[np.float32]

cfg.dump("path_to_dump_dir", "object_id")
cfg.load("path_to_dump_dir", "object_id")
```

`NumpyModel.load` requires the original mode, use `model_agnostic_load` when you have several models that may be the right model.

### Data type (dtype) support!

This package also comes with `pydantic_numpy.dtype`, which adds subtyping support such as `NDArray[float, pnd.float32]`. All subfields must be from this package as numpy dtypes have no Pydantic support, which is implemented in this package through the [generic class workflow](https://pydantic-docs.helpmanual.io/usage/types/#generic-classes-as-types).

## Considerations

The `NDArray` class from `pydantic-numpy` is daughter of `np.ndarray`. IDEs and linters might complain that you are passing an incorrect `type` to a model. The only solution is to merge these change into `numpy`.

You can also use the `typings` in `pydantic.validate_arguments`.

You can install from [cheind's](https://github.com/cheind/pydantic-numpy) repository if you want Python `3.8` support.

## History

The original idea originates from [this discussion](https://gist.github.com/danielhfrank/00e6b8556eed73fb4053450e602d2434), and forked from [cheind's](https://github.com/cheind/pydantic-numpy) repository.
