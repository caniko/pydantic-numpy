import sys

from pydantic_numpy.dtype import *
from pydantic_numpy.ndarray import NDArray, NPFileDesc, PotentialNDArray

if sys.version_info >= (3, 9):
    from pydantic_numpy.model import NumpyModel
