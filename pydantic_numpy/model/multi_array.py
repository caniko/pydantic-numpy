from functools import lru_cache

import numpy as np
import numpy.typing as npt
from pydantic import FilePath
from pydantic.dataclasses import dataclass


@dataclass(frozen=True)
class MultiArrayNumpyFile:
    path: FilePath
    key: str
    cached_load: bool = False

    def load(self) -> npt.NDArray:
        """
        Load the NDArray stored in the given path within the given key

        Returns
        -------
        NDArray
        """
        loaded = _cached_np_array_load(self.path) if self.cached_load else np.load(self.path)
        try:
            return loaded[self.key]
        except IndexError:
            msg = f"The given path points to an uncompressed numpy file, which only has one array in it: {self.path}"
            raise AttributeError(msg)


@lru_cache
def _cached_np_array_load(path: FilePath):
    """
    Store the loaded numpy object within LRU cache in case we need it several times

    Parameters
    ----------
    path: FilePath
        Path to the numpy file

    Returns
    -------
    Same as np.load
    """
    return np.load(path)


__all__ = ["MultiArrayNumpyFile"]
