from typing import Callable, Optional, Union

import numpy as np
import numpy.typing as npt
from numpy import floating, integer
from numpy.lib.npyio import NpzFile
from pydantic import FilePath

from pydantic_numpy.helper.typing import NumpyArrayTypeData, SupportedDTypes
from pydantic_numpy.model import MultiArrayNumpyFile


class PydanticNumpyMultiArrayNumpyFileOnFilePath(Exception):
    pass


def create_array_validator(
    dimensions: Optional[int], target_data_type: SupportedDTypes, strict_data_typing: bool
) -> Callable[[npt.NDArray], npt.NDArray]:
    """
    Creates a validator that ensures the numpy array has the defined dimensions and dtype (data_type).

    Parameters
    ----------
    dimensions: int | None
        Default to None; if set to an integer, enforce the dimension of the numpy array to that integer
    target_data_type: DTypeLike
        The data type the array must have after validation, arrays with different data types will be converted
        during validation. Float to integer is rounded (np.round) followed by an astype with target data type.
    strict_data_typing: bool
        Default False; if True, the incoming array must its dtype match the target_data_type. Strict mode.

    Returns
    -------
    Callable[[npt.NDArray], npt.NDArray]
    Validator for numpy array
    """

    def array_validator(array_data: Union[npt.NDArray, NumpyArrayTypeData]) -> npt.NDArray:
        array: npt.NDArray
        if isinstance(array_data, dict):
            array = np.array(array_data["data"], dtype=array_data["data_type"])
        else:
            array = array_data

        if dimensions and (array_dimensions := len(array.shape)) != dimensions:
            msg = f"Array {array_dimensions}-dimensional; the target dimensions is {dimensions}"
            raise ValueError(msg)

        if target_data_type and array.dtype.type != target_data_type:
            if strict_data_typing:
                msg = f"The data_type {array.dtype.type} does not coincide with type hint; {target_data_type}"
                raise ValueError(msg)

            if issubclass(target_data_type, integer) and issubclass(array.dtype.type, floating):
                array = np.round(array).astype(target_data_type, copy=False)
            else:
                array = array.astype(target_data_type, copy=True)

        return array

    return array_validator


def validate_numpy_array_file(v: FilePath) -> npt.NDArray:
    """
    Validate file path to numpy file by loading and return the respective numpy array

    Parameters
    ----------
    v: FilePath
        Path to the numpy file

    Returns
    -------
    NDArray
    """
    result = np.load(v)

    if isinstance(result, NpzFile):
        files = result.files
        if len(files) > 1:
            msg = (
                f"The provided file path is a multi array NpzFile, which is not supported; "
                f"convert to single array NpzFiles.\n"
                f"Path to multi array file: {result}\n"
                f"Array keys: {', '.join(result.files)}\n"
                f"Use pydantic_numpy.{MultiArrayNumpyFile.__name__} instead of a PathLike alone"
            )
            raise PydanticNumpyMultiArrayNumpyFileOnFilePath(msg)
        result = result[files[0]]

    return result


def validate_multi_array_numpy_file(v: MultiArrayNumpyFile) -> npt.NDArray:
    """
    Validation function for loading numpy array from a name mapping numpy file

    Parameters
    ----------
    v: MultiArrayNumpyFile
        MultiArrayNumpyFile to load

    Returns
    -------
    NDArray from MultiArrayNumpyFile
    """
    return v.load()
