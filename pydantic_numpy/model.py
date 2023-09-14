import pickle as pickle_pkg
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, ClassVar, Iterable, Optional

import compress_pickle
import numpy as np
import numpy.typing as npt
from pydantic import BaseModel, DirectoryPath, FilePath, computed_field, validate_call
from ruamel.yaml import YAML

from pydantic_numpy.util import np_general_all_close

yaml = YAML()


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


class NumpyModel(BaseModel):
    _dump_compression: ClassVar[str] = "lz4"
    _dump_numpy_savez_file_name: ClassVar[str] = "arrays.npz"
    _dump_non_array_file_stem: ClassVar[str] = "object_info"

    _directory_suffix: ClassVar[str] = ".pdnp"

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, NumpyModel):
            self_type = self.__pydantic_generic_metadata__["origin"] or self.__class__
            other_type = other.__pydantic_generic_metadata__["origin"] or other.__class__

            self_ndarray_field_to_array, self_other_field_to_value = self._dump_numpy_split_dict()
            other_ndarray_field_to_array, other_other_field_to_value = other._dump_numpy_split_dict()

            return (
                self_type == other_type
                and self_other_field_to_value == other_other_field_to_value
                and self.__pydantic_private__ == other.__pydantic_private__
                and self.__pydantic_extra__ == other.__pydantic_extra__
                and _compare_np_array_dicts(self_ndarray_field_to_array, other_ndarray_field_to_array)
            )
        elif isinstance(other, BaseModel):
            return super().__eq__(other)
        else:
            return NotImplemented  # delegate to the other item in the comparison

    @classmethod
    @validate_call
    def model_directory_path(cls, output_directory: DirectoryPath, object_id: str) -> DirectoryPath:
        return output_directory / f"{object_id}.{cls.__name__}{cls._directory_suffix}"

    @classmethod
    @validate_call
    def load(
        cls,
        output_directory: DirectoryPath,
        object_id: str,
        *,
        pre_load_modifier: Optional[Callable[[dict[str, Any]], dict[str, Any]]] = None,
    ):
        """
        Load NumpyModel instance

        Parameters
        ----------
        output_directory: DirectoryPath
            The root directory where all model instances of interest are stored
        object_id: String
            The ID of the model instance
        pre_load_modifier: Callable[[dict[str, Any]], dict[str, Any]] | None
            Optional function that modifies the loaded arrays

        Returns
        -------
        NumpyModel instance
        """
        object_directory_path = cls.model_directory_path(output_directory, object_id)

        npz_file = np.load(object_directory_path / cls._dump_numpy_savez_file_name)

        other_path: FilePath
        if (other_path := object_directory_path / cls._dump_compressed_pickle_file_name).exists():
            other_field_to_value = compress_pickle.load(other_path)
        elif (other_path := object_directory_path / cls._dump_pickle_file_name).exists():
            with open(other_path, "rb") as in_pickle:
                other_field_to_value = pickle_pkg.load(in_pickle)
        elif (other_path := object_directory_path / cls._dump_non_array_yaml_name).exists():
            with open(other_path, "r") as in_yaml:
                other_field_to_value = yaml.load(in_yaml)
        else:
            other_field_to_value = {}

        field_to_value = {**npz_file, **other_field_to_value}
        if pre_load_modifier:
            field_to_value = pre_load_modifier(field_to_value)

        return cls(**field_to_value)

    @validate_call
    def dump(
        self, output_directory: Path, object_id: str, *, compress: bool = True, pickle: bool = False
    ) -> DirectoryPath:
        assert "arbitrary_types_allowed" not in self.model_config or (
            self.model_config["arbitrary_types_allowed"] and pickle
        ), "Arbitrary types are only supported in pickle mode"

        dump_directory_path = self.model_directory_path(output_directory, object_id)
        dump_directory_path.mkdir(parents=True, exist_ok=True)

        ndarray_field_to_array, other_field_to_value = self._dump_numpy_split_dict()

        if ndarray_field_to_array:
            (np.savez_compressed if compress else np.savez)(
                dump_directory_path / self._dump_numpy_savez_file_name, **ndarray_field_to_array
            )

        if other_field_to_value:
            if pickle:
                if compress:
                    compress_pickle.dump(
                        other_field_to_value,
                        dump_directory_path / self._dump_compressed_pickle_file_name,
                        compression=self._dump_compression,
                    )
                else:
                    with open(dump_directory_path / self._dump_pickle_file_name, "wb") as out_pickle:
                        pickle_pkg.dump(other_field_to_value, out_pickle)

            else:
                with open(dump_directory_path / self._dump_non_array_yaml_name, "w") as out_yaml:
                    yaml.dump(other_field_to_value, out_yaml)

        return dump_directory_path

    def _dump_numpy_split_dict(self) -> tuple[dict, dict]:
        ndarray_field_to_array = {}
        other_field_to_value = {}

        for k, v in self.model_dump(exclude_unset=True).items():
            if isinstance(v, np.ndarray):
                ndarray_field_to_array[k] = v
            else:
                other_field_to_value[k] = v

        return ndarray_field_to_array, other_field_to_value

    @classmethod  # type: ignore[misc]
    @computed_field(return_type=str)
    @property
    def _dump_compressed_pickle_file_name(cls) -> str:
        return f"{cls._dump_non_array_file_stem}.pickle.{cls._dump_compression}"

    @classmethod  # type: ignore[misc]
    @computed_field(return_type=str)
    @property
    def _dump_pickle_file_name(cls) -> str:
        return f"{cls._dump_non_array_file_stem}.pickle"

    @classmethod  # type: ignore[misc]
    @computed_field(return_type=str)
    @property
    def _dump_non_array_yaml_name(cls) -> str:
        return f"{cls._dump_non_array_file_stem}.yaml"


def model_agnostic_load(
    output_directory: DirectoryPath,
    object_id: str,
    models: Iterable[type[NumpyModel]],
    not_found_error: bool = False,
    **load_kwargs,
) -> Optional[NumpyModel]:
    """
    Provided an Iterable containing possible models, and the directory where they have been dumped. Load the first
    instance of model that matches the provided object ID.

    Parameters
    ----------
    output_directory: DirectoryPath
        The root directory where all model instances of interest are stored
    object_id: String
        The ID of the model instance
    models: Iterable[type[NumpyModel]]
        All NumpyModel instances of interest, note that they should have differing names
    not_found_error: bool
        If True, throw error when the respective model instance was not found
    load_kwargs
        Key-word arguments to pass to the load function

    Returns
    -------
    NumpyModel instance if found
    """
    for model in models:
        if model.model_directory_path(output_directory, object_id).exists():
            return model.load(output_directory, object_id, **load_kwargs)

    if not_found_error:
        raise FileNotFoundError(
            f"Could not find NumpyModel with {object_id} in {output_directory}."
            f"Tried from following classes:\n{', '.join(model.__name__ for model in models)}"
        )

    return None


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


def _compare_np_array_dicts(
    dict_a: dict[str, npt.NDArray], dict_b: dict[str, npt.NDArray], rtol: float = 1e-05, atol: float = 1e-08
) -> bool:
    """
    Compare two dictionaries containing numpy arrays as values.

    Parameters:
    dict_a, dict_b: dictionaries to compare. They should have same keys.
    rtol, atol: relative and absolute tolerances for np.isclose()

    Returns:
    Boolean value for each key, True if corresponding arrays are close, else False.
    """

    keys1 = frozenset(dict_a.keys())
    keys2 = frozenset(dict_b.keys())

    if keys1 != keys2:
        raise ValueError("Dictionaries have different keys")

    for key in keys1:
        arr_a = dict_a[key]
        arr_b = dict_b[key]

        if arr_a.shape != arr_b.shape:
            raise ValueError(f"Arrays for key '{key}' have different shapes")

        if not np_general_all_close(arr_a, arr_b, rtol, atol):
            return False

    return True


__all__ = ["NumpyModel", "MultiArrayNumpyFile", "model_agnostic_load"]
