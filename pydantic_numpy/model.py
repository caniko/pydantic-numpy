import pickle as pickle_pkg
from pathlib import Path
from typing import (
    Any,
    Callable,
    ClassVar,
    Dict,
    Iterable,
    Optional,
    Tuple,
    Type,
    TypeVar,
)

import compress_pickle
import numpy as np
from pydantic import BaseModel, DirectoryPath, FilePath, validate_arguments
from ruamel.yaml import YAML

yaml = YAML()


class NumpyModel(BaseModel):
    _dump_compression: ClassVar[str] = "lz4"
    _dump_numpy_savez_file_name: ClassVar[str] = "arrays.npz"
    _dump_non_array_file_stem: ClassVar[str] = "object_info"

    _directory_suffix: ClassVar[str] = ".pdnp"

    @classmethod
    @validate_arguments
    def model_directory_path(cls, output_directory: DirectoryPath, object_id: str) -> DirectoryPath:
        return output_directory / f"{object_id}.{cls.__name__}{cls._directory_suffix}"

    def dump(
        self, output_directory: Path, object_id: str, compress: bool = True, pickle: bool = False
    ) -> DirectoryPath:
        assert not self.__config__.arbitrary_types_allowed or (
            self.__config__.arbitrary_types_allowed and pickle
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

    @classmethod
    def load(
        cls,
        output_directory: DirectoryPath,
        object_id: str,
        pre_load_modifier: Optional[Callable[[dict[str, Any]], dict[str, Any]]] = None,
    ) -> "NumpyModelVar":
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

    def _dump_numpy_split_dict(self) -> Tuple[Dict, Dict]:
        ndarray_field_to_array, other_field_to_value = {}, {}
        for k, v in self.dict(exclude_unset=True).items():
            if isinstance(v, np.ndarray):
                ndarray_field_to_array[k] = v
            else:
                other_field_to_value[k] = v

        return ndarray_field_to_array, other_field_to_value

    @classmethod
    @property
    def _dump_compressed_pickle_file_name(cls) -> str:
        return f"{cls._dump_non_array_file_stem}.pickle.{cls._dump_compression}"

    @classmethod
    @property
    def _dump_pickle_file_name(cls) -> str:
        return f"{cls._dump_non_array_file_stem}.pickle"

    @classmethod
    @property
    def _dump_non_array_yaml_name(cls) -> str:
        return f"{cls._dump_non_array_file_stem}.yaml"


NumpyModelVar = TypeVar("NumpyModelVar", bound=NumpyModel)
NumpyModel.update_forward_refs(NumpyModelVar=NumpyModelVar)


NumpyModelCLS = Type[NumpyModel]


def model_agnostic_load(
    output_directory: DirectoryPath,
    object_id: str,
    models: Iterable[NumpyModelCLS],
    not_found_error: bool = False,
    **load_kwargs,
) -> NumpyModelVar | None:
    for model in models:
        if model.model_directory_path(output_directory, object_id).exists():
            return model.load(output_directory, object_id, **load_kwargs)
    if not_found_error:
        raise FileNotFoundError(
            f"Could not find NumpyModel with {object_id} in {output_directory}."
            f"Tried from following classes:\n{', '.join(model.__name__ for model in models)}"
        )
    return None
