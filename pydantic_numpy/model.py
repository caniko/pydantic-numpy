import pickle as pickle_pkg
from pathlib import Path
from typing import ClassVar, Dict, Tuple, TypeVar

import compress_pickle
import numpy as np
from pydantic import BaseModel, DirectoryPath
from ruamel.yaml import YAML

yaml = YAML()


class NumpyModel(BaseModel):
    _dump_compression: ClassVar[str] = "lz4"
    _dump_numpy_savez_file_name: ClassVar[str] = "arrays.npz"
    _dump_non_array_file_stem: ClassVar[str] = "object_info"

    _directory_suffix: ClassVar[str] = ".pdnp"

    @classmethod
    def model_directory_path(cls, pre_model_path: DirectoryPath) -> DirectoryPath:
        return pre_model_path.parent / f"{pre_model_path.stem}-{cls.__name__}{cls._directory_suffix}"

    @property
    def _dump_numpy_split_dict(self) -> Tuple[Dict, Dict]:
        ndarray_field_to_array, other_field_to_value = {}, {}
        for k, v in self.dict().items():
            if isinstance(v, np.ndarray):
                ndarray_field_to_array[k] = v
            else:
                other_field_to_value[k] = v

        return ndarray_field_to_array, other_field_to_value

    def dump(self, dump_directory_path: Path, compress: bool = True, pickle: bool = False) -> None:
        assert not (
            self.Config.arbitrary_types_allowed and not pickle
        ), "Arbitrary types are only supported in pickle mode"

        dump_directory_path = self.model_directory_path(dump_directory_path)
        dump_directory_path.mkdir(parents=True, exist_ok=True)

        ndarray_field_to_array, other_field_to_value = self._dump_numpy_split_dict

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

    @classmethod
    def load(cls, object_directory_path: DirectoryPath) -> "NumpyModelVar":
        object_directory_path = cls.model_directory_path(object_directory_path)

        npz_file = np.load(object_directory_path / cls._dump_numpy_savez_file_name)

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

        return cls(**npz_file, **other_field_to_value)

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
