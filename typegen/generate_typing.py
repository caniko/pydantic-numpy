from pathlib import Path
from typing import Final


def write_annotations(output_folder: Path, strict: bool) -> None:
    """
    Generate and write typing annotations for the library

    Parameters
    ----------
    output_folder : Path
        path within a package that gets updated

    strict : bool
        whether to generate strict types or allow conversions

    Returns
    -------
    None
    """
    generate_template = _generate_strict_template if strict else _generate_union_template
    for dimensions in _DIMENSION_TYPES:
        contents = "\n".join(_annotate_type(dimensions, type_name, strict) for type_name in _DATA_TYPES)
        all_types = "\n".join(
            _indent(f"{_quote(full_type_name)},") for full_type_name in _list_all_types(dimensions, strict)
        )
        filename = output_folder / _DIMENSIONS_TO_FILENAME[dimensions]
        print(f"Writing {filename}..")
        with open(filename, "w") as f:
            f.write(generate_template(dimensions, contents, all_types))


_DATA_TYPES: Final = {
    "": "None",
    "Int64": "np.int64",
    "Int32": "np.int32",
    "Int16": "np.int16",
    "Int8": "np.int8",
    "Uint64": "np.uint64",
    "Uint32": "np.uint32",
    "Uint16": "np.uint16",
    "Uint8": "np.uint8",
    "FpLongDouble": "np.longdouble",
    "Fp64": "np.float64",
    "Fp32": "np.float32",
    "Fp16": "np.float16",
    "ComplexLongDouble": "np.clongdouble",
    "Complex128": "np.complex128",
    "Complex64": "np.complex64",
    "Bool": "np.bool_",
    "Datetime64": "np.datetime64",
    "Timedelta64": "np.timedelta64",
}

_DIMENSION_TYPES: Final = {
    0: "Any",
    1: "tuple[int]",
    2: "tuple[int, int]",
    3: "tuple[int, int, int]",
}

_DIMENSIONS_TO_PREFIX: Final = {
    0: "NDArray",
    1: "1DArray",
    2: "2DArray",
    3: "3DArray",
}

_DIMENSIONS_TO_FILENAME: Final = {
    0: "n_dimensional.py",
    1: "i_dimensional.py",
    2: "ii_dimensional.py",
    3: "iii_dimensional.py",
}

_SPACES: Final = "    "


def _unindent(text: str) -> str:
    return "\n".join(line.removeprefix(_SPACES) for line in text.split("\n"))


def _indent(text: str) -> str:
    return "\n".join(f"{_SPACES}{line}" for line in text.split("\n"))


def _quote(text: str) -> str:
    return f'"{text}"'


def _type_name_with_prefix(dimensions: int, type_name: str, strict: bool) -> str:
    strict_prefix = "Strict" if strict else ""
    dimension_prefix = _DIMENSIONS_TO_PREFIX[dimensions]
    return f"Np{strict_prefix}{dimension_prefix}{type_name}"


def _strict_type(dimension_type: str, dtype: str) -> str:
    return f"np.ndarray[{dimension_type}, np.dtype[{dtype}]]"


def _union_type(dimension_type: str, dtype: str) -> str:
    return f"Union[np.ndarray[{dimension_type}, np.dtype[{dtype}]], FilePath, MultiArrayNumpyFile]"


def _annotate_type(dimensions: int, type_name: str, strict: bool) -> str:
    dimension_type = _DIMENSION_TYPES[dimensions]
    type_with_prefix = _type_name_with_prefix(dimensions, type_name, strict)
    data_type = _DATA_TYPES[type_name]
    if data_type == "None" and strict:
        return ""

    dtype = "Any" if data_type == "None" else data_type
    T = _strict_type(dimension_type, dtype) if strict else _union_type(dimension_type, dtype)
    dim = dimensions if dimensions > 0 else None
    annotation = f"""{type_with_prefix} = Annotated[
        {T},
        NpArrayPydanticAnnotation.factory(data_type={data_type}, dimensions={dim}, strict_data_typing={strict}),
    ]
"""
    return _unindent(annotation)


def _generate_strict_template(dimensions: int, contents: str, all_types: str) -> str:
    from_typing = "from typing import Annotated, Any" if dimensions == 0 else "from typing import Annotated"
    template = f"""{from_typing}

import numpy as np

from pydantic_numpy.helper.annotation import NpArrayPydanticAnnotation

{contents.strip()}

__all__ = [
{all_types}
]
"""
    return template


def _generate_union_template(dimensions: int, contents: str, all_types: str) -> str:
    template = f"""from typing import Annotated, Any, Union

import numpy as np
from pydantic import FilePath

from pydantic_numpy.helper.annotation import NpArrayPydanticAnnotation
from pydantic_numpy.model import MultiArrayNumpyFile

{contents.strip()}

__all__ = [
{all_types}
]
"""
    return template


def _list_all_types(dimensions: int, strict: bool) -> list[str]:
    return [
        _type_name_with_prefix(dimensions, type_name, strict)
        for type_name in _DATA_TYPES
        if not (type_name == "" and strict)
    ]


if __name__ == "__main__":
    write_annotations(Path("pydantic_numpy/typing"), strict=False)
    write_annotations(Path("pydantic_numpy/typing/strict_data_type"), strict=True)
