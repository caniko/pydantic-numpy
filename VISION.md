# pydantic-numpy vision

pydantic-numpy integrates NumPy arrays into Pydantic models. It provides typed
array annotations (`NpNDArray`, `Np3DArrayFp64`, and the rest of the
`pydantic_numpy.typing` family) that validate dtype and dimensionality on both
`pydantic.BaseModel` and `pydantic.dataclass`, plus a `NumpyModel` base class
that can dump and load arrays alongside ordinary fields. It is used by
Python projects that keep arrays in their data models and still want Pydantic
validation, JSON, and file persistence.

## Problem

Pydantic has no native array type. Passing an `np.ndarray` through a model
either loses validation (any object is accepted), loses serialization (JSON
round-trips become lists and the dtype is gone), or forces every project to
write its own validators and encoders. pydantic-numpy makes the array a
first-class validated field with a stable JSON shape, so a model can be
validated, serialized to JSON, and persisted to disk without each project
inventing the format.

## Principles

- **Validation carries the dtype and rank.** An annotated type states its
  expected dtype and dimensions, and instantiation enforces it; the same
  annotations work on models and dataclasses.
- **One documented JSON shape.** Arrays serialize as
  `{"data_type": ..., "data": [...], "shape": [...]}` and deserialize back to
  the original dtype and shape. The generated JSON Schema is valid and states
  the same contract, so non-Python consumers can read a model's schema.
- **Several ways in, one representation.** A field can be filled from a Python
  list, a `.npy` path, or a key into an `.npz` through `MultiArrayNumpyFile`;
  all three become the same validated array.
- **Files are part of the model contract.** `NumpyModel.dump`/`load` persist a
  model to a directory with compressed pickling, so a saved model round-trips
  including its arrays.
- **Semantic versioning and documented contribution rules.** The project ships
  a contributor guide with GitHub Flow, SemVer and Conventional Commits, and
  publishes through PyPI trusted publishing rather than stored credentials.

## Scope

In scope: the typing families (`type_safe` and `with_loader` variants for 1-, 2-,
3- and N-dimensional arrays), the annotated-typing helper
`np_array_pydantic_annotated_typing`, validation helpers, `NumpyModel` and
`MultiArrayNumpyFile`, YAML/pickle persistence, JSON (de)serialization and
schema generation, and the type-check and JSON-schema test suites.

## Non-goals

- Not a replacement for NumPy I/O. It reads `.npy`/`.npz` and pickles; it does
  not define new array storage formats.
- Not a conversion layer of its own. A mismatched dtype is cast to the annotated
  type (floats rounded when the target is integer) unless `strict_data_typing`
  is set, in which case validation fails.

## Direction

Recent fixes are about correctness at the edges of the JSON shape: preserving
the empty-array shape and turning reshape failures into validation errors
instead of exceptions, followed by packaging and CI maintenance (license file in
the sdist, PyPI trusted publishing, generated workflow pins). The current
branch is `fix/empty-nd-json-and-pydantic-213`, which names the two live
concerns: empty N-dimensional JSON round-trips and Pydantic 2.13 compatibility.

## Place in the fleet

Atlas lists the repository as a Simit updater target but disables its
enrollment because the default branch requires pull requests, and `oul_analysis`
depends on `pydantic-numpy` from PyPI as part of the scientific Python stack.
