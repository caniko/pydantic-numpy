from functools import lru_cache

from pydantic import BaseModel


@lru_cache
def get_numpy_type_model(array_type_hint) -> type[BaseModel]:
    class ModelForTesting(BaseModel):
        array_field: array_type_hint

    return ModelForTesting
