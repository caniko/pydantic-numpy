from functools import cache

from pydantic import BaseModel


@cache
def cached_calculation(array_type_hint) -> type[BaseModel]:
    class ModelForTesting(BaseModel):
        array_field: array_type_hint

    return ModelForTesting
