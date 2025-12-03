test:
    uv run pytest tests

mypy:
    uv run --group type-check mypy src/

mypy_test:
    uv run --group type-check mypy tests/

typegen:
    uv run python typegen/generate_typing.py
