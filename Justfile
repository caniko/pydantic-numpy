test:
    uv run pytest tests

mypy:
    uv run --group type-check mypy .

pyright:
    uv run --group type-check pyright .

typegen:
    uv run python typegen/generate_typing.py
