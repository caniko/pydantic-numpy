

test:
    poetry run pytest tests


format:
    poetry run black .
    poetry run isort .
    poetry run ruff check --fix --exit-zero .
    @echo "Formatting complete 🎉"

mypy:
    poetry run mypy

mypy_test:
    poetry run mypy tests/

pyright:
    poetry run pyright pydantic_numpy

pyright_test:
    poetry run pyright tests/

typegen:
    poetry run python typegen/generate_typing.py

check: format pyright mypy test