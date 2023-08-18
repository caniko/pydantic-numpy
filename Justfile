

format:
	poetry run black .
	poetry run isort .
	poetry run ruff check . --fix
	@echo "Formatting complete 🎉"

mypy:
	poetry run mypy --ignore-missing-imports \
	--follow-imports=skip \
	--strict-optional \
	-p pydantic_numpy
