

format:
	black .
	isort .
	ruff check . --fix
	@echo "Formatting complete 🎉"

mypy:
	mypy --ignore-missing-imports \
	--follow-imports=skip \
	--strict-optional \
	-p pydantic_numpy
