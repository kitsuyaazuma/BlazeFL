format:
	ruff format .

lint:
	ruff check . --fix
	uv run mypy .

test:
	uv run pytest -v tests
