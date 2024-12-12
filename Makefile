format:
	uv run ruff format src
	uv run ruff format tests

lint:
	uv run ruff check src --fix
	uv run ruff check tests --fix
	uv run mypy src

test:
	uv run pytest -v tests
