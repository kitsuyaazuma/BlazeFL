format:
	uv run ruff format src

lint:
	uv run ruff check src --fix
	uv run mypy src
