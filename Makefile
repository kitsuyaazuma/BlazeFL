format:
	ruff format .

lint:
	ruff check . --fix
	mypy src

test:
	pytest -v tests

stubgen:
	stubgen -m blazefl.core -m blazefl.utils --no-analysis -o src
