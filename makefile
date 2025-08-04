sources = src/ scripts/ train.py

format:
	uv run ruff format $(sources)

lint:
	uv run ruff check $(sources) --fix --unsafe-fixes

fix:
	uv run ruff check $(sources) --fix --unsafe-fixes
	uv run ruff format $(sources)

activate:
	source .venv/bin/activate