.PHONY: setup demo reproduce portfolio-assets test lint audit clean

PYTHON ?= python3

setup:
	uv sync --all-extras

demo:
	uv run python scripts/generate_demo_data.py
	uv run python -m defi_security run --profile demo

reproduce:
	uv run python -m defi_security run --profile portfolio --robustness

portfolio-assets:
	uv run python scripts/generate_portfolio_assets.py

test:
	uv run pytest --cov=defi_security --cov-report=term-missing

lint:
	uv run ruff check src tests scripts
	uv run python scripts/check_readme_metrics.py

audit:
	uv run pip-audit

clean:
	rm -rf artifacts/demo .coverage htmlcov .pytest_cache .ruff_cache
