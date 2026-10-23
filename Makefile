.PHONY: install lint type test cov check monitor clean

LOGS ?= logs/my-run

install:  ## install runtime and dev dependencies
	uv sync

lint:  ## style and bug checks
	uv run ruff check .

type:  ## static type check
	uv run mypy

test:  ## unit and offline end-to-end tests (no model server or network needed)
	uv run pytest

cov:  ## tests with a coverage floor
	uv run pytest --cov --cov-fail-under=75

check: lint type cov  ## everything CI runs

monitor:  ## live monitor for a run folder: make monitor LOGS=logs/my-run
	uv run reliance monitor $(LOGS)

clean:  ## remove caches and coverage data
	rm -rf .pytest_cache .ruff_cache .mypy_cache .coverage htmlcov
