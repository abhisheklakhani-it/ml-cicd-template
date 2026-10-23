.PHONY: install lint format typecheck test train serve docker-build docker-run check

install:  ## install dependencies and the git hooks
	uv sync
	uv run pre-commit install

lint:
	uv run ruff check .
	uv run ruff format --check .

format:
	uv run ruff check --fix .
	uv run ruff format .

typecheck:
	uv run mypy

test:
	uv run pytest --cov --cov-report=term-missing

check: lint typecheck test  ## everything CI runs

train:
	uv run python -m ml_service.train --out models/model.joblib

serve: train
	uv run uvicorn ml_service.app:app --reload

docker-build:
	docker build -t ml-service:local .

docker-run:
	docker run --rm -p 8000:8000 ml-service:local
