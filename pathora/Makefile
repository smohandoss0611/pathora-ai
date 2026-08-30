.PHONY: install test lint typecheck run-api run-ui demo docker

install:
	pip install -e ".[dev,ui]"

test:
	pytest -q

lint:
	ruff check src tests
	ruff format --check src tests

typecheck:
	mypy src/pathora

run-api:
	uvicorn pathora.api.main:app --reload --port 8000

run-ui:
	streamlit run src/pathora/ui/app.py

demo:
	python scripts/demo.py

docker:
	docker compose up --build
