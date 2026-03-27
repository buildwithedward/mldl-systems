.PHONY: install test lint format docker-build run clean

install:
	pip install -r requirements.txt

test:
	pytest tests/ -v

lint:
	ruff check src/ tests/
	mypy src/ 

format:
	black src/ tests/

docker-build:
	docker build -t neuralcorp-scaffold .

run:
	uvicorn src.main:app --host 0.0.0.0 --port 8000 --reload

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -rf .pytest_cache .mypy_cache .ruff_cache htmlcov