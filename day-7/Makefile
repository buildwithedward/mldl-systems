.PHONY: install test lint docker-build clean help run

help:
	@echo "Available commands:"
	@echo "  make install       - Install dependencies"
	@echo "  make test          - Run pytest"
	@echo "  make lint          - Run black and ruff"
	@echo "  make run           - Run the pipeline"
	@echo "  make docker-build  - Build Docker image"
	@echo "  make clean         - Remove output and cache"

install:
	pip install -r requirements.txt

test:
	pytest tests/ -v --tb=short --cov=src --cov-report=term --cov-report=html

lint:
	black src/ tests/
	ruff check src/ tests/ --fix

run:
	python -m src.pipeline

docker-build:
	docker build -t day7-pipeline:latest .

clean:
	rm -rf output/ .pytest_cache/ .coverage htmlcov/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete