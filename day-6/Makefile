.PHONY: install test lint docker-build clean help

install:
	python3.11 -m venv venv
	. venv/bin/activate && pip install --upgrade pip setuptools wheel
	. venv/bin/activate && pip install -r requirements.txt

test:
	. venv/bin/activate && python -m pytest tests/ -v --tb=short --cov=src --cov-report=term --cov-report=html

lint:
	. venv/bin/activate && python -m pytest --co -q

clean:
	rm -rf venv __pycache__ .pytest_cache .coverage htmlcov
	rm -rf data/* logs/*
	find . -type d -name __pycache__ -exec rm -rf {} +

docker-build:
	docker build -t day-6-pipeline:latest .

help:
	@echo "Available targets:"
	@echo "  make install       - Create venv and install dependencies"
	@echo "  make test          - Run pytest tests with coverage"
	@echo "  make lint          - Run linting"
	@echo "  make clean         - Remove virtual env and cache"
	@echo "  make docker-build  - Build Docker image"