# Makefile for NeuralCorp ML Template

.PHONY: help install test run clean docker-build

help:
    @grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

install:
	python3 -m venv .venv
	.venv/bin/python -m pip install --upgrade pip
	.venv/bin/python -m pip install -r requirements.txt

test:
	.venv/bin/python -m pytest tests/ -v --tb=short

lint:
	.venv/bin/python -m flake8 src/ tests/ --max-line-length=100

clean:
	rm -rf .venv __pycache__ .pytest_cache dist build *.egg-info

docker-build:
	docker build -t neuralcorp-ml-template:latest .