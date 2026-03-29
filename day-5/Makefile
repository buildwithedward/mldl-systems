.PHONY: install test lint clean docker-build

install:
	python3.11 -m venv venv
	. venv/bin/activate && pip install --upgrade pip setuptools wheel
	. venv/bin/activate && pip install -r requirements.txt

test:
	. venv/bin/activate && python -m pytest tests/ -v --tb=short --cov=src --cov-report=term --cov-report=html

lint:
	. venv/bin/activate && black src/ tests/ --check
	. venv/bin/activate && ruff check src/ tests/

format:
	. venv/bin/activate && black src/ tests/

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -rf .pytest_cache .mypy_cache .coverage htmlcov

docker-build:
	docker build -t day-5-analyzer:latest .

docker-run:
	docker run --rm -v $(PWD)/data:/app/data day-5-analyzer:latest