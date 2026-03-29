.PHONY: install test lint docker-build run-demo

install:
	pip install -r requirements.txt

test:
	pytest tests/ -v --cov=src --cov-report=term-missing

lint:
	python -m py_compile src/*.py tests/*.py && echo "Syntax OK"

docker-build:
	docker build -t nc-004-python-foundations:latest .

run-demo:
	python3 -c "\
from src.model_skeleton import LinearRegressionModel; \
m = LinearRegressionModel(); \
m.fit([1,2,3,4,5],[2,4,6,8,10],learning_rate=0.01,num_steps=300); \
print(f'w={m.weight:.3f} b={m.bias:.3f}  predict(6)={m.predict(6.0):.3f}')"