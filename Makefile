install:
	pip install --upgrade pip && \
		pip install -r requirements.txt

format:
	isort */*.py && \
		black */*.py

lint:
	pylint --disable=R,C *.py

test:
	python -m pytest -vv --cov=siphotonics

all: install lint format test
