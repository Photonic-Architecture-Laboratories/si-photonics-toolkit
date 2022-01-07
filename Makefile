install:
	pip install --upgrade pip && \
		pip install -r requirements.txt

format:
	isort */*.py && \
		black siphotonics/*.py tests/*.py setup.py

lint:
	pylint --disable=R,C,W1514 \
 		siphotonics/*.py tests/*.py

test:
	python -m pytest -vv --cov=siphotonics

all: install lint format test
