TARGET_FILES=siphotonics/*.py tests/*.py setup.py

install:
	pip install --upgrade pip && \
		pip install -r requirements.txt

format:
	autoflake --in-place --remove-all-unused-imports \
		--remove-unused-variables --expand-star-imports \
		--ignore-init-module-imports \
			$(TARGET_FILES) && \
	isort --filter-files $(TARGET_FILES) && \
	black --line-length=120 $(TARGET_FILES)

lint:
	- pylint --disable=R0911,R0912,C0114,W1514 \
		--max-line-length=120  --output-format=text \
			$(TARGET_FILES) | tee pylint.txt
test:
	python -m pytest -vv --cov=siphotonics --disable-warnings

ci: install lint test

all: install format lint test
