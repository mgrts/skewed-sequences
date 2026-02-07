#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_NAME = skewed-sequences
PYTHON_VERSION = 3.11
PYTHON_INTERPRETER = python3

#################################################################################
# COMMANDS                                                                      #
#################################################################################


## Install Python Dependencies
.PHONY: install
install:
	poetry install

## Delete all compiled Python files
.PHONY: clean
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete

## Lint using flake8, isort, and black (use `make format` to do formatting)
.PHONY: lint
lint:
	poetry run flake8 skewed_sequences tests
	poetry run isort --check --diff --profile black skewed_sequences tests
	poetry run black --check --config pyproject.toml skewed_sequences tests

## Format source code with black and isort
.PHONY: format
format:
	poetry run isort --profile black skewed_sequences tests
	poetry run black --config pyproject.toml skewed_sequences tests

## Run tests
.PHONY: test
test:
	poetry run pytest

## Run pre-commit on all files
.PHONY: pre-commit
pre-commit:
	poetry run pre-commit run --all-files

## Build Docker image
.PHONY: docker-build
docker-build:
	docker build -t $(PROJECT_NAME) .


#################################################################################
# PROJECT RULES                                                                 #
#################################################################################


#################################################################################
# Self Documenting Commands                                                     #
#################################################################################

.DEFAULT_GOAL := help

define PRINT_HELP_PYSCRIPT
import re, sys; \
lines = '\n'.join([line for line in sys.stdin]); \
matches = re.findall(r'\n## (.*)\n[\s\S]+?\n([a-zA-Z_-]+):', lines); \
print('Available rules:\n'); \
print('\n'.join(['{:25}{}'.format(*reversed(match)) for match in matches]))
endef
export PRINT_HELP_PYSCRIPT

help:
	@$(PYTHON_INTERPRETER) -c "${PRINT_HELP_PYSCRIPT}" < $(MAKEFILE_LIST)
