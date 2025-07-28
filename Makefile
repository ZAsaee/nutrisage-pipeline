#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_NAME = nutrisage-mlops
PYTHON_VERSION = 3.11
PYTHON_INTERPRETER = python

#################################################################################
# COMMANDS                                                                      #
#################################################################################

## Install Python dependencies
.PHONY: requirements
requirements:
	$(PYTHON_INTERPRETER) -m pip install -U pip
	$(PYTHON_INTERPRETER) -m pip install -r requirements.txt

## Delete all compiled Python files
.PHONY: clean
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete

## Lint using flake8, black, and isort
.PHONY: lint
lint:
	flake8 src
	isort --check --diff src
	black --check src

## Format source code with black
.PHONY: format
format:
	isort src
	black src

## Run tests
.PHONY: test
test:
	$(PYTHON_INTERPRETER) -m pytest tests -v

## Run API server
.PHONY: api
api: requirements
	$(PYTHON_INTERPRETER) -m uvicorn src.api.app:app --host 0.0.0.0 --port 8000 --reload

## Build and run with Docker
.PHONY: docker-build
docker-build:
	docker build -t nutrisage-api .

.PHONY: docker-run
docker-run: docker-build
	docker-compose up -d

.PHONY: docker-stop
docker-stop:
	docker-compose down

## Create necessary directories
.PHONY: setup
setup:
	mkdir -p data/raw data/processed models reports/figures

## Run specific test files
.PHONY: test-data
test-data:
	python -m pytest tests/test_data_pipeline.py -v

.PHONY: test-preprocessing
test-preprocessing:
	python -m pytest tests/test_preprocessing_pipeline.py -v

.PHONY: test-training
test-training:
	python -m pytest tests/test_training_pipeline.py -v

.PHONY: test-prediction
test-prediction:
	python -m pytest tests/test_prediction_pipeline.py -v

#################################################################################
# PROJECT RULES                                                                 #
#################################################################################

## Load and process dataset
.PHONY: data
data: requirements
	$(PYTHON_INTERPRETER) -m src.dataset

## Preprocess data
.PHONY: preprocess
preprocess: requirements
	$(PYTHON_INTERPRETER) -m src.preprocessing

## Train the nutrition grade model
.PHONY: train
train: requirements
	$(PYTHON_INTERPRETER) -m src.modeling.train

## Train with preprocessing pipeline
.PHONY: train-full
train-full: requirements
	$(PYTHON_INTERPRETER) -m src.modeling.train --use-preprocessing-pipeline --sample-fraction 0.1

## Train with hyperparameter tuning
.PHONY: train-tune
train-tune: requirements
	$(PYTHON_INTERPRETER) -m src.modeling.train --use-preprocessing-pipeline --sample-fraction 0.1 --tune

## Make predictions with trained model
.PHONY: predict
predict: requirements
	$(PYTHON_INTERPRETER) -m src.modeling.predict

## Generate model evaluation plots
.PHONY: plots
plots: requirements
	$(PYTHON_INTERPRETER) -m src.plots

## Run complete pipeline (preprocess -> train -> predict)
.PHONY: pipeline
pipeline: setup requirements
	$(PYTHON_INTERPRETER) -m src.preprocessing
	$(PYTHON_INTERPRETER) -m src.modeling.train --use-preprocessing-pipeline
	$(PYTHON_INTERPRETER) -m src.modeling.predict

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
