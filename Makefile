# ==================================================================================== #
# VARIABLES
# ==================================================================================== #
# Fichier .env par défaut
ENV_FILE := .env

# ==================================================================================== #
# HELPERS
# ==================================================================================== #
## help: print this help message
.PHONY: help
help:
	@echo 'Usage:'
	@sed -n 's/^##//p' ${MAKEFILE_LIST} | column -t -s ":" | sed -e 's/^/ /'

# ==================================================================================== #
# TEST
# ==================================================================================== #

## test: test all app in project
.PHONY: test
test:
	PYTHONPATH=. pytest tests/*

# ==================================================================================== #
# FORMAT
# ==================================================================================== #

## black: do black for all project
.PHONY: black
black:
	black .

## black-check: check black for all project
.PHONY: black-check
black-check:
	black --check src/ tests/

# ==================================================================================== #
# FREEZE
# ==================================================================================== #

## freeze: Add dependancies to requirements.txt
.PHONY: freeze
freeze:
	pip freeze > requirements.txt

## freeze-dev: Add dependancies to requirements.txt
.PHONY: freeze-dev
freeze-dev:
	pip freeze > requirements-dev.txt

