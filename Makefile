# Kinetra Makefile
# Convenient commands for development

.PHONY: setup setup-mt5 install test lint format run clean help

# Default target
help:
	@echo "Kinetra Development Commands"
	@echo ""
	@echo "Setup:"
	@echo "  make setup       - Full dev environment setup (Python + dependencies)"
	@echo "  make setup-mt5   - Install MetaTrader 5 via Wine"
	@echo "  make install     - Install Python dependencies only"
	@echo ""
	@echo "Development:"
	@echo "  make test        - Run test suite"
	@echo "  make lint        - Run linters (ruff, mypy)"
	@echo "  make format      - Format code with black"
	@echo ""
	@echo "Running:"
	@echo "  make run         - Run example/demo"
	@echo "  make mt5         - Launch MetaTrader 5"
	@echo "  make jupyter     - Start Jupyter Lab"
	@echo ""
	@echo "Cleanup:"
	@echo "  make clean       - Remove cache files"

# Setup targets
setup:
	chmod +x scripts/setup_dev_env.sh
	./scripts/setup_dev_env.sh

setup-mt5:
	chmod +x scripts/setup_mt5_wine.sh
	./scripts/setup_mt5_wine.sh

install:
	pip install -r requirements.txt

# Development targets
test:
	pytest tests/ -v --cov=kinetra --cov-report=term-missing

lint:
	ruff check kinetra/ tests/
	mypy kinetra/ --ignore-missing-imports

format:
	black kinetra/ tests/ scripts/
	ruff check --fix kinetra/ tests/

# Running targets
run:
	python -c "from kinetra import PhysicsEngine; print('Kinetra ready!')"

mt5:
	./scripts/run_mt5.sh

jupyter:
	jupyter lab

# Cleanup
clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .mypy_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .ruff_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	find . -type f -name ".coverage" -delete 2>/dev/null || true

# Activate virtual environment reminder
venv:
	@echo "Run: source .venv/bin/activate"
