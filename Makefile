# Kinetra Makefile

.PHONY: setup setup-mt5 install test lint format run clean branch-status branch-setup branch-sync help

help:
	@echo "Kinetra Development Commands"
	@echo "  make setup         - Full dev environment setup"
	@echo "  make setup-mt5     - Install MetaTrader 5 via Wine"
	@echo "  make install       - Install Python dependencies"
	@echo "  make test          - Run test suite"
	@echo "  make branch-status - Show git branch status"
	@echo "  make branch-setup  - Set up local main branch"
	@echo "  make branch-sync   - Sync current branch with remote"

setup:
	chmod +x scripts/setup_dev_env.sh
	./scripts/setup_dev_env.sh

setup-mt5:
	chmod +x scripts/setup_mt5_wine.sh
	./scripts/setup_mt5_wine.sh

install:
	pip install -r requirements.txt
	pip install -e .

test:
	pytest tests/ -v

lint:
	ruff check kinetra/ tests/

format:
	black kinetra/ tests/

mt5:
	./scripts/run_mt5.sh

branch-status:
	python scripts/branch_manager.py --status

branch-setup:
	python scripts/branch_manager.py --setup

branch-sync:
	python scripts/branch_manager.py --sync

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
