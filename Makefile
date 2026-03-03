# Kinetra Makefile - Simplified Renko Brick Trade Engine
# ========================================================

.PHONY: help setup install test lint format clean launch \
        download dsp backtest full paper live

# Default symbol
SYMBOL ?= XAUUSD

help:
	@echo "Kinetra Renko Brick Trade Engine"
	@echo "================================"
	@echo ""
	@echo "SEQUENTIAL PIPELINE (same engine, all stages):"
	@echo ""
	@echo "  make download         - Download M1 historical data"
	@echo "  make dsp              - DSP analysis → find brick size"
	@echo "  make backtest         - Quick backtest (3 months)"
	@echo "  make full             - Full backtest (3 years, rolling OOS)"
	@echo "  make paper            - Paper trading (simulated)"
	@echo "  make live             - Live trading (micro → scaled)"
	@echo "  make all              - Run complete pipeline"
	@echo ""
	@echo "USAGE:"
	@echo "  make all SYMBOL=XAUUSD"
	@echo "  make backtest SYMBOL=XAUUSD MONTHS=6"
	@echo "  make live SYMBOL=XAUUSD SIZE=scaled"
	@echo ""
	@echo "PHILOSOPHY:"
	@echo "  - Same engine through all stages"
	@echo "  - No new code in live trading"
	@echo "  - All changes → backtest → paper → live"

# ═══════════════════════════════════════════════════════════════════════════════
# SEQUENTIAL PIPELINE
# ═══════════════════════════════════════════════════════════════════════════════

all:
	python scripts/renko_engine.py $(SYMBOL) --stage all

download:
	python scripts/renko_engine.py $(SYMBOL) --stage download

dsp:
	python scripts/renko_engine.py $(SYMBOL) --stage dsp

backtest:
	python scripts/renko_engine.py $(SYMBOL) --stage backtest --months $(MONTHS)

full:
	python scripts/renko_engine.py $(SYMBOL) --stage full

paper:
	python scripts/renko_engine.py $(SYMBOL) --stage paper

live:
	python scripts/renko_engine.py $(SYMBOL) --stage live --live-size $(SIZE)

# ═══════════════════════════════════════════════════════════════════════════════
# UTILITY
# ═══════════════════════════════════════════════════════════════════════════════

launch:
	chmod +x scripts/ctrader/launch.sh
	./scripts/ctrader/launch.sh

setup:
	chmod +x scripts/setup_dev_env.sh
	./scripts/setup_dev_env.sh

install:
	pip install -r requirements.txt
	pip install -e .

test:
	@if [ -d tests ]; then \
		pytest tests/ -v; \
	else \
		echo "tests/ is archived (see archive/production_cleanup_2026-03-03/repo/tests)"; \
	fi

lint:
	@if [ -d tests ]; then \
		ruff check kinetra/ tests/; \
	else \
		ruff check kinetra/; \
	fi

format:
	@if [ -d tests ]; then \
		black kinetra/ tests/; \
	else \
		black kinetra/; \
	fi

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true

# ═══════════════════════════════════════════════════════════════════════════════
# DEFAULTS
# ═══════════════════════════════════════════════════════════════════════════════

MONTHS ?= 3
SIZE ?= micro
