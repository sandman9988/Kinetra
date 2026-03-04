# Kinetra Makefile - Simplified Renko Brick Trade Engine
# ========================================================

.PHONY: help setup install test lint format clean launch \
        download dsp backtest full dry-run live

# Default symbol
SYMBOL ?= XAUUSD

help:
	@echo "Kinetra Renko Brick Trade Engine"
	@echo "================================"
	@echo ""
	@echo "PIPELINE (same engine and display across all modes):"
	@echo ""
	@echo "  make download         - Download/update M1 historical data + contract specs"
	@echo "  make dsp              - DSP analysis → find brick size"
	@echo "  make backtest         - Quick backtest (3 months, historical CSV)"
	@echo "  make full             - Full backtest (2 years, rolling OOS)"
	@echo "  make dry-run          - Live data, paper orders (same dashboard as live)"
	@echo "  make live             - Live data, real orders"
	@echo "  make all              - Run complete pipeline"
	@echo ""
	@echo "USAGE:"
	@echo "  make all SYMBOL=XAUUSD"
	@echo "  make backtest SYMBOL=XAUUSD MONTHS=6"
	@echo "  make dry-run SYMBOL=XAUUSD GATE=micro"
	@echo "  make live SYMBOL=XAUUSD GATE=micro"
	@echo "  make launch MODE=dry_run SYMBOL=XAUUSD PROFILE=balanced GATE=micro"
	@echo "  make launch MODE=live GATE=micro BRICK_SIZE=1.0 STOP_BRICKS=1.0"
	@echo "  make launch MODE=backtest ENTRY_LATENCY_MS=250 EXIT_LATENCY_MS=250 LATENCY_JITTER_MS=50"
	@echo "  make launch MODE=backtest PROFILE=bayesian BAYES_TRIALS=25 BAYES_MONTHS=3"
	@echo "  make launch MODE=live PROFILE=saved PROFILE_NAME=xauusd_balanced_20260303_180500"
	@echo ""
	@echo "PHILOSOPHY:"
	@echo "  - Same engine, same display through all stages"
	@echo "  - Only data source and dispatcher differ between modes"
	@echo "  - All changes → backtest → dry-run → live"

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

dry-run:
	chmod +x scripts/ctrader/launch.sh
	LAUNCH_MODE=dry_run LAUNCH_SYMBOL=$(SYMBOL) LAUNCH_GATE=$(GATE) ./scripts/ctrader/launch.sh

live:
	chmod +x scripts/ctrader/launch.sh
	LAUNCH_MODE=live LAUNCH_SYMBOL=$(SYMBOL) LAUNCH_GATE=$(GATE) ./scripts/ctrader/launch.sh

# ═══════════════════════════════════════════════════════════════════════════════
# UTILITY
# ═══════════════════════════════════════════════════════════════════════════════

launch:
	chmod +x scripts/ctrader/launch.sh
	LAUNCH_MODE="$(MODE)" \
	LAUNCH_SYMBOL="$(SYMBOL)" \
	LAUNCH_GATE="$(GATE)" \
	LAUNCH_MONTHS_SET="$(origin MONTHS)" \
	LAUNCH_MONTHS="$(MONTHS)" \
	LAUNCH_PROFILE="$(PROFILE)" \
	LAUNCH_PROFILE_NAME="$(PROFILE_NAME)" \
	LAUNCH_BAYES_TRIALS="$(BAYES_TRIALS)" \
	LAUNCH_BAYES_SEED="$(BAYES_SEED)" \
	LAUNCH_BAYES_MONTHS="$(BAYES_MONTHS)" \
	LAUNCH_BRICK_SIZE="$(BRICK_SIZE)" \
	LAUNCH_STOP_BRICKS="$(STOP_BRICKS)" \
	LAUNCH_TRAILING_MFE_FRACTION="$(TRAILING_MFE_FRACTION)" \
	LAUNCH_TRAILING_MFE_AFTER_BRICKS="$(TRAILING_MFE_AFTER_BRICKS)" \
	LAUNCH_PE_ENTRY_THRESHOLD="$(PE_ENTRY_THRESHOLD)" \
	LAUNCH_PE_EXIT_THRESHOLD="$(PE_EXIT_THRESHOLD)" \
	LAUNCH_STALE_BRICK_FACTOR="$(STALE_BRICK_FACTOR)" \
	LAUNCH_MARKOV_STALE_PENALTY="$(MARKOV_STALE_PENALTY)" \
	LAUNCH_USE_OBI_SLIPPAGE_BUFFER="$(USE_OBI_SLIPPAGE_BUFFER)" \
	LAUNCH_OBI_LEVELS="$(OBI_LEVELS)" \
	LAUNCH_OBI_MAX_BUFFER_BRICKS="$(OBI_MAX_BUFFER_BRICKS)" \
	LAUNCH_OBI_EMA_ALPHA="$(OBI_EMA_ALPHA)" \
	LAUNCH_TARGET_RISK="$(TARGET_RISK)" \
	LAUNCH_LOT_CEILING="$(LOT_CEILING)" \
	LAUNCH_MAX_LEVERAGE="$(MAX_LEVERAGE)" \
	LAUNCH_FLIPRATE_THRESHOLD="$(FLIPRATE_THRESHOLD)" \
	LAUNCH_MARKOV_THRESHOLD="$(MARKOV_THRESHOLD)" \
	LAUNCH_MIN_OMEGA="$(MIN_OMEGA)" \
	LAUNCH_MIN_TRADES="$(MIN_TRADES)" \
	LAUNCH_ENTRY_LATENCY_MS="$(ENTRY_LATENCY_MS)" \
	LAUNCH_EXIT_LATENCY_MS="$(EXIT_LATENCY_MS)" \
	LAUNCH_LATENCY_JITTER_MS="$(LATENCY_JITTER_MS)" \
	LAUNCH_LATENCY_SEED="$(LATENCY_SEED)" \
	LAUNCH_CONSERVATIVE_FILLS="$(CONSERVATIVE_FILLS)" \
	LAUNCH_SPREAD_SIDES="$(SPREAD_SIDES)" \
	LAUNCH_ENTRY_SLIP_BRICKS="$(ENTRY_SLIP_BRICKS)" \
	LAUNCH_EXIT_SLIP_BRICKS="$(EXIT_SLIP_BRICKS)" \
	LAUNCH_STOP_WORST_CASE_BRICKS="$(STOP_WORST_CASE_BRICKS)" \
	LAUNCH_FLIP_TRADE_THROUGH_BRICKS="$(FLIP_TRADE_THROUGH_BRICKS)" \
	LAUNCH_MONDAY_START="$(MONDAY_START)" \
	LAUNCH_FRIDAY_END="$(FRIDAY_END)" \
	LAUNCH_AUTO_DOWNLOAD="$(AUTO_DOWNLOAD)" \
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
GATE ?= micro
