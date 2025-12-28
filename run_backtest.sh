#!/bin/bash
#
# Kinetra Physics Backtester Runner
#
# Usage (from WSL2 terminal):
#   ./run_backtest.sh /path/to/your/data/*.csv
#   ./run_backtest.sh /home/renier/QuantumHunter/*.csv
#
# Example:
#   cd /path/to/Kinetra
#   ./run_backtest.sh ~/QuantumHunter/EURUSD_H1.csv
#   ./run_backtest.sh ~/QuantumHunter/*.csv  # All CSV files
#

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}=====================================${NC}"
echo -e "${GREEN} KINETRA PHYSICS BACKTESTER${NC}"
echo -e "${GREEN} Energy-Transfer Theorem v7.0${NC}"
echo -e "${GREEN}=====================================${NC}"

# Check if CSV files provided
if [ $# -eq 0 ]; then
    echo -e "${RED}Error: No CSV files specified${NC}"
    echo ""
    echo "Usage:"
    echo "  ./run_backtest.sh /path/to/data/*.csv"
    echo ""
    echo "Examples:"
    echo "  ./run_backtest.sh ~/QuantumHunter/*.csv"
    echo "  ./run_backtest.sh /home/renier/QuantumHunter/EURUSD_H1.csv"
    echo ""
    exit 1
fi

# Check Python environment
echo -e "\n${YELLOW}Checking environment...${NC}"

if ! command -v python3 &> /dev/null; then
    echo -e "${RED}Error: Python3 not found${NC}"
    exit 1
fi

# Check if dependencies are installed
python3 -c "import backtesting" 2>/dev/null || {
    echo -e "${YELLOW}Installing dependencies...${NC}"
    pip install -r requirements.txt
}

# Count CSV files
CSV_COUNT=0
for f in "$@"; do
    if [ -f "$f" ]; then
        ((CSV_COUNT++))
    fi
done

echo -e "${GREEN}Found $CSV_COUNT CSV file(s) to process${NC}"

# Run the backtest
echo -e "\n${YELLOW}Running backtests...${NC}\n"
python3 scripts/run_full_backtest.py "$@"

echo -e "\n${GREEN}=====================================${NC}"
echo -e "${GREEN} BACKTEST COMPLETE${NC}"
echo -e "${GREEN}=====================================${NC}"
echo ""
echo "Results saved to:"
echo "  - results_*_comparison.png (Strategy comparison charts)"
echo "  - results_*_equity.png (Equity curve charts)"
echo "  - backtest_results_combined.csv (Combined results)"
