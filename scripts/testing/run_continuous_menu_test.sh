#!/bin/bash
# Continuous Menu Testing Script
# Runs menu tests continuously, logs errors, and tracks failures

LOGDIR="logs/continuous_testing"
mkdir -p "$LOGDIR"

LOGFILE="$LOGDIR/test_$(date +%Y%m%d_%H%M%S).log"
ERRORFILE="$LOGDIR/errors_$(date +%Y%m%d_%H%M%S).log"
STATSFILE="$LOGDIR/stats.json"

echo "Starting continuous menu testing..." | tee -a "$LOGFILE"
echo "Log file: $LOGFILE" | tee -a "$LOGFILE"
echo "Error file: $ERRORFILE" | tee -a "$LOGFILE"
echo "" | tee -a "$LOGFILE"

iteration=0
errors=0
fixes=0

# Test menu options to try
test_sequences=(
    "2\n3\ny\n0\n0"  # Exploration -> Scientific Discovery
    "3\n3\n10\n0\n0"  # Backtesting -> Monte Carlo (10 runs)
    "3\n1\ny\n0\n0"   # Backtesting -> Quick Backtest
    "2\n4\ny\n0\n0"   # Exploration -> Agent Comparison
    "6\n1\n0\n0"      # System Status -> Health
    "5\n3\n0\n0"      # Data Management -> Check Missing
)

while true; do
    iteration=$((iteration + 1))
    
    echo "========================================" | tee -a "$LOGFILE"
    echo "Iteration $iteration - $(date)" | tee -a "$LOGFILE"
    echo "========================================" | tee -a "$LOGFILE"
    
    # Pick a test sequence
    idx=$((iteration % ${#test_sequences[@]}))
    test_seq="${test_sequences[$idx]}"
    
    echo "Testing menu sequence $idx..." | tee -a "$LOGFILE"
    
    # Run the test
    timeout 60 python kinetra_menu.py <<EOF 2>&1 | tee -a "$LOGFILE"
$test_seq
EOF
    
    exit_code=$?
    
    if [ $exit_code -ne 0 ]; then
        errors=$((errors + 1))
        echo "ERROR: Test failed with exit code $exit_code" | tee -a "$ERRORFILE"
        echo "Sequence: $test_seq" | tee -a "$ERRORFILE"
        echo "----------------------------------------" | tee -a "$ERRORFILE"
    else
        echo "✓ Test passed" | tee -a "$LOGFILE"
    fi
    
    # Update stats
    cat > "$STATSFILE" <<STATS
{
  "total_iterations": $iteration,
  "errors": $errors,
  "fixes": $fixes,
  "last_run": "$(date -Iseconds)",
  "error_rate": $(awk "BEGIN {printf \"%.2f\", ($errors / $iteration) * 100}")
}
STATS
    
    echo "" | tee -a "$LOGFILE"
    echo "Stats: Iterations=$iteration, Errors=$errors, Fixes=$fixes" | tee -a "$LOGFILE"
    echo "" | tee -a "$LOGFILE"
    
    # Wait before next iteration
    sleep 2
    
    # Check if we should stop (for now, run 10 iterations)
    if [ $iteration -ge 10 ]; then
        echo "Completed 10 test iterations" | tee -a "$LOGFILE"
        break
    fi
done

echo "" | tee -a "$LOGFILE"
echo "========================================" | tee -a "$LOGFILE"
echo "TESTING COMPLETE" | tee -a "$LOGFILE"
echo "========================================" | tee -a "$LOGFILE"
echo "Total iterations: $iteration" | tee -a "$LOGFILE"
echo "Total errors: $errors" | tee -a "$LOGFILE"
echo "Total fixes: $fixes" | tee -a "$LOGFILE"
echo "Error rate: $(awk "BEGIN {printf \"%.2f%%\", ($errors / $iteration) * 100}")" | tee -a "$LOGFILE"
echo "" | tee -a "$LOGFILE"
echo "Check $ERRORFILE for error details" | tee -a "$LOGFILE"
