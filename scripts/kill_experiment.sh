#!/bin/bash
# ============================================
# Kill all running experiment processes
# ============================================

echo "Killing all experiment processes..."

# Kill main experiment script
pkill -9 -f "experiment.sh" 2>/dev/null && echo "✓ Killed experiment.sh" || echo "  No experiment.sh found"

# Kill training processes
pkill -9 -f "train.py" 2>/dev/null && echo "✓ Killed train.py processes" || echo "  No train.py processes found"

# Kill feature computation
pkill -9 -f "compute_features.py" 2>/dev/null && echo "✓ Killed compute_features.py" || echo "  No compute_features.py found"

# Kill visualization scripts
pkill -9 -f "generate_visualizations.py" 2>/dev/null && echo "✓ Killed generate_visualizations.py" || echo "  No generate_visualizations.py found"
pkill -9 -f "generate_classification_report.py" 2>/dev/null && echo "✓ Killed generate_classification_report.py" || echo "  No generate_classification_report.py found"
pkill -9 -f "generate_gradcam.py" 2>/dev/null && echo "✓ Killed generate_gradcam.py" || echo "  No generate_gradcam.py found"

# Kill summary table generation
pkill -9 -f "aggregate_results_table.py" 2>/dev/null && echo "✓ Killed aggregate_results_table.py" || echo "  No aggregate_results_table.py found"

# Check for remaining processes
REMAINING=$(ps aux | grep -E "python3.*(train|compute_features|generate_|aggregate_results)" | grep -v grep | wc -l)
if [ "$REMAINING" -gt 0 ]; then
    echo ""
    echo "⚠ Warning: $REMAINING processes still running:"
    ps aux | grep -E "python3.*(train|compute_features|generate_|aggregate_results)" | grep -v grep
    echo ""
    echo "Force killing remaining processes..."
    ps aux | grep -E "python3.*(train|compute_features|generate_|aggregate_results)" | grep -v grep | awk '{print $2}' | xargs kill -9 2>/dev/null
    echo "✓ All processes killed"
else
    echo ""
    echo "✓ All experiment processes stopped"
fi

echo ""
echo "Done!"

