#!/bin/bash
# ============================================
# Pipeline Validation Script
# ============================================
# This script runs comprehensive validation of the entire pipeline
# Usage: bash scripts/validate_pipeline.sh <config_file> [options]
# ============================================

set -e

cd "$(dirname "$0")/.."

PROJECT_ROOT="$(pwd)"
VALIDATION_DIR="${PROJECT_ROOT}/validation"
RESULTS_DIR="${PROJECT_ROOT}/results"

# Default values
CONFIG_FILE=""
EXTRACTION_FILE=""
FEATURE_FILE=""
OUTPUT_FILE="${RESULTS_DIR}/validation_report.txt"
DEVICE="cuda"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --config)
            CONFIG_FILE="$2"
            shift 2
            ;;
        --extraction-file)
            EXTRACTION_FILE="$2"
            shift 2
            ;;
        --feature-file)
            FEATURE_FILE="$2"
            shift 2
            ;;
        --output)
            OUTPUT_FILE="$2"
            shift 2
            ;;
        --device)
            DEVICE="$2"
            shift 2
            ;;
        --help)
            echo "Usage: bash scripts/validate_pipeline.sh --config <config_file> [options]"
            echo ""
            echo "Options:"
            echo "  --config <file>          Training config YAML file (required)"
            echo "  --extraction-file <file> Path to extraction file to validate (optional)"
            echo "  --feature-file <file>    Path to feature file to validate (optional)"
            echo "  --output <file>          Output report file (default: results/validation_report.txt)"
            echo "  --device <device>        Device to use (default: cuda)"
            echo "  --help                   Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Check required arguments
if [ -z "$CONFIG_FILE" ]; then
    echo "ERROR: --config is required"
    echo "Use --help for usage information"
    exit 1
fi

if [ ! -f "$CONFIG_FILE" ]; then
    echo "ERROR: Config file not found: $CONFIG_FILE"
    exit 1
fi

# Create output directory
mkdir -p "$(dirname "$OUTPUT_FILE")"

# Run validation
echo "=" * 80
echo "PIPELINE VALIDATION"
echo "=" * 80
echo "Config: $CONFIG_FILE"
echo "Output: $OUTPUT_FILE"
echo "Device: $DEVICE"
if [ -n "$EXTRACTION_FILE" ]; then
    echo "Extraction file: $EXTRACTION_FILE"
fi
if [ -n "$FEATURE_FILE" ]; then
    echo "Feature file: $FEATURE_FILE"
fi
echo "=" * 80
echo ""

python "${VALIDATION_DIR}/pipeline_validator.py" \
    --config "$CONFIG_FILE" \
    ${EXTRACTION_FILE:+--extraction-file "$EXTRACTION_FILE"} \
    ${FEATURE_FILE:+--feature-file "$FEATURE_FILE"} \
    --output "$OUTPUT_FILE" \
    --device "$DEVICE"

EXIT_CODE=$?

echo ""
echo "=" * 80
if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ VALIDATION PASSED"
else
    echo "❌ VALIDATION FAILED"
fi
echo "=" * 80
echo "Report saved to: $OUTPUT_FILE"
echo ""

exit $EXIT_CODE

