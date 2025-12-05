#!/bin/bash
# Single-GPU memory bandwidth test (no distributed)
#
# Usage:
#   ./scripts/run_memory.sh record golden/
#   ./scripts/run_memory.sh validate golden/

set -e

MODE=${1:-record}
GOLDEN_DIR=${2:-golden}
DURATION=${DURATION:-quick}

if [ "$MODE" = "record" ]; then
    echo "Recording memory test golden to $GOLDEN_DIR"
    python -m torch_validator.stress_tests.runner \
        --test memory \
        --mode "$DURATION" \
        --output "$GOLDEN_DIR"
elif [ "$MODE" = "validate" ]; then
    echo "Validating memory test against $GOLDEN_DIR"
    python -m torch_validator.stress_tests.runner \
        --test memory \
        --mode "$DURATION" \
        --validate \
        --golden "$GOLDEN_DIR"
else
    echo "Usage: $0 [record|validate] <golden_dir>"
    echo "  Set DURATION=long for 1-hour test"
    exit 1
fi
