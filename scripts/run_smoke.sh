#!/bin/bash
# Smoke test - full suite in 10 minutes (150s per test)
#
# Usage:
#   ./scripts/run_smoke.sh record golden/
#   ./scripts/run_smoke.sh validate golden/
#
# Environment:
#   NGPU=8         # Number of GPUs (default: 8)
#   SIZE=small     # Model size: small, medium, large (default: small)

set -e

MODE=${1:-record}
GOLDEN_DIR=${2:-golden}
NGPU=${NGPU:-8}
SIZE=${SIZE:-small}

if [ "$MODE" = "record" ]; then
    echo "Recording smoke test golden to $GOLDEN_DIR (gpus=$NGPU, size=$SIZE, ~10min total)"
    torchrun --nproc_per_node=$NGPU -m torch_validator.stress_tests.runner \
        --test quick_suite \
        --mode smoke \
        --model-size "$SIZE" \
        --nvml \
        --output "$GOLDEN_DIR"
elif [ "$MODE" = "validate" ]; then
    echo "Validating smoke test against $GOLDEN_DIR (gpus=$NGPU, size=$SIZE, ~10min total)"
    torchrun --nproc_per_node=$NGPU -m torch_validator.stress_tests.runner \
        --test quick_suite \
        --mode smoke \
        --model-size "$SIZE" \
        --nvml \
        --validate \
        --golden "$GOLDEN_DIR"
else
    echo "Usage: $0 [record|validate] <golden_dir>"
    echo "  Runs quick_suite (4 tests) at 150s each = ~10 minutes total"
    echo "  Set NGPU=N to change GPU count (default: 8)"
    echo "  Set SIZE=small|medium|large for model size (default: small)"
    exit 1
fi
