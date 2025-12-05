#!/bin/bash
# Run quick validation suite (most likely to catch issues)
#
# Usage:
#   ./scripts/run_suite.sh record golden/
#   ./scripts/run_suite.sh validate golden/
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
    echo "Recording quick_suite golden to $GOLDEN_DIR (gpus=$NGPU, size=$SIZE)"
    torchrun --nproc_per_node=$NGPU -m torch_validator.stress_tests.runner \
        --test quick_suite \
        --mode quick \
        --model-size "$SIZE" \
        --nvml \
        --output "$GOLDEN_DIR"
elif [ "$MODE" = "validate" ]; then
    echo "Validating quick_suite against $GOLDEN_DIR (gpus=$NGPU, size=$SIZE)"
    torchrun --nproc_per_node=$NGPU -m torch_validator.stress_tests.runner \
        --test quick_suite \
        --mode quick \
        --model-size "$SIZE" \
        --nvml \
        --validate \
        --golden "$GOLDEN_DIR"
else
    echo "Usage: $0 [record|validate] <golden_dir>"
    echo "  Set NGPU=N to change GPU count (default: 8)"
    echo "  Set SIZE=small|medium|large for model size (default: small)"
    exit 1
fi
