#!/bin/bash
# Quick stress test (10 minutes) - record or validate
#
# Usage:
#   ./scripts/run_quick.sh record golden/    # Record golden data
#   ./scripts/run_quick.sh validate golden/  # Validate against golden
#
# Environment:
#   NGPU=8         # Number of GPUs (default: 8)
#   SIZE=small     # Model size: small, medium, large (default: small)

set -e

MODE=${1:-record}
GOLDEN_DIR=${2:-golden}
TEST=${3:-transformer}
NGPU=${NGPU:-8}
SIZE=${SIZE:-small}

if [ "$MODE" = "record" ]; then
    echo "Recording golden data to $GOLDEN_DIR (test=$TEST, gpus=$NGPU, size=$SIZE)"
    torchrun --nproc_per_node=$NGPU -m torch_validator.stress_tests.runner \
        --test "$TEST" \
        --mode quick \
        --model-size "$SIZE" \
        --output "$GOLDEN_DIR"
elif [ "$MODE" = "validate" ]; then
    echo "Validating against $GOLDEN_DIR (test=$TEST, gpus=$NGPU, size=$SIZE)"
    torchrun --nproc_per_node=$NGPU -m torch_validator.stress_tests.runner \
        --test "$TEST" \
        --mode quick \
        --model-size "$SIZE" \
        --validate \
        --golden "$GOLDEN_DIR"
else
    echo "Usage: $0 [record|validate] <golden_dir> [test_name]"
    echo "  Tests: transformer, memory, nccl_mixed, fsdp_pattern, quick_suite, all"
    echo "  Set SIZE=small|medium|large for model size (default: small)"
    exit 1
fi
