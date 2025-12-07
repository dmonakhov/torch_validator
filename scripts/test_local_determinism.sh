#!/bin/bash
#
# Local GPU determinism test - all GPUs run identical workload
#
# Each GPU initializes with the SAME seed, runs the SAME model,
# with the SAME input. After each step, checksums are compared
# via all_reduce. If torch.compile is deterministic, all GPUs
# should have identical checksums.
#
# Usage:
#   ./scripts/test_local_determinism.sh
#
#   # With custom settings
#   STEPS=500 ./scripts/test_local_determinism.sh
#   NPROC=4 ./scripts/test_local_determinism.sh
#   NO_COMPILE=1 ./scripts/test_local_determinism.sh  # baseline without torch.compile

set -euo pipefail

# Determinism settings
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export TORCHINDUCTOR_COMPILE_THREADS=1

# Configuration
NPROC="${NPROC:-8}"
STEPS="${STEPS:-250}"
MODEL_SIZE="${MODEL_SIZE:-large}"
NO_COMPILE="${NO_COMPILE:-0}"

# Build arguments
ARGS="--steps $STEPS --model-size $MODEL_SIZE"
if [[ "$NO_COMPILE" == "1" ]]; then
    ARGS="$ARGS --no-compile"
fi

echo "============================================================"
echo "Local GPU Determinism Test"
echo "============================================================"
echo "GPUs:         $NPROC"
echo "Steps:        $STEPS"
echo "Model:        $MODEL_SIZE"
echo "Compile:      $([ "$NO_COMPILE" == "1" ] && echo "disabled" || echo "enabled")"
echo "============================================================"
echo

torchrun --nproc_per_node=$NPROC \
    -m torch_validator.stress_tests.test_compile_local \
    $ARGS
