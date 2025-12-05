#!/bin/bash
#
# Multi-host determinism validation script
#
# Run this on multiple hosts in parallel on a shared filesystem.
# Each host creates a unique directory based on its first GPU UUID.
#
# Usage:
#   # On shared FS, from torch_validator directory:
#   ./scripts/multihost_validate.sh /shared/validation_results
#
#   # Or with custom duration (default: 300s = 5min)
#   DURATION=600 ./scripts/multihost_validate.sh /shared/validation_results
#
#   # Or with specific model size (default: large to match production)
#   MODEL_SIZE=small ./scripts/multihost_validate.sh /shared/validation_results
#
# Output structure:
#   /shared/validation_results/
#     GPU-abc123.../              # Directory per host (first GPU UUID)
#       env_fingerprint.json      # System/GPU configuration
#       record/                   # Golden data from record run
#         compile_rank0.golden.json
#         compile_rank0.log.json
#         ...
#       validate/                 # Validation results
#         summary.json
#         ...
#       validate.log              # Full validation output
#       record.log                # Full record output
#       status.txt                # PASS/FAIL status
#

set -euo pipefail

# Determinism: required for cuBLAS reproducibility
export CUBLAS_WORKSPACE_CONFIG=:4096:8

# Configuration
OUTPUT_BASE="${1:?Usage: $0 <output_dir>}"
DURATION="${DURATION:-300}"
MODEL_SIZE="${MODEL_SIZE:-large}"  # Match production (malibu_v2_mini: 32 layers, 4096 dim)
NPROC="${NPROC:-8}"

# Get first GPU UUID for unique directory name
get_gpu_uuid() {
    nvidia-smi --query-gpu=uuid --format=csv,noheader -i 0
}

# Get hostname for logging
HOSTNAME=$(hostname -s)

# Get GPU UUID
GPU_UUID=$(get_gpu_uuid)
if [[ -z "$GPU_UUID" ]]; then
    echo "[ERROR] Failed to get GPU UUID. Is nvidia-smi available?"
    exit 1
fi

# Create host-specific output directory
HOST_DIR="${OUTPUT_BASE}/${GPU_UUID}"
mkdir -p "${HOST_DIR}"

# Logging
LOG_FILE="${HOST_DIR}/run_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG_FILE") 2>&1

echo "============================================================"
echo "Multi-host Determinism Validation"
echo "============================================================"
echo "Timestamp:    $(date -Iseconds)"
echo "Hostname:     ${HOSTNAME}"
echo "GPU UUID:     ${GPU_UUID}"
echo "Output Dir:   ${HOST_DIR}"
echo "Duration:     ${DURATION}s"
echo "Model Size:   ${MODEL_SIZE}"
echo "Num GPUs:     ${NPROC}"
echo "============================================================"
echo

# Step 1: Collect environment fingerprint
echo "[1/3] Collecting environment fingerprint..."
python3 -m torch_validator.env_fingerprint -o "${HOST_DIR}/env_fingerprint.json"
echo "      Saved to: ${HOST_DIR}/env_fingerprint.json"
echo

# Step 2: Run compile test in RECORD mode (create golden)
echo "[2/3] Running compile test (RECORD mode)..."
RECORD_DIR="${HOST_DIR}/record"
mkdir -p "${RECORD_DIR}"

torchrun --nproc_per_node=${NPROC} \
    -m torch_validator.stress_tests.runner \
    --test compile \
    --duration ${DURATION} \
    --model-size ${MODEL_SIZE} \
    --output "${RECORD_DIR}" \
    2>&1 | tee "${HOST_DIR}/record.log"

RECORD_STATUS=$?
if [[ $RECORD_STATUS -ne 0 ]]; then
    echo "[ERROR] Record run failed with status ${RECORD_STATUS}"
    echo "FAIL: record" > "${HOST_DIR}/status.txt"
    exit 1
fi
echo "      Golden data saved to: ${RECORD_DIR}"
echo

# Step 3: Run compile test in VALIDATE mode (check against golden)
echo "[3/3] Running compile test (VALIDATE mode)..."
VALIDATE_DIR="${HOST_DIR}/validate"
mkdir -p "${VALIDATE_DIR}"

torchrun --nproc_per_node=${NPROC} \
    -m torch_validator.stress_tests.runner \
    --test compile \
    --duration ${DURATION} \
    --model-size ${MODEL_SIZE} \
    --validate \
    --golden "${RECORD_DIR}" \
    --output "${VALIDATE_DIR}" \
    2>&1 | tee "${HOST_DIR}/validate.log"

VALIDATE_STATUS=$?

echo
echo "============================================================"
echo "RESULTS"
echo "============================================================"

if [[ $VALIDATE_STATUS -eq 0 ]]; then
    echo "Status: PASS"
    echo "PASS" > "${HOST_DIR}/status.txt"
else
    echo "Status: FAIL"
    echo "FAIL: validate" > "${HOST_DIR}/status.txt"

    # Show first few failures
    echo
    echo "First failures:"
    grep -m 5 "MISMATCH\|FAIL" "${HOST_DIR}/validate.log" || true
fi

echo
echo "Output directory: ${HOST_DIR}"
echo "  env_fingerprint.json  - System configuration"
echo "  record/               - Golden reference data"
echo "  validate/             - Validation results"
echo "  record.log            - Record run output"
echo "  validate.log          - Validation run output"
echo "  status.txt            - PASS/FAIL status"
echo
echo "Completed at: $(date -Iseconds)"
