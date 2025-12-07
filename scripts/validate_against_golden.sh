#!/bin/bash
#
# Validate against pre-recorded golden data with portable compile cache
#
# Run this on multiple hosts in parallel to validate against shared golden.
# Uses bundled Triton/Inductor caches for cross-host determinism.
#
# Usage:
#   ./scripts/validate_against_golden.sh <golden_dir> [output_dir]
#
# Examples:
#   # Validate against shared golden, results to /tmp
#   ./scripts/validate_against_golden.sh /data/shared/vcjob-v6.golden
#
#   # Validate with custom output directory
#   ./scripts/validate_against_golden.sh /data/shared/vcjob-v6.golden /data/results
#
#   # Custom step count (must match golden)
#   STEPS=500 ./scripts/validate_against_golden.sh /data/shared/vcjob-v6.golden
#
# Output structure:
#   <output_dir>/
#     GPU-abc123.../              # Directory per host (first GPU UUID)
#       env_fingerprint.json      # System/GPU configuration
#       validate/                 # Validation results
#         summary.json
#         ...
#       validate.log              # Full validation output
#       status.txt                # PASS/FAIL status
#

set -euo pipefail

# Determinism settings
export CUBLAS_WORKSPACE_CONFIG=:4096:8        # cuBLAS reproducibility
export TORCHINDUCTOR_COMPILE_THREADS=1        # Single-threaded torch.compile for reproducible kernels

# Configuration
GOLDEN_DIR="${1:?Usage: $0 <golden_dir> [output_dir]}"
OUTPUT_BASE="${2:-/tmp/validate_results}"
STEPS="${STEPS:-250}"  # Must match golden recording
MODEL_SIZE="${MODEL_SIZE:-large}"
NPROC="${NPROC:-8}"

# Verify golden directory exists
if [[ ! -d "$GOLDEN_DIR" ]]; then
    echo "[ERROR] Golden directory not found: $GOLDEN_DIR"
    exit 1
fi

# Verify golden has cache for portable determinism
if [[ ! -d "$GOLDEN_DIR/cache" ]]; then
    echo "[WARNING] No portable cache found in golden directory"
    echo "         Cross-host validation may fail due to compile cache differences"
fi

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
echo "Validate Against Golden (Portable Determinism)"
echo "============================================================"
echo "Timestamp:    $(date -Iseconds)"
echo "Hostname:     ${HOSTNAME}"
echo "GPU UUID:     ${GPU_UUID}"
echo "Golden Dir:   ${GOLDEN_DIR}"
echo "Output Dir:   ${HOST_DIR}"
echo "Steps:        ${STEPS}"
echo "Model Size:   ${MODEL_SIZE}"
echo "Num GPUs:     ${NPROC}"
echo "============================================================"
echo

# Step 1: Collect environment fingerprint
echo "[1/2] Collecting environment fingerprint..."
python3 -m torch_validator.env_fingerprint -o "${HOST_DIR}/env_fingerprint.json"
echo "      Saved to: ${HOST_DIR}/env_fingerprint.json"
echo

# Step 2: Run compile test in VALIDATE mode
echo "[2/2] Running compile test (VALIDATE mode)..."
VALIDATE_DIR="${HOST_DIR}/validate"
mkdir -p "${VALIDATE_DIR}"

torchrun --nproc_per_node=${NPROC} \
    -m torch_validator.stress_tests.runner \
    --test compile \
    --steps ${STEPS} \
    --model-size ${MODEL_SIZE} \
    --validate \
    --golden "${GOLDEN_DIR}" \
    --output "${VALIDATE_DIR}" \
    --portable \
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
    grep -m 5 "MISMATCH\|FAIL\|EARLY MISMATCH" "${HOST_DIR}/validate.log" || true
fi

echo
echo "Output directory: ${HOST_DIR}"
echo "  env_fingerprint.json  - System configuration"
echo "  validate/             - Validation results"
echo "  validate.log          - Validation run output"
echo "  status.txt            - PASS/FAIL status"
echo
echo "Completed at: $(date -Iseconds)"
