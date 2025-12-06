#!/bin/bash
#
# Cross-host determinism validation script
#
# Validates current host against golden data from a DIFFERENT (reference) host.
# This is the key test for detecting cross-host divergence.
#
# Usage:
#   # First, create golden on a known-good reference host:
#   ./scripts/multihost_validate.sh /shared/validation_results
#   # This creates /shared/validation_results/GPU-xxxx/record/
#
#   # Then on OTHER hosts, validate against that golden:
#   ./scripts/crosshost_validate.sh /shared/validation_results/GPU-xxxx/record /shared/validation_results
#
#   # Or specify the reference host by UUID:
#   GOLDEN_HOST=GPU-abc123... ./scripts/crosshost_validate.sh auto /shared/validation_results
#
# Environment variables:
#   GOLDEN_HOST   - GPU UUID of reference host (for auto mode)
#   STEPS         - Number of steps to run (default: 250)
#   MODEL_SIZE    - Model size: small, medium, large (default: large)
#   NPROC         - Number of GPUs per node (default: 8)
#

set -euo pipefail

# Determinism settings
export CUBLAS_WORKSPACE_CONFIG=:4096:8        # cuBLAS reproducibility
export TORCHINDUCTOR_COMPILE_THREADS=1        # Single-threaded torch.compile for reproducible kernels

GOLDEN_DIR="${1:?Usage: $0 <golden_dir|auto> <output_base>}"
OUTPUT_BASE="${2:?Usage: $0 <golden_dir|auto> <output_base>}"
STEPS="${STEPS:-250}"  # Explicit step count for reproducibility
MODEL_SIZE="${MODEL_SIZE:-large}"  # Match production (malibu_v2_mini: 32 layers, 4096 dim)
NPROC="${NPROC:-8}"

# Get first GPU UUID for unique directory name
get_gpu_uuid() {
    nvidia-smi --query-gpu=uuid --format=csv,noheader -i 0
}

HOSTNAME=$(hostname -s)
GPU_UUID=$(get_gpu_uuid)

if [[ -z "$GPU_UUID" ]]; then
    echo "[ERROR] Failed to get GPU UUID. Is nvidia-smi available?"
    exit 1
fi

# Handle "auto" mode - find golden from GOLDEN_HOST
if [[ "$GOLDEN_DIR" == "auto" ]]; then
    if [[ -z "${GOLDEN_HOST:-}" ]]; then
        echo "[ERROR] GOLDEN_HOST not set. Required for auto mode."
        echo "Usage: GOLDEN_HOST=GPU-xxx ./scripts/crosshost_validate.sh auto /shared/results"
        exit 1
    fi
    GOLDEN_DIR="${OUTPUT_BASE}/${GOLDEN_HOST}/record"
fi

# Verify golden directory exists
if [[ ! -d "$GOLDEN_DIR" ]]; then
    echo "[ERROR] Golden directory not found: ${GOLDEN_DIR}"
    exit 1
fi

# Check if we're validating against ourselves (warning)
GOLDEN_GPU_UUID=$(basename "$(dirname "$GOLDEN_DIR")")
if [[ "$GPU_UUID" == "$GOLDEN_GPU_UUID" ]]; then
    echo "[WARNING] Validating against golden from THIS host."
    echo "          For cross-host validation, use golden from a different host."
    echo
fi

# Create host-specific output directory
HOST_DIR="${OUTPUT_BASE}/${GPU_UUID}"
mkdir -p "${HOST_DIR}"

# Logging
LOG_FILE="${HOST_DIR}/crosshost_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG_FILE") 2>&1

echo "============================================================"
echo "Cross-host Determinism Validation"
echo "============================================================"
echo "Timestamp:       $(date -Iseconds)"
echo "This Host:       ${HOSTNAME} (${GPU_UUID})"
echo "Golden From:     ${GOLDEN_GPU_UUID}"
echo "Golden Dir:      ${GOLDEN_DIR}"
echo "Output Dir:      ${HOST_DIR}"
echo "Steps:           ${STEPS}"
echo "Model Size:      ${MODEL_SIZE}"
echo "Num GPUs:        ${NPROC}"
echo "============================================================"
echo

# Step 1: Collect environment fingerprint
echo "[1/2] Collecting environment fingerprint..."
python3 -m torch_validator.env_fingerprint -o "${HOST_DIR}/env_fingerprint.json"
echo "      Saved to: ${HOST_DIR}/env_fingerprint.json"

# Compare with golden host fingerprint if available
GOLDEN_FP="${OUTPUT_BASE}/${GOLDEN_GPU_UUID}/env_fingerprint.json"
if [[ -f "$GOLDEN_FP" ]]; then
    echo
    echo "      Comparing with reference host:"
    python3 -m torch_validator.env_fingerprint \
        --compare "$GOLDEN_FP" "${HOST_DIR}/env_fingerprint.json" \
        2>/dev/null | sed 's/^/      /' || true
fi
echo

# Step 2: Run compile test in VALIDATE mode against cross-host golden
echo "[2/2] Running compile test (VALIDATE against ${GOLDEN_GPU_UUID})..."
VALIDATE_DIR="${HOST_DIR}/crosshost_validate"
mkdir -p "${VALIDATE_DIR}"

torchrun --nproc_per_node=${NPROC} \
    -m torch_validator.stress_tests.runner \
    --test compile \
    --steps ${STEPS} \
    --model-size ${MODEL_SIZE} \
    --validate \
    --golden "${GOLDEN_DIR}" \
    --output "${VALIDATE_DIR}" \
    2>&1 | tee "${HOST_DIR}/crosshost_validate.log"

VALIDATE_STATUS=$?

echo
echo "============================================================"
echo "RESULTS"
echo "============================================================"

if [[ $VALIDATE_STATUS -eq 0 ]]; then
    echo "Status: PASS"
    echo "This host matches the reference host (${GOLDEN_GPU_UUID})"
    echo "PASS: crosshost vs ${GOLDEN_GPU_UUID}" > "${HOST_DIR}/crosshost_status.txt"
else
    echo "Status: FAIL"
    echo "This host DIVERGES from reference host (${GOLDEN_GPU_UUID})"
    echo "FAIL: crosshost vs ${GOLDEN_GPU_UUID}" > "${HOST_DIR}/crosshost_status.txt"

    # Show first few failures
    echo
    echo "First failures:"
    grep -m 5 "MISMATCH\|FAIL" "${HOST_DIR}/crosshost_validate.log" 2>/dev/null | sed 's/^/  /' || true

    # Check if fingerprints differ
    if [[ -f "$GOLDEN_FP" ]]; then
        echo
        echo "Check env_fingerprint differences above for potential causes."
    fi
fi

echo
echo "Output directory: ${HOST_DIR}"
echo "  env_fingerprint.json     - System configuration"
echo "  crosshost_validate/      - Validation results"
echo "  crosshost_validate.log   - Full validation output"
echo "  crosshost_status.txt     - PASS/FAIL status"
echo
echo "Completed at: $(date -Iseconds)"

exit $VALIDATE_STATUS
