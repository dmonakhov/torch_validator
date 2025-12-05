#!/bin/bash
#
# Summarize multi-host validation results
#
# Run this after multihost_validate.sh has completed on all hosts.
# Compares environment fingerprints and shows pass/fail status.
#
# Usage:
#   ./scripts/multihost_summary.sh /shared/validation_results
#

set -euo pipefail

OUTPUT_BASE="${1:?Usage: $0 <output_dir>}"

echo "============================================================"
echo "Multi-host Validation Summary"
echo "============================================================"
echo "Results directory: ${OUTPUT_BASE}"
echo "Timestamp: $(date -Iseconds)"
echo

# Count hosts
TOTAL=0
PASSED=0
FAILED=0

echo "Host Status:"
echo "------------"

for host_dir in "${OUTPUT_BASE}"/GPU-*; do
    if [[ ! -d "$host_dir" ]]; then
        continue
    fi

    TOTAL=$((TOTAL + 1))
    GPU_UUID=$(basename "$host_dir")

    # Get hostname and instance ID from fingerprint
    HOSTNAME=""
    INSTANCE_ID=""
    if [[ -f "${host_dir}/env_fingerprint.json" ]]; then
        HOSTNAME=$(python3 -c "import json; d=json.load(open('${host_dir}/env_fingerprint.json')); print(d.get('system',{}).get('hostname',''))" 2>/dev/null || echo "")
        INSTANCE_ID=$(python3 -c "import json; d=json.load(open('${host_dir}/env_fingerprint.json')); print(d.get('system',{}).get('instance_id',''))" 2>/dev/null || echo "")
    fi

    # Get status
    STATUS="UNKNOWN"
    if [[ -f "${host_dir}/status.txt" ]]; then
        STATUS=$(cat "${host_dir}/status.txt")
    fi

    # Count pass/fail
    if [[ "$STATUS" == "PASS" ]]; then
        PASSED=$((PASSED + 1))
        STATUS_DISPLAY="[PASS]"
    elif [[ "$STATUS" == FAIL* ]]; then
        FAILED=$((FAILED + 1))
        STATUS_DISPLAY="[FAIL]"
    else
        STATUS_DISPLAY="[????]"
    fi

    printf "  %-8s %-20s %-20s %s\n" "$STATUS_DISPLAY" "$HOSTNAME" "$INSTANCE_ID" "$GPU_UUID"
done

echo
echo "Summary: ${PASSED}/${TOTAL} passed, ${FAILED} failed"
echo

# Compare fingerprints if there are multiple hosts
if [[ $TOTAL -gt 1 ]]; then
    echo "============================================================"
    echo "Environment Comparison"
    echo "============================================================"
    echo

    # Get first host as reference
    REF_HOST=""
    for host_dir in "${OUTPUT_BASE}"/GPU-*; do
        if [[ -f "${host_dir}/env_fingerprint.json" ]]; then
            REF_HOST="$host_dir"
            break
        fi
    done

    if [[ -n "$REF_HOST" ]]; then
        REF_NAME=$(basename "$REF_HOST")
        echo "Reference: ${REF_NAME}"
        echo

        for host_dir in "${OUTPUT_BASE}"/GPU-*; do
            if [[ "$host_dir" == "$REF_HOST" ]]; then
                continue
            fi
            if [[ ! -f "${host_dir}/env_fingerprint.json" ]]; then
                continue
            fi

            HOST_NAME=$(basename "$host_dir")
            echo "Comparing: ${HOST_NAME}"

            python3 -m torch_validator.env_fingerprint \
                --compare "${REF_HOST}/env_fingerprint.json" "${host_dir}/env_fingerprint.json" \
                2>/dev/null | grep -v "^Comparing:" || true
            echo
        done
    fi
fi

# Show failed hosts details
if [[ $FAILED -gt 0 ]]; then
    echo "============================================================"
    echo "Failed Host Details"
    echo "============================================================"
    echo

    for host_dir in "${OUTPUT_BASE}"/GPU-*; do
        if [[ ! -f "${host_dir}/status.txt" ]]; then
            continue
        fi

        STATUS=$(cat "${host_dir}/status.txt")
        if [[ "$STATUS" != FAIL* ]]; then
            continue
        fi

        GPU_UUID=$(basename "$host_dir")
        echo "Host: ${GPU_UUID}"
        echo "Status: ${STATUS}"

        if [[ -f "${host_dir}/validate.log" ]]; then
            echo "First failures:"
            grep -m 3 "MISMATCH\|FAIL" "${host_dir}/validate.log" 2>/dev/null | sed 's/^/  /' || true
        fi
        echo
    done
fi

echo "============================================================"
echo "Done"
echo "============================================================"
