# Torch Validator

Hardware validation for LLM training via golden reference comparison.

Detects silent GPU corruption by comparing training metrics against a known-good reference.

## Quick Start

```bash
# Install
pip install -e .

# Step 1: Record golden data on known-good host
python examples/train_with_validation.py --mode record --output golden.json --steps 100

# Step 2: Validate on suspected host
python examples/train_with_validation.py --mode validate --golden golden.json --steps 100
```

## Usage

### Record Mode (known-good host)

```python
from torch_validator import GoldenVerifier, set_deterministic_mode

# Enable deterministic mode for reproducibility
seed_state = set_deterministic_mode(seed=42)

# Create verifier in record mode
verifier = GoldenVerifier(
    mode="record",
    output_file="golden.json",
    seed_state=seed_state,
)

for step in range(num_steps):
    # ... training code ...
    loss.backward()

    # Record metrics (after backward, before optimizer.step)
    verifier.record(step, loss=loss, model=model)

    optimizer.step()

# Save golden data
verifier.save()
```

### Validate Mode (suspected host)

```python
from torch_validator import GoldenVerifier, set_deterministic_mode

# Use same seed as recording
set_deterministic_mode(seed=42)

# Create verifier in validate mode
verifier = GoldenVerifier(
    mode="validate",
    golden_file="golden.json",
)

for step in range(num_steps):
    # ... training code ...
    loss.backward()

    # Validate metrics against golden reference
    verifier.validate(step, loss=loss, model=model)

    optimizer.step()

# Generate report
result = verifier.report()
if not result.passed:
    print("HARDWARE VALIDATION FAILED")
    exit(1)
```

## What It Validates

For each training step, the verifier compares:

| Metric | Description | Default Tolerance |
|--------|-------------|-------------------|
| Loss | Training loss value | rtol=1e-5, atol=1e-8 |
| Grad Norm | L2 norm of all gradients | rtol=1e-4, atol=1e-8 |
| Weight Norm | L2 norm of all weights | rtol=1e-5, atol=1e-8 |
| Checksum | MD5 hash of model parameters | Exact match |

## TorchTitan Integration

See `examples/torchtitan_integration.py` for integration with TorchTitan training scripts.

Minimal changes required:

```python
# 1. Import
from torch_validator import GoldenVerifier, set_deterministic_mode

# 2. Initialize (before training loop)
if validation_mode == "record":
    seed_state = set_deterministic_mode(seed=42)
    verifier = GoldenVerifier(mode="record", output_file="golden.json")
elif validation_mode == "validate":
    set_deterministic_mode(seed=42)
    verifier = GoldenVerifier(mode="validate", golden_file="golden.json")

# 3. In training loop (after backward)
if verifier:
    verifier.record(step, loss, model)  # or verifier.validate(...)

# 4. After training
if verifier:
    verifier.save()  # or verifier.report()
```

## Configuration

### Tolerance Configuration

```python
from torch_validator.verifier import ToleranceConfig

# Stricter tolerances for debugging
strict = ToleranceConfig(
    loss_rtol=1e-6,
    grad_norm_rtol=1e-5,
    weight_norm_rtol=1e-6,
    checksum_exact=True,
)

# Looser tolerances for FP16/BF16
loose = ToleranceConfig(
    loss_rtol=1e-3,
    grad_norm_rtol=1e-2,
    weight_norm_rtol=1e-3,
    checksum_exact=False,  # Checksums may differ with mixed precision
)

verifier = GoldenVerifier(mode="validate", golden_file="golden.json", tolerance=loose)
```

### Deterministic Mode

For accurate validation, enable deterministic mode before training:

```python
from torch_validator import set_deterministic_mode

# Sets:
# - torch.manual_seed(seed)
# - torch.cuda.manual_seed_all(seed)
# - torch.backends.cudnn.deterministic = True
# - torch.backends.cudnn.benchmark = False
# - torch.use_deterministic_algorithms(True)
seed_state = set_deterministic_mode(seed=42)
```

Also set environment variables before importing torch:

```bash
export CUBLAS_WORKSPACE_CONFIG=:4096:8
```

## Golden Data Format

Golden data is stored as JSON:

```json
{
  "version": "1.0",
  "seed_state": {
    "python_seed": 42,
    "torch_seed": 42,
    "cuda_seed": 42
  },
  "tolerance": {
    "loss_rtol": 1e-5,
    "grad_norm_rtol": 1e-4,
    "checksum_exact": true
  },
  "steps": [
    {
      "step": 0,
      "loss": 2.345678,
      "grad_norm": 1.234567,
      "weight_norm": 12.345678,
      "model_checksum": "abc123..."
    }
  ]
}
```

## Validation Report

```
============================================================
VALIDATION REPORT
============================================================
Status: FAIL
Total steps: 100
Failed steps: 3
NaN/Inf errors: 0

Deviations:
  Step 50: checksum - expected=abc123..., actual=def456..., rel_err=None
  Step 51: loss - expected=2.345678, actual=2.567890, rel_err=0.0947
  Step 52: grad_norm - expected=1.234567, actual=1.567890, rel_err=0.2700
============================================================
```

## Testing

```bash
# Run tests
pytest tests/ -v

# Test corruption detection
python examples/train_with_validation.py --mode record --output golden.json --steps 50
python examples/train_with_validation.py --mode validate --golden golden.json --steps 50 --inject-error
```

## Stress Tests

Standalone stress tests to expose hardware faults without full training setup.

### Available Tests

| Test | Purpose | Distributed |
|------|---------|-------------|
| `memory` | HBM bandwidth saturation | No |
| `nccl_allreduce` | AllReduce collective stress | Yes |
| `nccl_mixed` | Combined NCCL patterns | Yes |
| `fsdp_pattern` | FSDP communication simulation | Yes |
| `transformer` | Minimal LLaMA3-like model + FSDP | Yes |

### Usage

```bash
# Record golden on known-good node
torchrun --nproc_per_node=8 -m torch_validator.stress_tests.runner \
    --test transformer --mode quick --output golden/

# Validate on suspected node
torchrun --nproc_per_node=8 -m torch_validator.stress_tests.runner \
    --test transformer --mode quick --validate --golden golden/

# Run quick validation suite
torchrun --nproc_per_node=8 -m torch_validator.stress_tests.runner \
    --test quick_suite --mode quick --validate --golden golden/

# Single-GPU memory test (no distributed)
python -m torch_validator.stress_tests.runner --test memory --mode quick --output golden/

# With GPU monitoring (requires pynvml)
torchrun --nproc_per_node=8 -m torch_validator.stress_tests.runner \
    --test transformer --mode long --nvml --output golden/

# List all available tests
python -m torch_validator.stress_tests.runner --list
```

### Duration Modes

- `--mode quick`: 600 seconds (10 minutes)
- `--mode long`: 3600 seconds (1 hour)
- `--duration N`: Override with custom duration in seconds

### Test Groups

- `all`: All tests
- `quick_suite`: memory, nccl_mixed, fsdp_layer, transformer
- `local`: Single-GPU tests (memory, dense_local)
- `nccl_tests`: All NCCL tests
- `fsdp_tests`: All FSDP tests

## Requirements

- Python >= 3.9
- PyTorch >= 2.0.0
- pynvml (optional, for GPU monitoring)
