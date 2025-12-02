"""
Torch Validator - Hardware validation for LLM training via golden reference comparison.

Usage:
    # Record mode (on known-good host)
    from torch_validator import GoldenVerifier

    verifier = GoldenVerifier(mode="record", output_file="golden.json")
    for step in range(num_steps):
        loss = train_step(...)
        verifier.record(step, loss=loss, model=model)
    verifier.save()

    # Validate mode (on suspected host)
    verifier = GoldenVerifier(mode="validate", golden_file="golden.json")
    for step in range(num_steps):
        loss = train_step(...)
        verifier.validate(step, loss=loss, model=model)
    result = verifier.report()
    # result.passed -> True/False
"""

from torch_validator.verifier import GoldenVerifier, ValidationResult
from torch_validator.metrics import MetricsCollector
from torch_validator.deterministic import set_seed, set_deterministic_mode, get_seed_state

__version__ = "0.1.0"
__all__ = [
    "GoldenVerifier",
    "ValidationResult",
    "MetricsCollector",
    "set_seed",
    "set_deterministic_mode",
    "get_seed_state",
]
