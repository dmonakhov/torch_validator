"""
Stress tests for hardware validation.

These tests are designed to expose silent GPU/interconnect corruption
by creating sustained memory and communication stress patterns.
"""

from torch_validator.stress_tests.base import StressTest, StressTestConfig
from torch_validator.stress_tests.test_memory import MemoryBandwidthTest
from torch_validator.stress_tests.test_nccl import NCCLAllReduceTest
from torch_validator.stress_tests.test_fsdp import FSDPPatternTest
from torch_validator.stress_tests.test_transformer import MinimalTransformerTest

__all__ = [
    "StressTest",
    "StressTestConfig",
    "MemoryBandwidthTest",
    "NCCLAllReduceTest",
    "FSDPPatternTest",
    "MinimalTransformerTest",
]
