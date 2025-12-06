#!/usr/bin/env python3
"""
CLI runner for stress tests.

Usage:
    # Record golden on known-good node (single GPU)
    python -m torch_validator.stress_tests.runner --test memory --mode quick --output golden/

    # Record golden on known-good node (distributed)
    torchrun --nproc_per_node=8 -m torch_validator.stress_tests.runner \
        --test transformer --mode quick --output golden/

    # Validate on suspected node
    torchrun --nproc_per_node=8 -m torch_validator.stress_tests.runner \
        --test transformer --mode quick --validate --golden golden/

    # Run all tests
    torchrun --nproc_per_node=8 -m torch_validator.stress_tests.runner \
        --test all --mode long --validate --golden golden/
"""

import argparse
import json
import logging
import os
import sys
from datetime import timedelta
from pathlib import Path
from typing import Dict, List, Type

import torch
import torch.distributed as dist

from torch_validator.stress_tests.base import StressTest, StressTestConfig

# Import all test classes
from torch_validator.stress_tests.test_memory import MemoryBandwidthTest, MemoryPatternTest
from torch_validator.stress_tests.test_nccl import (
    NCCLAllReduceTest,
    NCCLAllGatherTest,
    NCCLReduceScatterTest,
    NCCLMixedTest,
)
from torch_validator.stress_tests.test_fsdp import FSDPPatternTest, FSDPLayerTest
from torch_validator.stress_tests.test_transformer import MinimalTransformerTest, MinimalDenseTest
from torch_validator.stress_tests.test_compile import CompileDeterminismTest, CompileDisabledTest, CompileTestConfig

# Registry of available tests
TEST_REGISTRY: Dict[str, Type[StressTest]] = {
    "memory": MemoryBandwidthTest,
    "memory_pattern": MemoryPatternTest,
    "nccl_allreduce": NCCLAllReduceTest,
    "nccl_allgather": NCCLAllGatherTest,
    "nccl_reducescatter": NCCLReduceScatterTest,
    "nccl_mixed": NCCLMixedTest,
    "fsdp_pattern": FSDPPatternTest,
    "fsdp_layer": FSDPLayerTest,
    "transformer": MinimalTransformerTest,
    "dense_local": MinimalDenseTest,
    "compile": CompileDeterminismTest,
    "compile_disabled": CompileDisabledTest,
}

# Test groups for convenience
TEST_GROUPS = {
    "all": list(TEST_REGISTRY.keys()),
    "memory_tests": ["memory", "memory_pattern"],
    "nccl_tests": ["nccl_allreduce", "nccl_allgather", "nccl_reducescatter", "nccl_mixed"],
    "fsdp_tests": ["fsdp_pattern", "fsdp_layer"],
    "model_tests": ["transformer", "dense_local"],
    "compile_tests": ["compile", "compile_disabled"],
    # Quick validation suite (most likely to catch issues)
    "quick_suite": ["memory", "nccl_mixed", "fsdp_layer", "transformer"],
    # Cross-host determinism validation
    "determinism_suite": ["compile", "compile_disabled"],
    # Single-GPU tests (no distributed required)
    "local": ["memory", "memory_pattern", "dense_local"],
}

# Model size presets (to fit different GPU memory)
MODEL_PRESETS = {
    # Small: ~1B params, fits on most GPUs
    "small": {
        "hidden_dim": 2048,
        "num_layers": 16,
        "intermediate_size": 5504,
        "num_heads": 16,
        "num_kv_heads": 4,
        "batch_size": 2,
        "seq_len": 2048,
    },
    # Medium: ~3B params
    "medium": {
        "hidden_dim": 3072,
        "num_layers": 24,
        "intermediate_size": 8192,
        "num_heads": 24,
        "num_kv_heads": 8,
        "batch_size": 2,
        "seq_len": 2048,
    },
    # Large: ~8B params (LLaMA3-8B like), needs H100/A100 80GB
    "large": {
        "hidden_dim": 4096,
        "num_layers": 32,
        "intermediate_size": 14336,
        "num_heads": 32,
        "num_kv_heads": 8,
        "batch_size": 2,
        "seq_len": 4096,
    },
}

logger = logging.getLogger(__name__)


def setup_logging(rank: int):
    """Configure logging with rank prefix."""
    level = logging.INFO
    if os.environ.get("DEBUG"):
        level = logging.DEBUG

    logging.basicConfig(
        level=level,
        format=f"%(asctime)s [%(levelname)s] [rank {rank}] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def setup_distributed(timeout_sec: int = 60):
    """Initialize distributed training if not already initialized.

    Args:
        timeout_sec: Timeout in seconds for NCCL operations (default 60s).
                     Helps debug stuck collectives by raising error instead of hanging.
    """
    if dist.is_initialized():
        return

    # Check if running under torchrun
    if "RANK" in os.environ:
        timeout = timedelta(seconds=timeout_sec)
        dist.init_process_group(backend="nccl", timeout=timeout)
        torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))
        logger.info(
            f"Initialized distributed: rank={dist.get_rank()}, world={dist.get_world_size()}, "
            f"timeout={timeout_sec}s"
        )
    else:
        logger.info("Running in single-GPU mode (no distributed)")


def get_tests_to_run(test_name: str) -> List[str]:
    """Expand test name or group to list of test names."""
    if test_name in TEST_GROUPS:
        return TEST_GROUPS[test_name]
    elif test_name in TEST_REGISTRY:
        return [test_name]
    else:
        raise ValueError(f"Unknown test: {test_name}. Available: {list(TEST_REGISTRY.keys()) + list(TEST_GROUPS.keys())}")


def run_test(test_cls: Type[StressTest], config: StressTestConfig) -> Dict:
    """Run a single test and return results."""
    # Special handling for compile tests - they need CompileTestConfig
    if test_cls in (CompileDeterminismTest, CompileDisabledTest):
        compile_config = CompileTestConfig(
            mode=config.mode,
            duration_override=config.duration_override,
            max_steps=config.max_steps,
            check_interval=config.check_interval,
            seed=config.seed,
            hidden_dim=config.hidden_dim,
            num_layers=config.num_layers,
            intermediate_size=config.intermediate_size,
            num_heads=config.num_heads,
            num_kv_heads=config.num_kv_heads,
            batch_size=config.batch_size,
            seq_len=config.seq_len,
            dtype=config.dtype,
            enable_nvml=config.enable_nvml,
            output_dir=config.output_dir,
            golden_dir=config.golden_dir,
            validate_mode=config.validate_mode,
        )
        test = test_cls(compile_config)
    else:
        test = test_cls(config)

    try:
        return test.run()
    except Exception as e:
        logger.error(f"Test {test_cls.name} failed with exception: {e}")
        return {
            "test": test_cls.name,
            "passed": False,
            "error": str(e),
        }


def main():
    parser = argparse.ArgumentParser(description="Hardware stress test runner")

    # Test selection
    parser.add_argument(
        "--test", "-t",
        default="quick_suite",
        help=f"Test name or group. Available tests: {list(TEST_REGISTRY.keys())}. "
             f"Groups: {list(TEST_GROUPS.keys())}"
    )

    # Duration mode
    parser.add_argument(
        "--mode", "-m",
        choices=["smoke", "quick", "long"],
        default="quick",
        help="Duration mode: smoke (150s), quick (600s), or long (3600s)"
    )
    parser.add_argument(
        "--duration",
        type=int,
        help="Override duration in seconds"
    )
    parser.add_argument(
        "--steps",
        type=int,
        help="Run exactly this many steps (overrides duration)"
    )

    # Validation mode
    parser.add_argument(
        "--validate", "-v",
        action="store_true",
        help="Validate against golden data"
    )
    parser.add_argument(
        "--golden", "-g",
        help="Directory containing golden data"
    )
    parser.add_argument(
        "--output", "-o",
        help="Output directory for recording golden data"
    )

    # Model configuration
    parser.add_argument(
        "--model-size", "-s",
        choices=["small", "medium", "large"],
        default="small",
        help="Model size preset: small (~1B), medium (~3B), large (~8B, needs 80GB GPU)"
    )
    parser.add_argument("--hidden_dim", type=int)
    parser.add_argument("--num_layers", type=int)
    parser.add_argument("--intermediate_size", type=int)
    parser.add_argument("--num_heads", type=int)
    parser.add_argument("--num_kv_heads", type=int)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--seq_len", type=int)
    parser.add_argument("--dtype", default="bfloat16", choices=["float32", "float16", "bfloat16"])

    # Other options
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--check_interval", type=int, default=10)
    parser.add_argument("--nvml", action="store_true", help="Enable NVML GPU monitoring")
    parser.add_argument("--list", action="store_true", help="List available tests and exit")
    parser.add_argument(
        "--timeout", type=int, default=60,
        help="NCCL operation timeout in seconds (default: 60). Raises error instead of hanging."
    )

    args = parser.parse_args()

    # List tests and exit
    if args.list:
        print("Available tests:")
        for name, cls in TEST_REGISTRY.items():
            print(f"  {name}: {cls.__doc__.strip().split(chr(10))[0] if cls.__doc__ else 'No description'}")
        print("\nTest groups:")
        for name, tests in TEST_GROUPS.items():
            print(f"  {name}: {', '.join(tests)}")
        print("\nModel size presets:")
        for name, cfg in MODEL_PRESETS.items():
            print(f"  {name}: hidden={cfg['hidden_dim']}, layers={cfg['num_layers']}, seq_len={cfg['seq_len']}")
        return 0

    # Setup distributed
    setup_distributed(timeout_sec=args.timeout)
    rank = dist.get_rank() if dist.is_initialized() else 0
    setup_logging(rank)

    # Validate arguments
    if args.validate and not args.golden:
        parser.error("--golden required when using --validate")

    # Get model preset and allow overrides
    preset = MODEL_PRESETS[args.model_size]
    model_config = {
        "hidden_dim": args.hidden_dim or preset["hidden_dim"],
        "num_layers": args.num_layers or preset["num_layers"],
        "intermediate_size": args.intermediate_size or preset["intermediate_size"],
        "num_heads": args.num_heads or preset["num_heads"],
        "num_kv_heads": args.num_kv_heads or preset["num_kv_heads"],
        "batch_size": args.batch_size or preset["batch_size"],
        "seq_len": args.seq_len or preset["seq_len"],
    }

    # Build config
    config = StressTestConfig(
        mode=args.mode,
        duration_override=args.duration,
        max_steps=args.steps,
        check_interval=args.check_interval,
        seed=args.seed,
        dtype=args.dtype,
        enable_nvml=args.nvml,
        output_dir=args.output,
        golden_dir=args.golden,
        validate_mode=args.validate,
        **model_config,
    )

    logger.info(f"Model config: {args.model_size} preset, {model_config}")

    # Get tests to run
    test_names = get_tests_to_run(args.test)
    logger.info(f"Running tests: {test_names}")

    # Check if distributed tests can run
    requires_dist = {"nccl_allreduce", "nccl_allgather", "nccl_reducescatter",
                     "nccl_mixed", "fsdp_pattern", "fsdp_layer", "transformer",
                     "compile", "compile_disabled"}
    if not dist.is_initialized():
        test_names = [t for t in test_names if t not in requires_dist]
        if not test_names:
            logger.error("No tests can run without distributed initialization. Use torchrun.")
            return 1
        logger.warning(f"Skipping distributed tests. Running: {test_names}")

    # Run tests
    results = []
    all_passed = True

    for test_name in test_names:
        test_cls = TEST_REGISTRY[test_name]
        logger.info(f"=" * 60)
        logger.info(f"Starting test: {test_name}")
        logger.info(f"=" * 60)

        result = run_test(test_cls, config)
        results.append(result)

        if not result.get("passed", False):
            all_passed = False
            logger.error(f"Test {test_name} FAILED")

            # In validate mode, stop on first failure
            if args.validate:
                logger.error("Stopping due to validation failure")
                break
        else:
            logger.info(f"Test {test_name} PASSED")

        # Sync all ranks between tests
        if dist.is_initialized():
            dist.barrier()

    # Summary
    logger.info("=" * 60)
    logger.info("TEST SUMMARY")
    logger.info("=" * 60)

    for result in results:
        status = "PASS" if result.get("passed") else "FAIL"
        test_name = result.get("test", "unknown")
        if "error" in result:
            logger.info(f"  {test_name}: {status} (error: {result['error']})")
        elif "num_failures" in result:
            logger.info(f"  {test_name}: {status} ({result['num_failures']} failures)")
        else:
            logger.info(f"  {test_name}: {status}")

    # Save summary if output directory specified
    if args.output and rank == 0:
        summary_file = Path(args.output) / "summary.json"
        summary_file.parent.mkdir(parents=True, exist_ok=True)
        with open(summary_file, "w") as f:
            json.dump({
                "config": config.to_dict(),
                "results": results,
                "all_passed": all_passed,
            }, f, indent=2)
        logger.info(f"Summary saved to {summary_file}")

    # Cleanup distributed
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
