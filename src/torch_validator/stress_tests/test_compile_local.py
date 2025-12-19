"""
Local GPU determinism test - all GPUs run identical workload, compare via all_reduce.

Each GPU initializes identically (same seed, same model, same input) and runs
independently. After each step, checksums are compared via all_reduce.
If torch.compile is deterministic, all GPUs should have identical checksums.

This eliminates the need for golden files - divergence is detected in real-time.

Usage:
    torchrun --nproc_per_node=8 -m torch_validator.stress_tests.test_compile_local

    # With custom steps
    torchrun --nproc_per_node=8 -m torch_validator.stress_tests.test_compile_local --steps 500
"""

# CRITICAL: Set CUDA_VISIBLE_DEVICES before importing torch
# This ensures each process sees its GPU as cuda:0, which makes
# Triton/Inductor compile identical kernels across all ranks.
import os
_local_rank = os.environ.get("LOCAL_RANK")
if _local_rank is not None:
    os.environ["CUDA_VISIBLE_DEVICES"] = _local_rank

import argparse
import ctypes
import hashlib
import logging
import socket
import struct
import sys
import time
from dataclasses import dataclass
from typing import List

import torch
import torch.distributed as dist
import torch.nn as nn

from torch_validator._version import get_version_info


def get_gpu_uuids() -> List[str]:
    """Get GPU UUIDs for all devices using pynvml."""
    try:
        import pynvml
        pynvml.nvmlInit()
        count = pynvml.nvmlDeviceGetCount()
        uuids = []
        for i in range(count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            uuid = pynvml.nvmlDeviceGetUUID(handle)
            uuids.append(uuid)
        pynvml.nvmlShutdown()
        return uuids
    except Exception:
        return []

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


@dataclass
class LocalTestConfig:
    """Configuration for local determinism test."""
    seed: int = 42  # Same seed for ALL GPUs
    steps: int = 5000 # for large model runtime is ~1hour
    check_interval: int = 10
    hidden_dim: int = 4096
    num_layers: int = 32
    intermediate_size: int = 14336
    num_heads: int = 32
    num_kv_heads: int = 8
    batch_size: int = 2
    seq_len: int = 4096
    dtype: str = "bfloat16"
    use_compile: bool = True
    no_autotune: bool = False  # Disable autotuning for guaranteed determinism


class RMSNorm(nn.Module):
    """RMSNorm - deterministic normalization."""
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        norm = x.float().pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return (x * norm).type_as(x) * self.weight


class MLP(nn.Module):
    """MLP block matching LLaMA architecture."""
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.gate_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.up_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, dim, bias=False)
        self.act = nn.SiLU()

    def forward(self, x):
        return self.down_proj(self.act(self.gate_proj(x)) * self.up_proj(x))


class TransformerBlock(nn.Module):
    """Transformer block without attention (MLP only for determinism testing)."""
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.norm1 = RMSNorm(dim)
        self.mlp = MLP(dim, hidden_dim)

    def forward(self, x):
        return x + self.mlp(self.norm1(x))


class LocalTestModel(nn.Module):
    """Model for local determinism testing."""
    def __init__(self, config: LocalTestConfig):
        super().__init__()
        self.embed = nn.Linear(config.hidden_dim, config.hidden_dim, bias=False)
        self.layers = nn.ModuleList([
            TransformerBlock(config.hidden_dim, config.intermediate_size)
            for _ in range(config.num_layers)
        ])
        self.norm = RMSNorm(config.hidden_dim)
        self.output = nn.Linear(config.hidden_dim, config.hidden_dim, bias=False)

    def forward(self, x):
        x = self.embed(x)
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        return self.output(x)


def compute_checksum_tensor(tensor: torch.Tensor) -> torch.Tensor:
    """Compute checksum as a tensor (for all_reduce comparison)."""
    t = tensor.detach().float().cpu().contiguous()
    # Get raw bytes via ctypes (no numpy required)
    ptr = t.data_ptr()
    size = t.numel() * t.element_size()
    buffer = (ctypes.c_char * size).from_address(ptr)
    h = hashlib.sha256(buffer).digest()[:8]  # 8 bytes = 64 bits
    # Convert to int64 for all_reduce
    value = struct.unpack('<q', h)[0]
    return torch.tensor([value], dtype=torch.int64, device=tensor.device)


def compute_checksum_str(tensor: torch.Tensor) -> str:
    """Compute checksum as hex string (for logging)."""
    t = tensor.detach().float().cpu().contiguous()
    ptr = t.data_ptr()
    size = t.numel() * t.element_size()
    buffer = (ctypes.c_char * size).from_address(ptr)
    return hashlib.sha256(buffer).hexdigest()[:16]


def set_deterministic_seed(seed: int):
    """Set all seeds for determinism - SAME seed on all GPUs."""
    import random
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # Deterministic algorithms
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Inductor determinism
    try:
        torch._inductor.config.deterministic_algorithms = True
        torch._inductor.config.fallback_random = True
        # Skip on-device benchmarking that causes non-determinism
        torch._inductor.config.deterministic = True
    except (AttributeError, ModuleNotFoundError):
        pass


def disable_autotuning():
    """Disable all inductor autotuning for guaranteed determinism.

    This forces default kernel configs without benchmarking, eliminating
    run-to-run variability from autotune timing. May reduce performance
    by 10-30% but guarantees identical compilation across runs.
    """
    try:
        torch._inductor.config.max_autotune = False
        torch._inductor.config.max_autotune_pointwise = False
        torch._inductor.config.max_autotune_gemm = False
    except (AttributeError, ModuleNotFoundError):
        pass


def check_all_equal(checksum: torch.Tensor, rank: int, world_size: int) -> bool:
    """Check if all ranks have the same checksum using all_reduce."""
    # Gather all checksums
    all_checksums = [torch.zeros_like(checksum) for _ in range(world_size)]
    dist.all_gather(all_checksums, checksum)

    # Check if all equal
    reference = all_checksums[0].item()
    for i, cs in enumerate(all_checksums):
        if cs.item() != reference:
            return False
    return True


def get_divergent_ranks(checksum: torch.Tensor, rank: int, world_size: int) -> list:
    """Get list of ranks that have different checksums."""
    all_checksums = [torch.zeros_like(checksum) for _ in range(world_size)]
    dist.all_gather(all_checksums, checksum)

    reference = all_checksums[0].item()
    divergent = []
    for i, cs in enumerate(all_checksums):
        if cs.item() != reference:
            divergent.append(i)
    return divergent


def run_local_test(config: LocalTestConfig) -> dict:
    """Run local determinism test."""

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    # Each process sees only its GPU as cuda:0 (CUDA_VISIBLE_DEVICES set at import)
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    dtype = getattr(torch, config.dtype)

    # Get GPU UUID for this process (only one GPU visible)
    gpu_uuids = get_gpu_uuids()
    my_uuid = gpu_uuids[0] if gpu_uuids else "unknown"
    # Gather all UUIDs for logging
    all_uuids = [None] * world_size
    dist.all_gather_object(all_uuids, my_uuid)

    if rank == 0:
        logger.info("=" * 60)
        logger.info("Local GPU Determinism Test")
        logger.info("=" * 60)
        logger.info(f"Hostname: {socket.gethostname()}")
        logger.info(f"World size: {world_size}")
        logger.info(f"Steps: {config.steps}")
        logger.info(f"Seed: {config.seed} (same on all GPUs)")
        logger.info(f"Model: {config.num_layers} layers, {config.hidden_dim} dim")
        logger.info(f"Compile: {config.use_compile}")
        logger.info(f"Version: {get_version_info()}")
        logger.info("GPU UUIDs:")
        for i, uuid in enumerate(all_uuids):
            logger.info(f"  rank {i}: {uuid}")
        logger.info("=" * 60)

    # Set SAME seed on all GPUs
    set_deterministic_seed(config.seed)

    # Disable autotuning if requested (for guaranteed determinism)
    if config.no_autotune:
        disable_autotuning()
        if rank == 0:
            logger.info("Autotuning disabled (--no-autotune)")

    # Create model (identical on all GPUs due to same seed)
    model = LocalTestModel(config).to(device=device, dtype=dtype)

    # Apply torch.compile if enabled
    if config.use_compile:
        if rank == 0:
            logger.info("Applying torch.compile (backend=inductor)...")
        model = torch.compile(model, backend="inductor", mode="default")

    # Create identical input on all GPUs (same seed)
    set_deterministic_seed(config.seed)
    x = torch.randn(config.batch_size, config.seq_len, config.hidden_dim,
                    device=device, dtype=dtype)

    # Warmup pass: populate compile cache before validation
    # CRITICAL: Serialize compilation to avoid autotuning race conditions
    # Rank 0 compiles first and populates the shared Triton/Inductor cache,
    # then other ranks compile and hit the cache (getting identical kernels)
    if config.use_compile:
        if rank == 0:
            logger.info("Rank 0 warming up compile cache (others waiting)...")
            with torch.no_grad():
                _ = model(x)
            torch.cuda.synchronize()
        dist.barrier()  # Other ranks wait for rank 0 to finish compilation

        # Now other ranks compile - they should hit rank 0's cached kernels
        if rank != 0:
            with torch.no_grad():
                _ = model(x)
            torch.cuda.synchronize()
        dist.barrier()  # Ensure all ranks have compiled

        # Reset input to ensure identical starting state
        set_deterministic_seed(config.seed)
        x = torch.randn(config.batch_size, config.seq_len, config.hidden_dim,
                        device=device, dtype=dtype)

    # Verify input is identical across GPUs
    input_cs = compute_checksum_tensor(x)
    if not check_all_equal(input_cs, rank, world_size):
        logger.error(f"[rank {rank}] INPUT MISMATCH - GPUs not initialized identically!")
        return {"passed": False, "error": "input_mismatch"}

    if rank == 0:
        logger.info(f"Input checksum (all GPUs): {compute_checksum_str(x)}")
        logger.info("Starting test loop...")

    # Run test
    start_time = time.time()
    failures = []
    last_log_time = start_time

    for step in range(config.steps):
        # Forward pass (no communication)
        with torch.no_grad():
            output = model(x)

        # Check every N steps
        if step % config.check_interval == 0:
            cs_tensor = compute_checksum_tensor(output)
            cs_str = compute_checksum_str(output)

            all_equal = check_all_equal(cs_tensor, rank, world_size)

            now = time.time()
            should_log = (now - last_log_time) >= 2.0

            if not all_equal:
                divergent = get_divergent_ranks(cs_tensor, rank, world_size)
                divergent_uuids = [all_uuids[r] if r < len(all_uuids) else "unknown" for r in divergent]
                failures.append({"step": step, "divergent_ranks": divergent, "divergent_uuids": divergent_uuids})
                logger.error(f"[rank {rank}] Step {step}: DIVERGENCE - ranks {divergent} differ, my checksum={cs_str}")
                if rank == 0:
                    for r in divergent:
                        uuid = all_uuids[r] if r < len(all_uuids) else "unknown"
                        logger.error(f"  BAD GPU: rank {r} -> {uuid}")
            elif should_log:
                if rank == 0:
                    logger.info(f"Step {step}: ALL MATCH checksum={cs_str}")
                last_log_time = now

        # Use output as next input
        x = output.clone()

        # Sync all ranks
        dist.barrier()

    elapsed = time.time() - start_time

    # Summary
    passed = len(failures) == 0
    if rank == 0:
        logger.info("=" * 60)
        if passed:
            logger.info(f"PASSED: All {config.steps} steps matched across {world_size} GPUs")
        else:
            logger.error(f"FAILED: {len(failures)} divergences detected")
            for f in failures[:5]:
                logger.error(f"  Step {f['step']}: ranks {f['divergent_ranks']} diverged")
        logger.info(f"Elapsed: {elapsed:.1f}s ({config.steps/elapsed:.1f} steps/sec)")
        logger.info("=" * 60)

    return {
        "passed": passed,
        "rank": rank,
        "world_size": world_size,
        "total_steps": config.steps,
        "num_failures": len(failures),
        "failures": failures[:10],
        "elapsed_sec": elapsed,
    }


def main():
    parser = argparse.ArgumentParser(description="Local GPU determinism test")
    parser.add_argument("--steps", type=int, default=250)
    parser.add_argument("--check-interval", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--hidden-dim", type=int, default=4096)
    parser.add_argument("--num-layers", type=int, default=32)
    parser.add_argument("--no-compile", action="store_true", help="Disable torch.compile")
    parser.add_argument("--no-autotune", action="store_true",
                        help="Disable inductor autotuning (slower but guaranteed determinism)")
    parser.add_argument("--model-size", choices=["small", "medium", "large"], default="large")

    args = parser.parse_args()

    # Initialize distributed
    # Note: CUDA_VISIBLE_DEVICES is set at module load time, so each process
    # sees only its assigned GPU as cuda:0
    torch.cuda.set_device(0)
    dist.init_process_group(backend="nccl", device_id=torch.device("cuda:0"))
    rank = dist.get_rank()

    # Model presets
    presets = {
        "small": {"hidden_dim": 2048, "num_layers": 16, "intermediate_size": 5504},
        "medium": {"hidden_dim": 3072, "num_layers": 24, "intermediate_size": 8192},
        "large": {"hidden_dim": 4096, "num_layers": 32, "intermediate_size": 14336},
    }
    preset = presets[args.model_size]

    config = LocalTestConfig(
        seed=args.seed,
        steps=args.steps,
        check_interval=args.check_interval,
        hidden_dim=args.hidden_dim or preset["hidden_dim"],
        num_layers=args.num_layers or preset["num_layers"],
        intermediate_size=preset["intermediate_size"],
        use_compile=not args.no_compile,
        no_autotune=args.no_autotune,
    )

    result = run_local_test(config)

    dist.barrier()
    dist.destroy_process_group()

    return 0 if result["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
