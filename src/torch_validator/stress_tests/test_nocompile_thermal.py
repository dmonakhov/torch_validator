"""
No-compile transformer test with thermal monitoring.

Same as test_compile_local but:
- NO torch.compile (eager mode only)
- Logs GPU temperature at each checkpoint
- Simpler code path for debugging

Usage:
    torchrun --nproc_per_node=8 -m torch_validator.stress_tests.test_nocompile_thermal --steps 500
"""

import argparse
import ctypes
import hashlib
import logging
import os
import socket
import struct
import sys
import time
from dataclasses import dataclass
from typing import List, Optional

import torch
import torch.distributed as dist
import torch.nn as nn

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ============================================================
# NVML Temperature Monitoring
# ============================================================

_nvml_initialized = False
_nvml_handles = {}


def init_nvml():
    """Initialize NVML for temperature monitoring."""
    global _nvml_initialized, _nvml_handles
    if _nvml_initialized:
        return True
    try:
        import pynvml
        pynvml.nvmlInit()
        count = pynvml.nvmlDeviceGetCount()
        for i in range(count):
            _nvml_handles[i] = pynvml.nvmlDeviceGetHandleByIndex(i)
        _nvml_initialized = True
        return True
    except Exception as e:
        logger.warning(f"NVML init failed: {e}")
        return False


def get_gpu_temperature(device_idx: int) -> Optional[int]:
    """Get GPU temperature in Celsius."""
    if not _nvml_initialized:
        if not init_nvml():
            return None
    try:
        import pynvml
        handle = _nvml_handles.get(device_idx)
        if handle:
            return pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
    except Exception:
        pass
    return None


def get_gpu_power(device_idx: int) -> Optional[int]:
    """Get GPU power usage in watts."""
    if not _nvml_initialized:
        if not init_nvml():
            return None
    try:
        import pynvml
        handle = _nvml_handles.get(device_idx)
        if handle:
            return pynvml.nvmlDeviceGetPowerUsage(handle) // 1000  # mW to W
    except Exception:
        pass
    return None


def get_gpu_uuid(device_idx: int) -> str:
    """Get GPU UUID."""
    if not _nvml_initialized:
        if not init_nvml():
            return f"cuda:{device_idx}"
    try:
        import pynvml
        handle = _nvml_handles.get(device_idx)
        if handle:
            return pynvml.nvmlDeviceGetUUID(handle)
    except Exception:
        pass
    return f"cuda:{device_idx}"


def get_all_gpu_uuids() -> List[str]:
    """Get all GPU UUIDs."""
    if not init_nvml():
        return []
    return [get_gpu_uuid(i) for i in range(len(_nvml_handles))]


# ============================================================
# Model Definition (same as test_compile_local)
# ============================================================

@dataclass
class TestConfig:
    """Configuration for test."""
    seed: int = 42
    steps: int = 500
    check_interval: int = 10
    hidden_dim: int = 4096
    num_layers: int = 32
    intermediate_size: int = 14336
    batch_size: int = 2
    seq_len: int = 4096
    dtype: str = "bfloat16"


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
    """Transformer block without attention (MLP only)."""
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.norm1 = RMSNorm(dim)
        self.mlp = MLP(dim, hidden_dim)

    def forward(self, x):
        return x + self.mlp(self.norm1(x))


class TestModel(nn.Module):
    """Model for determinism testing."""
    def __init__(self, config: TestConfig):
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


# ============================================================
# Checksum and Comparison
# ============================================================

def compute_checksum_tensor(tensor: torch.Tensor) -> torch.Tensor:
    """Compute checksum as tensor for all_gather."""
    t = tensor.detach().float().cpu().contiguous()
    ptr = t.data_ptr()
    size = t.numel() * t.element_size()
    buffer = (ctypes.c_char * size).from_address(ptr)
    h = hashlib.sha256(buffer).digest()[:8]
    value = struct.unpack('<q', h)[0]
    return torch.tensor([value], dtype=torch.int64, device=tensor.device)


def compute_checksum_str(tensor: torch.Tensor) -> str:
    """Compute checksum as hex string."""
    t = tensor.detach().float().cpu().contiguous()
    ptr = t.data_ptr()
    size = t.numel() * t.element_size()
    buffer = (ctypes.c_char * size).from_address(ptr)
    return hashlib.sha256(buffer).hexdigest()[:16]


def set_deterministic_seed(seed: int):
    """Set all seeds for determinism."""
    import random
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def check_all_equal(checksum: torch.Tensor, world_size: int) -> bool:
    """Check if all ranks have same checksum."""
    all_checksums = [torch.zeros_like(checksum) for _ in range(world_size)]
    dist.all_gather(all_checksums, checksum)
    reference = all_checksums[0].item()
    return all(cs.item() == reference for cs in all_checksums)


def get_divergent_ranks(checksum: torch.Tensor, world_size: int) -> list:
    """Get ranks with different checksums."""
    all_checksums = [torch.zeros_like(checksum) for _ in range(world_size)]
    dist.all_gather(all_checksums, checksum)

    # Find majority checksum
    values = [cs.item() for cs in all_checksums]
    from collections import Counter
    majority = Counter(values).most_common(1)[0][0]

    return [i for i, v in enumerate(values) if v != majority]


# ============================================================
# Main Test
# ============================================================

def run_test(config: TestConfig) -> dict:
    """Run no-compile determinism test with thermal monitoring."""

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)
    dtype = getattr(torch, config.dtype)

    # Initialize NVML
    init_nvml()
    gpu_uuids = get_all_gpu_uuids()
    my_uuid = gpu_uuids[rank] if rank < len(gpu_uuids) else "unknown"

    if rank == 0:
        logger.info("=" * 60)
        logger.info("No-Compile Thermal Monitoring Test")
        logger.info("=" * 60)
        logger.info(f"Hostname: {socket.gethostname()}")
        logger.info(f"World size: {world_size}")
        logger.info(f"Steps: {config.steps}")
        logger.info(f"Seed: {config.seed}")
        logger.info(f"Model: {config.num_layers} layers, {config.hidden_dim} dim")
        logger.info(f"Compile: DISABLED (eager mode)")
        logger.info("GPU UUIDs:")
        for i, uuid in enumerate(gpu_uuids):
            temp = get_gpu_temperature(i)
            power = get_gpu_power(i)
            logger.info(f"  rank {i}: {uuid} (temp={temp}C, power={power}W)")
        logger.info("=" * 60)

    # Set deterministic seed
    set_deterministic_seed(config.seed)

    # Create model (NO torch.compile)
    model = TestModel(config).to(device=device, dtype=dtype)

    # Create input
    set_deterministic_seed(config.seed)
    x = torch.randn(config.batch_size, config.seq_len, config.hidden_dim,
                    device=device, dtype=dtype)

    # Verify input
    input_cs = compute_checksum_tensor(x)
    if not check_all_equal(input_cs, world_size):
        logger.error(f"[rank {rank}] INPUT MISMATCH!")
        return {"passed": False, "error": "input_mismatch"}

    if rank == 0:
        logger.info(f"Input checksum (all GPUs): {compute_checksum_str(x)}")
        logger.info("Starting test loop...")

    # Track thermal data
    thermal_log = []
    start_time = time.time()
    failures = []
    last_log_time = start_time

    for step in range(config.steps):
        # Forward pass (eager mode, no compile)
        with torch.no_grad():
            output = model(x)

        # Check at intervals
        if step % config.check_interval == 0:
            cs_tensor = compute_checksum_tensor(output)
            cs_str = compute_checksum_str(output)

            all_equal = check_all_equal(cs_tensor, world_size)
            temp = get_gpu_temperature(rank)
            power = get_gpu_power(rank)

            thermal_log.append({
                "step": step,
                "temp": temp,
                "power": power,
                "elapsed": time.time() - start_time,
            })

            now = time.time()
            should_log = (now - last_log_time) >= 2.0

            if not all_equal:
                divergent = get_divergent_ranks(cs_tensor, world_size)
                failures.append({
                    "step": step,
                    "divergent_ranks": divergent,
                    "temp": temp,
                    "power": power,
                })
                logger.error(
                    f"[rank {rank}] Step {step}: DIVERGENCE - ranks {divergent} differ, "
                    f"checksum={cs_str}, temp={temp}C, power={power}W"
                )
                if rank in divergent:
                    logger.error(f"  BAD GPU: rank {rank} -> {my_uuid}")
            elif should_log:
                if rank == 0:
                    # Log all GPU temps
                    temps = [get_gpu_temperature(i) for i in range(world_size)]
                    logger.info(f"Step {step}: ALL MATCH checksum={cs_str} temps={temps}C")
                last_log_time = now

        # Chain output to input
        x = output.clone()
        dist.barrier()

    elapsed = time.time() - start_time
    passed = len(failures) == 0

    if rank == 0:
        logger.info("=" * 60)
        if passed:
            logger.info(f"PASSED: All {config.steps} steps matched")
        else:
            logger.error(f"FAILED: {len(failures)} divergences")
            for f in failures[:5]:
                logger.error(f"  Step {f['step']}: ranks {f['divergent_ranks']} (temp={f['temp']}C)")
        logger.info(f"Elapsed: {elapsed:.1f}s ({config.steps/elapsed:.1f} steps/sec)")

        # Thermal summary
        if thermal_log:
            temps = [t["temp"] for t in thermal_log if t["temp"]]
            if temps:
                logger.info(f"Temperature: min={min(temps)}C, max={max(temps)}C, final={temps[-1]}C")
        logger.info("=" * 60)

    return {
        "passed": passed,
        "rank": rank,
        "failures": failures,
        "thermal_log": thermal_log,
        "elapsed_sec": elapsed,
    }


def main():
    parser = argparse.ArgumentParser(description="No-compile thermal test")
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--check-interval", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model-size", choices=["small", "medium", "large"], default="large")

    args = parser.parse_args()

    # Initialize distributed
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl", device_id=torch.device(f"cuda:{local_rank}"))

    presets = {
        "small": {"hidden_dim": 2048, "num_layers": 16, "intermediate_size": 5504},
        "medium": {"hidden_dim": 3072, "num_layers": 24, "intermediate_size": 8192},
        "large": {"hidden_dim": 4096, "num_layers": 32, "intermediate_size": 14336},
    }
    preset = presets[args.model_size]

    config = TestConfig(
        seed=args.seed,
        steps=args.steps,
        check_interval=args.check_interval,
        hidden_dim=preset["hidden_dim"],
        num_layers=preset["num_layers"],
        intermediate_size=preset["intermediate_size"],
    )

    result = run_test(config)

    dist.barrier()
    dist.destroy_process_group()

    return 0 if result["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
