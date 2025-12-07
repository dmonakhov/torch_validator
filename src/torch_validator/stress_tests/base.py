"""
Base class for stress tests.
"""

import ctypes
import hashlib
import json
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torch.distributed as dist

logger = logging.getLogger(__name__)


@dataclass
class StressTestConfig:
    """Configuration for stress tests."""

    # Duration modes
    mode: str = "quick"  # "smoke" (150s), "quick" (600s), or "long" (3600s)

    # Override duration (seconds) or max steps
    duration_override: Optional[int] = None
    max_steps: Optional[int] = None  # If set, run exactly this many steps

    # Verification
    check_interval: int = 10  # Verify every N steps
    seed: int = 42

    # Model params (matching LLaMA3-8B architecture)
    hidden_dim: int = 4096
    num_layers: int = 32
    intermediate_size: int = 14336
    num_heads: int = 32
    num_kv_heads: int = 8
    batch_size: int = 2
    seq_len: int = 4096
    dtype: str = "bfloat16"

    # NVML telemetry
    enable_nvml: bool = False
    nvml_interval: int = 10  # Log GPU stats every N steps

    # Output
    output_dir: Optional[str] = None
    golden_dir: Optional[str] = None
    validate_mode: bool = False  # True = validate against golden

    # Portable determinism - bundle compile caches with golden
    portable: bool = False

    @property
    def duration_sec(self) -> int:
        if self.duration_override:
            return self.duration_override
        durations = {"smoke": 150, "quick": 600, "long": 3600}
        return durations.get(self.mode, 600)

    @property
    def torch_dtype(self) -> torch.dtype:
        return getattr(torch, self.dtype)

    def to_dict(self) -> dict:
        return {
            "mode": self.mode,
            "duration_sec": self.duration_sec,
            "check_interval": self.check_interval,
            "seed": self.seed,
            "hidden_dim": self.hidden_dim,
            "num_layers": self.num_layers,
            "intermediate_size": self.intermediate_size,
            "batch_size": self.batch_size,
            "seq_len": self.seq_len,
            "dtype": self.dtype,
        }


@dataclass
class StepMetrics:
    """Metrics collected at each verification step."""

    step: int
    elapsed_sec: float
    checksum: str
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "step": self.step,
            "elapsed_sec": self.elapsed_sec,
            "checksum": self.checksum,
            "extra": self.extra,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "StepMetrics":
        return cls(
            step=d["step"],
            elapsed_sec=d["elapsed_sec"],
            checksum=d["checksum"],
            extra=d.get("extra", {}),
        )


class NVMLMonitor:
    """Optional NVML-based GPU monitoring."""

    def __init__(self):
        self.enabled = False
        self.handle = None
        try:
            import pynvml
            pynvml.nvmlInit()
            device_idx = torch.cuda.current_device()
            self.handle = pynvml.nvmlDeviceGetHandleByIndex(device_idx)
            self.pynvml = pynvml
            self.enabled = True
            logger.info("NVML monitoring enabled")
        except Exception as e:
            logger.warning(f"NVML monitoring disabled: {e}")

    def get_stats(self) -> Dict[str, Any]:
        if not self.enabled:
            return {}
        try:
            temp = self.pynvml.nvmlDeviceGetTemperature(
                self.handle, self.pynvml.NVML_TEMPERATURE_GPU
            )
            power = self.pynvml.nvmlDeviceGetPowerUsage(self.handle) / 1000.0  # mW -> W
            mem_info = self.pynvml.nvmlDeviceGetMemoryInfo(self.handle)
            util = self.pynvml.nvmlDeviceGetUtilizationRates(self.handle)
            return {
                "gpu_temp_c": temp,
                "gpu_power_w": power,
                "gpu_util_pct": util.gpu,
                "mem_used_gb": mem_info.used / (1024**3),
                "mem_total_gb": mem_info.total / (1024**3),
            }
        except Exception as e:
            logger.warning(f"NVML query failed: {e}")
            return {}

    def shutdown(self):
        if self.enabled:
            try:
                self.pynvml.nvmlShutdown()
            except Exception:
                pass


def compute_checksum(tensor: torch.Tensor) -> str:
    """Compute SHA256 checksum of tensor (no numpy required)."""
    t = tensor.detach().float().cpu().contiguous()
    # Get raw bytes via ctypes (avoids slow Python iteration)
    ptr = t.data_ptr()
    size = t.numel() * t.element_size()
    buffer = (ctypes.c_char * size).from_address(ptr)
    return hashlib.sha256(buffer).hexdigest()[:16]


def get_rank() -> int:
    """Get distributed rank, or 0 if not distributed."""
    if dist.is_initialized():
        return dist.get_rank()
    return 0


def get_world_size() -> int:
    """Get world size, or 1 if not distributed."""
    if dist.is_initialized():
        return dist.get_world_size()
    return 1


class StressTest(ABC):
    """Base class for stress tests."""

    name: str = "base"

    # Minimum seconds between log messages
    LOG_INTERVAL_SEC: float = 2.0

    def __init__(self, config: StressTestConfig):
        self.config = config
        self.rank = get_rank()
        self.world_size = get_world_size()
        self.metrics: List[StepMetrics] = []
        self.golden_metrics: Dict[int, StepMetrics] = {}
        self.failures: List[Dict] = []
        self.nvml = NVMLMonitor() if config.enable_nvml else None
        self.start_time: float = 0
        self._last_log_time: float = 0

        # Load golden data if validating
        if config.validate_mode and config.golden_dir:
            self._load_golden()

    def _load_golden(self):
        """Load golden checksums from file."""
        golden_file = Path(self.config.golden_dir) / f"{self.name}_rank{self.rank}.golden.json"
        if not golden_file.exists():
            raise FileNotFoundError(f"Golden file not found: {golden_file}")

        with open(golden_file) as f:
            data = json.load(f)

        # New format: {"checksums": {"0": "abc...", "10": "def..."}}
        for step_str, checksum in data["checksums"].items():
            step = int(step_str)
            self.golden_metrics[step] = StepMetrics(step=step, elapsed_sec=0, checksum=checksum)

        logger.info(f"[{self.name}][rank {self.rank}] Loaded {len(self.golden_metrics)} golden checksums from {golden_file}")

    def _save_metrics(self):
        """Save golden checksums and run log to separate files."""
        if not self.config.output_dir:
            return

        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Golden file: just checksums (stable between runs)
        golden_file = output_dir / f"{self.name}_rank{self.rank}.golden.json"
        golden_data = {
            "test": self.name,
            "rank": self.rank,
            "checksums": {str(m.step): m.checksum for m in self.metrics},
        }
        with open(golden_file, "w") as f:
            json.dump(golden_data, f, indent=2)

        # Log file: full details including timing (variable between runs)
        log_file = output_dir / f"{self.name}_rank{self.rank}.log.json"
        log_data = {
            "test": self.name,
            "config": self.config.to_dict(),
            "rank": self.rank,
            "world_size": self.world_size,
            "steps": [m.to_dict() for m in self.metrics],
        }
        with open(log_file, "w") as f:
            json.dump(log_data, f, indent=2)

        logger.info(f"[{self.name}][rank {self.rank}] Saved {len(self.metrics)} checksums to {golden_file}")

    def _should_log(self) -> bool:
        """Check if enough time has passed to log again."""
        now = time.time()
        if now - self._last_log_time >= self.LOG_INTERVAL_SEC:
            self._last_log_time = now
            return True
        return False

    def _verify_step(self, step: int, checksum: str, extra: Dict[str, Any] = None) -> bool:
        """Verify step against golden data."""
        elapsed = time.time() - self.start_time
        extra = extra or {}
        should_log = self._should_log()

        # Add NVML stats if enabled (only when logging)
        if self.nvml and should_log:
            extra.update(self.nvml.get_stats())

        metrics = StepMetrics(step=step, elapsed_sec=elapsed, checksum=checksum, extra=extra)
        self.metrics.append(metrics)

        if not self.config.validate_mode:
            # Record mode - log progress at time intervals
            if should_log:
                nvml_str = ""
                if extra.get("gpu_temp_c"):
                    nvml_str = f", temp={extra['gpu_temp_c']}C, power={extra.get('gpu_power_w', 0):.0f}W"
                logger.info(f"[{self.name}][rank {self.rank}] Step {step}: checksum={checksum}{nvml_str}")
            return True

        # Validate mode - compare against golden
        golden = self.golden_metrics.get(step)
        if golden is None:
            if should_log:
                logger.warning(f"[{self.name}][rank {self.rank}] Step {step}: no golden data")
            return True

        if golden.checksum != checksum:
            failure = {
                "step": step,
                "expected": golden.checksum,
                "actual": checksum,
                "elapsed_sec": elapsed,
            }
            self.failures.append(failure)
            logger.error(
                f"[{self.name}][rank {self.rank}] Step {step}: CHECKSUM MISMATCH - "
                f"expected={golden.checksum}, actual={checksum}"
            )
            return False

        if should_log:
            logger.info(f"[{self.name}][rank {self.rank}] Step {step}: [PASS] checksum={checksum}")
        return True

    @abstractmethod
    def setup(self):
        """Initialize test resources."""
        pass

    @abstractmethod
    def step(self, step_num: int) -> str:
        """Execute one test step. Returns checksum."""
        pass

    @abstractmethod
    def cleanup(self):
        """Clean up test resources."""
        pass

    def _should_continue(self, current_step: int) -> bool:
        """Check if test should continue. Rank 0 decides, broadcasts to all."""
        # If max_steps is set, use step-based termination (simpler, no broadcast needed)
        if self.config.max_steps is not None:
            return current_step < self.config.max_steps

        if not dist.is_initialized():
            # Single GPU mode - just check time
            return (time.time() - self.start_time) < self.config.duration_sec

        # Rank 0 decides based on time
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        continue_flag = torch.tensor([1], dtype=torch.int32, device=device)

        if self.rank == 0:
            if (time.time() - self.start_time) >= self.config.duration_sec:
                continue_flag[0] = 0  # Signal stop

        # Broadcast decision from rank 0 to all
        dist.broadcast(continue_flag, src=0)

        return continue_flag.item() == 1

    def run(self) -> Dict[str, Any]:
        """Run the stress test."""
        if self.config.max_steps is not None:
            logger.info(f"[{self.name}][rank {self.rank}] Starting test, max_steps={self.config.max_steps}")
        else:
            logger.info(f"[{self.name}][rank {self.rank}] Starting test, duration={self.config.duration_sec}s")

        torch.manual_seed(self.config.seed + self.rank)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.config.seed + self.rank)

        self.setup()
        self.start_time = time.time()

        step = 0
        try:
            while True:
                checksum = self.step(step)

                if step % self.config.check_interval == 0:
                    self._verify_step(step, checksum)
                    # Coordinated shutdown check - all ranks check together
                    if not self._should_continue(step):
                        if self.config.max_steps is not None:
                            logger.info(f"[{self.name}][rank {self.rank}] Max steps reached, stopping at step {step}")
                        else:
                            logger.info(f"[{self.name}][rank {self.rank}] Duration reached, stopping at step {step}")
                        break

                step += 1

        except Exception as e:
            logger.error(f"[{self.name}][rank {self.rank}] Test failed at step {step}: {e}")
            raise
        finally:
            self.cleanup()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            if self.nvml:
                self.nvml.shutdown()

        # Save metrics
        if not self.config.validate_mode:
            self._save_metrics()

        # Generate report
        elapsed = time.time() - self.start_time
        result = {
            "test": self.name,
            "rank": self.rank,
            "total_steps": step,
            "elapsed_sec": elapsed,
            "steps_per_sec": step / elapsed if elapsed > 0 else 0,
            "passed": len(self.failures) == 0,
            "num_failures": len(self.failures),
            "failures": self.failures[:10],  # First 10 failures
        }

        status = "PASSED" if result["passed"] else "FAILED"
        logger.info(
            f"[{self.name}][rank {self.rank}] {status}: "
            f"{step} steps in {elapsed:.1f}s ({result['steps_per_sec']:.1f} steps/sec), "
            f"{len(self.failures)} failures"
        )

        return result
