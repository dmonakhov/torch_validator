"""
Deterministic mode utilities for reproducible training.

Ensures identical results across runs on same hardware for validation.
"""

import os
import random
from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class SeedState:
    """Captured seed state for reproducibility."""

    python_seed: int
    torch_seed: int
    cuda_seed: Optional[int]
    numpy_seed: Optional[int]

    def to_dict(self) -> dict:
        """Convert to JSON-serializable dict."""
        return {
            "python_seed": self.python_seed,
            "torch_seed": self.torch_seed,
            "cuda_seed": self.cuda_seed,
            "numpy_seed": self.numpy_seed,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "SeedState":
        """Create from dict."""
        return cls(
            python_seed=d["python_seed"],
            torch_seed=d["torch_seed"],
            cuda_seed=d.get("cuda_seed"),
            numpy_seed=d.get("numpy_seed"),
        )


def set_seed(seed: int = 42) -> SeedState:
    """
    Set random seeds only (fast, no performance impact).

    This is sufficient for hardware validation - both record and validate
    runs will produce identical results if using same seed on same hardware.

    Args:
        seed: Base seed for all RNGs

    Returns:
        SeedState with captured seed values
    """
    # Python random
    random.seed(seed)

    # PyTorch
    torch.manual_seed(seed)

    # CUDA
    cuda_seed = None
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        cuda_seed = seed

    # NumPy (if available)
    numpy_seed = None
    try:
        import numpy as np
        np.random.seed(seed)
        numpy_seed = seed
    except ImportError:
        pass

    return SeedState(
        python_seed=seed,
        torch_seed=seed,
        cuda_seed=cuda_seed,
        numpy_seed=numpy_seed,
    )


def set_deterministic_mode(
    seed: int = 42,
    cudnn_deterministic: bool = True,
    cudnn_benchmark: bool = False,
    warn_only: bool = False,
) -> SeedState:
    """
    Set all random seeds and enable FULL deterministic mode.

    WARNING: This can cause 2-3x slowdown! For hardware validation,
    consider using set_seed() instead - it's sufficient if both runs
    use the same seed on the same hardware type.

    Args:
        seed: Base seed for all RNGs
        cudnn_deterministic: Enable cuDNN deterministic algorithms (slower)
        cudnn_benchmark: Enable cuDNN benchmark (disable for determinism)
        warn_only: If True, use warn_only mode for torch.use_deterministic_algorithms

    Returns:
        SeedState with captured seed values

    Note:
        For full determinism, also set environment variables BEFORE importing torch:
        - CUBLAS_WORKSPACE_CONFIG=:4096:8 (or :16:8)
        - CUDA_LAUNCH_BLOCKING=1 (optional, for debugging)
    """
    # torch.compile determinism: single-threaded compilation for reproducible kernels
    os.environ.setdefault("TORCHINDUCTOR_COMPILE_THREADS", "1")

    # Set seeds first
    state = set_seed(seed)

    # Python hash seed (from torchtitan)
    os.environ["PYTHONHASHSEED"] = str(seed % 2**32)

    # cuDNN deterministic (slower)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = cudnn_deterministic
        torch.backends.cudnn.benchmark = cudnn_benchmark

    # PyTorch deterministic algorithms (much slower)
    try:
        torch.use_deterministic_algorithms(True, warn_only=warn_only)
    except Exception:
        # Older PyTorch versions
        pass

    # Inductor/Triton determinism (for torch.compile)
    try:
        torch._inductor.config.deterministic_algorithms = True
    except (AttributeError, ModuleNotFoundError):
        pass

    # Disable non-deterministic SDPA backends
    # Flash/cuDNN attention have non-deterministic backward pass
    if torch.cuda.is_available():
        try:
            torch.backends.cuda.enable_flash_sdp(False)
            torch.backends.cuda.enable_cudnn_sdp(False)
            torch.backends.cuda.enable_mem_efficient_sdp(True)
            torch.backends.cuda.enable_math_sdp(True)
        except AttributeError:
            pass

    return state


def get_seed_state() -> SeedState:
    """
    Get current seed state (for logging purposes).

    Note: This captures the initial seed values, not the current RNG state.
    For reproducibility, use set_deterministic_mode() at the start of training.
    """
    # We can't reliably get the current seed from most RNGs
    # This is mainly for documentation purposes
    return SeedState(
        python_seed=-1,  # Cannot retrieve
        torch_seed=torch.initial_seed() if hasattr(torch, "initial_seed") else -1,
        cuda_seed=-1,  # Cannot retrieve
        numpy_seed=-1,  # Cannot retrieve
    )


def set_cublas_workspace() -> None:
    """
    Set CUBLAS workspace configuration for determinism.

    Must be called BEFORE any CUDA operations.
    Alternatively, set CUBLAS_WORKSPACE_CONFIG=:4096:8 environment variable.
    """
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")


def get_determinism_env_vars() -> dict:
    """
    Get recommended environment variables for deterministic training.

    Returns:
        Dict of environment variable names to recommended values
    """
    return {
        "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        # Optional for debugging non-determinism:
        # "CUDA_LAUNCH_BLOCKING": "1",
    }


def check_determinism_config() -> dict:
    """
    Check current determinism configuration.

    Returns:
        Dict with configuration status
    """
    config = {
        "cublas_workspace": os.environ.get("CUBLAS_WORKSPACE_CONFIG", "NOT SET"),
        "cudnn_deterministic": None,
        "cudnn_benchmark": None,
        "torch_deterministic": None,
    }

    if hasattr(torch.backends, "cudnn"):
        config["cudnn_deterministic"] = torch.backends.cudnn.deterministic
        config["cudnn_benchmark"] = torch.backends.cudnn.benchmark

    try:
        config["torch_deterministic"] = torch.are_deterministic_algorithms_enabled()
    except AttributeError:
        pass

    return config
