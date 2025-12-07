"""
Deterministic mode utilities for reproducible training.

Ensures identical results across runs on same hardware for validation.

Portable Determinism:
    For cross-host determinism, Triton/Inductor caches must be shared.
    Use save_compile_cache() during record and load_compile_cache() during validate
    to bundle caches with golden traces.
"""

import logging
import os
import random
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import torch

logger = logging.getLogger(__name__)


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


# =============================================================================
# Portable Determinism: Compile Cache Management
# =============================================================================

def get_default_cache_dirs() -> Tuple[Path, Path]:
    """
    Get default Triton and TorchInductor cache directories.

    Returns:
        Tuple of (triton_cache_dir, inductor_cache_dir)
    """
    # Triton cache: TRITON_CACHE_DIR or ~/.triton/cache
    triton_dir = Path(os.environ.get("TRITON_CACHE_DIR", Path.home() / ".triton" / "cache"))

    # TorchInductor cache: TORCHINDUCTOR_CACHE_DIR or ~/.cache/torch_inductor
    inductor_dir = Path(os.environ.get(
        "TORCHINDUCTOR_CACHE_DIR",
        Path.home() / ".cache" / "torch_inductor"
    ))

    return triton_dir, inductor_dir


def save_compile_cache(output_dir: str, rank: int = 0) -> Optional[Path]:
    """
    Save Triton/Inductor compile caches to output directory.

    Only rank 0 saves the cache (all ranks compile identical kernels).

    Args:
        output_dir: Directory to save caches (typically golden_dir)
        rank: Current process rank (only rank 0 saves)

    Returns:
        Path to cache directory if saved, None otherwise
    """
    if rank != 0:
        return None

    output_path = Path(output_dir)
    cache_dir = output_path / "cache"

    triton_src, inductor_src = get_default_cache_dirs()

    # Save Triton cache
    if triton_src.exists():
        triton_dst = cache_dir / "triton"
        if triton_dst.exists():
            shutil.rmtree(triton_dst)
        shutil.copytree(triton_src, triton_dst)
        logger.info(f"Saved Triton cache: {triton_src} -> {triton_dst}")

    # Save Inductor cache
    if inductor_src.exists():
        inductor_dst = cache_dir / "inductor"
        if inductor_dst.exists():
            shutil.rmtree(inductor_dst)
        shutil.copytree(inductor_src, inductor_dst)
        logger.info(f"Saved Inductor cache: {inductor_src} -> {inductor_dst}")

    if cache_dir.exists():
        # Calculate total size
        total_size = sum(f.stat().st_size for f in cache_dir.rglob("*") if f.is_file())
        logger.info(f"Total cache size: {total_size / 1024 / 1024:.1f} MB")
        return cache_dir

    return None


def load_compile_cache(golden_dir: str) -> bool:
    """
    Load Triton/Inductor compile caches from golden directory.

    Sets TRITON_CACHE_DIR and TORCHINDUCTOR_CACHE_DIR environment variables
    to point to the bundled caches. Must be called BEFORE torch.compile().

    Args:
        golden_dir: Directory containing golden data and caches

    Returns:
        True if caches were loaded, False if not found
    """
    golden_path = Path(golden_dir)
    cache_dir = golden_path / "cache"

    if not cache_dir.exists():
        logger.warning(f"No portable cache found at {cache_dir}")
        return False

    triton_cache = cache_dir / "triton"
    inductor_cache = cache_dir / "inductor"

    loaded = False

    if triton_cache.exists():
        os.environ["TRITON_CACHE_DIR"] = str(triton_cache)
        logger.info(f"Using portable Triton cache: {triton_cache}")
        loaded = True

    if inductor_cache.exists():
        os.environ["TORCHINDUCTOR_CACHE_DIR"] = str(inductor_cache)
        logger.info(f"Using portable Inductor cache: {inductor_cache}")
        loaded = True

    return loaded


def setup_portable_cache(golden_dir: str, output_dir: Optional[str] = None) -> None:
    """
    Setup compile cache for portable determinism.

    In validate mode: loads cache from golden_dir
    In record mode: prepares to save cache to output_dir

    Args:
        golden_dir: Directory containing golden data (for validate mode)
        output_dir: Directory to save golden data (for record mode)
    """
    if golden_dir:
        # Validate mode - load existing cache
        if load_compile_cache(golden_dir):
            logger.info("Portable determinism: using bundled compile cache")
        else:
            logger.warning("Portable determinism: no cache found, results may differ across hosts")


def clear_compile_cache() -> None:
    """
    Clear local Triton/Inductor caches.

    Useful for ensuring fresh compilation when testing.
    """
    triton_dir, inductor_dir = get_default_cache_dirs()

    if triton_dir.exists():
        shutil.rmtree(triton_dir)
        logger.info(f"Cleared Triton cache: {triton_dir}")

    if inductor_dir.exists():
        shutil.rmtree(inductor_dir)
        logger.info(f"Cleared Inductor cache: {inductor_dir}")
