"""
Compile determinism stress test.

This test is designed to detect cross-host divergence caused by:
- torch.compile kernel selection differences
- cuDNN/cuBLAS autotuning differences
- Floating-point operation ordering

Based on research/030: Production training showed deterministic divergence
starting at step 54 (after warmup), with errors accumulating over time.

Key characteristics that trigger the divergence:
1. torch.compile enabled
2. Warmup phase -> training transition
3. FSDP all-reduce (propagates small errors)
4. bfloat16 precision

Usage:
    # Record golden on known-good host
    torchrun --nproc_per_node=8 -m torch_validator.stress_tests.runner \
        --test compile --mode quick --output golden/

    # Validate on suspect host
    torchrun --nproc_per_node=8 -m torch_validator.stress_tests.runner \
        --test compile --mode quick --validate --golden golden/
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import ShardingStrategy
from dataclasses import dataclass, field
from typing import Dict, Any, Optional
import json

from torch_validator.stress_tests.base import (
    StressTest,
    StressTestConfig,
    compute_checksum,
    logger,
)
from torch_validator.deterministic import set_deterministic_mode, set_cublas_workspace, set_seed


@dataclass
class CompileTestConfig(StressTestConfig):
    """Extended config for compile determinism test."""

    # Compile settings
    compile_enabled: bool = True
    compile_backend: str = "inductor"
    compile_mode: str = "default"  # "default", "reduce-overhead", "max-autotune"

    # Warmup settings (critical for triggering divergence)
    warmup_steps: int = 50
    base_lr: float = 1e-5
    warmup_lr: float = 0.0
    end_lr: float = 1e-6

    # Extended metrics (match production verification)
    track_loss: bool = True
    track_grad_norm: bool = True
    track_weight_norm: bool = True

    # Tolerance for validation (based on research/030 findings)
    loss_rtol: float = 1e-6  # Initial divergence is ~1e-5
    grad_norm_rtol: float = 1e-4


class SwiGLU(nn.Module):
    """SwiGLU activation (LLaMA style)."""

    def __init__(self, hidden_dim: int, intermediate_dim: int):
        super().__init__()
        self.w1 = nn.Linear(hidden_dim, intermediate_dim, bias=False)
        self.w2 = nn.Linear(intermediate_dim, hidden_dim, bias=False)
        self.w3 = nn.Linear(hidden_dim, intermediate_dim, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class RMSNorm(nn.Module):
    """RMSNorm (LLaMA style)."""

    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        norm = x.float().pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return (x * norm).type_as(x) * self.weight


class Attention(nn.Module):
    """Multi-head attention with GQA support."""

    def __init__(self, hidden_dim: int, num_heads: int, num_kv_heads: int):
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = hidden_dim // num_heads

        self.q_proj = nn.Linear(hidden_dim, num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(hidden_dim, num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_dim, num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(num_heads * self.head_dim, hidden_dim, bias=False)

    def forward(self, x):
        batch, seq_len, _ = x.shape

        q = self.q_proj(x).view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

        # GQA: repeat KV heads
        if self.num_kv_heads < self.num_heads:
            repeat_factor = self.num_heads // self.num_kv_heads
            k = k.repeat_interleave(repeat_factor, dim=1)
            v = v.repeat_interleave(repeat_factor, dim=1)

        # SDPA - flash/cuDNN disabled globally in setup() for determinism
        out = F.scaled_dot_product_attention(q, k, v)

        out = out.transpose(1, 2).contiguous().view(batch, seq_len, -1)
        return self.o_proj(out)


class TransformerBlock(nn.Module):
    """Single transformer block."""

    def __init__(self, config: StressTestConfig):
        super().__init__()
        self.attention_norm = RMSNorm(config.hidden_dim)
        self.attention = Attention(
            config.hidden_dim,
            config.num_heads,
            config.num_kv_heads
        )
        self.ffn_norm = RMSNorm(config.hidden_dim)
        self.ffn = SwiGLU(config.hidden_dim, config.intermediate_size)

    def forward(self, x):
        x = x + self.attention(self.attention_norm(x))
        x = x + self.ffn(self.ffn_norm(x))
        return x


class CompileTestModel(nn.Module):
    """Transformer model for compile determinism testing."""

    def __init__(self, config: StressTestConfig):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerBlock(config)
            for _ in range(config.num_layers)
        ])
        self.norm = RMSNorm(config.hidden_dim)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)


class CosineWithWarmupLR:
    """Cosine LR scheduler with linear warmup (matches production)."""

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_steps: int,
        total_steps: int,
        base_lr: float,
        warmup_lr: float = 0.0,
        end_lr: float = 1e-6,
    ):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.base_lr = base_lr
        self.warmup_lr = warmup_lr
        self.end_lr = end_lr
        self.current_step = 0

    def get_lr(self, step: int) -> float:
        if step < self.warmup_steps:
            # Linear warmup
            progress = step / self.warmup_steps
            return self.warmup_lr + progress * (self.base_lr - self.warmup_lr)
        else:
            # Cosine decay
            progress = (step - self.warmup_steps) / (self.total_steps - self.warmup_steps)
            import math
            return self.end_lr + 0.5 * (self.base_lr - self.end_lr) * (1 + math.cos(math.pi * progress))

    def step(self):
        lr = self.get_lr(self.current_step)
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        self.current_step += 1
        return lr


class CompileDeterminismTest(StressTest):
    """
    Test for detecting torch.compile determinism issues across hosts.

    This test specifically targets the divergence pattern observed in
    production (research/030):
    - Divergence starts after warmup phase
    - Initially tiny errors (~1e-5) accumulate over training
    - FSDP all-reduce propagates errors across ranks
    """

    name = "compile"

    def __init__(self, config: CompileTestConfig):
        # Set check_interval to 1 for this test - we check every step
        config.check_interval = 1
        super().__init__(config)
        self.compile_config = config
        self.model: nn.Module = None
        self.optimizer: torch.optim.Optimizer = None
        self.scheduler: CosineWithWarmupLR = None
        self.step_metrics: Dict[int, Dict[str, float]] = {}

    def _compute_grad_norm(self) -> float:
        """Compute total gradient norm across parameters."""
        total_norm_sq = 0.0
        for p in self.model.parameters():
            if p.grad is not None:
                total_norm_sq += p.grad.detach().float().pow(2).sum().item()
        return total_norm_sq ** 0.5

    def _compute_weight_norm(self) -> float:
        """Compute total weight norm across parameters."""
        total_norm_sq = 0.0
        for p in self.model.parameters():
            total_norm_sq += p.detach().float().pow(2).sum().item()
        return total_norm_sq ** 0.5

    def setup(self):
        if not dist.is_initialized():
            raise RuntimeError("Compile test requires distributed initialization")

        device = torch.device("cuda")

        # Deterministic settings - use centralized utility
        set_cublas_workspace()  # Must be called before CUDA ops
        set_deterministic_mode(seed=self.config.seed, warn_only=False)

        # Create model
        model = CompileTestModel(self.config)
        model = model.to(device=device, dtype=self.config.torch_dtype)

        # Apply torch.compile if enabled
        if self.compile_config.compile_enabled:
            logger.info(f"[rank {self.rank}] Applying torch.compile (backend={self.compile_config.compile_backend}, mode={self.compile_config.compile_mode})")
            model = torch.compile(
                model,
                backend=self.compile_config.compile_backend,
                mode=self.compile_config.compile_mode,
            )

        # Wrap in FSDP
        self.model = FSDP(
            model,
            sharding_strategy=ShardingStrategy.FULL_SHARD,
            use_orig_params=True,
        )

        # Optimizer (match production: AdamW with betas=(0.9, 0.95))
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.compile_config.base_lr,
            betas=(0.9, 0.95),
            weight_decay=0.0,
            fused=True,
        )

        # LR scheduler with warmup
        # Estimate total steps from duration (rough)
        estimated_total_steps = self.config.duration_sec * 10  # ~10 steps/sec estimate
        self.scheduler = CosineWithWarmupLR(
            self.optimizer,
            warmup_steps=self.compile_config.warmup_steps,
            total_steps=estimated_total_steps,
            base_lr=self.compile_config.base_lr,
            warmup_lr=self.compile_config.warmup_lr,
            end_lr=self.compile_config.end_lr,
        )

        num_params = sum(p.numel() for p in self.model.parameters())
        logger.info(f"[rank {self.rank}] Model: {num_params:,} params, warmup={self.compile_config.warmup_steps} steps")

        torch.cuda.synchronize()

    def step(self, step_num: int) -> str:
        # Generate deterministic input - seed all RNGs per step
        set_seed(self.config.seed + step_num)
        x = torch.randn(
            self.config.batch_size,
            self.config.seq_len,
            self.config.hidden_dim,
            device="cuda",
            dtype=self.config.torch_dtype,
        )

        # Forward pass
        self.optimizer.zero_grad()
        output = self.model(x)

        # Simple loss (mean squared)
        loss = (output ** 2).mean()
        loss_val = loss.item()

        # Backward pass
        loss.backward()

        # Compute grad norm before clipping
        grad_norm = self._compute_grad_norm()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

        # Optimizer step
        self.optimizer.step()

        # LR scheduler step
        lr = self.scheduler.step()

        torch.cuda.synchronize()

        # Compute weight norm
        weight_norm = self._compute_weight_norm()

        # Store extended metrics for this step
        self.step_metrics[step_num] = {
            "loss": loss_val,
            "grad_norm": grad_norm,
            "weight_norm": weight_norm,
            "lr": lr,
        }

        # Build checksum from multiple metrics
        checksum_data = f"{loss_val:.15e}:{grad_norm:.15e}:{weight_norm:.15e}"
        return compute_checksum(torch.tensor([loss_val, grad_norm, weight_norm]))

    def _verify_step(self, step: int, checksum: str, extra: Dict[str, Any] = None) -> bool:
        """Override to add extended metric comparison."""
        extra = extra or {}

        # Add step metrics to extra
        if step in self.step_metrics:
            extra.update(self.step_metrics[step])

        # Call parent implementation
        return super()._verify_step(step, checksum, extra)

    def _save_metrics(self):
        """Override to save extended metrics format."""
        if not self.config.output_dir:
            return

        from pathlib import Path
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Golden file: checksums + metrics (for detailed comparison)
        golden_file = output_dir / f"{self.name}_rank{self.rank}.golden.json"
        golden_data = {
            "test": self.name,
            "rank": self.rank,
            "compile_enabled": self.compile_config.compile_enabled,
            "warmup_steps": self.compile_config.warmup_steps,
            "checksums": {str(m.step): m.checksum for m in self.metrics},
            "metrics": {
                str(step): metrics
                for step, metrics in self.step_metrics.items()
            },
        }
        with open(golden_file, "w") as f:
            json.dump(golden_data, f, indent=2)

        logger.info(f"[{self.name}][rank {self.rank}] Saved {len(self.metrics)} steps to {golden_file}")

    def cleanup(self):
        del self.model, self.optimizer, self.scheduler
        torch.cuda.empty_cache()


class CompileDisabledTest(CompileDeterminismTest):
    """
    Same as CompileDeterminismTest but with torch.compile disabled.

    Use this as a baseline to isolate compile-related divergence.
    """

    name = "compile_disabled"

    def __init__(self, config: CompileTestConfig):
        config.compile_enabled = False
        super().__init__(config)
