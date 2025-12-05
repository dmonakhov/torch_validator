"""
Minimal transformer stress test.

A minimal dense transformer model that reproduces the compute and
memory patterns of LLaMA3-8B without external dependencies.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import ShardingStrategy

from torch_validator.stress_tests.base import (
    StressTest,
    StressTestConfig,
    compute_checksum,
    logger,
)


class SwiGLU(nn.Module):
    """SwiGLU activation function used in LLaMA."""

    def __init__(self, hidden_dim: int, intermediate_dim: int):
        super().__init__()
        self.w1 = nn.Linear(hidden_dim, intermediate_dim, bias=False)
        self.w2 = nn.Linear(intermediate_dim, hidden_dim, bias=False)
        self.w3 = nn.Linear(hidden_dim, intermediate_dim, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class RMSNorm(nn.Module):
    """RMSNorm used in LLaMA."""

    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        norm = x.float().pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return (x * norm).type_as(x) * self.weight


class MinimalAttention(nn.Module):
    """
    Simplified attention without rotary embeddings.

    Captures the compute pattern without the complexity of RoPE.
    """

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

        # Repeat KV heads for GQA
        if self.num_kv_heads < self.num_heads:
            repeat_factor = self.num_heads // self.num_kv_heads
            k = k.repeat_interleave(repeat_factor, dim=1)
            v = v.repeat_interleave(repeat_factor, dim=1)

        # Scaled dot-product attention
        scale = self.head_dim ** -0.5
        attn = torch.matmul(q, k.transpose(-2, -1)) * scale
        attn = F.softmax(attn, dim=-1)
        out = torch.matmul(attn, v)

        out = out.transpose(1, 2).contiguous().view(batch, seq_len, -1)
        return self.o_proj(out)


class MinimalTransformerBlock(nn.Module):
    """Single transformer block matching LLaMA architecture."""

    def __init__(self, config: StressTestConfig):
        super().__init__()
        self.attention_norm = RMSNorm(config.hidden_dim)
        self.attention = MinimalAttention(
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


class MinimalTransformer(nn.Module):
    """
    Minimal transformer matching LLaMA3-8B architecture.

    Architecture:
    - 32 layers
    - 4096 hidden dim
    - 32 attention heads (GQA with 8 KV heads)
    - 14336 intermediate size (SwiGLU)
    - RMSNorm

    No embeddings or output head - just the core transformer blocks.
    """

    def __init__(self, config: StressTestConfig):
        super().__init__()
        self.layers = nn.ModuleList([
            MinimalTransformerBlock(config)
            for _ in range(config.num_layers)
        ])
        self.norm = RMSNorm(config.hidden_dim)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)


class MinimalTransformerTest(StressTest):
    """
    Stress test using minimal transformer with FSDP.

    This test creates a real PyTorch model wrapped in FSDP,
    running actual forward/backward passes with random inputs.
    This is the closest simulation to real LLaMA training.
    """

    name = "transformer"

    def __init__(self, config: StressTestConfig):
        super().__init__(config)
        self.model: nn.Module = None
        self.optimizer: torch.optim.Optimizer = None

    def setup(self):
        if not dist.is_initialized():
            raise RuntimeError("Transformer test requires distributed initialization")

        device = torch.device("cuda")

        torch.manual_seed(self.config.seed)
        torch.cuda.manual_seed(self.config.seed)

        # Create model
        model = MinimalTransformer(self.config)

        # Convert to bfloat16
        model = model.to(device=device, dtype=self.config.torch_dtype)

        # Wrap in FSDP
        self.model = FSDP(
            model,
            sharding_strategy=ShardingStrategy.FULL_SHARD,
            use_orig_params=True,
        )

        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=1e-5,
            betas=(0.9, 0.95),
            weight_decay=0.0,
        )

        # Count parameters
        num_params = sum(p.numel() for p in self.model.parameters())
        logger.info(f"[rank {self.rank}] Model initialized with {num_params:,} parameters")

        torch.cuda.synchronize()

    def step(self, step_num: int) -> str:
        # Generate random input
        torch.manual_seed(self.config.seed + step_num)
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

        # Simple loss (mean of squared outputs)
        loss = (output ** 2).mean()

        # Backward pass
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

        # Optimizer step
        self.optimizer.step()

        torch.cuda.synchronize()

        # Only compute expensive CPU checksum at verification intervals
        if step_num % self.config.check_interval == 0:
            return compute_checksum(output.detach().flatten()[:10000])
        return ""

    def cleanup(self):
        del self.model, self.optimizer
        torch.cuda.empty_cache()


class MinimalDenseTest(StressTest):
    """
    Simplified dense model test without FSDP.

    For testing memory bandwidth without NCCL overhead.
    Useful for isolating memory vs communication issues.
    """

    name = "dense_local"

    def __init__(self, config: StressTestConfig):
        super().__init__(config)
        self.model: nn.Module = None
        self.optimizer: torch.optim.Optimizer = None

    def setup(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        torch.manual_seed(self.config.seed + self.rank)

        # Simpler model without FSDP
        layers = []
        for _ in range(self.config.num_layers):
            layers.extend([
                nn.Linear(self.config.hidden_dim, self.config.intermediate_size, bias=False),
                nn.GELU(),
                nn.Linear(self.config.intermediate_size, self.config.hidden_dim, bias=False),
            ])

        self.model = nn.Sequential(*layers)
        self.model = self.model.to(device=device, dtype=self.config.torch_dtype)

        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=1e-5)

        num_params = sum(p.numel() for p in self.model.parameters())
        logger.info(f"[rank {self.rank}] Dense model: {num_params:,} parameters")

        if torch.cuda.is_available():
            torch.cuda.synchronize()

    def step(self, step_num: int) -> str:
        torch.manual_seed(self.config.seed + step_num + self.rank)

        x = torch.randn(
            self.config.batch_size,
            self.config.seq_len,
            self.config.hidden_dim,
            device=next(self.model.parameters()).device,
            dtype=self.config.torch_dtype,
        )

        self.optimizer.zero_grad()
        output = self.model(x)
        loss = (output ** 2).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        # Only compute expensive CPU checksum at verification intervals
        if step_num % self.config.check_interval == 0:
            first_param = next(self.model.parameters())
            return compute_checksum(first_param.data.flatten()[:1000])
        return ""

    def cleanup(self):
        del self.model, self.optimizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
