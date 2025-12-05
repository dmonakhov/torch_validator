"""
FSDP pattern stress test.

Simulates the exact communication and compute pattern of FSDP
without requiring the full FSDP wrapper.
"""

import torch
import torch.distributed as dist

from torch_validator.stress_tests.base import (
    StressTest,
    StressTestConfig,
    compute_checksum,
)


class FSDPPatternTest(StressTest):
    """
    Simulates FSDP full-shard communication pattern.

    This test reproduces the exact memory and communication pattern
    of FSDP training without the complexity of the full FSDP wrapper:
    1. All-gather weights (forward)
    2. Matrix multiply (compute)
    3. Reduce-scatter gradients (backward)
    4. Local update (optimizer)

    This pattern creates sustained memory bandwidth and NCCL traffic
    similar to dense model training.
    """

    name = "fsdp_pattern"

    def __init__(self, config: StressTestConfig):
        super().__init__(config)

        # Simulate model parameters
        # For 8B model with 8 GPUs: ~1B params per rank
        # 1B params in bfloat16 = 2GB per shard
        self.params_per_layer = config.hidden_dim * config.intermediate_size
        self.num_layers = config.num_layers

        # Sharded parameters (local to this rank)
        self.weight_shards: list = []
        self.grad_shards: list = []

        # Buffers for gathered weights
        self.gather_buffers: list = []

        # Input/output buffers
        self.input_buffer: torch.Tensor = None
        self.output_buffer: torch.Tensor = None

    def setup(self):
        if not dist.is_initialized():
            raise RuntimeError("FSDP test requires distributed initialization")

        device = torch.device("cuda")
        dtype = self.config.torch_dtype

        torch.manual_seed(self.config.seed + self.rank)

        # Calculate shard size
        total_params = self.params_per_layer * self.num_layers
        shard_size = total_params // self.world_size

        # Initialize weight shards for each "layer"
        params_per_shard = shard_size // self.num_layers

        for layer_idx in range(self.num_layers):
            # Local weight shard
            shard = torch.randn(params_per_shard, device=device, dtype=dtype) * 0.02
            self.weight_shards.append(shard)

            # Gradient buffer
            grad = torch.zeros(params_per_shard, device=device, dtype=dtype)
            self.grad_shards.append(grad)

            # Gather buffers (for all-gather result)
            gather_buf = [
                torch.empty(params_per_shard, device=device, dtype=dtype)
                for _ in range(self.world_size)
            ]
            self.gather_buffers.append(gather_buf)

        # Input/output buffers for compute
        batch_tokens = self.config.batch_size * self.config.seq_len
        self.input_buffer = torch.randn(
            batch_tokens, self.config.hidden_dim,
            device=device, dtype=dtype
        )
        self.output_buffer = torch.zeros(
            batch_tokens, self.config.hidden_dim,
            device=device, dtype=dtype
        )

        torch.cuda.synchronize()

    def step(self, step_num: int) -> str:
        # Simulate one training step with FSDP pattern

        # Reset gradients
        for grad in self.grad_shards:
            grad.zero_()

        # Forward pass through layers
        x = self.input_buffer.clone()

        for layer_idx in range(self.num_layers):
            # 1. All-gather weights for this layer
            dist.all_gather(
                self.gather_buffers[layer_idx],
                self.weight_shards[layer_idx]
            )
            full_weights = torch.cat(self.gather_buffers[layer_idx])

            # 2. Compute (matrix multiply)
            # Reshape weights to matrix form and multiply
            weight_matrix = full_weights[:self.config.hidden_dim * self.config.hidden_dim].view(
                self.config.hidden_dim, self.config.hidden_dim
            )
            x = torch.mm(x, weight_matrix)

        self.output_buffer = x

        # Backward pass (simplified)
        grad_output = self.output_buffer * 0.001  # Simulated loss gradient

        for layer_idx in reversed(range(self.num_layers)):
            # Generate gradient for this layer
            grad_full = grad_output.sum(dim=0).expand(
                len(self.gather_buffers[layer_idx][0]) * self.world_size
            )

            # 3. Reduce-scatter gradients
            dist.reduce_scatter_tensor(
                self.grad_shards[layer_idx],
                grad_full[:len(self.grad_shards[layer_idx]) * self.world_size]
            )

        # 4. Optimizer step (SGD-like update)
        for layer_idx in range(self.num_layers):
            self.weight_shards[layer_idx] -= 0.01 * self.grad_shards[layer_idx]

        torch.cuda.synchronize()

        # Only compute expensive CPU checksum at verification intervals
        if step_num % self.config.check_interval == 0:
            return compute_checksum(self.weight_shards[0])
        return ""

    def cleanup(self):
        del self.weight_shards, self.grad_shards, self.gather_buffers
        del self.input_buffer, self.output_buffer
        torch.cuda.empty_cache()


class FSDPLayerTest(StressTest):
    """
    Simplified FSDP test focusing on single-layer pattern.

    Tests the core FSDP communication loop without multi-layer complexity.
    Useful for isolating communication vs compute issues.
    """

    name = "fsdp_layer"

    def __init__(self, config: StressTestConfig):
        super().__init__(config)

        # Single large layer
        self.weight_shard: torch.Tensor = None
        self.grad_shard: torch.Tensor = None
        self.gather_buffer: list = None

        # Match LLaMA FFN size: hidden * intermediate * 2 (up + down projections)
        self.layer_params = config.hidden_dim * config.intermediate_size * 2

    def setup(self):
        if not dist.is_initialized():
            raise RuntimeError("FSDP test requires distributed initialization")

        device = torch.device("cuda")
        dtype = self.config.torch_dtype

        torch.manual_seed(self.config.seed + self.rank)

        shard_size = self.layer_params // self.world_size

        self.weight_shard = torch.randn(shard_size, device=device, dtype=dtype) * 0.02
        self.grad_shard = torch.zeros(shard_size, device=device, dtype=dtype)

        self.gather_buffer = [
            torch.empty(shard_size, device=device, dtype=dtype)
            for _ in range(self.world_size)
        ]

        torch.cuda.synchronize()

    def step(self, step_num: int) -> str:
        # Forward: all-gather
        dist.all_gather(self.gather_buffer, self.weight_shard)
        full_weights = torch.cat(self.gather_buffer)

        # Compute: simple operation on full weights
        output = full_weights.sum() + full_weights.mean()

        # Backward: generate gradient and reduce-scatter
        grad_full = full_weights * 0.001
        dist.reduce_scatter_tensor(self.grad_shard, grad_full)

        # Update
        self.weight_shard -= 0.01 * self.grad_shard

        torch.cuda.synchronize()

        # Only compute expensive CPU checksum at verification intervals
        if step_num % self.config.check_interval == 0:
            return compute_checksum(self.weight_shard)
        return ""

    def cleanup(self):
        del self.weight_shard, self.grad_shard, self.gather_buffer
        torch.cuda.empty_cache()
