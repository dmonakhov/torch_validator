"""
NCCL communication stress test.

Stresses GPU interconnect (NVLink, PCIe, EFA) with continuous
collective operations to expose communication-related hardware faults.
"""

import torch
import torch.distributed as dist

from torch_validator.stress_tests.base import (
    StressTest,
    StressTestConfig,
    compute_checksum,
    get_rank,
    get_world_size,
)


class NCCLAllReduceTest(StressTest):
    """
    Stress test for NCCL AllReduce operations.

    Creates sustained high-bandwidth collective communication
    to test:
    - NVLink reliability
    - PCIe/InfiniBand/EFA stability
    - NCCL ring algorithm correctness
    """

    name = "nccl_allreduce"

    def __init__(self, config: StressTestConfig):
        super().__init__(config)
        self.tensor: torch.Tensor = None

        # Large tensor for all-reduce (512MB in bfloat16)
        self.tensor_elements = 256 * 1024 * 1024

    def setup(self):
        if not dist.is_initialized():
            raise RuntimeError("NCCL test requires distributed initialization")

        device = torch.device("cuda")
        dtype = self.config.torch_dtype

        torch.manual_seed(self.config.seed + self.rank)

        self.tensor = torch.randn(
            self.tensor_elements,
            device=device,
            dtype=dtype
        )

        torch.cuda.synchronize()

    def step(self, step_num: int) -> str:
        # All-reduce: sum across all ranks
        dist.all_reduce(self.tensor, op=dist.ReduceOp.SUM)

        # Scale down to prevent overflow
        self.tensor = self.tensor / self.world_size

        # Add small perturbation for next iteration
        self.tensor = self.tensor + 0.0001

        torch.cuda.synchronize()

        # Only compute expensive CPU checksum at verification intervals
        if step_num % self.config.check_interval == 0:
            return compute_checksum(self.tensor[:10000])
        return ""

    def cleanup(self):
        del self.tensor
        torch.cuda.empty_cache()


class NCCLAllGatherTest(StressTest):
    """
    Stress test for NCCL AllGather operations.

    Tests the all-gather pattern used by FSDP forward pass.
    """

    name = "nccl_allgather"

    def __init__(self, config: StressTestConfig):
        super().__init__(config)
        self.local_tensor: torch.Tensor = None
        self.gathered_tensors: list = None

        # Shard size per rank (128MB in bfloat16)
        self.shard_elements = 64 * 1024 * 1024

    def setup(self):
        if not dist.is_initialized():
            raise RuntimeError("NCCL test requires distributed initialization")

        device = torch.device("cuda")
        dtype = self.config.torch_dtype

        torch.manual_seed(self.config.seed + self.rank)

        self.local_tensor = torch.randn(
            self.shard_elements,
            device=device,
            dtype=dtype
        )

        # Pre-allocate gather buffers
        self.gathered_tensors = [
            torch.empty(self.shard_elements, device=device, dtype=dtype)
            for _ in range(self.world_size)
        ]

        torch.cuda.synchronize()

    def step(self, step_num: int) -> str:
        # All-gather: collect shards from all ranks
        dist.all_gather(self.gathered_tensors, self.local_tensor)

        # Compute on gathered data (simulates forward pass)
        full_tensor = torch.cat(self.gathered_tensors)
        result = full_tensor.sum()

        # Update local tensor for next iteration
        self.local_tensor = self.local_tensor * 0.9999 + 0.0001

        torch.cuda.synchronize()

        # Only compute expensive CPU checksum at verification intervals
        if step_num % self.config.check_interval == 0:
            return compute_checksum(self.local_tensor[:10000])
        return ""

    def cleanup(self):
        del self.local_tensor, self.gathered_tensors
        torch.cuda.empty_cache()


class NCCLReduceScatterTest(StressTest):
    """
    Stress test for NCCL ReduceScatter operations.

    Tests the reduce-scatter pattern used by FSDP backward pass.
    """

    name = "nccl_reducescatter"

    def __init__(self, config: StressTestConfig):
        super().__init__(config)
        self.input_tensor: torch.Tensor = None
        self.output_tensor: torch.Tensor = None

        # Total size = shard_size * world_size
        self.shard_elements = 64 * 1024 * 1024

    def setup(self):
        if not dist.is_initialized():
            raise RuntimeError("NCCL test requires distributed initialization")

        device = torch.device("cuda")
        dtype = self.config.torch_dtype

        torch.manual_seed(self.config.seed + self.rank)

        total_elements = self.shard_elements * self.world_size
        self.input_tensor = torch.randn(
            total_elements,
            device=device,
            dtype=dtype
        )

        self.output_tensor = torch.empty(
            self.shard_elements,
            device=device,
            dtype=dtype
        )

        torch.cuda.synchronize()

    def step(self, step_num: int) -> str:
        # Reduce-scatter: reduce and distribute shards
        dist.reduce_scatter_tensor(self.output_tensor, self.input_tensor)

        # Update input for next iteration
        self.input_tensor = self.input_tensor * 0.9999 + 0.0001

        torch.cuda.synchronize()

        # Only compute expensive CPU checksum at verification intervals
        if step_num % self.config.check_interval == 0:
            return compute_checksum(self.output_tensor[:10000])
        return ""

    def cleanup(self):
        del self.input_tensor, self.output_tensor
        torch.cuda.empty_cache()


class NCCLMixedTest(StressTest):
    """
    Mixed NCCL operations stress test.

    Combines all-gather, reduce-scatter, and all-reduce in a pattern
    similar to real FSDP training.
    """

    name = "nccl_mixed"

    def __init__(self, config: StressTestConfig):
        super().__init__(config)
        self.shard: torch.Tensor = None
        self.grad_shard: torch.Tensor = None
        self.gathered: list = None

        self.shard_elements = 64 * 1024 * 1024

    def setup(self):
        if not dist.is_initialized():
            raise RuntimeError("NCCL test requires distributed initialization")

        device = torch.device("cuda")
        dtype = self.config.torch_dtype

        torch.manual_seed(self.config.seed + self.rank)

        self.shard = torch.randn(
            self.shard_elements,
            device=device,
            dtype=dtype
        )

        self.grad_shard = torch.empty(
            self.shard_elements,
            device=device,
            dtype=dtype
        )

        self.gathered = [
            torch.empty(self.shard_elements, device=device, dtype=dtype)
            for _ in range(self.world_size)
        ]

        torch.cuda.synchronize()

    def step(self, step_num: int) -> str:
        # Phase 1: All-gather (FSDP forward)
        dist.all_gather(self.gathered, self.shard)
        full_weights = torch.cat(self.gathered)

        # Phase 2: Compute (simulate matmul)
        compute_result = full_weights.sum()

        # Phase 3: Generate gradients
        grad_full = full_weights * 0.001

        # Phase 4: Reduce-scatter (FSDP backward)
        dist.reduce_scatter_tensor(self.grad_shard, grad_full)

        # Phase 5: Update shard (optimizer step)
        self.shard = self.shard - 0.01 * self.grad_shard

        torch.cuda.synchronize()

        # Only compute expensive CPU checksum at verification intervals
        if step_num % self.config.check_interval == 0:
            return compute_checksum(self.shard[:10000])
        return ""

    def cleanup(self):
        del self.shard, self.grad_shard, self.gathered
        torch.cuda.empty_cache()
