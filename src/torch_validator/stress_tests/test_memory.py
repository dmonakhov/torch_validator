"""
Memory bandwidth stress test.

Saturates HBM bandwidth with continuous large matrix operations
to expose memory-related hardware faults.
"""

import torch

from torch_validator.stress_tests.base import StressTest, StressTestConfig, compute_checksum


class MemoryBandwidthTest(StressTest):
    """
    Stress test for GPU memory bandwidth.

    Creates sustained high-bandwidth memory access patterns using
    large matrix multiplications. This tests:
    - HBM read/write reliability
    - Memory controller stability
    - Thermal behavior under sustained memory load
    """

    name = "memory_bandwidth"

    def __init__(self, config: StressTestConfig):
        super().__init__(config)
        self.A: torch.Tensor = None
        self.B: torch.Tensor = None
        self.C: torch.Tensor = None

        # Matrix size to saturate bandwidth
        # 8192 x 8192 bfloat16 = 128MB per matrix
        # Each matmul reads 256MB, writes 128MB
        self.matrix_size = 8192

    def setup(self):
        """Initialize large matrices."""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dtype = self.config.torch_dtype

        # Initialize with deterministic values
        torch.manual_seed(self.config.seed + self.rank)

        self.A = torch.randn(
            self.matrix_size, self.matrix_size,
            device=device, dtype=dtype
        )
        self.B = torch.randn(
            self.matrix_size, self.matrix_size,
            device=device, dtype=dtype
        )
        self.C = torch.zeros(
            self.matrix_size, self.matrix_size,
            device=device, dtype=dtype
        )

        # Synchronize before starting
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    def step(self, step_num: int) -> str:
        """Execute one step of memory stress."""
        # Matrix multiply: reads A and B, writes C
        self.C = torch.mm(self.A, self.B)

        # Accumulate into A: reads C and A, writes A
        self.A = self.A + self.C * 0.001  # Small factor to prevent overflow

        # Sync GPU to prevent command queue overflow
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        # Only compute expensive CPU checksum at verification intervals
        if step_num % self.config.check_interval == 0:
            return compute_checksum(self.A.flatten()[:10000])
        return ""

    def cleanup(self):
        """Release memory."""
        del self.A, self.B, self.C
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


class MemoryPatternTest(StressTest):
    """
    Alternative memory test with varied access patterns.

    Tests different memory access patterns that may expose
    specific failure modes:
    - Sequential access
    - Strided access
    - Random access
    """

    name = "memory_pattern"

    def __init__(self, config: StressTestConfig):
        super().__init__(config)
        self.tensor: torch.Tensor = None
        self.indices: torch.Tensor = None

        # Large tensor for memory stress
        self.tensor_size = 256 * 1024 * 1024  # 256M elements

    def setup(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dtype = self.config.torch_dtype

        torch.manual_seed(self.config.seed + self.rank)

        self.tensor = torch.randn(self.tensor_size, device=device, dtype=dtype)

        # Random indices for gather/scatter operations
        self.indices = torch.randint(
            0, self.tensor_size, (self.tensor_size // 4,),
            device=device
        )

        if torch.cuda.is_available():
            torch.cuda.synchronize()

    def step(self, step_num: int) -> str:
        # Alternate between access patterns
        pattern = step_num % 3

        if pattern == 0:
            # Sequential: element-wise operations
            self.tensor = self.tensor * 1.0001 + 0.0001

        elif pattern == 1:
            # Strided: reshape and transpose
            size = int(self.tensor_size ** 0.5)
            reshaped = self.tensor[:size * size].view(size, size)
            transposed = reshaped.t().contiguous()
            self.tensor[:size * size] = transposed.view(-1)

        else:
            # Random: gather and scatter
            gathered = self.tensor[self.indices]
            self.tensor[self.indices] = gathered * 1.0001

        # Sync GPU to prevent command queue overflow
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        # Only compute expensive CPU checksum at verification intervals
        if step_num % self.config.check_interval == 0:
            return compute_checksum(self.tensor[:1000])
        return ""

    def cleanup(self):
        del self.tensor, self.indices
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
