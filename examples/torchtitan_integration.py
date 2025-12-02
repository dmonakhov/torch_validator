#!/usr/bin/env python
"""
Example: Integration with TorchTitan training script.

Shows how to add golden validation to existing TorchTitan training code
with minimal changes.

Integration pattern:
    1. Import GoldenVerifier
    2. Create verifier in record or validate mode
    3. Call verifier.record() or verifier.validate() after each step
    4. Call verifier.save() or verifier.report() at the end
"""

# ============================================================================
# MINIMAL INTEGRATION PATTERN
# ============================================================================
#
# Add these changes to your existing TorchTitan train.py:
#
# 1. At the top of the file:
#
#     from torch_validator import GoldenVerifier, set_deterministic_mode
#
# 2. After config parsing, before training loop:
#
#     # For hardware validation runs only
#     if config.validation_mode == "record":
#         seed_state = set_deterministic_mode(seed=config.seed)
#         verifier = GoldenVerifier(
#             mode="record",
#             output_file=config.golden_output,
#             seed_state=seed_state,
#         )
#     elif config.validation_mode == "validate":
#         seed_state = set_deterministic_mode(seed=config.seed)
#         verifier = GoldenVerifier(
#             mode="validate",
#             golden_file=config.golden_file,
#         )
#     else:
#         verifier = None
#
# 3. Inside training loop, after backward() and before optimizer.step():
#
#     if verifier:
#         if verifier.mode.value == "record":
#             verifier.record(step, loss=loss, model=model)
#         else:
#             verifier.validate(step, loss=loss, model=model)
#
# 4. After training loop:
#
#     if verifier:
#         if verifier.mode.value == "record":
#             verifier.save()
#         else:
#             result = verifier.report()
#             if not result.passed:
#                 raise RuntimeError("Hardware validation failed")
#
# ============================================================================

from typing import Optional

import torch
import torch.nn as nn

# Import torch-validator
from torch_validator import GoldenVerifier, set_deterministic_mode
from torch_validator.verifier import VerifierMode


class TitanValidationWrapper:
    """
    Wrapper to add golden validation to TorchTitan training.

    Usage:
        # Initialize once
        wrapper = TitanValidationWrapper(
            mode="validate",
            golden_file="golden.json",
            seed=42,
        )

        # In training loop
        for step in range(num_steps):
            loss = train_step(...)
            wrapper.step(step, loss, model)

        # After training
        wrapper.finalize()
    """

    def __init__(
        self,
        mode: Optional[str] = None,  # "record", "validate", or None
        output_file: Optional[str] = None,
        golden_file: Optional[str] = None,
        seed: int = 42,
        fail_fast: bool = False,
        verbose: bool = True,
    ):
        """
        Initialize validation wrapper.

        Args:
            mode: "record", "validate", or None (disabled)
            output_file: Output path for golden data (record mode)
            golden_file: Golden data path (validate mode)
            seed: Random seed for deterministic training
            fail_fast: Stop on first validation failure
            verbose: Print progress messages
        """
        self.mode = mode
        self.verifier: Optional[GoldenVerifier] = None
        self.seed_state = None

        if mode is None:
            if verbose:
                print("[VALIDATOR] Validation disabled")
            return

        # Enable deterministic mode
        self.seed_state = set_deterministic_mode(seed=seed)
        if verbose:
            print(f"[VALIDATOR] Deterministic mode enabled, seed={seed}")

        # Create verifier
        if mode == "record":
            self.verifier = GoldenVerifier(
                mode="record",
                output_file=output_file or "golden.json",
                seed_state=self.seed_state,
                verbose=verbose,
            )
        elif mode == "validate":
            if not golden_file:
                raise ValueError("golden_file required for validate mode")
            self.verifier = GoldenVerifier(
                mode="validate",
                golden_file=golden_file,
                fail_fast=fail_fast,
                verbose=verbose,
            )
        else:
            raise ValueError(f"Invalid mode: {mode}")

    def step(
        self,
        step: int,
        loss: torch.Tensor,
        model: nn.Module,
    ) -> None:
        """
        Process a training step.

        Call this after backward() and before optimizer.step().

        Args:
            step: Current training step
            loss: Loss tensor
            model: Model with gradients populated
        """
        if self.verifier is None:
            return

        if self.verifier.mode == VerifierMode.RECORD:
            self.verifier.record(step, loss=loss, model=model)
        else:
            self.verifier.validate(step, loss=loss, model=model)

    def finalize(self) -> bool:
        """
        Finalize validation.

        Returns:
            True if validation passed (or disabled), False if failed
        """
        if self.verifier is None:
            return True

        if self.verifier.mode == VerifierMode.RECORD:
            self.verifier.save()
            return True
        else:
            result = self.verifier.report()
            return result.passed


# ============================================================================
# EXAMPLE: Simulated TorchTitan training loop
# ============================================================================

def example_torchtitan_train():
    """Example showing integration with TorchTitan-style training."""
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--validation-mode", choices=["record", "validate", "none"], default="none")
    parser.add_argument("--golden-output", type=str, default="golden.json")
    parser.add_argument("--golden-file", type=str)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=50)
    args = parser.parse_args()

    # === TorchTitan-style setup ===
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Simple model (replace with actual TorchTitan model)
    model = nn.Sequential(
        nn.Linear(512, 512),
        nn.ReLU(),
        nn.Linear(512, 512),
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    loss_fn = nn.MSELoss()

    # === Add validation wrapper ===
    validation = TitanValidationWrapper(
        mode=args.validation_mode if args.validation_mode != "none" else None,
        output_file=args.golden_output,
        golden_file=args.golden_file,
        seed=args.seed,
    )

    # === Training loop ===
    for step in range(args.steps):
        # Deterministic data generation
        torch.manual_seed(args.seed + step)
        data = torch.randn(32, 512, device=device)
        target = torch.randn(32, 512, device=device)

        # Forward
        optimizer.zero_grad()
        output = model(data)
        loss = loss_fn(output, target)

        # Backward
        loss.backward()

        # === Validation hook (after backward, before optimizer.step) ===
        validation.step(step, loss, model)

        # Optimizer step
        optimizer.step()

        if step % 10 == 0:
            print(f"Step {step}: loss={loss.item():.6f}")

    # === Finalize validation ===
    passed = validation.finalize()
    if not passed:
        print("HARDWARE VALIDATION FAILED")
        exit(1)
    print("Training complete")


if __name__ == "__main__":
    example_torchtitan_train()
