#!/usr/bin/env python
"""
Example: LLM training with golden validation.

This example shows how to integrate torch-validator with a training loop.

Usage:
    # Step 1: Record golden data on known-good host
    python train_with_validation.py --mode record --output golden.json --steps 100

    # Step 2: Validate on suspected host
    python train_with_validation.py --mode validate --golden golden.json --steps 100

    # Step 3: Test corruption detection (inject error)
    python train_with_validation.py --mode validate --golden golden.json --steps 100 --inject-error
"""

import argparse
import os

import torch
import torch.nn as nn

from torch_validator import GoldenVerifier, set_deterministic_mode


class SimpleModel(nn.Module):
    """Simple model for testing."""

    def __init__(self, hidden_size: int = 1024, num_layers: int = 4):
        super().__init__()
        layers = []
        for _ in range(num_layers):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(nn.ReLU())
        self.layers = nn.Sequential(*layers)
        self.output = nn.Linear(hidden_size, hidden_size)

    def forward(self, x):
        return self.output(self.layers(x))


def main():
    parser = argparse.ArgumentParser(description="Training with golden validation")
    parser.add_argument("--mode", choices=["record", "validate"], required=True)
    parser.add_argument("--output", type=str, help="Output file for golden data (record mode)")
    parser.add_argument("--golden", type=str, help="Golden file to validate against (validate mode)")
    parser.add_argument("--steps", type=int, default=100, help="Number of training steps")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--hidden-size", type=int, default=1024, help="Model hidden size")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--inject-error", action="store_true", help="Inject error to test detection")
    parser.add_argument("--fail-fast", action="store_true", help="Stop on first validation failure")
    args = parser.parse_args()

    # Set deterministic mode for reproducibility
    seed_state = set_deterministic_mode(seed=args.seed)
    print(f"Deterministic mode enabled, seed={args.seed}")

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create model
    model = SimpleModel(hidden_size=args.hidden_size).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    loss_fn = nn.MSELoss()

    # Create verifier
    if args.mode == "record":
        if not args.output:
            args.output = "golden.json"
        verifier = GoldenVerifier(
            mode="record",
            output_file=args.output,
            seed_state=seed_state,
        )
    else:
        if not args.golden:
            raise ValueError("--golden required for validate mode")
        verifier = GoldenVerifier(
            mode="validate",
            golden_file=args.golden,
            fail_fast=args.fail_fast,
        )

    # Training loop
    for step in range(args.steps):
        # Generate deterministic data
        torch.manual_seed(args.seed + step)
        data = torch.randn(args.batch_size, args.hidden_size, device=device)
        target = torch.randn(args.batch_size, args.hidden_size, device=device)

        # Forward pass
        optimizer.zero_grad()
        output = model(data)
        loss = loss_fn(output, target)

        # Backward pass
        loss.backward()

        # === VERIFIER HOOK: after backward, before optimizer.step() ===
        # This captures gradients before they're consumed by optimizer
        if args.mode == "record":
            verifier.record(step, loss=loss, model=model)
        else:
            verifier.validate(step, loss=loss, model=model)

        # Inject error if requested (for testing) - after validation to corrupt future steps
        if args.inject_error and step == 50:
            print(f"\n[INJECTING ERROR] Corrupting model weights at step {step}")
            with torch.no_grad():
                for param in model.parameters():
                    param.data += torch.randn_like(param) * 0.1
                    break  # Corrupt just one parameter

        # Optimizer step
        optimizer.step()

    # Finalize
    if args.mode == "record":
        verifier.save()
        print(f"\nGolden data saved to: {args.output}")
    else:
        result = verifier.report()
        print(f"\nValidation result: {'PASS' if result.passed else 'FAIL'}")
        exit(0 if result.passed else 1)


if __name__ == "__main__":
    main()
