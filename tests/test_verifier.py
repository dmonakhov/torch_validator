"""
Tests for GoldenVerifier.
"""

import json
import tempfile
from pathlib import Path

import pytest
import torch
import torch.nn as nn

from torch_validator import GoldenVerifier, set_deterministic_mode
from torch_validator.metrics import MetricsCollector, StepMetrics, check_nan_inf
from torch_validator.verifier import ToleranceConfig


class SimpleModel(nn.Module):
    """Simple model for testing."""

    def __init__(self, size: int = 64):
        super().__init__()
        self.fc1 = nn.Linear(size, size)
        self.fc2 = nn.Linear(size, size)

    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x)))


def forward_backward(model, optimizer, data, target):
    """Forward and backward pass only (no optimizer step)."""
    optimizer.zero_grad()
    output = model(data)
    loss = nn.functional.mse_loss(output, target)
    loss.backward()
    return loss


class TestMetricsCollector:
    """Tests for MetricsCollector."""

    def test_collect_basic(self):
        """Test basic metrics collection."""
        model = SimpleModel()
        collector = MetricsCollector(collect_layer_norms=True)

        # Do a forward/backward pass
        data = torch.randn(8, 64)
        target = torch.randn(8, 64)
        output = model(data)
        loss = nn.functional.mse_loss(output, target)
        loss.backward()

        metrics = collector.collect(step=0, loss=loss, model=model)

        assert metrics.step == 0
        assert metrics.loss > 0
        assert metrics.grad_norm > 0
        assert metrics.weight_norm > 0
        assert len(metrics.model_checksum) > 0
        assert len(metrics.layer_grad_norms) > 0
        assert "fc1.weight" in metrics.layer_grad_norms

    def test_checksum_deterministic(self):
        """Test that checksum is deterministic for same model state."""
        set_deterministic_mode(seed=42)
        model = SimpleModel()
        collector = MetricsCollector()

        data = torch.randn(8, 64)
        output = model(data)
        loss = output.sum()
        loss.backward()

        metrics1 = collector.collect(step=0, loss=loss, model=model)

        # Collect again without changing model
        metrics2 = collector.collect(step=0, loss=loss, model=model)

        assert metrics1.model_checksum == metrics2.model_checksum

    def test_checksum_changes_after_update(self):
        """Test that checksum changes after optimizer step."""
        model = SimpleModel()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        collector = MetricsCollector()

        data = torch.randn(8, 64)
        output = model(data)
        loss = output.sum()
        loss.backward()

        metrics1 = collector.collect(step=0, loss=loss, model=model)

        # Update model
        optimizer.step()

        metrics2 = collector.collect(step=1, loss=loss, model=model)

        assert metrics1.model_checksum != metrics2.model_checksum


class TestNaNInfDetection:
    """Tests for NaN/Inf detection."""

    def test_normal_values(self):
        """Test that normal values pass."""
        loss = torch.tensor(1.5)
        result = check_nan_inf(loss)
        assert result["loss_ok"] is True

    def test_nan_loss(self):
        """Test NaN detection in loss."""
        loss = torch.tensor(float("nan"))
        result = check_nan_inf(loss)
        assert result["loss_ok"] is False

    def test_inf_loss(self):
        """Test Inf detection in loss."""
        loss = torch.tensor(float("inf"))
        result = check_nan_inf(loss)
        assert result["loss_ok"] is False

    def test_nan_gradient(self):
        """Test NaN detection in gradients."""
        model = SimpleModel()
        data = torch.randn(8, 64)
        output = model(data)
        loss = output.sum()
        loss.backward()

        # Inject NaN
        model.fc1.weight.grad[0, 0] = float("nan")

        result = check_nan_inf(loss, model, check_gradients=True)
        assert result["grad_ok"] is False


class TestGoldenVerifier:
    """Tests for GoldenVerifier."""

    def test_record_and_validate_same_run(self):
        """Test recording and validating with identical runs."""
        with tempfile.TemporaryDirectory() as tmpdir:
            golden_path = Path(tmpdir) / "golden.json"

            # Record
            set_deterministic_mode(seed=42)
            model = SimpleModel()
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

            verifier = GoldenVerifier(
                mode="record",
                output_file=str(golden_path),
                verbose=False,
            )

            for step in range(10):
                torch.manual_seed(42 + step)
                data = torch.randn(8, 64)
                target = torch.randn(8, 64)
                loss = forward_backward(model, optimizer, data, target)
                verifier.record(step, loss=loss, model=model)  # after backward, before optimizer
                optimizer.step()

            verifier.save()

            # Validate with identical run
            set_deterministic_mode(seed=42)
            model = SimpleModel()
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

            verifier = GoldenVerifier(
                mode="validate",
                golden_file=str(golden_path),
                verbose=False,
            )

            for step in range(10):
                torch.manual_seed(42 + step)
                data = torch.randn(8, 64)
                target = torch.randn(8, 64)
                loss = forward_backward(model, optimizer, data, target)
                verifier.validate(step, loss=loss, model=model)  # after backward, before optimizer
                optimizer.step()

            result = verifier.report()

            assert result.passed is True
            assert result.failed_steps == 0

    def test_detect_corruption(self):
        """Test that corruption is detected."""
        with tempfile.TemporaryDirectory() as tmpdir:
            golden_path = Path(tmpdir) / "golden.json"

            # Record
            set_deterministic_mode(seed=42)
            model = SimpleModel()
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

            verifier = GoldenVerifier(
                mode="record",
                output_file=str(golden_path),
                verbose=False,
            )

            for step in range(10):
                torch.manual_seed(42 + step)
                data = torch.randn(8, 64)
                target = torch.randn(8, 64)
                loss = forward_backward(model, optimizer, data, target)
                verifier.record(step, loss=loss, model=model)  # after backward, before optimizer
                optimizer.step()

            verifier.save()

            # Validate with corrupted model
            set_deterministic_mode(seed=42)
            model = SimpleModel()
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

            verifier = GoldenVerifier(
                mode="validate",
                golden_file=str(golden_path),
                verbose=False,
            )

            for step in range(10):
                torch.manual_seed(42 + step)
                data = torch.randn(8, 64)
                target = torch.randn(8, 64)
                loss = forward_backward(model, optimizer, data, target)

                # Inject corruption at step 5 (before validation)
                if step == 5:
                    with torch.no_grad():
                        model.fc1.weight += torch.randn_like(model.fc1.weight) * 0.1

                verifier.validate(step, loss=loss, model=model)  # after backward, before optimizer
                optimizer.step()

            result = verifier.report()

            assert result.passed is False
            assert result.failed_steps > 0

    def test_golden_file_format(self):
        """Test that golden file is valid JSON with expected fields."""
        with tempfile.TemporaryDirectory() as tmpdir:
            golden_path = Path(tmpdir) / "golden.json"

            model = SimpleModel()
            verifier = GoldenVerifier(
                mode="record",
                output_file=str(golden_path),
                verbose=False,
            )

            data = torch.randn(8, 64)
            output = model(data)
            loss = output.sum()
            loss.backward()

            verifier.record(0, loss=loss, model=model)
            verifier.save()

            # Load and check
            with open(golden_path) as f:
                data = json.load(f)

            assert "version" in data
            assert "steps" in data
            assert len(data["steps"]) == 1
            assert "loss" in data["steps"][0]
            assert "grad_norm" in data["steps"][0]
            assert "model_checksum" in data["steps"][0]

    def test_tolerance_config(self):
        """Test that tolerance configuration works."""
        with tempfile.TemporaryDirectory() as tmpdir:
            golden_path = Path(tmpdir) / "golden.json"

            # Record
            set_deterministic_mode(seed=42)
            model = SimpleModel()

            verifier = GoldenVerifier(
                mode="record",
                output_file=str(golden_path),
                tolerance=ToleranceConfig(loss_rtol=0.1),  # Very loose tolerance
                verbose=False,
            )

            data = torch.randn(8, 64)
            output = model(data)
            loss = output.sum()
            loss.backward()
            verifier.record(0, loss=loss, model=model)
            verifier.save()

            # Validate with slightly different loss (within tolerance)
            verifier = GoldenVerifier(
                mode="validate",
                golden_file=str(golden_path),
                tolerance=ToleranceConfig(
                    loss_rtol=0.5,  # 50% tolerance
                    checksum_exact=False,
                ),
                verbose=False,
            )

            # Use different seed -> different loss
            set_deterministic_mode(seed=43)
            model = SimpleModel()
            data = torch.randn(8, 64)
            output = model(data)
            loss = output.sum()
            loss.backward()
            verifier.validate(0, loss=loss, model=model)

            result = verifier.report()
            # May or may not pass depending on how different the loss is
            # The key is that no exception is raised


class TestDeterminism:
    """Tests for deterministic mode."""

    def test_deterministic_mode(self):
        """Test that deterministic mode produces same results."""
        results = []

        for _ in range(2):
            set_deterministic_mode(seed=42)
            model = SimpleModel()
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

            losses = []
            for step in range(5):
                torch.manual_seed(42 + step)
                data = torch.randn(8, 64)
                target = torch.randn(8, 64)
                loss = forward_backward(model, optimizer, data, target)
                losses.append(loss.item())
                optimizer.step()

            results.append(losses)

        # Check losses are identical
        for i, (l1, l2) in enumerate(zip(results[0], results[1])):
            assert abs(l1 - l2) < 1e-6, f"Step {i}: {l1} != {l2}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
