"""
GoldenVerifier - Main validation class for hardware qualification.

Two modes:
1. RECORD: Capture metrics on known-good host, save as golden reference
2. VALIDATE: Compare metrics against golden reference on suspected host
"""

import json
import logging
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Union

import torch
import torch.nn as nn

from torch_validator.deterministic import SeedState, check_determinism_config
from torch_validator.metrics import MetricsCollector, StepMetrics, check_nan_inf

logger = logging.getLogger(__name__)


class VerifierMode(Enum):
    """Operating mode for GoldenVerifier."""

    RECORD = "record"
    VALIDATE = "validate"


@dataclass
class ToleranceConfig:
    """Tolerance thresholds for validation comparison."""

    # Relative tolerance for floating point comparisons
    loss_rtol: float = 1e-5
    grad_norm_rtol: float = 1e-4
    weight_norm_rtol: float = 1e-5

    # Absolute tolerance (used when value is near zero)
    loss_atol: float = 1e-8
    grad_norm_atol: float = 1e-8
    weight_norm_atol: float = 1e-8

    # Checksum must match exactly
    checksum_exact: bool = True

    def to_dict(self) -> dict:
        """Convert to JSON-serializable dict."""
        return {
            "loss_rtol": self.loss_rtol,
            "grad_norm_rtol": self.grad_norm_rtol,
            "weight_norm_rtol": self.weight_norm_rtol,
            "loss_atol": self.loss_atol,
            "grad_norm_atol": self.grad_norm_atol,
            "weight_norm_atol": self.weight_norm_atol,
            "checksum_exact": self.checksum_exact,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ToleranceConfig":
        """Create from dict."""
        return cls(**d)


@dataclass
class StepDeviation:
    """Deviation details for a single step."""

    step: int
    field: str  # "loss", "grad_norm", "weight_norm", "checksum"
    expected: Union[float, str]
    actual: Union[float, str]
    relative_error: Optional[float] = None  # For numeric fields
    passed: bool = False

    def to_dict(self) -> dict:
        """Convert to JSON-serializable dict."""
        return {
            "step": self.step,
            "field": self.field,
            "expected": self.expected,
            "actual": self.actual,
            "relative_error": self.relative_error,
            "passed": self.passed,
        }


@dataclass
class ValidationResult:
    """Result of validation run."""

    passed: bool
    total_steps: int
    failed_steps: int
    deviations: List[StepDeviation] = field(default_factory=list)
    nan_inf_errors: List[Dict] = field(default_factory=list)
    summary: str = ""

    def to_dict(self) -> dict:
        """Convert to JSON-serializable dict."""
        return {
            "passed": self.passed,
            "total_steps": self.total_steps,
            "failed_steps": self.failed_steps,
            "deviations": [d.to_dict() for d in self.deviations],
            "nan_inf_errors": self.nan_inf_errors,
            "summary": self.summary,
        }


@dataclass
class GoldenData:
    """Golden reference data."""

    version: str = "1.0"
    seed_state: Optional[SeedState] = None
    determinism_config: Optional[Dict] = None
    tolerance: Optional[ToleranceConfig] = None
    steps: List[StepMetrics] = field(default_factory=list)
    metadata: Dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Convert to JSON-serializable dict."""
        return {
            "version": self.version,
            "seed_state": self.seed_state.to_dict() if self.seed_state else None,
            "determinism_config": self.determinism_config,
            "tolerance": self.tolerance.to_dict() if self.tolerance else None,
            "steps": [s.to_dict() for s in self.steps],
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "GoldenData":
        """Create from dict."""
        return cls(
            version=d.get("version", "1.0"),
            seed_state=SeedState.from_dict(d["seed_state"]) if d.get("seed_state") else None,
            determinism_config=d.get("determinism_config"),
            tolerance=ToleranceConfig.from_dict(d["tolerance"]) if d.get("tolerance") else None,
            steps=[StepMetrics.from_dict(s) for s in d.get("steps", [])],
            metadata=d.get("metadata", {}),
        )


class GoldenVerifier:
    """
    Hardware validation via golden reference comparison.

    Usage:
        # Record mode (on known-good host)
        verifier = GoldenVerifier(mode="record", output_file="golden.json")
        for step in range(num_steps):
            loss = train_step(...)
            verifier.record(step, loss=loss, model=model)
        verifier.save()

        # Validate mode (on suspected host)
        verifier = GoldenVerifier(mode="validate", golden_file="golden.json")
        for step in range(num_steps):
            loss = train_step(...)
            verifier.validate(step, loss=loss, model=model)
        result = verifier.report()
    """

    def __init__(
        self,
        mode: Union[str, VerifierMode] = "record",
        output_file: Optional[str] = None,
        golden_file: Optional[str] = None,
        rank: Optional[int] = None,
        tolerance: Optional[ToleranceConfig] = None,
        seed_state: Optional[SeedState] = None,
        collect_layer_norms: bool = False,
        checksum_enabled: bool = False,
        fail_fast: bool = False,
        verbose: bool = True,
    ):
        """
        Initialize GoldenVerifier.

        Args:
            mode: "record" or "validate"
            output_file: Path to save golden data (record mode). Use {rank} placeholder for rank.
            golden_file: Path to load golden data (validate mode). Use {rank} placeholder for rank.
            rank: Distributed rank for file naming. If None, tries to get from torch.distributed.
            tolerance: Tolerance configuration for validation
            seed_state: Seed state to record (record mode)
            collect_layer_norms: If True, collect per-layer norms (adds overhead)
            checksum_enabled: If True, compute model checksum (expensive, requires CPU transfer)
            fail_fast: If True, raise exception on first validation failure
            verbose: If True, print progress messages
        """
        if isinstance(mode, str):
            mode = VerifierMode(mode.lower())
        self.mode = mode

        # Get rank for file naming
        if rank is None:
            rank = self._get_rank()
        self.rank = rank

        # Format file paths with rank
        self.output_file = self._format_path(output_file, rank)
        self.golden_file = self._format_path(golden_file, rank)

        # Debug logging for rank issues
        logger.debug(f"GoldenVerifier init: passed rank={rank}, golden_file param={golden_file}, formatted={self.golden_file}")
        self.tolerance = tolerance or ToleranceConfig()
        self.fail_fast = fail_fast
        self.verbose = verbose

        self._collector = MetricsCollector(
            collect_layer_norms=collect_layer_norms,
            checksum_enabled=checksum_enabled,
        )
        self._golden_data: Optional[GoldenData] = None
        self._step_lookup: Dict[int, StepMetrics] = {}  # step_number -> metrics
        self._recorded_steps: List[StepMetrics] = []
        self._validated_steps: List[StepMetrics] = []  # Store validation run metrics too
        self._deviations: List[StepDeviation] = []
        self._nan_inf_errors: List[Dict] = []
        self._failed_steps: set = set()

        # Load golden data in validate mode
        if self.mode == VerifierMode.VALIDATE:
            if not golden_file:
                raise ValueError("golden_file required for validate mode")
            self._load_golden()
            if self.verbose:
                logger.info(f"Validate mode: rank={self.rank}, golden={self.golden_file}")

        # Record seed state and determinism config
        if self.mode == VerifierMode.RECORD:
            self._seed_state = seed_state
            self._determinism_config = check_determinism_config()
            if self.verbose:
                logger.info(f"Record mode: rank={self.rank}, output={self.output_file}")

    @staticmethod
    def _get_rank() -> int:
        """Get distributed rank, or 0 if not distributed."""
        try:
            import torch.distributed as dist
            if dist.is_initialized():
                return dist.get_rank()
        except Exception:
            pass
        return 0

    @staticmethod
    def _format_path(path: Optional[str], rank: int) -> Optional[Path]:
        """Format path with rank placeholder."""
        if path is None:
            return None
        # Support {rank} placeholder
        formatted = path.format(rank=rank)
        return Path(formatted)

    def _load_golden(self) -> None:
        """Load golden data from file."""
        logger.info(f"[rank {self.rank}] Loading golden file: {self.golden_file}")
        with open(self.golden_file, "r") as f:
            data = json.load(f)
        self._golden_data = GoldenData.from_dict(data)

        # Build step lookup dict (step_number -> metrics)
        self._step_lookup = {s.step: s for s in self._golden_data.steps}
        step_range = sorted(self._step_lookup.keys())
        logger.info(f"[rank {self.rank}] Golden steps: {step_range[0]}-{step_range[-1]} ({len(step_range)} steps)")

        # Use tolerance from golden file if not specified
        if self._golden_data.tolerance and self.tolerance == ToleranceConfig():
            self.tolerance = self._golden_data.tolerance

    def record(
        self,
        step: int,
        loss: Union[float, torch.Tensor],
        model: nn.Module,
    ) -> StepMetrics:
        """
        Record metrics at a training step (record mode).

        Args:
            step: Current training step
            loss: Loss value
            model: Model with gradients populated

        Returns:
            StepMetrics collected at this step
        """
        if self.mode != VerifierMode.RECORD:
            raise RuntimeError("record() only available in record mode")

        # Check for NaN/Inf
        nan_inf = check_nan_inf(loss, model, check_gradients=True, check_weights=True)
        if not all(nan_inf.values()):
            error = {"step": step, **nan_inf}
            self._nan_inf_errors.append(error)
            if self.verbose:
                logger.warning(f"Step {step}: NaN/Inf detected: {nan_inf}")

        # Collect metrics
        metrics = self._collector.collect(step, loss, model)
        self._recorded_steps.append(metrics)

        if self.verbose and step % 10 == 0:
            checksum_str = metrics.model_checksum[:8] + "..." if metrics.model_checksum else "disabled"
            logger.info(f"Step {step}: loss={metrics.loss:.6f}, grad_norm={metrics.grad_norm:.6f}, checksum={checksum_str}")

        return metrics

    def validate(
        self,
        step: int,
        loss: Union[float, torch.Tensor],
        model: nn.Module,
    ) -> List[StepDeviation]:
        """
        Validate metrics against golden reference (validate mode).

        Args:
            step: Current training step
            loss: Loss value
            model: Model with gradients populated

        Returns:
            List of deviations detected at this step (empty if passed)
        """
        if self.mode != VerifierMode.VALIDATE:
            raise RuntimeError("validate() only available in validate mode")

        # Check for NaN/Inf first
        nan_inf = check_nan_inf(loss, model, check_gradients=True, check_weights=True)
        if not all(nan_inf.values()):
            error = {"step": step, **nan_inf}
            self._nan_inf_errors.append(error)
            self._failed_steps.add(step)
            if self.verbose:
                logger.error(f"Step {step}: [FAIL] NaN/Inf detected: {nan_inf}")
            if self.fail_fast:
                raise RuntimeError(f"NaN/Inf detected at step {step}")

        # Collect current metrics
        metrics = self._collector.collect(step, loss, model)
        self._validated_steps.append(metrics)

        # Get golden metrics for this step - lookup by step number, not array index
        golden = self._step_lookup.get(step)
        if golden is None:
            raise RuntimeError(
                f"Step {step} not found in golden data (available: {list(self._step_lookup.keys())[:10]}...)"
            )

        # Log step 0 actual values to help diagnose seed issues
        if step == 0 and self.verbose:
            logger.info(f"Actual step 0: loss={metrics.loss:.6f}, grad_norm={metrics.grad_norm:.6f}, weight_norm={metrics.weight_norm:.6f}")

        # Compare metrics
        step_deviations = self._compare_metrics(step, metrics, golden)

        if step_deviations:
            self._failed_steps.add(step)
            self._deviations.extend(step_deviations)
            if self.verbose:
                for dev in step_deviations:
                    logger.error(
                        f"[rank {self.rank}] Step {step}: [FAIL] {dev.field}: "
                        f"expected={dev.expected}, actual={dev.actual}, "
                        f"rel_err={dev.relative_error}"
                    )
            if self.fail_fast:
                raise RuntimeError(f"Validation failed at step {step}: {step_deviations}")
        elif self.verbose and step % 10 == 0:
            logger.info(f"[rank {self.rank}] Step {step}: [PASS] loss={metrics.loss:.6f} (golden={golden.loss:.6f})")

        return step_deviations

    def _compare_metrics(
        self,
        step: int,
        actual: StepMetrics,
        golden: StepMetrics,
    ) -> List[StepDeviation]:
        """Compare actual metrics against golden reference."""
        deviations = []

        # Compare loss
        dev = self._compare_float(
            step, "loss", golden.loss, actual.loss,
            self.tolerance.loss_rtol, self.tolerance.loss_atol
        )
        if dev:
            deviations.append(dev)

        # Compare grad_norm
        dev = self._compare_float(
            step, "grad_norm", golden.grad_norm, actual.grad_norm,
            self.tolerance.grad_norm_rtol, self.tolerance.grad_norm_atol
        )
        if dev:
            deviations.append(dev)

        # Compare weight_norm
        dev = self._compare_float(
            step, "weight_norm", golden.weight_norm, actual.weight_norm,
            self.tolerance.weight_norm_rtol, self.tolerance.weight_norm_atol
        )
        if dev:
            deviations.append(dev)

        # Compare checksum (skip if disabled - empty checksum)
        if self.tolerance.checksum_exact and actual.model_checksum and golden.model_checksum:
            if actual.model_checksum != golden.model_checksum:
                deviations.append(StepDeviation(
                    step=step,
                    field="checksum",
                    expected=golden.model_checksum,
                    actual=actual.model_checksum,
                    relative_error=None,
                    passed=False,
                ))

        return deviations

    def _compare_float(
        self,
        step: int,
        field: str,
        expected: float,
        actual: float,
        rtol: float,
        atol: float,
    ) -> Optional[StepDeviation]:
        """Compare two float values with tolerance."""
        # Compute relative error
        if abs(expected) > atol:
            rel_err = abs(actual - expected) / abs(expected)
        else:
            rel_err = abs(actual - expected)

        # Check if within tolerance
        passed = abs(actual - expected) <= atol + rtol * abs(expected)

        if not passed:
            return StepDeviation(
                step=step,
                field=field,
                expected=expected,
                actual=actual,
                relative_error=rel_err,
                passed=False,
            )
        return None

    def save(self, output_file: Optional[str] = None) -> Path:
        """
        Save recorded golden data to file (record mode).

        Args:
            output_file: Override output path

        Returns:
            Path to saved file
        """
        if self.mode != VerifierMode.RECORD:
            raise RuntimeError("save() only available in record mode")

        path = Path(output_file) if output_file else self.output_file
        if not path:
            raise ValueError("No output file specified")

        golden_data = GoldenData(
            version="1.0",
            seed_state=self._seed_state,
            determinism_config=self._determinism_config,
            tolerance=self.tolerance,
            steps=self._recorded_steps,
            metadata={
                "total_steps": len(self._recorded_steps),
                "nan_inf_errors": len(self._nan_inf_errors),
            },
        )

        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(golden_data.to_dict(), f, indent=2)

        if self.verbose:
            logger.info(f"Saved golden data: {len(self._recorded_steps)} steps -> {path}")

        return path

    def report(self, output_file: Optional[str] = None) -> ValidationResult:
        """
        Generate validation report (validate mode).

        Args:
            output_file: Optional path to save report JSON

        Returns:
            ValidationResult with pass/fail status and details
        """
        if self.mode != VerifierMode.VALIDATE:
            raise RuntimeError("report() only available in validate mode")

        total_steps = len(self._golden_data.steps)
        failed_steps = len(self._failed_steps)
        passed = failed_steps == 0 and len(self._nan_inf_errors) == 0

        # Generate summary
        if passed:
            summary = f"PASSED: All {total_steps} steps validated successfully"
        else:
            summary = (
                f"FAILED: {failed_steps}/{total_steps} steps failed, "
                f"{len(self._nan_inf_errors)} NaN/Inf errors"
            )

        result = ValidationResult(
            passed=passed,
            total_steps=total_steps,
            failed_steps=failed_steps,
            deviations=self._deviations,
            nan_inf_errors=self._nan_inf_errors,
            summary=summary,
        )

        # Log report
        if self.verbose:
            log_fn = logger.info if passed else logger.error
            log_fn(f"VALIDATION REPORT: Status={'PASS' if passed else 'FAIL'}, "
                   f"total_steps={total_steps}, failed_steps={failed_steps}, "
                   f"nan_inf_errors={len(self._nan_inf_errors)}")

            if self._deviations:
                for dev in self._deviations[:20]:  # Show first 20
                    logger.error(
                        f"Deviation: Step {dev.step}: {dev.field} - "
                        f"expected={dev.expected}, actual={dev.actual}, "
                        f"rel_err={dev.relative_error}"
                    )
                if len(self._deviations) > 20:
                    logger.error(f"... and {len(self._deviations) - 20} more deviations")

        # Save report if requested
        if output_file:
            with open(output_file, "w") as f:
                json.dump(result.to_dict(), f, indent=2)

        return result

    def save_validation_metrics(self, output_file: str) -> Path:
        """
        Save validation run metrics to file for debugging (validate mode).

        This allows comparing the actual run against golden side-by-side.

        Args:
            output_file: Path to save metrics. Use {rank} placeholder for rank.

        Returns:
            Path to saved file
        """
        if self.mode != VerifierMode.VALIDATE:
            raise RuntimeError("save_validation_metrics() only available in validate mode")

        path = self._format_path(output_file, self.rank)

        data = {
            "version": "1.0",
            "mode": "validation_run",
            "steps": [s.to_dict() for s in self._validated_steps],
        }

        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

        if self.verbose:
            logger.info(f"Saved validation metrics: {len(self._validated_steps)} steps -> {path}")

        return path

    def __enter__(self) -> "GoldenVerifier":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        # Auto-save in record mode
        if self.mode == VerifierMode.RECORD and self.output_file:
            self.save()
