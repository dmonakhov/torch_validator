"""
Metrics collection for golden reference validation.

Collects:
- Loss value
- Gradient L2 norm (global and per-layer)
- Weight L2 norm (global and per-layer)
- Model checksum (xxhash of flattened parameters)
"""

import hashlib
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union

import torch
import torch.nn as nn


def _to_local_tensor(tensor: torch.Tensor) -> torch.Tensor:
    """
    Convert tensor subclasses (DTensor, FSDP FlatParameter) to local tensor.

    Handles:
    - DTensor: calls .to_local() to get local shard
    - FSDP FlatParameter: uses ._local_tensor if available
    - Regular tensor: returns as-is
    """
    # Handle DTensor (torch.distributed.tensor)
    if hasattr(tensor, "to_local"):
        tensor = tensor.to_local()

    # Handle FSDP FlatParameter
    if hasattr(tensor, "_local_tensor"):
        tensor = tensor._local_tensor

    # Ensure it's a regular tensor (not a subclass)
    if type(tensor) is not torch.Tensor:
        # Last resort: clone to regular tensor
        tensor = tensor.clone().detach()

    return tensor


@dataclass
class StepMetrics:
    """Metrics collected at a single training step."""

    step: int
    loss: float
    grad_norm: float
    weight_norm: float
    model_checksum: str
    layer_grad_norms: Dict[str, float] = field(default_factory=dict)
    layer_weight_norms: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Convert to JSON-serializable dict."""
        return {
            "step": self.step,
            "loss": self.loss,
            "grad_norm": self.grad_norm,
            "weight_norm": self.weight_norm,
            "model_checksum": self.model_checksum,
            "layer_grad_norms": self.layer_grad_norms,
            "layer_weight_norms": self.layer_weight_norms,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "StepMetrics":
        """Create from dict."""
        return cls(
            step=d["step"],
            loss=d["loss"],
            grad_norm=d["grad_norm"],
            weight_norm=d["weight_norm"],
            model_checksum=d["model_checksum"],
            layer_grad_norms=d.get("layer_grad_norms", {}),
            layer_weight_norms=d.get("layer_weight_norms", {}),
        )


class MetricsCollector:
    """
    Collects training metrics from model and loss.

    Thread-safe for single-threaded training loops.
    """

    def __init__(
        self,
        collect_layer_norms: bool = False,
        checksum_enabled: bool = False,
        checksum_algorithm: str = "sha256",
    ):
        """
        Initialize MetricsCollector.

        Args:
            collect_layer_norms: If True, collect per-layer gradient and weight norms (expensive)
            checksum_enabled: If True, compute model checksum (very expensive, requires CPU transfer)
            checksum_algorithm: Hash algorithm for model checksum (default: "sha256", HW accelerated)
        """
        self.collect_layer_norms = collect_layer_norms
        self.checksum_enabled = checksum_enabled
        self.checksum_algorithm = checksum_algorithm

    def collect(
        self,
        step: int,
        loss: Union[float, torch.Tensor],
        model: nn.Module,
    ) -> StepMetrics:
        """
        Collect metrics at a training step.

        Args:
            step: Current training step
            loss: Loss value (scalar tensor or float)
            model: Model with gradients populated (after backward())

        Returns:
            StepMetrics with all collected values
        """
        # Convert loss to float
        if isinstance(loss, torch.Tensor):
            loss_val = loss.detach().item()
        else:
            loss_val = float(loss)

        # Compute gradient norm
        grad_norm, layer_grad_norms = self._compute_grad_norm(model)

        # Compute weight norm
        weight_norm, layer_weight_norms = self._compute_weight_norm(model)

        # Compute model checksum (expensive - disabled by default)
        model_checksum = self._compute_checksum(model) if self.checksum_enabled else ""

        return StepMetrics(
            step=step,
            loss=loss_val,
            grad_norm=grad_norm,
            weight_norm=weight_norm,
            model_checksum=model_checksum,
            layer_grad_norms=layer_grad_norms if self.collect_layer_norms else {},
            layer_weight_norms=layer_weight_norms if self.collect_layer_norms else {},
        )

    def _compute_grad_norm(
        self, model: nn.Module
    ) -> tuple[float, Dict[str, float]]:
        """Compute global and per-layer gradient L2 norms."""
        total_norm_sq = 0.0
        layer_norms = {}

        for name, param in model.named_parameters():
            if param.grad is not None:
                grad = _to_local_tensor(param.grad.data)
                param_norm_sq = grad.norm(2).item() ** 2
                total_norm_sq += param_norm_sq
                if self.collect_layer_norms:
                    layer_norms[name] = param_norm_sq ** 0.5

        return total_norm_sq ** 0.5, layer_norms

    def _compute_weight_norm(
        self, model: nn.Module
    ) -> tuple[float, Dict[str, float]]:
        """Compute global and per-layer weight L2 norms."""
        total_norm_sq = 0.0
        layer_norms = {}

        for name, param in model.named_parameters():
            data = _to_local_tensor(param.data)
            param_norm_sq = data.norm(2).item() ** 2
            total_norm_sq += param_norm_sq
            if self.collect_layer_norms:
                layer_norms[name] = param_norm_sq ** 0.5

        return total_norm_sq ** 0.5, layer_norms

    def _compute_checksum(self, model: nn.Module) -> str:
        """
        Compute deterministic checksum of model parameters.

        Uses raw bytes of parameters in consistent order.
        Handles tensor subclasses (DTensor, FSDP) by converting to local tensors.
        """
        hasher = hashlib.new(self.checksum_algorithm)

        for name, param in sorted(model.named_parameters()):
            # Convert to local tensor, then to CPU numpy bytes
            local_tensor = _to_local_tensor(param.data.detach())
            data = local_tensor.cpu().numpy().tobytes()
            hasher.update(name.encode("utf-8"))
            hasher.update(data)

        return hasher.hexdigest()


def check_nan_inf(
    loss: Union[float, torch.Tensor],
    model: Optional[nn.Module] = None,
    check_gradients: bool = True,
    check_weights: bool = False,
) -> Dict[str, bool]:
    """
    Check for NaN/Inf in loss, gradients, and weights.

    Args:
        loss: Loss value to check
        model: Model to check gradients/weights (optional)
        check_gradients: If True, check gradients for NaN/Inf
        check_weights: If True, check weights for NaN/Inf

    Returns:
        Dict with "loss_ok", "grad_ok", "weight_ok" boolean values
    """
    result = {"loss_ok": True, "grad_ok": True, "weight_ok": True}

    # Check loss
    if isinstance(loss, torch.Tensor):
        loss_local = _to_local_tensor(loss)
        result["loss_ok"] = not (torch.isnan(loss_local).any() or torch.isinf(loss_local).any())
    else:
        import math
        result["loss_ok"] = not (math.isnan(loss) or math.isinf(loss))

    if model is None:
        return result

    # Check gradients
    if check_gradients:
        for param in model.parameters():
            if param.grad is not None:
                grad = _to_local_tensor(param.grad)
                if torch.isnan(grad).any() or torch.isinf(grad).any():
                    result["grad_ok"] = False
                    break

    # Check weights
    if check_weights:
        for param in model.parameters():
            data = _to_local_tensor(param.data)
            if torch.isnan(data).any() or torch.isinf(data).any():
                result["weight_ok"] = False
                break

    return result
