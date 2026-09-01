#!/usr/bin/env python3
"""Outer-oracle entropy/task gradient alignment on frozen source-train data.

The supervised gradient is diagnostic evidence only.  It is never returned as
an adaptation direction and this module rejects test splits, test labels, or
any method-visible label access.  The primary quantities are per-group
``cos(g_entropy, g_supervised)`` and ``g_supervised^T d`` for the actual or
proposed unlabeled adaptation step ``d``.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import json
import math
from pathlib import Path
import sys
from typing import Any

import torch
from torch import Tensor


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis.source_train_provenance import SourceTrainAnalysisProvenance
from tta.parameter_vector import (
    NamedTensorItems,
    ParameterVectorError,
    vectorize_named_tensors,
)


ANALYSIS_SCHEMA_VERSION = 1
ANALYSIS_TYPE = "entropy_task_gradient_alignment"


class GradientAlignmentError(ValueError):
    """Gradient vectors or their outer-oracle provenance are invalid."""


def _finite_nonnegative(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise GradientAlignmentError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise GradientAlignmentError(f"{label} must be finite and non-negative")
    return result


def _norm(value: Tensor) -> float:
    return float(torch.linalg.vector_norm(value).item())


def _dot(left: Tensor, right: Tensor) -> float:
    return float(torch.dot(left, right).item())


def _cosine(left: Tensor, right: Tensor) -> float | None:
    left_norm = torch.linalg.vector_norm(left)
    right_norm = torch.linalg.vector_norm(right)
    if float(left_norm.item()) == 0.0 or float(right_norm.item()) == 0.0:
        return None
    cosine = torch.dot(left, right) / (left_norm * right_norm)
    return float(torch.clamp(cosine, -1.0, 1.0).item())


def _effect(dot_value: float, tolerance: float) -> str:
    if dot_value < -tolerance:
        return "predicted_task_loss_decrease"
    if dot_value > tolerance:
        return "predicted_task_loss_increase"
    return "first_order_neutral"


def _alignment_metrics(
    entropy_gradient: Tensor,
    supervised_gradient: Tensor,
    adaptation_step: Tensor,
    *,
    zero_tolerance: float,
) -> dict[str, Any]:
    supervised_dot_step = _dot(supervised_gradient, adaptation_step)
    return {
        "scalar_count": entropy_gradient.numel(),
        "entropy_gradient_norm": _norm(entropy_gradient),
        "supervised_gradient_norm": _norm(supervised_gradient),
        "adaptation_step_norm": _norm(adaptation_step),
        "entropy_supervised_dot": _dot(
            entropy_gradient, supervised_gradient
        ),
        "entropy_supervised_cosine": _cosine(
            entropy_gradient, supervised_gradient
        ),
        "supervised_dot_adaptation_step": supervised_dot_step,
        "first_order_task_loss_change": supervised_dot_step,
        "first_order_task_effect": _effect(
            supervised_dot_step, zero_tolerance
        ),
    }


def analyze_entropy_task_alignment(
    *,
    entropy_gradients: NamedTensorItems,
    supervised_gradients: NamedTensorItems,
    adaptation_step: NamedTensorItems,
    provenance: SourceTrainAnalysisProvenance,
    parameter_groups: Mapping[str, str] | None = None,
    first_order_zero_tolerance: float = 0.0,
) -> dict[str, Any]:
    """Compute global and per-group oracle task-alignment diagnostics."""

    if not isinstance(provenance, SourceTrainAnalysisProvenance):
        raise GradientAlignmentError(
            "provenance must be SourceTrainAnalysisProvenance"
        )
    if (
        not provenance.oracle_analysis
        or provenance.outer_evaluator_label_accesses <= 0
    ):
        raise GradientAlignmentError(
            "alignment requires source-train labels in the outer oracle only"
        )
    tolerance = _finite_nonnegative(
        first_order_zero_tolerance, "first_order_zero_tolerance"
    )
    layout, entropy = vectorize_named_tensors(
        entropy_gradients, label="entropy_gradients"
    )
    supervised = layout.flatten(
        supervised_gradients, label="supervised_gradients"
    )
    step = layout.flatten(adaptation_step, label="adaptation_step")
    assignment = layout.validated_group_assignment(parameter_groups)
    assignment_by_name = dict(assignment)

    per_group: dict[str, Any] = {}
    for group, indices in layout.group_indices(assignment_by_name).items():
        metrics = _alignment_metrics(
            entropy[indices],
            supervised[indices],
            step[indices],
            zero_tolerance=tolerance,
        )
        metrics["parameter_tensor_count"] = sum(
            assigned_group == group for _, assigned_group in assignment
        )
        per_group[group] = metrics

    return {
        "schema_version": ANALYSIS_SCHEMA_VERSION,
        "analysis_type": ANALYSIS_TYPE,
        "scope": provenance.to_dict(),
        "label_isolation": {
            "entropy_gradient_label_free": True,
            "adaptation_step_label_free": True,
            "supervised_gradient_used_by_method": False,
            "supervised_gradient_role": "outer_oracle_train_labels_only",
            "method_label_accesses": 0,
            "outer_evaluator_label_accesses": (
                provenance.outer_evaluator_label_accesses
            ),
        },
        "parameter_layout": {
            "parameter_tensor_count": len(layout.names),
            "parameter_scalar_count": layout.total_numel,
            "parameter_names_sha256": layout.parameter_names_sha256,
            "topology_sha256": layout.topology_sha256,
        },
        "first_order_zero_tolerance": tolerance,
        "global": _alignment_metrics(
            entropy,
            supervised,
            step,
            zero_tolerance=tolerance,
        ),
        "per_group": per_group,
    }


def _tensor_mapping(value: Any, label: str) -> dict[str, Tensor]:
    if not isinstance(value, Mapping) or not value:
        raise GradientAlignmentError(f"{label} must be a non-empty mapping")
    result: dict[str, Tensor] = {}
    for name, raw in value.items():
        if not isinstance(name, str) or not name:
            raise GradientAlignmentError(f"{label} has invalid parameter name")
        if raw is None:
            raise GradientAlignmentError(f"{label}.{name} cannot be null")
        try:
            result[name] = torch.tensor(raw, dtype=torch.float64)
        except (TypeError, ValueError, RuntimeError) as exc:
            raise GradientAlignmentError(f"{label}.{name} is not numeric") from exc
    return result


def analyze_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        raise GradientAlignmentError("input root must be a JSON object")
    provenance = SourceTrainAnalysisProvenance.from_mapping(
        payload.get("provenance"), require_outer_oracle=True
    )
    groups = payload.get("parameter_groups")
    if groups is not None and not isinstance(groups, Mapping):
        raise GradientAlignmentError("parameter_groups must be a mapping")
    return analyze_entropy_task_alignment(
        entropy_gradients=_tensor_mapping(
            payload.get("entropy_gradients"), "entropy_gradients"
        ),
        supervised_gradients=_tensor_mapping(
            payload.get("supervised_gradients"), "supervised_gradients"
        ),
        adaptation_step=_tensor_mapping(
            payload.get("adaptation_step"), "adaptation_step"
        ),
        provenance=provenance,
        parameter_groups=groups,
        first_order_zero_tolerance=payload.get(
            "first_order_zero_tolerance", 0.0
        ),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-json", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        payload = json.loads(args.input_json.read_text(encoding="utf-8"))
        result = analyze_payload(payload)
    except (OSError, json.JSONDecodeError, ValueError, ParameterVectorError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ANALYSIS_SCHEMA_VERSION",
    "ANALYSIS_TYPE",
    "GradientAlignmentError",
    "analyze_entropy_task_alignment",
    "analyze_payload",
    "main",
]
