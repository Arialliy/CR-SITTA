"""Pure train-only outer analysis for formal P3 Stage-A episodes.

The adaptation process supplies only label-free tensors.  A separate outer
evaluator supplies the frozen Pilot64 train target and supervised Source
gradient after the complete label-free cell receipt has been verified.  This
module is pure: it performs no file access, model update, candidate selection,
or publication.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import math
from typing import Any

import numpy as np
import torch
from torch import Tensor

from analysis.analyze_entropy_task_alignment import (
    analyze_entropy_task_alignment,
)
from analysis.d0_v2_diagnostics import (
    MARGIN_BIN_SPECS,
    MARGIN_OUTSIDE_LOWER_BOUND,
    analyze_noop_episode_v2,
)
from analysis.source_train_provenance import SourceTrainAnalysisProvenance
from metrics.irstd_metrics import probabilities_from_logits
from tta.diagnostics import NoOpThresholds


SCHEMA_VERSION = 3
ARTIFACT_TYPE = "cr_sitta_p3_stage_a_outer_episode"
LAYOUT_PROTOCOL = "cr-sitta-d0-v3-flat-bn-affine-layout-v1"


class D0V3OuterAnalyzerError(ValueError):
    """Formal label-free tensors or outer-oracle inputs are inconsistent."""


def _canonical_sha256(value: Any) -> str:
    try:
        payload = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise D0V3OuterAnalyzerError("value is not canonical-JSON safe") from exc
    return hashlib.sha256(payload).hexdigest()


def _shape(value: Any, *, label: str) -> tuple[int, ...]:
    if (
        isinstance(value, (str, bytes))
        or not isinstance(value, Sequence)
        or not value
    ):
        raise D0V3OuterAnalyzerError(f"{label} must be a non-empty shape")
    result: list[int] = []
    for index, item in enumerate(value):
        if isinstance(item, bool) or not isinstance(item, int) or item <= 0:
            raise D0V3OuterAnalyzerError(
                f"{label}[{index}] must be a positive integer"
            )
        result.append(int(item))
    return tuple(result)


@dataclass(frozen=True)
class FlatParameterLayout:
    """Exact ordered topology for one flattened all-BN affine vector."""

    names: tuple[str, ...]
    shapes: tuple[tuple[int, ...], ...]
    offsets: tuple[int, ...]
    scalar_count: int
    layout_sha256: str

    @classmethod
    def from_named_tensors(
        cls, values: Sequence[tuple[str, Tensor]]
    ) -> "FlatParameterLayout":
        if isinstance(values, (str, bytes)):
            raise D0V3OuterAnalyzerError("named tensors must be a sequence")
        materialized = tuple(values)
        if not materialized:
            raise D0V3OuterAnalyzerError("named tensors cannot be empty")
        names: list[str] = []
        shapes: list[tuple[int, ...]] = []
        offsets: list[int] = []
        cursor = 0
        for index, item in enumerate(materialized):
            if not isinstance(item, tuple) or len(item) != 2:
                raise D0V3OuterAnalyzerError(
                    f"named tensor {index} is not a pair"
                )
            name, tensor = item
            if not isinstance(name, str) or not name:
                raise D0V3OuterAnalyzerError(
                    f"named tensor {index} has an invalid name"
                )
            if not isinstance(tensor, Tensor) or tensor.layout != torch.strided:
                raise D0V3OuterAnalyzerError(
                    f"named tensor {name!r} is not strided"
                )
            if not torch.is_floating_point(tensor) or not bool(
                torch.isfinite(tensor).all().item()
            ):
                raise D0V3OuterAnalyzerError(
                    f"named tensor {name!r} must be finite floating-point"
                )
            names.append(name)
            shapes.append(tuple(int(value) for value in tensor.shape))
            offsets.append(cursor)
            cursor += int(tensor.numel())
        if len(set(names)) != len(names):
            raise D0V3OuterAnalyzerError("parameter names must be unique")
        descriptor = {
            "protocol": LAYOUT_PROTOCOL,
            "names": names,
            "shapes": [list(value) for value in shapes],
            "offsets": offsets,
            "scalar_count": cursor,
            "dtype": "torch.float32",
        }
        return cls(
            names=tuple(names),
            shapes=tuple(shapes),
            offsets=tuple(offsets),
            scalar_count=cursor,
            layout_sha256=_canonical_sha256(descriptor),
        )

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "FlatParameterLayout":
        if not isinstance(value, Mapping):
            raise D0V3OuterAnalyzerError("layout must be a mapping")
        expected = {
            "protocol",
            "names",
            "shapes",
            "offsets",
            "scalar_count",
            "dtype",
            "layout_sha256",
        }
        if set(value) != expected:
            raise D0V3OuterAnalyzerError("layout fields must be exact")
        if value["protocol"] != LAYOUT_PROTOCOL or value["dtype"] != "torch.float32":
            raise D0V3OuterAnalyzerError("layout protocol/dtype differs")
        names_raw = value["names"]
        shapes_raw = value["shapes"]
        offsets_raw = value["offsets"]
        if (
            isinstance(names_raw, (str, bytes))
            or not isinstance(names_raw, Sequence)
            or not isinstance(shapes_raw, Sequence)
            or not isinstance(offsets_raw, Sequence)
            or not names_raw
            or len(names_raw) != len(shapes_raw)
            or len(names_raw) != len(offsets_raw)
        ):
            raise D0V3OuterAnalyzerError("layout sequences are invalid")
        names = tuple(names_raw)
        if not all(isinstance(name, str) and name for name in names):
            raise D0V3OuterAnalyzerError("layout names are invalid")
        if len(set(names)) != len(names):
            raise D0V3OuterAnalyzerError("layout names are duplicated")
        shapes = tuple(
            _shape(item, label=f"layout.shapes[{index}]")
            for index, item in enumerate(shapes_raw)
        )
        offsets: list[int] = []
        cursor = 0
        for index, (raw, shape) in enumerate(zip(offsets_raw, shapes, strict=True)):
            if isinstance(raw, bool) or not isinstance(raw, int) or raw != cursor:
                raise D0V3OuterAnalyzerError(
                    f"layout offset is not canonical at index {index}"
                )
            offsets.append(raw)
            cursor += math.prod(shape)
        scalar_count = value["scalar_count"]
        if (
            isinstance(scalar_count, bool)
            or not isinstance(scalar_count, int)
            or scalar_count != cursor
        ):
            raise D0V3OuterAnalyzerError("layout scalar_count differs")
        descriptor = {
            "protocol": LAYOUT_PROTOCOL,
            "names": list(names),
            "shapes": [list(item) for item in shapes],
            "offsets": offsets,
            "scalar_count": cursor,
            "dtype": "torch.float32",
        }
        digest = _canonical_sha256(descriptor)
        if value["layout_sha256"] != digest:
            raise D0V3OuterAnalyzerError("layout SHA-256 differs")
        return cls(names, shapes, tuple(offsets), cursor, digest)

    def to_dict(self) -> dict[str, Any]:
        return {
            "protocol": LAYOUT_PROTOCOL,
            "names": list(self.names),
            "shapes": [list(value) for value in self.shapes],
            "offsets": list(self.offsets),
            "scalar_count": self.scalar_count,
            "dtype": "torch.float32",
            "layout_sha256": self.layout_sha256,
        }

    def pack(self, values: Mapping[str, Any]) -> Tensor:
        if not isinstance(values, Mapping) or tuple(values) != self.names:
            raise D0V3OuterAnalyzerError("named tensors differ from flat layout")
        flattened: list[Tensor] = []
        for name, shape in zip(self.names, self.shapes, strict=True):
            raw = values[name]
            tensor = raw.detach() if isinstance(raw, Tensor) else torch.as_tensor(raw)
            tensor = tensor.cpu().contiguous()
            if tuple(tensor.shape) != shape or tensor.dtype != torch.float32:
                raise D0V3OuterAnalyzerError(
                    f"tensor shape/dtype differs from layout: {name}"
                )
            if not bool(torch.isfinite(tensor).all().item()):
                raise D0V3OuterAnalyzerError(f"tensor is non-finite: {name}")
            flattened.append(tensor.reshape(-1))
        result = torch.cat(flattened)
        if result.numel() != self.scalar_count:
            raise D0V3OuterAnalyzerError("packed scalar count differs")
        return result

    def unpack(self, value: Any, *, label: str) -> dict[str, Tensor]:
        tensor = value.detach() if isinstance(value, Tensor) else torch.as_tensor(value)
        tensor = tensor.cpu().contiguous()
        if (
            tensor.dtype != torch.float32
            or tensor.ndim != 1
            or tensor.numel() != self.scalar_count
            or not bool(torch.isfinite(tensor).all().item())
        ):
            raise D0V3OuterAnalyzerError(
                f"{label} must be finite float32 [{self.scalar_count}]"
            )
        result: dict[str, Tensor] = {}
        for index, (name, shape, start) in enumerate(
            zip(self.names, self.shapes, self.offsets, strict=True)
        ):
            end = (
                self.offsets[index + 1]
                if index + 1 < len(self.offsets)
                else self.scalar_count
            )
            result[name] = tensor[start:end].reshape(shape).clone()
        return result


def _single_probability(value: Any, *, label: str) -> np.ndarray:
    if isinstance(value, Tensor):
        value = value.detach().cpu().numpy()
    array = np.asarray(value)
    if array.ndim == 4 and array.shape[:2] == (1, 1):
        array = array[0, 0]
    elif array.ndim == 3 and array.shape[0] == 1:
        array = array[0]
    if array.ndim != 2 or array.size == 0:
        raise D0V3OuterAnalyzerError(f"{label} must be one 2-D image")
    result = array.astype(np.float64, copy=False)
    if not np.isfinite(result).all() or (result < 0).any() or (result > 1).any():
        raise D0V3OuterAnalyzerError(f"{label} must be finite in [0,1]")
    return result


def _margin_response(probability_pre: Any, probability_post: Any) -> dict[str, Any]:
    pre = _single_probability(probability_pre, label="probability_pre")
    post = _single_probability(probability_post, label="probability_post")
    if pre.shape != post.shape:
        raise D0V3OuterAnalyzerError("pre/post probability shapes differ")
    margin = np.abs(pre - 0.5)
    delta = np.abs(post - pre)
    pre_fg = pre > 0.5
    post_fg = post > 0.5
    result: dict[str, Any] = {}
    partition = 0
    specs = (*MARGIN_BIN_SPECS, ("ge_5e_minus_2", MARGIN_OUTSIDE_LOWER_BOUND, None))
    for key, lower, upper in specs:
        selector = margin >= lower
        if upper is not None:
            selector = np.logical_and(selector, margin < upper)
        count = int(selector.sum())
        partition += count
        selected = delta[selector]
        result[key] = {
            "pixel_count": count,
            "mean_abs_delta_probability": (
                float(selected.mean(dtype=np.float64)) if count else None
            ),
            "p95_abs_delta_probability": (
                float(np.quantile(selected, 0.95, method="linear"))
                if count
                else None
            ),
            "max_abs_delta_probability": (
                float(selected.max()) if count else None
            ),
            "bg_to_fg_pixel_count": int(
                np.logical_and(selector, np.logical_and(~pre_fg, post_fg)).sum()
            ),
            "fg_to_bg_pixel_count": int(
                np.logical_and(selector, np.logical_and(pre_fg, ~post_fg)).sum()
            ),
        }
    if partition != pre.size:
        raise D0V3OuterAnalyzerError("margin response is not a pixel partition")
    return result


def _validate_group_assignment(
    layout: FlatParameterLayout, value: Mapping[str, str]
) -> dict[str, str]:
    if not isinstance(value, Mapping) or tuple(value) != layout.names:
        raise D0V3OuterAnalyzerError("fine-group assignment differs from layout")
    result = dict(value)
    if not all(isinstance(group, str) and group for group in result.values()):
        raise D0V3OuterAnalyzerError("fine-group IDs must be non-empty strings")
    return result


def analyze_formal_stage_a_episode(
    *,
    layout: FlatParameterLayout,
    source_parameters_flat: Any,
    parameters_after_flat: Any,
    entropy_gradient_flat: Any,
    supervised_gradient_flat: Any,
    logits_pre: Any,
    logits_post: Any,
    target: Any,
    thresholds: NoOpThresholds,
    provenance: SourceTrainAnalysisProvenance,
    fine_group_assignment: Mapping[str, str],
    first_order_zero_tolerance: float,
) -> dict[str, Any]:
    """Merge one sealed label-free episode with one train-only oracle view."""

    if not isinstance(layout, FlatParameterLayout):
        raise D0V3OuterAnalyzerError("layout must be FlatParameterLayout")
    source = layout.unpack(source_parameters_flat, label="source_parameters_flat")
    after = layout.unpack(parameters_after_flat, label="parameters_after_flat")
    entropy_gradient = layout.unpack(
        entropy_gradient_flat, label="entropy_gradient_flat"
    )
    supervised_gradient = layout.unpack(
        supervised_gradient_flat, label="supervised_gradient_flat"
    )
    step = {
        name: after[name] - source[name]
        for name in layout.names
    }
    assignment = _validate_group_assignment(layout, fine_group_assignment)
    noop = analyze_noop_episode_v2(
        parameter_pre=source,
        parameter_post=after,
        logits_pre=logits_pre,
        logits_post=logits_post,
        target=target,
        thresholds=thresholds,
    )
    alignment = analyze_entropy_task_alignment(
        entropy_gradients=entropy_gradient,
        supervised_gradients=supervised_gradient,
        adaptation_step=step,
        provenance=provenance,
        parameter_groups=assignment,
        first_order_zero_tolerance=first_order_zero_tolerance,
    )
    probability_pre = probabilities_from_logits(logits_pre)
    probability_post = probabilities_from_logits(logits_post)
    response = _margin_response(probability_pre, probability_post)
    return {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": ARTIFACT_TYPE,
        "scope": provenance.to_dict(),
        "label_isolation": {
            "label_free_payload_complete_before_target_open": True,
            "method_label_accesses": 0,
            "outer_evaluator_label_accesses": (
                provenance.outer_evaluator_label_accesses
            ),
            "supervised_gradient_used_by_adaptation": False,
            "adaptation_optimizer_executed_by_outer_evaluator": False,
            "test_payload_accesses": 0,
        },
        "layout_sha256": layout.layout_sha256,
        "noop": noop,
        "threshold_margin_bin_response": response,
        "entropy_task_alignment": alignment,
        "scientific_selection_performed": False,
        "stage2_authorized": False,
    }


__all__ = [
    "ARTIFACT_TYPE",
    "D0V3OuterAnalyzerError",
    "FlatParameterLayout",
    "LAYOUT_PROTOCOL",
    "SCHEMA_VERSION",
    "analyze_formal_stage_a_episode",
]
