"""Rebuild formal P3 Stage-A replicate evidence from per-episode records.

All scientific quantities are derived from integer sufficient statistics or
the explicitly stored diagnostics.  Cell metrics first pool the 64 Pilot
images; dataset/corruption/clean/non-clean summaries then use equal-cell
macros.  No candidate ranking or threshold is implemented here.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import math
from statistics import median
from typing import Any

from analysis.d0_v3_formal_contract import (
    CONDITIONS,
    DATASETS,
    FINE_ALIGNMENT_GROUP_IDS,
    FROZEN_CANDIDATES,
    FormalCandidate,
)
from analysis.d0_v3_science_gate import (
    CORRUPTION_FAMILIES,
    EPISODES_PER_CANDIDATE_REPLICATE,
    SAFETY_STRATA,
    parse_r0_eligibility_receipt,
    parse_replicate_evidence,
)


SCHEMA_VERSION = 3
EPISODE_ARTIFACT_TYPE = "cr_sitta_d0_v3_stage_a_outer_episode_record"
EVIDENCE_ARTIFACT_TYPE = "cr_sitta_d0_v3_stage_a_replicate_evidence"
FUNCTIONAL_LOGIT_THRESHOLD = 1.0e-6
IMAGES_PER_CELL = 64
ALL_BN_PARAMETER_TENSOR_COUNT = 106

_GLOBAL_ALIGNMENT_FIELDS = {
    "scalar_count",
    "entropy_gradient_norm",
    "supervised_gradient_norm",
    "adaptation_step_norm",
    "entropy_supervised_dot",
    "entropy_supervised_cosine",
    "supervised_dot_adaptation_step",
    "first_order_task_loss_change",
    "first_order_task_effect",
}
_GROUP_ALIGNMENT_FIELDS = _GLOBAL_ALIGNMENT_FIELDS | {"parameter_tensor_count"}
_FIRST_ORDER_EFFECTS = {
    "predicted_task_loss_decrease",
    "predicted_task_loss_increase",
    "first_order_neutral",
}


class D0V3ReplicateAggregateError(ValueError):
    """Episode records cannot form one complete formal replicate."""


_RECORD_FIELDS = {
    "schema_version",
    "artifact_type",
    "dataset",
    "condition",
    "replicate_id",
    "image_index",
    "image_id",
    "candidate",
    "finite_gradient",
    "changed_parameter_tensor_count",
    "analysis",
}
_COUNT_FIELDS = {
    "detected_targets",
    "false_alarm_pixels",
    "false_negative_pixels",
    "false_positive_components",
    "false_positive_pixels",
    "intersection_pixels",
    "predicted_positive_pixels",
    "target_positive_pixels",
    "total_image_pixels",
    "total_targets",
    "true_negative_pixels",
    "union_pixels",
}


def _mapping(value: Any, *, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise D0V3ReplicateAggregateError(f"{field} must be a mapping")
    return value


def _exact(value: Any, expected: set[str], *, field: str) -> Mapping[str, Any]:
    mapping = _mapping(value, field=field)
    if set(mapping) != expected:
        missing = sorted(expected - set(mapping))
        unknown = sorted(set(mapping) - expected)
        raise D0V3ReplicateAggregateError(
            f"{field} fields must be exact; missing={missing}, unknown={unknown}"
        )
    return mapping


def _integer(value: Any, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise D0V3ReplicateAggregateError(
            f"{field} must be a non-negative integer"
        )
    return value


def _finite(value: Any, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise D0V3ReplicateAggregateError(f"{field} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise D0V3ReplicateAggregateError(f"{field} must be finite")
    return result


def _alignment_metrics(
    value: Any, *, field: str, require_parameter_count: bool
) -> tuple[int, int | None, float, float, float | None]:
    """Validate one complete global or fine-group alignment observation."""

    expected = (
        _GROUP_ALIGNMENT_FIELDS if require_parameter_count else _GLOBAL_ALIGNMENT_FIELDS
    )
    metrics = _exact(value, expected, field=field)
    scalar_count = _integer(metrics["scalar_count"], field=f"{field}.scalar_count")
    if scalar_count <= 0:
        raise D0V3ReplicateAggregateError(f"{field}.scalar_count must be positive")
    parameter_count: int | None = None
    if require_parameter_count:
        parameter_count = _integer(
            metrics["parameter_tensor_count"],
            field=f"{field}.parameter_tensor_count",
        )
        if parameter_count <= 0:
            raise D0V3ReplicateAggregateError(
                f"{field}.parameter_tensor_count must be positive"
            )
    entropy_norm = _finite(
        metrics["entropy_gradient_norm"], field=f"{field}.entropy_gradient_norm"
    )
    supervised_norm = _finite(
        metrics["supervised_gradient_norm"],
        field=f"{field}.supervised_gradient_norm",
    )
    adaptation_norm = _finite(
        metrics["adaptation_step_norm"], field=f"{field}.adaptation_step_norm"
    )
    if min(entropy_norm, supervised_norm, adaptation_norm) < 0.0:
        raise D0V3ReplicateAggregateError(f"{field} norms must be non-negative")
    _finite(metrics["entropy_supervised_dot"], field=f"{field}.entropy_supervised_dot")
    supervised_dot = _finite(
        metrics["supervised_dot_adaptation_step"],
        field=f"{field}.supervised_dot_adaptation_step",
    )
    first_order = _finite(
        metrics["first_order_task_loss_change"],
        field=f"{field}.first_order_task_loss_change",
    )
    if first_order != supervised_dot:
        raise D0V3ReplicateAggregateError(
            f"{field} first-order loss/dot aliases differ"
        )
    expected_effect = (
        "predicted_task_loss_decrease"
        if supervised_dot < 0.0
        else "predicted_task_loss_increase"
        if supervised_dot > 0.0
        else "first_order_neutral"
    )
    effect = metrics["first_order_task_effect"]
    if effect not in _FIRST_ORDER_EFFECTS or effect != expected_effect:
        raise D0V3ReplicateAggregateError(f"{field} first-order effect differs")

    both_nonzero = entropy_norm > 0.0 and supervised_norm > 0.0
    cosine_raw = metrics["entropy_supervised_cosine"]
    cosine: float | None
    if both_nonzero:
        cosine = _finite(cosine_raw, field=f"{field}.entropy_supervised_cosine")
        if not -1.0 <= cosine <= 1.0:
            raise D0V3ReplicateAggregateError(f"{field} cosine is outside [-1,1]")
    else:
        if cosine_raw is not None:
            raise D0V3ReplicateAggregateError(
                f"{field} zero-norm cosine must be null"
            )
        cosine = None
    return scalar_count, parameter_count, entropy_norm, supervised_norm, cosine


def _candidate(value: Any) -> FormalCandidate:
    mapping = _exact(
        value,
        {"candidate_id", "optimizer", "learning_rate"},
        field="record.candidate",
    )
    matches = tuple(
        candidate
        for candidate in FROZEN_CANDIDATES
        if mapping["candidate_id"] == candidate.candidate_id
        and mapping["optimizer"] == candidate.optimizer
        and _finite(
            mapping["learning_rate"], field="candidate.learning_rate"
        )
        == candidate.learning_rate
    )
    if len(matches) != 1:
        raise D0V3ReplicateAggregateError("record candidate is not frozen")
    return matches[0]


def _counts(value: Any, *, field: str) -> dict[str, int]:
    mapping = _exact(value, _COUNT_FIELDS, field=field)
    result = {key: _integer(mapping[key], field=f"{field}.{key}") for key in mapping}
    intersection = result["intersection_pixels"]
    fp = result["false_positive_pixels"]
    fn = result["false_negative_pixels"]
    tn = result["true_negative_pixels"]
    if result["predicted_positive_pixels"] != intersection + fp:
        raise D0V3ReplicateAggregateError(f"{field} predicted-positive mismatch")
    if result["target_positive_pixels"] != intersection + fn:
        raise D0V3ReplicateAggregateError(f"{field} target-positive mismatch")
    if result["union_pixels"] != intersection + fp + fn:
        raise D0V3ReplicateAggregateError(f"{field} union mismatch")
    if result["total_image_pixels"] != intersection + fp + fn + tn:
        raise D0V3ReplicateAggregateError(f"{field} total-pixel mismatch")
    if result["detected_targets"] > result["total_targets"]:
        raise D0V3ReplicateAggregateError(f"{field} detected-target mismatch")
    if result["false_alarm_pixels"] > result["predicted_positive_pixels"]:
        raise D0V3ReplicateAggregateError(f"{field} false-alarm pixel mismatch")
    return result


def _sum_counts(values: Sequence[Mapping[str, int]]) -> dict[str, int]:
    if not values:
        raise D0V3ReplicateAggregateError("cannot pool an empty cell")
    return {key: sum(value[key] for value in values) for key in _COUNT_FIELDS}


def _metrics(counts: Mapping[str, int]) -> dict[str, float]:
    union = counts["union_pixels"]
    targets = counts["total_targets"]
    pixels = counts["total_image_pixels"]
    if pixels <= 0:
        raise D0V3ReplicateAggregateError("pooled cell has no image pixels")
    return {
        "iou": counts["intersection_pixels"] / union if union else 1.0,
        "pd": counts["detected_targets"] / targets if targets else 0.0,
        "fa_per_million": counts["false_alarm_pixels"] / pixels * 1.0e6,
        "foreground_fraction": counts["predicted_positive_pixels"] / pixels,
    }


def _mean(values: Sequence[float]) -> float:
    if not values:
        raise D0V3ReplicateAggregateError("cannot average an empty stratum")
    result = math.fsum(values) / len(values)
    if not math.isfinite(result):
        raise D0V3ReplicateAggregateError("macro result is not finite")
    return result


def _family(condition: str) -> str | None:
    if condition == "clean_S0":
        return None
    matches = tuple(
        family for family in CORRUPTION_FAMILIES if condition.startswith(f"{family}_S")
    )
    if len(matches) != 1:
        raise D0V3ReplicateAggregateError(
            f"condition has no unique corruption family: {condition}"
        )
    return matches[0]


def _stratum_cells(
    cells: Mapping[tuple[str, str], Mapping[str, Any]], key: str
) -> list[Mapping[str, Any]]:
    values: list[Mapping[str, Any]] = []
    for (dataset, condition), cell in cells.items():
        family = _family(condition)
        include = False
        if key == "overall":
            include = True
        elif key == "nonclean":
            include = family is not None
        elif key == "clean":
            include = family is None
        elif key.startswith("dataset:"):
            # Dataset safety is intentionally the 12-cell non-clean slice;
            # the one clean cell is separately protected by clean_dataset_iou.
            include = dataset == key.partition(":")[2] and family is not None
        elif key.startswith("corruption_family:"):
            include = family == key.partition(":")[2]
        else:  # pragma: no cover - SAFETY_STRATA is frozen upstream.
            raise D0V3ReplicateAggregateError(f"unknown safety stratum {key}")
        if include:
            values.append(cell)
    return values


def _science_candidate(value: FormalCandidate) -> dict[str, Any]:
    return {
        "candidate_id": value.candidate_id,
        "optimizer": value.optimizer,
        "learning_rate": value.learning_rate,
    }


def build_candidate_replicate_evidence(
    records: Sequence[Mapping[str, Any]],
    *,
    candidate: FormalCandidate,
    replicate_id: str,
) -> dict[str, Any]:
    """Build one strict 2,496-episode science-gate input record."""

    if candidate not in FROZEN_CANDIDATES:
        raise D0V3ReplicateAggregateError("candidate is not frozen")
    if replicate_id not in {"R0", "R1", "R2"}:
        raise D0V3ReplicateAggregateError("replicate_id must be R0, R1, or R2")
    if isinstance(records, (str, bytes)) or len(records) != EPISODES_PER_CANDIDATE_REPLICATE:
        raise D0V3ReplicateAggregateError(
            f"candidate replicate must contain {EPISODES_PER_CANDIDATE_REPLICATE} records"
        )

    by_key: dict[tuple[str, str, int], Mapping[str, Any]] = {}
    ordered_ids: dict[tuple[str, str], dict[int, str]] = {}
    activity = {
        "finite_gradient_episodes": 0,
        "parameter_changed_episodes": 0,
        "functional_logit_changed_episodes": 0,
        "threshold_crossing_episodes": 0,
        "metric_sufficient_count_changed_episodes": 0,
        "entropy_decrease_episodes": 0,
        "both_gradients_nonzero_episodes": 0,
        "fine_group_alignment_observation_count": 0,
    }
    global_cosines: list[tuple[str, float]] = []
    cell_counts: dict[
        tuple[str, str], dict[str, list[dict[str, int]]]
    ] = {}
    for position, raw in enumerate(records):
        record = _exact(raw, _RECORD_FIELDS, field=f"records[{position}]")
        if record["schema_version"] != SCHEMA_VERSION or record[
            "artifact_type"
        ] != EPISODE_ARTIFACT_TYPE:
            raise D0V3ReplicateAggregateError("episode record identity differs")
        dataset = record["dataset"]
        condition = record["condition"]
        if dataset not in DATASETS or condition not in CONDITIONS:
            raise D0V3ReplicateAggregateError("episode dataset/condition is not frozen")
        if record["replicate_id"] != replicate_id or _candidate(record["candidate"]) != candidate:
            raise D0V3ReplicateAggregateError("episode candidate/replicate differs")
        image_index = _integer(record["image_index"], field="image_index")
        image_id = record["image_id"]
        if image_index >= IMAGES_PER_CELL or not isinstance(image_id, str) or not image_id:
            raise D0V3ReplicateAggregateError("episode image identity is invalid")
        key = (dataset, condition, image_index)
        if key in by_key:
            raise D0V3ReplicateAggregateError(f"duplicate episode key: {key}")
        by_key[key] = record
        cell_ids = ordered_ids.setdefault((dataset, condition), {})
        cell_ids[image_index] = image_id

        finite_gradient = record["finite_gradient"]
        if not isinstance(finite_gradient, bool):
            raise D0V3ReplicateAggregateError("finite_gradient must be bool")
        activity["finite_gradient_episodes"] += int(finite_gradient)
        changed = _integer(
            record["changed_parameter_tensor_count"],
            field="changed_parameter_tensor_count",
        )
        if changed > ALL_BN_PARAMETER_TENSOR_COUNT:
            raise D0V3ReplicateAggregateError(
                "changed parameter tensor count exceeds all-BN inventory"
            )
        activity["parameter_changed_episodes"] += int(changed > 0)

        analysis = _mapping(record["analysis"], field="record.analysis")
        if (
            analysis.get("schema_version") != 3
            or analysis.get("artifact_type") != "cr_sitta_p3_stage_a_outer_episode"
            or analysis.get("stage2_authorized") is not False
            or analysis.get("scientific_selection_performed") is not False
        ):
            raise D0V3ReplicateAggregateError("outer analysis identity differs")
        noop = _mapping(analysis.get("noop"), field="analysis.noop")
        logit_delta = _finite(
            noop.get("logit_delta_abs_max"), field="noop.logit_delta_abs_max"
        )
        if logit_delta < 0.0:
            raise D0V3ReplicateAggregateError(
                "noop.logit_delta_abs_max must be non-negative"
            )
        activity["functional_logit_changed_episodes"] += int(
            logit_delta > FUNCTIONAL_LOGIT_THRESHOLD
        )
        xor = _integer(
            noop.get("binary_pixel_xor_count"),
            field="noop.binary_pixel_xor_count",
        )
        identical = noop.get("metric_counts_identical")
        if not isinstance(identical, bool):
            raise D0V3ReplicateAggregateError(
                "noop.metric_counts_identical must be bool"
            )
        entropy_pre = _finite(noop.get("entropy_pre"), field="noop.entropy_pre")
        entropy_post = _finite(noop.get("entropy_post"), field="noop.entropy_post")
        activity["entropy_decrease_episodes"] += int(entropy_post < entropy_pre)
        metric_counts = _exact(
            noop.get("metric_counts"), {"identical", "pre", "post"}, field="noop.metric_counts"
        )
        if not isinstance(metric_counts["identical"], bool):
            raise D0V3ReplicateAggregateError(
                "noop.metric_counts.identical must be bool"
            )
        if metric_counts["identical"] is not identical:
            raise D0V3ReplicateAggregateError("metric identical aliases differ")
        pre_counts = _counts(metric_counts["pre"], field="metric_counts.pre")
        post_counts = _counts(metric_counts["post"], field="metric_counts.post")
        recomputed_identical = pre_counts == post_counts
        if identical is not recomputed_identical:
            raise D0V3ReplicateAggregateError(
                "metric identical flag differs from recomputed pre/post counts"
            )
        if xor > pre_counts["total_image_pixels"]:
            raise D0V3ReplicateAggregateError(
                "binary pixel XOR count exceeds image pixel count"
            )
        if xor == 0 and not recomputed_identical:
            raise D0V3ReplicateAggregateError(
                "zero binary XOR cannot change official sufficient counts"
            )
        activity["threshold_crossing_episodes"] += int(xor > 0)
        activity["metric_sufficient_count_changed_episodes"] += int(
            not recomputed_identical
        )
        for conserved in ("total_image_pixels", "total_targets", "target_positive_pixels"):
            if pre_counts[conserved] != post_counts[conserved]:
                raise D0V3ReplicateAggregateError(
                    f"target/image sufficient count changed: {conserved}"
                )
        bucket = cell_counts.setdefault((dataset, condition), {"pre": [], "post": []})
        bucket["pre"].append(pre_counts)
        bucket["post"].append(post_counts)

        alignment = _mapping(
            analysis.get("entropy_task_alignment"),
            field="analysis.entropy_task_alignment",
        )
        (
            global_scalar_count,
            global_parameter_count,
            entropy_norm,
            supervised_norm,
            cosine,
        ) = _alignment_metrics(
            alignment.get("global"),
            field="alignment.global",
            require_parameter_count=False,
        )
        if global_parameter_count is not None:  # pragma: no cover - helper contract.
            raise D0V3ReplicateAggregateError(
                "global alignment unexpectedly has a parameter count"
            )
        both_nonzero = entropy_norm > 0.0 and supervised_norm > 0.0
        activity["both_gradients_nonzero_episodes"] += int(both_nonzero)
        if both_nonzero:
            assert cosine is not None
            global_cosines.append((dataset, cosine))
        per_group = _mapping(alignment.get("per_group"), field="alignment.per_group")
        if tuple(per_group) != FINE_ALIGNMENT_GROUP_IDS:
            raise D0V3ReplicateAggregateError(
                "fine alignment groups/order differ from frozen inventory"
            )
        group_scalar_count = 0
        group_parameter_count = 0
        valid_group_observations = 0
        for group in FINE_ALIGNMENT_GROUP_IDS:
            scalar_count, parameter_count, *_ = _alignment_metrics(
                per_group[group],
                field=f"alignment.per_group.{group}",
                require_parameter_count=True,
            )
            assert parameter_count is not None
            group_scalar_count += scalar_count
            group_parameter_count += parameter_count
            valid_group_observations += 1
        if group_scalar_count != global_scalar_count:
            raise D0V3ReplicateAggregateError(
                "fine-group scalar counts do not partition the global alignment"
            )
        if group_parameter_count != ALL_BN_PARAMETER_TENSOR_COUNT:
            raise D0V3ReplicateAggregateError(
                "fine-group parameter counts do not match the all-BN inventory"
            )
        activity["fine_group_alignment_observation_count"] += (
            valid_group_observations
        )

    expected_keys = {
        (dataset, condition, index)
        for dataset in DATASETS
        for condition in CONDITIONS
        for index in range(IMAGES_PER_CELL)
    }
    if set(by_key) != expected_keys:
        raise D0V3ReplicateAggregateError("episode grid is not exact 3x13x64")
    for cell, ids in ordered_ids.items():
        if set(ids) != set(range(IMAGES_PER_CELL)) or len(set(ids.values())) != IMAGES_PER_CELL:
            raise D0V3ReplicateAggregateError(
                f"cell Pilot IDs are incomplete or duplicated: {cell}"
            )

    cells: dict[tuple[str, str], dict[str, Any]] = {}
    for cell in ((dataset, condition) for dataset in DATASETS for condition in CONDITIONS):
        bucket = cell_counts[cell]
        if len(bucket["pre"]) != IMAGES_PER_CELL or len(bucket["post"]) != IMAGES_PER_CELL:
            raise D0V3ReplicateAggregateError(f"cell count differs from 64: {cell}")
        cells[cell] = {
            "source": _metrics(_sum_counts(bucket["pre"])),
            "adapted": _metrics(_sum_counts(bucket["post"])),
        }

    def macro_delta(metric: str, selected: Sequence[Mapping[str, Any]]) -> float:
        return _mean(
            [cell["adapted"][metric] - cell["source"][metric] for cell in selected]
        )

    dataset_nonclean_iou = {
        dataset: macro_delta(
            "iou", _stratum_cells(cells, f"dataset:{dataset}")
        )
        for dataset in DATASETS
    }
    dataset_nonclean_pd = {
        dataset: macro_delta(
            "pd", _stratum_cells(cells, f"dataset:{dataset}")
        )
        for dataset in DATASETS
    }
    family_iou = {
        family: macro_delta(
            "iou", _stratum_cells(cells, f"corruption_family:{family}")
        )
        for family in CORRUPTION_FAMILIES
    }
    clean_dataset_iou = {
        dataset: (
            cells[(dataset, "clean_S0")]["adapted"]["iou"]
            - cells[(dataset, "clean_S0")]["source"]["iou"]
        )
        for dataset in DATASETS
    }
    source_fa: dict[str, float] = {}
    fa_delta: dict[str, float] = {}
    source_fg: dict[str, float] = {}
    adapted_fg: dict[str, float] = {}
    for key in SAFETY_STRATA:
        selected = _stratum_cells(cells, key)
        source_fa[key] = _mean([cell["source"]["fa_per_million"] for cell in selected])
        fa_delta[key] = _mean(
            [cell["adapted"]["fa_per_million"] - cell["source"]["fa_per_million"] for cell in selected]
        )
        source_fg[key] = _mean([cell["source"]["foreground_fraction"] for cell in selected])
        adapted_fg[key] = _mean([cell["adapted"]["foreground_fraction"] for cell in selected])

    valid_all = [value for _dataset, value in global_cosines]
    alignment_macro = _mean(valid_all) if valid_all else 0.0
    alignment_dataset = {
        dataset: (
            float(median([value for name, value in global_cosines if name == dataset]))
            if any(name == dataset for name, _value in global_cosines)
            else 0.0
        )
        for dataset in DATASETS
    }
    evidence = {
        "schema_version": 3,
        "artifact_type": EVIDENCE_ARTIFACT_TYPE,
        "candidate": _science_candidate(candidate),
        "replicate_id": replicate_id,
        "episode_count": EPISODES_PER_CANDIDATE_REPLICATE,
        "metrics": {
            "nonclean_macro_delta_iou": macro_delta("iou", _stratum_cells(cells, "nonclean")),
            "overall_macro_delta_iou": macro_delta("iou", _stratum_cells(cells, "overall")),
            "dataset_nonclean_delta_iou": dataset_nonclean_iou,
            "family_delta_iou": family_iou,
            "clean_macro_delta_iou": macro_delta("iou", _stratum_cells(cells, "clean")),
            "clean_dataset_delta_iou": clean_dataset_iou,
            "nonclean_macro_delta_pd": macro_delta("pd", _stratum_cells(cells, "nonclean")),
            "dataset_nonclean_delta_pd": dataset_nonclean_pd,
            "clean_macro_delta_pd": macro_delta("pd", _stratum_cells(cells, "clean")),
        },
        "safety": {
            "source_fa_per_million": source_fa,
            "fa_delta_per_million": fa_delta,
            "source_foreground_fraction": source_fg,
            "adapted_foreground_fraction": adapted_fg,
        },
        "activity": {
            **activity,
            "functional_logit_threshold": FUNCTIONAL_LOGIT_THRESHOLD,
        },
        "alignment": {
            "macro_cosine": alignment_macro,
            "dataset_median_cosine": alignment_dataset,
        },
    }
    parse_replicate_evidence(evidence)
    return evidence


def build_replicate_evidence_set(
    records: Sequence[Mapping[str, Any]],
    *,
    replicate_id: str,
    r0_eligibility_receipt: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Validate and aggregate the exact candidate set required by a replicate."""

    if isinstance(records, (str, bytes)):
        raise D0V3ReplicateAggregateError("records must be a sequence")
    if replicate_id == "R0":
        if r0_eligibility_receipt is not None:
            raise D0V3ReplicateAggregateError(
                "R0 aggregation must not receive an R0 eligibility receipt"
            )
        expected_candidates = FROZEN_CANDIDATES
    elif replicate_id in {"R1", "R2"}:
        if r0_eligibility_receipt is None:
            raise D0V3ReplicateAggregateError(
                "R1/R2 aggregation requires the canonical R0 eligibility receipt"
            )
        expected_candidates = parse_r0_eligibility_receipt(
            r0_eligibility_receipt
        )
        if not expected_candidates:
            raise D0V3ReplicateAggregateError(
                "R0 had no eligible candidates; R1/R2 execution is forbidden"
            )
    else:
        raise D0V3ReplicateAggregateError("replicate_id must be R0, R1, or R2")

    by_candidate: dict[FormalCandidate, list[Mapping[str, Any]]] = {
        candidate: [] for candidate in expected_candidates
    }
    for raw in records:
        mapping = _mapping(raw, field="record")
        candidate = _candidate(mapping.get("candidate"))
        if candidate not in by_candidate:
            raise D0V3ReplicateAggregateError(
                "replicate contains a candidate outside the receipt-defined set: "
                f"{candidate.candidate_id}"
            )
        by_candidate[candidate].append(mapping)
    missing = tuple(
        candidate.candidate_id
        for candidate in expected_candidates
        if not by_candidate[candidate]
    )
    if missing:
        raise D0V3ReplicateAggregateError(
            f"replicate is missing receipt-required candidates: {list(missing)}"
        )
    return [
        build_candidate_replicate_evidence(
            by_candidate[candidate],
            candidate=candidate,
            replicate_id=replicate_id,
        )
        for candidate in expected_candidates
    ]


__all__ = [
    "D0V3ReplicateAggregateError",
    "EPISODE_ARTIFACT_TYPE",
    "EVIDENCE_ARTIFACT_TYPE",
    "FUNCTIONAL_LOGIT_THRESHOLD",
    "build_candidate_replicate_evidence",
    "build_replicate_evidence_set",
]
