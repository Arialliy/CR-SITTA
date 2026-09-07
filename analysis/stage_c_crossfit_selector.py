"""Train-only four-fold cross-fitting primitives for CR-SITTA Stage-C.

Fold assignment depends only on canonical train image IDs.  The module has no
filesystem or model side effects and deliberately accepts neither masks nor
validation/test records.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import math
from numbers import Real
from typing import Any


FOLD_IDS = ("F0", "F1", "F2", "F3")
_HASH_DOMAIN = "cr-sitta-stage-c1-crossfit-v1"


class StageCCrossFitError(ValueError):
    """The train-only cross-fit contract is incomplete or inconsistent."""


@dataclass(frozen=True)
class CrossFitRotation:
    held_out_fold: str
    development_folds: tuple[str, ...]
    held_out_image_ids: tuple[str, ...]
    development_image_ids: tuple[str, ...]


@dataclass(frozen=True)
class CrossFitPlan:
    dataset: str
    folds: Mapping[str, tuple[str, ...]]
    rotations: tuple[CrossFitRotation, ...]
    ordered_ids_sha256: str
    assignment_sha256: str


@dataclass(frozen=True)
class HeldOutCandidateAggregate:
    candidate_id: str
    fold_count: int
    record_count: int
    metrics: Mapping[str, float]


def _canonical_id(value: Any) -> str:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise StageCCrossFitError(
            "every image ID must be a non-empty canonical string"
        )
    if "\0" in value or value.startswith("/") or ".." in value.split("/"):
        raise StageCCrossFitError("image ID is not a safe relative identifier")
    return value


def _digest(dataset: str, image_id: str) -> str:
    payload = f"{_HASH_DOMAIN}\0{dataset}\0{image_id}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def build_train_pilot64_crossfit(
    image_ids: Sequence[str],
    *,
    dataset: str,
) -> CrossFitPlan:
    """Assign exactly 64 train IDs to four deterministic 16-image folds."""

    if isinstance(image_ids, (str, bytes)):
        raise TypeError("image_ids must be a sequence of strings")
    if not isinstance(dataset, str) or not dataset:
        raise StageCCrossFitError("dataset must be a non-empty string")
    ids = tuple(_canonical_id(value) for value in image_ids)
    if len(ids) != 64 or len(set(ids)) != 64:
        raise StageCCrossFitError("cross-fitting requires exactly 64 unique train IDs")
    ordered = tuple(sorted(ids, key=lambda value: (_digest(dataset, value), value)))
    folds = {
        fold_id: ordered[index * 16 : (index + 1) * 16]
        for index, fold_id in enumerate(FOLD_IDS)
    }
    rotations: list[CrossFitRotation] = []
    for held_out in FOLD_IDS:
        development_folds = tuple(value for value in FOLD_IDS if value != held_out)
        development_ids = tuple(
            image_id
            for fold_id in development_folds
            for image_id in folds[fold_id]
        )
        rotations.append(
            CrossFitRotation(
                held_out_fold=held_out,
                development_folds=development_folds,
                held_out_image_ids=folds[held_out],
                development_image_ids=development_ids,
            )
        )
    ordered_hash = hashlib.sha256(("\n".join(ids) + "\n").encode()).hexdigest()
    assignment_lines = tuple(
        f"{fold_id}\t{image_id}"
        for fold_id in FOLD_IDS
        for image_id in folds[fold_id]
    )
    assignment_hash = hashlib.sha256(
        ("\n".join(assignment_lines) + "\n").encode()
    ).hexdigest()
    return CrossFitPlan(
        dataset=dataset,
        folds=folds,
        rotations=tuple(rotations),
        ordered_ids_sha256=ordered_hash,
        assignment_sha256=assignment_hash,
    )


def aggregate_held_out_confirmation(
    records: Sequence[Mapping[str, Any]],
    *,
    candidate_ids: Sequence[str],
    metric_names: Sequence[str],
) -> tuple[HeldOutCandidateAggregate, ...]:
    """Macro-average only four-fold held-out train confirmation records."""

    if isinstance(records, (str, bytes)):
        raise TypeError("records must be a sequence")
    candidates = tuple(candidate_ids)
    metrics = tuple(metric_names)
    if not candidates or len(set(candidates)) != len(candidates):
        raise StageCCrossFitError("candidate_ids must be non-empty and unique")
    if not metrics or len(set(metrics)) != len(metrics):
        raise StageCCrossFitError("metric_names must be non-empty and unique")
    allowed_fields = {
        "candidate_id",
        "fold_id",
        "role",
        "split_role",
        "metrics",
        "validation_payload_accesses",
        "test_payload_accesses",
    }
    by_candidate: dict[str, dict[str, Mapping[str, Any]]] = {
        candidate: {} for candidate in candidates
    }
    for raw in records:
        if not isinstance(raw, Mapping) or set(raw) != allowed_fields:
            raise StageCCrossFitError("held-out record fields differ from contract")
        if raw["role"] != "held_out_confirmation" or raw["split_role"] != "train":
            raise StageCCrossFitError("only held-out train confirmation is accepted")
        if raw["validation_payload_accesses"] != 0 or raw["test_payload_accesses"] != 0:
            raise StageCCrossFitError("validation/test payload access is forbidden")
        candidate = raw["candidate_id"]
        fold_id = raw["fold_id"]
        if candidate not in by_candidate or fold_id not in FOLD_IDS:
            raise StageCCrossFitError("unknown candidate or fold")
        if fold_id in by_candidate[candidate]:
            raise StageCCrossFitError("duplicate candidate/fold held-out record")
        observed = raw["metrics"]
        if not isinstance(observed, Mapping) or set(observed) != set(metrics):
            raise StageCCrossFitError("held-out metric fields differ from contract")
        checked: dict[str, float] = {}
        for name in metrics:
            value = observed[name]
            if isinstance(value, bool) or not isinstance(value, Real):
                raise StageCCrossFitError(f"metric {name} must be real")
            number = float(value)
            if not math.isfinite(number):
                raise StageCCrossFitError(f"metric {name} must be finite")
            checked[name] = number
        by_candidate[candidate][fold_id] = checked

    aggregates: list[HeldOutCandidateAggregate] = []
    for candidate in candidates:
        fold_records = by_candidate[candidate]
        if tuple(sorted(fold_records)) != FOLD_IDS:
            raise StageCCrossFitError(
                f"candidate {candidate!r} lacks four held-out folds"
            )
        aggregates.append(
            HeldOutCandidateAggregate(
                candidate_id=candidate,
                fold_count=4,
                record_count=4,
                metrics={
                    name: math.fsum(
                        float(fold_records[fold][name]) for fold in FOLD_IDS
                    )
                    / 4.0
                    for name in metrics
                },
            )
        )
    return tuple(aggregates)


__all__ = [
    "CrossFitPlan",
    "CrossFitRotation",
    "FOLD_IDS",
    "HeldOutCandidateAggregate",
    "StageCCrossFitError",
    "aggregate_held_out_confirmation",
    "build_train_pilot64_crossfit",
]
