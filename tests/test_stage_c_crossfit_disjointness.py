from __future__ import annotations

import copy

import pytest

from analysis.stage_c_crossfit_selector import (
    FOLD_IDS,
    StageCCrossFitError,
    aggregate_held_out_confirmation,
    build_train_pilot64_crossfit,
)


def _ids() -> tuple[str, ...]:
    return tuple(f"train/image_{index:03d}" for index in range(64))


def test_four_folds_are_deterministic_disjoint_and_exhaustive() -> None:
    first = build_train_pilot64_crossfit(_ids(), dataset="NUAA-SIRST")
    second = build_train_pilot64_crossfit(_ids(), dataset="NUAA-SIRST")
    assert first == second
    assert tuple(first.folds) == FOLD_IDS
    assert all(len(first.folds[fold]) == 16 for fold in FOLD_IDS)
    flattened = tuple(value for fold in FOLD_IDS for value in first.folds[fold])
    assert len(flattened) == len(set(flattened)) == 64
    assert set(flattened) == set(_ids())


def test_every_id_is_held_out_exactly_once_and_never_in_its_development_set() -> None:
    plan = build_train_pilot64_crossfit(_ids(), dataset="IRSTD-1K")
    held_out_counts = {image_id: 0 for image_id in _ids()}
    for rotation in plan.rotations:
        assert len(rotation.held_out_image_ids) == 16
        assert len(rotation.development_image_ids) == 48
        assert set(rotation.held_out_image_ids).isdisjoint(
            rotation.development_image_ids
        )
        for image_id in rotation.held_out_image_ids:
            held_out_counts[image_id] += 1
    assert set(held_out_counts.values()) == {1}


def test_fold_assignment_requires_exact_unique_pilot64() -> None:
    with pytest.raises(StageCCrossFitError, match="exactly 64 unique train IDs"):
        build_train_pilot64_crossfit(_ids()[:-1], dataset="NUDT-SIRST")
    with pytest.raises(StageCCrossFitError, match="exactly 64 unique train IDs"):
        build_train_pilot64_crossfit((*_ids()[:-1], _ids()[0]), dataset="NUDT-SIRST")


def _held_out_records():
    return [
        {
            "candidate_id": candidate,
            "fold_id": fold,
            "role": "held_out_confirmation",
            "split_role": "train",
            "metrics": {"delta_iou": float(index), "delta_pd": -float(index)},
            "validation_payload_accesses": 0,
            "test_payload_accesses": 0,
        }
        for candidate in ("C1", "C2")
        for index, fold in enumerate(FOLD_IDS)
    ]


def test_aggregate_uses_exactly_four_held_out_train_records() -> None:
    output = aggregate_held_out_confirmation(
        _held_out_records(),
        candidate_ids=("C1", "C2"),
        metric_names=("delta_iou", "delta_pd"),
    )
    assert tuple(item.candidate_id for item in output) == ("C1", "C2")
    assert all(item.fold_count == item.record_count == 4 for item in output)
    assert all(item.metrics["delta_iou"] == pytest.approx(1.5) for item in output)


@pytest.mark.parametrize(
    "field,value,match",
    [
        ("split_role", "test", "held-out train"),
        ("split_role", "validation", "held-out train"),
        ("role", "development_selection", "held-out train"),
        ("test_payload_accesses", 1, "payload access"),
        ("validation_payload_accesses", 1, "payload access"),
    ],
)
def test_aggregate_rejects_development_validation_or_test_records(
    field: str, value: object, match: str
) -> None:
    records = copy.deepcopy(_held_out_records())
    records[0][field] = value
    with pytest.raises(StageCCrossFitError, match=match):
        aggregate_held_out_confirmation(
            records,
            candidate_ids=("C1", "C2"),
            metric_names=("delta_iou", "delta_pd"),
        )
