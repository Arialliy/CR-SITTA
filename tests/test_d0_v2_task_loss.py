from __future__ import annotations

from dataclasses import FrozenInstanceError
import json
import math

import pytest
import torch

from analysis.d0_v2_task_loss import (
    BCE_REDUCTION,
    EMPTY_TARGET_CONVENTION,
    SOFT_IOU_REDUCTION,
    D0V2TaskLossConfig,
    D0V2TaskLossError,
    compute_d0_v2_task_loss,
)
from analysis.source_train_provenance import SourceTrainAnalysisProvenance


def _config_mapping() -> dict:
    return {
        "lambda_bce": 2.0,
        "lambda_soft_iou": 3.0,
        "eps": 0.25,
        "bce_reduction": BCE_REDUCTION,
        "soft_iou_reduction": SOFT_IOU_REDUCTION,
        "empty_target_convention": EMPTY_TARGET_CONVENTION,
    }


def _provenance_mapping() -> dict:
    return {
        "dataset": "NUAA-SIRST",
        "split_name": "train",
        "split_sha256": "a" * 64,
        "checkpoint_sha256": "b" * 64,
        "seed": 0,
        "source_train_derived": True,
        "paper_test_result": False,
        "use_test_images": False,
        "use_test_labels": False,
        "oracle_analysis": True,
        "method_label_accesses": 0,
        "outer_evaluator_label_accesses": 1,
        "adaptation_gradient_uses_labels": False,
        "supervised_gradient_role": "outer_oracle_train_labels_only",
    }


def _provenance() -> SourceTrainAnalysisProvenance:
    return SourceTrainAnalysisProvenance.from_mapping(
        _provenance_mapping(), require_outer_oracle=True
    )


def test_manual_bce_and_soft_iou_value_and_json_audit() -> None:
    config = D0V2TaskLossConfig.from_mapping(_config_mapping())
    logits = torch.zeros((1, 1, 1, 2), dtype=torch.float64, requires_grad=True)
    target = torch.tensor([[[[1.0, 0.0]]]], dtype=torch.float64)

    total, audit = compute_d0_v2_task_loss(
        logits=logits,
        target=target,
        config=config,
        provenance=_provenance(),
    )

    expected_bce = math.log(2.0)
    expected_iou = (0.5 + 0.25) / (1.5 + 0.25)
    expected_soft_iou_loss = 1.0 - expected_iou
    expected_total = 2.0 * expected_bce + 3.0 * expected_soft_iou_loss
    assert total.item() == pytest.approx(expected_total)
    assert total.requires_grad is True
    assert total.grad_fn is not None
    assert audit["components"]["bce_with_logits"] == pytest.approx(expected_bce)
    assert audit["components"]["soft_iou_score"] == pytest.approx(expected_iou)
    assert audit["components"]["total_loss"] == pytest.approx(expected_total)
    assert audit["label_isolation"]["used_by_adaptation"] is False
    assert audit["scope"]["split_name"] == "train"
    assert audit["scope"]["use_test_labels"] is False
    json.dumps(audit, allow_nan=False)


def test_gradient_is_finite_and_total_loss_remains_connected() -> None:
    logits = torch.tensor(
        [[[[0.3, -0.8], [1.2, -1.5]]]],
        dtype=torch.float64,
        requires_grad=True,
    )
    target = torch.tensor(
        [[[[1.0, 0.0], [0.4, 0.0]]]], dtype=torch.float64
    )
    total, _ = compute_d0_v2_task_loss(
        logits=logits,
        target=target,
        config=D0V2TaskLossConfig.from_mapping(_config_mapping()),
        provenance=_provenance(),
    )
    gradient = torch.autograd.grad(total, logits)[0]
    assert gradient.shape == logits.shape
    assert bool(torch.isfinite(gradient).all())
    assert bool(torch.count_nonzero(gradient))


def test_empty_target_uses_epsilon_smoothed_union_with_nonzero_gradient() -> None:
    config_map = _config_mapping()
    config_map.update(lambda_bce=0.0, lambda_soft_iou=1.0, eps=0.5)
    logits = torch.zeros((1, 1, 1, 2), dtype=torch.float64, requires_grad=True)
    target = torch.zeros_like(logits, requires_grad=False)
    total, audit = compute_d0_v2_task_loss(
        logits=logits,
        target=target,
        config=config_map,
        provenance=_provenance(),
    )
    expected = 1.0 - 0.5 / (1.0 + 0.5)
    assert total.item() == pytest.approx(expected)
    assert audit["input"]["target_is_empty"] is True
    assert audit["input"]["empty_target_convention_applied"] is True
    gradient = torch.autograd.grad(total, logits)[0]
    assert bool(torch.isfinite(gradient).all())
    assert bool(torch.count_nonzero(gradient))


def test_config_is_frozen_and_round_trips_exactly() -> None:
    mapping = _config_mapping()
    config = D0V2TaskLossConfig.from_mapping(mapping)
    assert config.to_dict() == mapping
    with pytest.raises(FrozenInstanceError):
        config.eps = 1.0  # type: ignore[misc]


@pytest.mark.parametrize(
    ("update", "message"),
    (
        ({"lambda_bce": -1.0}, "lambda_bce must be non-negative"),
        ({"lambda_soft_iou": -1.0}, "lambda_soft_iou must be non-negative"),
        (
            {"lambda_bce": 0.0, "lambda_soft_iou": 0.0},
            "at least one",
        ),
        ({"lambda_bce": True}, "not bool"),
        ({"lambda_soft_iou": float("nan")}, "must be finite"),
        ({"eps": 0.0}, "eps must be positive"),
        ({"eps": float("inf")}, "eps must be finite"),
        ({"bce_reduction": "sum"}, "bce_reduction must be exactly"),
        (
            {"soft_iou_reduction": "sum"},
            "soft_iou_reduction must be exactly",
        ),
        (
            {"empty_target_convention": "return_zero"},
            "empty_target_convention must be exactly",
        ),
    ),
)
def test_config_rejects_invalid_values(update: dict, message: str) -> None:
    mapping = _config_mapping()
    mapping.update(update)
    with pytest.raises(D0V2TaskLossError, match=message):
        D0V2TaskLossConfig.from_mapping(mapping)


def test_config_mapping_rejects_missing_unknown_and_non_mapping() -> None:
    missing = _config_mapping()
    del missing["eps"]
    with pytest.raises(D0V2TaskLossError, match="fields must be exact"):
        D0V2TaskLossConfig.from_mapping(missing)
    unknown = _config_mapping()
    unknown["allow_test"] = True
    with pytest.raises(D0V2TaskLossError, match="fields must be exact"):
        D0V2TaskLossConfig.from_mapping(unknown)
    with pytest.raises(D0V2TaskLossError, match="must be a mapping"):
        D0V2TaskLossConfig.from_mapping([])  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("logits", "target", "message"),
    (
        (
            torch.zeros((1, 2, 2), dtype=torch.float32),
            torch.zeros((1, 2, 2), dtype=torch.float32),
            "shape must be exactly",
        ),
        (
            torch.zeros((2, 1, 2, 2), dtype=torch.float32),
            torch.zeros((2, 1, 2, 2), dtype=torch.float32),
            "shape must be exactly",
        ),
        (
            torch.zeros((1, 2, 2, 2), dtype=torch.float32),
            torch.zeros((1, 2, 2, 2), dtype=torch.float32),
            "shape must be exactly",
        ),
        (
            torch.empty((1, 1, 0, 2), dtype=torch.float32),
            torch.empty((1, 1, 0, 2), dtype=torch.float32),
            "H and W must be positive",
        ),
        (
            torch.zeros((1, 1, 2, 2), dtype=torch.int64),
            torch.zeros((1, 1, 2, 2), dtype=torch.int64),
            "floating-point",
        ),
        (
            torch.zeros((1, 1, 2, 2), dtype=torch.float32),
            torch.zeros((1, 1, 2, 3), dtype=torch.float32),
            "shapes must match",
        ),
        (
            torch.zeros((1, 1, 2, 2), dtype=torch.float32),
            torch.zeros((1, 1, 2, 2), dtype=torch.float64),
            "dtypes must match",
        ),
    ),
)
def test_tensor_shape_and_dtype_contract_fails_closed(
    logits: torch.Tensor, target: torch.Tensor, message: str
) -> None:
    with pytest.raises(D0V2TaskLossError, match=message):
        compute_d0_v2_task_loss(
            logits=logits,
            target=target,
            config=_config_mapping(),
            provenance=_provenance(),
        )


@pytest.mark.parametrize(
    ("which", "value", "message"),
    (
        ("logits", float("nan"), "logits must contain only finite"),
        ("logits", float("inf"), "logits must contain only finite"),
        ("target", float("nan"), "target must contain only finite"),
        ("target", -0.1, "closed interval"),
        ("target", 1.1, "closed interval"),
    ),
)
def test_nonfinite_and_out_of_range_values_fail_closed(
    which: str, value: float, message: str
) -> None:
    logits = torch.zeros((1, 1, 2, 2), dtype=torch.float64)
    target = torch.zeros_like(logits)
    (logits if which == "logits" else target)[0, 0, 0, 0] = value
    with pytest.raises(D0V2TaskLossError, match=message):
        compute_d0_v2_task_loss(
            logits=logits,
            target=target,
            config=_config_mapping(),
            provenance=_provenance(),
        )


def test_target_requiring_gradient_fails_closed() -> None:
    with pytest.raises(D0V2TaskLossError, match="target must not require"):
        compute_d0_v2_task_loss(
            logits=torch.zeros((1, 1, 2, 2), dtype=torch.float64),
            target=torch.zeros(
                (1, 1, 2, 2), dtype=torch.float64, requires_grad=True
            ),
            config=_config_mapping(),
            provenance=_provenance(),
        )


def test_outer_oracle_provenance_is_mandatory_and_revalidated() -> None:
    tensors = {
        "logits": torch.zeros((1, 1, 2, 2), dtype=torch.float64),
        "target": torch.zeros((1, 1, 2, 2), dtype=torch.float64),
    }
    with pytest.raises(D0V2TaskLossError, match="SourceTrainAnalysisProvenance"):
        compute_d0_v2_task_loss(
            **tensors,
            config=_config_mapping(),
            provenance=_provenance_mapping(),  # type: ignore[arg-type]
        )

    label_free_mapping = _provenance_mapping()
    label_free_mapping.update(
        oracle_analysis=False,
        outer_evaluator_label_accesses=0,
        supervised_gradient_role="none",
    )
    label_free = SourceTrainAnalysisProvenance.from_mapping(
        label_free_mapping, require_outer_oracle=False
    )
    with pytest.raises(D0V2TaskLossError, match="train-only outer-oracle"):
        compute_d0_v2_task_loss(
            **tensors,
            config=_config_mapping(),
            provenance=label_free,
        )


def test_computation_does_not_initialize_cuda() -> None:
    initialized_before = torch.cuda.is_initialized()
    compute_d0_v2_task_loss(
        logits=torch.zeros((1, 1, 2, 2), dtype=torch.float64),
        target=torch.zeros((1, 1, 2, 2), dtype=torch.float64),
        config=_config_mapping(),
        provenance=_provenance(),
    )
    assert torch.cuda.is_initialized() is initialized_before
