"""Pure D0-B projection of a Stage-C0 gradient-gate decision.

The inherited Stage-C0 gate decides parameter-space eligibility.  This module
changes only the stage transition: D0-B may authorize D1 train-internal OOF,
never the retired C1 name and never any test phase.  It performs no I/O.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from analysis.stage_c_science_gate_v1 import StageCProtocolError, StageCReceipt


@dataclass(frozen=True, slots=True)
class D1Authorization:
    schema_version: str
    protocol_status: str
    scientific_status: str
    d1_train_internal_oof_allowed: bool
    eligible_parameter_space_ids: tuple[str, ...]
    authorization_scope: str
    formal_test_allowed: bool
    validation_access_count: int
    test_access_count: int
    reason: str

    def to_receipt(self) -> dict[str, Any]:
        value = asdict(self)
        value["eligible_parameter_space_ids"] = list(
            self.eligible_parameter_space_ids
        )
        return value


def _validate_receipt(receipt: StageCReceipt) -> tuple[str, ...]:
    if not isinstance(receipt, StageCReceipt):
        raise StageCProtocolError("D0-B authorization requires a StageCReceipt")
    if receipt.protocol_status != "protocol_complete":
        raise StageCProtocolError("D0-B authorization requires protocol-complete evidence")
    derived = tuple(
        evaluation.parameter_space
        for evaluation in receipt.mechanism_evaluation.space_evaluations
        if evaluation.eligible
    )
    if derived != receipt.eligible_space_ids:
        raise StageCProtocolError("D0-B eligible-space projection is inconsistent")
    expected_status = "scientific_eligible" if derived else "scientific_no_eligible"
    if receipt.scientific_status != expected_status:
        raise StageCProtocolError("D0-B scientific status contradicts eligibility")
    return derived


def build_d0b_science_receipt(receipt: StageCReceipt) -> dict[str, Any]:
    """Return a D0-B receipt with no legacy C1 authorization fields."""

    eligible = _validate_receipt(receipt)
    return {
        "schema_version": "cr_sitta_d0b_science_receipt_v1",
        "receipt_type": "d0b_checkpoint_rebound_train_only_gradient_gate",
        "stage": "D0-B",
        "protocol_status": receipt.protocol_status,
        "scientific_status": receipt.scientific_status,
        "development_only": True,
        "paper_result": False,
        "data_role": "train_fixed_Pilot64",
        "validation_access_count": 0,
        "test_access_count": 0,
        "mechanism_evaluation": receipt.mechanism_evaluation.to_receipt(),
        "eligible_parameter_space_ids": list(eligible),
        "d1_train_internal_oof_allowed": bool(eligible),
        "formal_test_allowed": False,
        "exit_semantics": (
            "continue_to_preregistered_D1_train_internal_OOF"
            if eligible
            else "normal_scientific_early_stop_before_D1_and_test"
        ),
    }


def authorize_d1(receipt: StageCReceipt) -> D1Authorization:
    """Project eligibility to D1 only; never authorize formal test."""

    eligible = _validate_receipt(receipt)
    allowed = bool(eligible)
    return D1Authorization(
        schema_version="cr_sitta_d0b_d1_authorization_v1",
        protocol_status=receipt.protocol_status,
        scientific_status=receipt.scientific_status,
        d1_train_internal_oof_allowed=allowed,
        eligible_parameter_space_ids=eligible,
        authorization_scope="D1_train_internal_OOF_only",
        formal_test_allowed=False,
        validation_access_count=0,
        test_access_count=0,
        reason=(
            "eligible_parameter_space_exists"
            if allowed
            else "scientific_no_eligible_stop"
        ),
    )


__all__ = [
    "D1Authorization",
    "authorize_d1",
    "build_d0b_science_receipt",
]
