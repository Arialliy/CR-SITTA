"""Fail-closed provenance for source-train-only outer diagnostics."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import re
from typing import Any, Literal


SHA256_PATTERN = re.compile(r"[0-9a-f]{64}\Z")
OUTER_ORACLE_ROLE = "outer_oracle_train_labels_only"
NO_SUPERVISED_GRADIENT_ROLE = "none"


class AnalysisProvenanceError(ValueError):
    """An analysis attempted to use non-train or method-visible label evidence."""


def _required_bool(value: Mapping[str, Any], key: str, expected: bool) -> bool:
    observed = value.get(key)
    if observed is not expected:
        raise AnalysisProvenanceError(f"{key} must be exactly {expected}")
    return expected


def _sha256(value: Any, key: str) -> str:
    if not isinstance(value, str) or SHA256_PATTERN.fullmatch(value) is None:
        raise AnalysisProvenanceError(f"{key} must be lowercase 64-hex SHA-256")
    return value


@dataclass(frozen=True)
class SourceTrainAnalysisProvenance:
    """Immutable scope receipt for geometry or outer-oracle alignment analysis."""

    dataset: str
    split_name: Literal["train"]
    split_sha256: str
    checkpoint_sha256: str
    seed: int
    oracle_analysis: bool
    outer_evaluator_label_accesses: int
    supervised_gradient_role: Literal[
        "none", "outer_oracle_train_labels_only"
    ]

    def __post_init__(self) -> None:
        if not isinstance(self.dataset, str) or not self.dataset:
            raise AnalysisProvenanceError("dataset must be a non-empty string")
        if self.split_name != "train":
            raise AnalysisProvenanceError(
                "outer diagnostics are restricted to the frozen train split"
            )
        _sha256(self.split_sha256, "split_sha256")
        _sha256(self.checkpoint_sha256, "checkpoint_sha256")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or self.seed < 0:
            raise AnalysisProvenanceError("seed must be a non-negative integer")
        if not isinstance(self.oracle_analysis, bool):
            raise AnalysisProvenanceError("oracle_analysis must be bool")
        accesses = self.outer_evaluator_label_accesses
        if isinstance(accesses, bool) or not isinstance(accesses, int):
            raise AnalysisProvenanceError(
                "outer_evaluator_label_accesses must be an integer"
            )
        expected_role = (
            OUTER_ORACLE_ROLE
            if self.oracle_analysis
            else NO_SUPERVISED_GRADIENT_ROLE
        )
        if self.supervised_gradient_role != expected_role:
            raise AnalysisProvenanceError(
                "oracle_analysis and supervised_gradient_role are inconsistent"
            )
        if self.oracle_analysis and accesses <= 0:
            raise AnalysisProvenanceError(
                "outer oracle alignment requires positive outer label accesses"
            )
        if not self.oracle_analysis and accesses != 0:
            raise AnalysisProvenanceError(
                "label-free geometry requires zero outer label accesses"
            )

    @classmethod
    def from_mapping(
        cls,
        value: Mapping[str, Any],
        *,
        require_outer_oracle: bool,
    ) -> "SourceTrainAnalysisProvenance":
        if not isinstance(value, Mapping):
            raise AnalysisProvenanceError("analysis provenance must be a mapping")
        expected_fields = {
            "dataset",
            "split_name",
            "split_sha256",
            "checkpoint_sha256",
            "seed",
            "source_train_derived",
            "paper_test_result",
            "use_test_images",
            "use_test_labels",
            "oracle_analysis",
            "method_label_accesses",
            "outer_evaluator_label_accesses",
            "adaptation_gradient_uses_labels",
            "supervised_gradient_role",
        }
        missing = sorted(expected_fields - set(value))
        unknown = sorted(set(value) - expected_fields)
        if missing or unknown:
            raise AnalysisProvenanceError(
                "analysis provenance fields must be exact; "
                f"missing={missing}, unknown={unknown}"
            )
        dataset = value.get("dataset")
        if not isinstance(dataset, str) or not dataset:
            raise AnalysisProvenanceError("dataset must be a non-empty string")
        if value.get("split_name") != "train":
            raise AnalysisProvenanceError(
                "outer diagnostics are restricted to the frozen train split"
            )
        split_sha256 = _sha256(value.get("split_sha256"), "split_sha256")
        checkpoint_sha256 = _sha256(
            value.get("checkpoint_sha256"), "checkpoint_sha256"
        )
        seed = value.get("seed")
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            raise AnalysisProvenanceError("seed must be a non-negative integer")

        _required_bool(value, "source_train_derived", True)
        _required_bool(value, "paper_test_result", False)
        _required_bool(value, "use_test_images", False)
        _required_bool(value, "use_test_labels", False)
        _required_bool(value, "adaptation_gradient_uses_labels", False)
        accesses = value.get("method_label_accesses")
        if isinstance(accesses, bool) or not isinstance(accesses, int) or accesses != 0:
            raise AnalysisProvenanceError("method_label_accesses must be exactly zero")

        outer_accesses = value.get("outer_evaluator_label_accesses")
        if isinstance(outer_accesses, bool) or not isinstance(outer_accesses, int):
            raise AnalysisProvenanceError(
                "outer_evaluator_label_accesses must be an integer"
            )
        expected_oracle = bool(require_outer_oracle)
        _required_bool(value, "oracle_analysis", expected_oracle)
        expected_role = (
            OUTER_ORACLE_ROLE
            if require_outer_oracle
            else NO_SUPERVISED_GRADIENT_ROLE
        )
        if value.get("supervised_gradient_role") != expected_role:
            raise AnalysisProvenanceError(
                f"supervised_gradient_role must be {expected_role!r}"
            )
        if require_outer_oracle:
            if outer_accesses <= 0:
                raise AnalysisProvenanceError(
                    "outer oracle alignment requires positive outer label accesses"
                )
        elif outer_accesses != 0:
            raise AnalysisProvenanceError(
                "optimizer geometry must not access labels in any role"
            )

        return cls(
            dataset=dataset,
            split_name="train",
            split_sha256=split_sha256,
            checkpoint_sha256=checkpoint_sha256,
            seed=seed,
            oracle_analysis=expected_oracle,
            outer_evaluator_label_accesses=outer_accesses,
            supervised_gradient_role=expected_role,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset": self.dataset,
            "split_name": self.split_name,
            "split_sha256": self.split_sha256,
            "checkpoint_sha256": self.checkpoint_sha256,
            "seed": self.seed,
            "source_train_derived": True,
            "paper_test_result": False,
            "use_test_images": False,
            "use_test_labels": False,
            "oracle_analysis": self.oracle_analysis,
            "method_label_accesses": 0,
            "outer_evaluator_label_accesses": (
                self.outer_evaluator_label_accesses
            ),
            "adaptation_gradient_uses_labels": False,
            "supervised_gradient_role": self.supervised_gradient_role,
        }


__all__ = [
    "AnalysisProvenanceError",
    "NO_SUPERVISED_GRADIENT_ROLE",
    "OUTER_ORACLE_ROLE",
    "SourceTrainAnalysisProvenance",
]
