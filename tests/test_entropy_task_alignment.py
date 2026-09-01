from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis.analyze_entropy_task_alignment import (
    analyze_entropy_task_alignment,
    analyze_payload,
    main,
)
from analysis.source_train_provenance import (
    AnalysisProvenanceError,
    SourceTrainAnalysisProvenance,
)
from tta.parameter_vector import ParameterVectorError


def _provenance_mapping() -> dict:
    return {
        "dataset": "NUDT-SIRST",
        "split_name": "train",
        "split_sha256": "c" * 64,
        "checkpoint_sha256": "d" * 64,
        "seed": 7,
        "source_train_derived": True,
        "paper_test_result": False,
        "use_test_images": False,
        "use_test_labels": False,
        "oracle_analysis": True,
        "method_label_accesses": 0,
        "outer_evaluator_label_accesses": 64,
        "adaptation_gradient_uses_labels": False,
        "supervised_gradient_role": "outer_oracle_train_labels_only",
    }


def _provenance() -> SourceTrainAnalysisProvenance:
    return SourceTrainAnalysisProvenance.from_mapping(
        _provenance_mapping(), require_outer_oracle=True
    )


def test_per_group_cosine_and_supervised_dot_step_have_expected_signs() -> None:
    report = analyze_entropy_task_alignment(
        entropy_gradients={
            "encoder.weight": torch.tensor([1.0, 0.0]),
            "decoder.weight": torch.tensor([1.0]),
        },
        supervised_gradients={
            "encoder.weight": torch.tensor([1.0, 0.0]),
            "decoder.weight": torch.tensor([-1.0]),
        },
        adaptation_step={
            "encoder.weight": torch.tensor([-1.0, 0.0]),
            "decoder.weight": torch.tensor([-1.0]),
        },
        provenance=_provenance(),
        parameter_groups={
            "encoder.weight": "encoder",
            "decoder.weight": "decoder",
        },
    )
    encoder = report["per_group"]["encoder"]
    decoder = report["per_group"]["decoder"]
    assert encoder["entropy_supervised_cosine"] == pytest.approx(1.0)
    assert encoder["supervised_dot_adaptation_step"] == pytest.approx(-1.0)
    assert encoder["first_order_task_effect"] == "predicted_task_loss_decrease"
    assert decoder["entropy_supervised_cosine"] == pytest.approx(-1.0)
    assert decoder["supervised_dot_adaptation_step"] == pytest.approx(1.0)
    assert decoder["first_order_task_effect"] == "predicted_task_loss_increase"
    assert report["global"]["first_order_task_effect"] == "first_order_neutral"
    assert report["label_isolation"]["supervised_gradient_used_by_method"] is False


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("split_name", "test", "train split"),
        ("use_test_labels", True, "use_test_labels"),
        ("method_label_accesses", 1, "method_label_accesses"),
        ("outer_evaluator_label_accesses", 0, "positive outer label"),
        ("adaptation_gradient_uses_labels", True, "adaptation_gradient"),
        ("oracle_analysis", False, "oracle_analysis"),
    ),
)
def test_outer_oracle_provenance_fails_closed(
    field: str, value, message: str
) -> None:
    provenance = _provenance_mapping()
    provenance[field] = value
    with pytest.raises(AnalysisProvenanceError, match=message):
        SourceTrainAnalysisProvenance.from_mapping(
            provenance, require_outer_oracle=True
        )


def test_outer_oracle_provenance_rejects_missing_or_unknown_metadata() -> None:
    missing = _provenance_mapping()
    del missing["checkpoint_sha256"]
    with pytest.raises(AnalysisProvenanceError, match="fields must be exact"):
        SourceTrainAnalysisProvenance.from_mapping(
            missing, require_outer_oracle=True
        )
    unknown = _provenance_mapping()
    unknown["paper_result_override"] = True
    with pytest.raises(AnalysisProvenanceError, match="fields must be exact"):
        SourceTrainAnalysisProvenance.from_mapping(
            unknown, require_outer_oracle=True
        )


def test_alignment_rejects_missing_or_reshaped_gradient() -> None:
    with pytest.raises(ParameterVectorError, match="topology mismatch"):
        analyze_entropy_task_alignment(
            entropy_gradients={"a": torch.tensor([1.0]), "b": torch.tensor([2.0])},
            supervised_gradients={"a": torch.tensor([1.0])},
            adaptation_step={"a": torch.tensor([-1.0]), "b": torch.tensor([-2.0])},
            provenance=_provenance(),
        )


def test_zero_gradient_cosine_is_null_not_nan() -> None:
    report = analyze_entropy_task_alignment(
        entropy_gradients={"a": torch.tensor([0.0])},
        supervised_gradients={"a": torch.tensor([1.0])},
        adaptation_step={"a": torch.tensor([0.0])},
        provenance=_provenance(),
    )
    assert report["global"]["entropy_supervised_cosine"] is None
    json.dumps(report, allow_nan=False)


def test_alignment_payload_and_cli_preserve_outer_only_metadata(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    payload = {
        "provenance": _provenance_mapping(),
        "entropy_gradients": {"a": [1.0]},
        "supervised_gradients": {"a": [1.0]},
        "adaptation_step": {"a": [-0.1]},
        "parameter_groups": {"a": "decoder_low"},
    }
    direct = analyze_payload(payload)
    assert direct["scope"]["oracle_analysis"] is True
    assert direct["scope"]["paper_test_result"] is False
    assert direct["scope"]["method_label_accesses"] == 0
    input_path = tmp_path / "alignment.json"
    input_path.write_text(json.dumps(payload), encoding="utf-8")
    assert main(["--input-json", str(input_path)]) == 0
    emitted = json.loads(capsys.readouterr().out)
    assert emitted["per_group"]["decoder_low"][
        "first_order_task_effect"
    ] == "predicted_task_loss_decrease"


def test_alignment_analysis_does_not_change_cuda_initialization_state() -> None:
    initialized_before = torch.cuda.is_initialized()
    analyze_entropy_task_alignment(
        entropy_gradients={"a": torch.tensor([1.0])},
        supervised_gradients={"a": torch.tensor([1.0])},
        adaptation_step={"a": torch.tensor([-0.1])},
        provenance=_provenance(),
    )
    assert torch.cuda.is_initialized() is initialized_before
