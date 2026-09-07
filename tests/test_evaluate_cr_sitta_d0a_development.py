"""Focused contracts for the independent completed-NUDT development evaluator."""

from copy import deepcopy
from pathlib import Path

import pytest
import torch

import evaluate_cr_sitta_d0a_development as evaluator


def safe_payload():
    counters = {f"{prefix}_{suffix}": 0 for prefix in ("test", "validation")
                for suffix in ("split_reads", "image_opens", "mask_opens")}
    return {"provenance": {
        "schema_version": 1, "artifact_type": evaluator.safe_export.SAFE_ARTIFACT_TYPE,
        "method_name": "CR-SITTA", "method_stage": "D0-A", "epoch": 1000,
        "dataset": evaluator.DATASET, "architecture": "MSHNet_NSFPN", "test_selected": False,
        "selection_rule": "fixed_final_epoch_train_only", "development_only": True,
        "state_dict_keys": 505, "split_manifest": counters.copy(), "access_firewall": counters.copy(),
    }, "state_dict": {str(index): torch.zeros(1) for index in range(505)}}


def test_final_epoch_metadata_accepted_without_best_metric_metadata():
    payload = safe_payload()
    assert evaluator.validate_checkpoint_metadata(payload)["epoch"] == 1000
    assert "selection_metric" not in payload["provenance"]


@pytest.mark.parametrize("field,value", [
    ("dataset", "IRSTD-1K"), ("epoch", 999), ("test_selected", True),
    ("selection_rule", "maximize_miou_then_pd_then_minimize_fa"), ("development_only", False),
])
def test_wrong_checkpoint_metadata_rejected(field, value):
    payload = safe_payload()
    payload["provenance"][field] = value
    with pytest.raises((ValueError, evaluator.safe_export.SafeCheckpointExportError)):
        evaluator.validate_checkpoint_metadata(payload)


def test_training_access_counter_and_nonfinite_tensor_rejected():
    payload = safe_payload()
    payload["provenance"]["access_firewall"]["test_image_opens"] = 1
    with pytest.raises(ValueError, match="access_firewall"):
        evaluator.validate_checkpoint_metadata(payload)
    payload = safe_payload()
    payload["state_dict"]["0"] = torch.tensor([float("nan")])
    with pytest.raises(ValueError, match="non-finite"):
        evaluator.validate_checkpoint_metadata(payload)


def baseline_metrics():
    numbers = {"miou": 0.8, "pd": 0.97, "fa_per_pixel_x1e6": 18.0}
    return {
        "dataset": evaluator.DATASET, "split_sha256": evaluator.SPLIT_SHA256,
        "split_file": str(evaluator.SPLIT), "dataset_root": str(evaluator.DATA_ROOT),
        "available_images": 664, "evaluated_images": 664, "complete_fixed_test_split": True,
        "image_size": 256, "threshold_rule": evaluator.THRESHOLD_RULE, "seed": 42,
        "official_reported_operating_point": numbers.copy(),
        "checkpoint_recorded_metric_comparison": {"passed": True},
        "checkpoint_metadata": {"dataset": evaluator.DATASET, "test_selected": True,
                                "selection_metric": "miou", "test_metrics": {**numbers, "images": 664},
                                "split_manifest": {"test_split_sha256": evaluator.SPLIT_SHA256}},
    }


def test_matching_baseline_protocol_accepted():
    evaluator.validate_baseline_protocol(baseline_metrics(), {"metric.py": "abc"},
                                        axis="best_miou", expected_evaluator_hashes={"metric.py": "abc"})


@pytest.mark.parametrize("field,value", [
    ("split_sha256", "wrong"), ("evaluated_images", 64), ("image_size", 224),
    ("threshold_rule", "sigmoid(logit) >= 0.5"), ("seed", 0),
])
def test_protocol_drift_rejected(field, value):
    metrics = baseline_metrics()
    metrics[field] = value
    with pytest.raises(ValueError):
        evaluator.validate_baseline_protocol(metrics, {}, axis="best_miou", expected_evaluator_hashes={})


def test_evaluator_hash_drift_rejected():
    with pytest.raises(ValueError, match="evaluator"):
        evaluator.validate_baseline_protocol(baseline_metrics(), {"metric.py": "old"},
                                            axis="best_miou", expected_evaluator_hashes={"metric.py": "new"})


def test_comparison_units_directions_and_no_promotion():
    baseline = {"metrics": baseline_metrics(), "metrics_file": {"path": "/baseline", "sha256": "a"}}
    comparison = evaluator.build_comparison({"miou": 0.81, "pd": 0.96, "fa_per_pixel_x1e6": 17.0},
                                           {"best_miou": baseline, "best_pd": deepcopy(baseline)})
    delta = comparison["comparisons"]["best_miou"]
    assert delta["delta_percentage_points"]["miou"] == pytest.approx(1.0)
    assert delta["improved"] == {"miou": True, "pd": False, "fa_per_pixel_x1e6": True}
    assert comparison["checkpoint_test_selected"] is False
    assert comparison["paper_result"] is False
    assert comparison["d0b_promotion_authorized"] is False
    assert comparison["d1_promotion_authorized"] is False


def test_refuses_existing_partial_or_complete_output(tmp_path):
    output = tmp_path / "run"
    evaluator.reserve_output(output)
    (output / "user_file.txt").write_text("preserve")
    with pytest.raises(FileExistsError):
        evaluator.reserve_output(output)
    assert (output / "user_file.txt").read_text() == "preserve"
    with pytest.raises(FileExistsError):
        evaluator._new_json(output / "user_file.txt", {})


def test_refuses_symlink_output(tmp_path):
    output = tmp_path / "run"
    output.symlink_to(tmp_path / "missing")
    with pytest.raises(FileExistsError):
        evaluator.reserve_output(output)
