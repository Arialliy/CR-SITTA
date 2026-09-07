"""Synthetic runner boundary checks; no actual weights/images or GPU."""
import argparse
import copy
import json
from pathlib import Path
import sys
import types

import numpy as np
from PIL import Image
import pytest
import torch

from analysis import audit_ipma_micro16_v1 as runner
from analysis.ipma_micro16_data_v1 import ACCESS_FIELDS, FORBIDDEN_ACCESS_FIELDS


def cases(fit_complete=True):
    ids = [f"{index:06d}" for index in range(16)]
    rows = [{"image_id": name, "episode_index": index} for index, name in enumerate(ids)]
    fit = copy.deepcopy(rows[:8]) if fit_complete else []
    access = {name: 0 for name in ACCESS_FIELDS}
    access.update(train_image_opens=16, train_mask_opens=8 if fit_complete else 0,
                  fit_mask_opens=8 if fit_complete else 0)
    return rows, fit, ids, access


@pytest.mark.parametrize("fit_complete", [False, True])
def test_coverage_exact(fit_complete):
    runner.validate_coverage(*cases(fit_complete), fit_complete=fit_complete)


@pytest.mark.parametrize("field", FORBIDDEN_ACCESS_FIELDS)
def test_coverage_forbidden_counts(field):
    rows, fit, ids, access = cases()
    access[field] = 1
    with pytest.raises(ValueError, match="forbidden"):
        runner.validate_coverage(rows, fit, ids, access)


@pytest.mark.parametrize("mutation", ["reorder", "missing", "duplicate", "index", "check_gt", "count", "bool_count"])
def test_coverage_invalid(mutation):
    rows, fit, ids, access = cases()
    if mutation == "reorder":
        rows.reverse()
    elif mutation == "missing":
        rows.pop()
    elif mutation == "duplicate":
        ids[-1] = ids[0]
    elif mutation == "index":
        rows[0]["episode_index"] = 20
    elif mutation == "check_gt":
        fit[-1]["image_id"] = ids[8]
    elif mutation == "count":
        access["fit_mask_opens"] = 16
    elif mutation == "bool_count":
        access["test_mask_opens"] = False
    with pytest.raises(ValueError):
        runner.validate_coverage(rows, fit, ids, access)


def test_pairing_rejects_missing_or_changed():
    runner.assert_pairing({"sha": "a"}, {"sha": "a"}, ["sha"])
    for actual in ({}, {"sha": "b"}):
        with pytest.raises(RuntimeError, match="sealed R2"):
            runner.assert_pairing(actual, {"sha": "a"}, ["sha"])


def test_lf_rows_exact_order_and_duplicate_guard():
    records = [{"image_id": name} for name in ["b", "a"]]
    rows = [{"dataset": "NUDT-SIRST", "view": "train_crop_224", "probe_id": "L4a", "image_id": name}
            for name in ["a", "b", "c"]]
    assert [r["image_id"] for r in runner.ordered_lf_rows(rows, records)] == ["b", "a"]
    with pytest.raises(ValueError, match="duplicate"):
        runner.ordered_lf_rows([*rows, rows[0]], records)
    with pytest.raises(ValueError, match="missing"):
        runner.ordered_lf_rows(rows[:1], records)


def test_native_sigmoid_threshold_is_strict_and_rounding_preserved():
    logits = torch.tensor([-1.0, 0.0, 1.0e-8, 1.0], dtype=torch.float32).view(1, 1, 1, 4)
    probs, mask = runner.native_prediction(logits)
    assert torch.equal(probs, logits.sigmoid())
    assert mask.flatten().tolist() == [0, 0, 0, 255]
    # Double sigmoid would turn the tiny positive logit into foreground.
    assert logits.double().sigmoid()[0, 0, 0, 2] > 0.5


@pytest.mark.parametrize("logits", [torch.zeros(1, 1, 2, 2).double(), torch.zeros(1, 2, 2, 2),
    torch.zeros(1, 1, 2), torch.full((1, 1, 2, 2), float("nan"))])
def test_prediction_rejects_wrong_precision_shape_nonfinite(logits):
    with pytest.raises(ValueError):
        runner.native_prediction(logits)


def test_save_all_masks_and_reject_overwrite(tmp_path):
    for name in runner.PREDICTION_NAMES:
        (tmp_path / "predictions" / name).mkdir(parents=True)
    logits = torch.tensor([[-1., 0.], [1., 1.0e-8]]).view(1, 1, 2, 2)
    tensors = {f"{name}_logits": logits for name in runner.PREDICTION_NAMES}
    paths = runner.save_prediction_set(tmp_path, "example", tensors)
    assert set(paths) == set(runner.PREDICTION_NAMES)
    for item in paths.values():
        with Image.open(item["path"]) as image:
            assert np.asarray(image).tolist() == [[0, 0], [255, 0]]
    with pytest.raises(FileExistsError):
        runner.save_prediction_set(tmp_path, "example", tensors)


def test_materialized_probabilities_used_without_recomputing_sigmoid(tmp_path, monkeypatch):
    for name in runner.PREDICTION_NAMES:
        (tmp_path / "predictions" / name).mkdir(parents=True)
    logits = torch.zeros(1, 1, 2, 2)
    probs = torch.tensor([0., .5, .5001, 1.]).view(1, 1, 2, 2)
    tensors = {f"{name}_logits": logits for name in runner.PREDICTION_NAMES}
    monkeypatch.setattr(runner, "native_prediction", lambda *args: pytest.fail("must not recompute sigmoid"))
    paths = runner.save_prediction_set(tmp_path, "example", tensors,
        probabilities={name: probs for name in runner.PREDICTION_NAMES})
    for item in paths.values():
        with Image.open(item["path"]) as image:
            assert np.asarray(image).tolist() == [[0, 0], [255, 255]]


def test_tensor_artifact_weights_only_and_no_overwrite(tmp_path):
    path = tmp_path / "audit.pth.tar"
    runner.save_tensor_artifact(path, {"trained": False, "tensors": {"x": torch.ones(1)}})
    payload = torch.load(path, map_location="cpu", weights_only=True)
    assert payload["trained"] is False
    assert torch.equal(payload["tensors"]["x"], torch.ones(1))
    with pytest.raises(FileExistsError):
        runner.save_tensor_artifact(path, {})


def test_fit_requires_sealed_unlabeled_phase(tmp_path, monkeypatch):
    rows, _, ids, access = cases(False)
    with pytest.raises(RuntimeError, match="completed label-free"):
        runner.require_unlabeled_phase_sealed(tmp_path, rows, ids, access)
    (tmp_path / "LABEL_FREE_COMPLETE.json").write_text("{}", encoding="utf-8")
    for row in rows:
        row["prediction_masks"] = {name: {"path": str(tmp_path / "missing"), "sha256": "bad"}
                                   for name in runner.PREDICTION_NAMES}
        row["cache"] = row["episode_tensors"] = row["prediction_masks"]["source"]
    with pytest.raises((FileNotFoundError, RuntimeError, ValueError)):
        runner.require_unlabeled_phase_sealed(tmp_path, rows, ids, access)
    import analysis.lf_repair_contract_v2 as contract
    calls = []
    monkeypatch.setattr(contract, "assert_bindings", lambda items: calls.append(items))
    runner.require_unlabeled_phase_sealed(tmp_path, rows, ids, access)
    assert len(calls) == 16 and all(len(items) == 6 for items in calls)
    rows[0]["prediction_masks"].pop("post")
    with pytest.raises(RuntimeError, match="all prediction masks"):
        runner.require_unlabeled_phase_sealed(tmp_path, rows, ids, access)


def test_readonly_default_has_no_execute_side_effects(monkeypatch, tmp_path):
    import analysis.ipma_micro16_contract_v1 as contract
    called = []
    monkeypatch.setattr(contract, "prepare_run", lambda path: called.append(path) or {"preflight": {"ready": True}})
    monkeypatch.setattr(runner, "execute_prepared", lambda *args: pytest.fail("must not execute"))
    args = argparse.Namespace(execute=False, engineering_only=False, config=tmp_path / "cfg", device="cuda:0")
    assert runner.run(args) == {"ready": True}
    assert called == [args.config]
    assert list(tmp_path.iterdir()) == []


def test_execute_needs_two_flags_before_prepare(monkeypatch):
    import analysis.ipma_micro16_contract_v1 as contract
    monkeypatch.setattr(contract, "prepare_run", lambda *args: pytest.fail("premature prepare"))
    with pytest.raises(ValueError, match="engineering-only"):
        runner.run(argparse.Namespace(execute=True, engineering_only=False))


@pytest.mark.parametrize("passed", [False, True])
def test_summary_never_promotes_even_if_engineering_passed(passed):
    result = runner.nonpromoting_summary({"engineering_passed": passed}, access={},
        label_free=[{"label_free_signal_passed": True}] * 16, fit=[{"target_empty": False}] * 8,
        metrics={}, transitions={}, replay={}, host_binding={})
    for name in ("ipma_meta_training_executed", "ipma_meta_training_allowed", "full_source_training_allowed",
                 "formal_test_allowed", "new_validation_split", "paper_result", "task_improvement_used_for_gate"):
        assert result[name] is False
    assert result["outer_optimizer_steps"] == 0
    assert result["requires_frozen_r5_contract"] is True
    assert result["prediction_mask_count"] == 64


def test_real_loader_uses_safe_api_and_strict_original_model(monkeypatch, tmp_path):
    from analysis import d0a_v7_common as common
    import export_cr_sitta_d0a_safe_checkpoint as exporter
    host = {"safe_checkpoint": str(tmp_path / "safe"), "safe_checkpoint_sha256": "safe_sha",
            "source_checkpoint_sha256": "source_sha", "run_contract_sha256": "contract_sha"}
    provenance = {"architecture": "MSHNet_NSFPN", "dataset": "NUDT-SIRST", "epoch": 1000,
        "global_optimizer_step": 41000, "state_dict_keys": 505, "selection_rule": "fixed_final_epoch_train_only",
        "test_selected": False, "source_checkpoint_sha256": "source_sha", "run_contract_sha256": "contract_sha"}
    calls = []
    class FakeModel(torch.nn.Module):
        def __init__(self, input_channels):
            super().__init__()
            assert input_channels == 3
            self.weight = torch.nn.Parameter(torch.zeros(1))
        def load_state_dict(self, state, strict):
            calls.append(("strict_load", strict))
            return super().load_state_dict(state, strict=strict)
    monkeypatch.setitem(sys.modules, "model.MSHNet_NSFPN", types.SimpleNamespace(MSHNet_NSFPN=FakeModel))
    monkeypatch.setattr(common, "sha256_file", lambda path: "safe_sha")
    monkeypatch.setattr(exporter, "_weights_only_load", lambda path: calls.append(("weights_only_load", path)) or {})
    monkeypatch.setattr(exporter, "_validate_safe_payload", lambda payload: (provenance, {"weight": torch.ones(1)}))
    model, _ = runner.load_frozen_host({"host": host}, "cpu")
    assert calls[0][0] == "weights_only_load" and calls[1] == ("strict_load", True)
    assert not model.training and not model.weight.requires_grad
    provenance["test_selected"] = True
    with pytest.raises(RuntimeError, match="test_selected"):
        runner.load_frozen_host({"host": host}, "cpu")


def test_metric_protocol_and_native_probability_interface():
    from metrics.irstd_metrics import UnifiedResearchEvaluator
    protocol = runner.metric_protocol()
    assert protocol.fixed_probability_threshold == 0.5 and protocol.connectivity == 2
    logits = torch.tensor([0., 1.0e-8, 1., -1.]).view(1, 1, 2, 2)
    probs, mask = runner.native_prediction(logits)
    target = (mask > 0).float()
    evaluator = UnifiedResearchEvaluator(protocol)
    evaluator.update_probabilities(probs, target)
    assert evaluator.compute().fixed.pixel.intersection_over_union == 1.0
