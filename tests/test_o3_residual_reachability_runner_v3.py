"""Synthetic integration only: no real inputs, checkpoints, or GPU initialization."""
import copy
import json

import numpy as np
import pytest
import torch

from analysis.o3_reachability_objects_v3 import object_transitions
from analysis.o3_residual_reachability_v3 import analyze_reachability
from scripts import run_o3_residual_reachability_v3 as runner


def test_exact_replay_and_probability_boundary():
    logits = np.array([-1., 0., 1.], dtype=np.float32)
    probability = torch.sigmoid(torch.from_numpy(logits)).numpy()
    runner.check_replay(logits, probability, probability.copy(), "fixture")


def test_float32_sigmoid_rounding_is_not_silently_ignored():
    logits = np.array([1e-20], dtype=np.float32)
    probability = torch.sigmoid(torch.from_numpy(logits)).numpy()
    assert probability[0] == .5 and logits[0] > 0
    with pytest.raises(RuntimeError, match="masks disagree"):
        runner.check_replay(logits, probability, probability.copy(), "fixture")


@pytest.mark.parametrize("kind", ["ulp", "logit_dtype", "probability_dtype", "shape", "nan"])
def test_replay_rejects_bad_or_changed_endpoint(kind):
    logits = np.array([1.], dtype=np.float32)
    expected = torch.sigmoid(torch.from_numpy(logits)).numpy()
    actual = expected.copy()
    if kind == "ulp": actual[0] = np.nextafter(actual[0], np.float32(1.))
    elif kind == "logit_dtype": logits = logits.astype(np.float64)
    elif kind == "probability_dtype": actual = actual.astype(np.float64)
    elif kind == "shape": logits = logits.reshape(1, 1)
    elif kind == "nan": logits[0] = np.nan
    with pytest.raises(RuntimeError, match="not bit-exact"):
        runner.check_replay(logits, actual, expected, "fixture")


@pytest.mark.parametrize("kwargs", [{"kernel_size": 3}, {"stride": 2}, {"padding": 1},
                                    {"dilation": 2}, {"in_channels": 8}, {"out_channels": 2}])
def test_proof_rejects_nonoriginal_head(kwargs):
    arguments = {"in_channels": 16, "out_channels": 1, "kernel_size": 1}
    arguments.update(kwargs)
    with torch.random.fork_rng(devices=[]):
        head = torch.nn.Conv2d(**arguments)
    with pytest.raises(ValueError, match="affine pointwise"):
        runner.validate_head(head)


def test_affine_head_scalar_gain_logit_identity():
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(13)
        head = torch.nn.Conv2d(16, 1, 1).double()
        runner.validate_head(head)
        h0, h1 = torch.randn(2, 16, 7, 8, dtype=torch.float64), torch.randn(2, 16, 7, 8, dtype=torch.float64)
        gain = torch.rand(2, 1, 7, 8, dtype=torch.float64) * 2
        z0, z1 = head(h0), head(h1)
        assert torch.allclose(head(h0 + gain * (h1 - h0)), z0 + gain * (z1 - z0), rtol=1e-12, atol=1e-12)


def _fixture():
    target = np.zeros((12, 12), dtype=np.float32)
    target[3, 3:5] = 1
    z0 = np.full(target.shape, -2., dtype=np.float32)
    z1 = z0.copy()
    z0[3, 3], z1[3, 3] = -2., -.5  # FN reachable by doubling.
    z0[3, 4], z1[3, 4] = -4., -3.  # FN not reachable.
    z0[9, 9], z1[9, 9] = -1., .5   # FP removable by reverting.
    result = analyze_reachability(z0, z1, target)
    objects = object_transitions(target, z1 > 0, result["arrays"]["oracle_mask"])
    objects["targets"][0]["reachable_gt_positive_pixels"] = 1
    record = {"counts": result["counts"], "objects": objects}
    methods = {method: {metric: float(index + 1) for index, metric in enumerate(
        ("iou", "normalized_iou", "pd", "fa_per_million"))} for method in runner.METHODS}
    return record, methods, result, target


def test_aggregate_counts_not_means_and_metrics_equal_condition_mean():
    record, cell, result, _ = _fixture()
    second = copy.deepcopy(cell)
    second["v1"]["iou"] = 3.
    actual = runner.aggregate([record, copy.deepcopy(record)], [cell, second])
    assert actual["pixel_counts"]["v1_fn"] == 4
    assert actual["pixel_counts"]["repairable_fn"] == 2
    assert actual["pixel_counts"]["unrepairable_fn"] == 2
    assert actual["pixel_counts"]["v1_fp"] == actual["pixel_counts"]["repairable_fp"] == 2
    assert actual["v1_fn_unrepairable_fraction"] == .5
    assert actual["v1_fp_unrepairable_fraction"] == 0.
    assert actual["target_observations"] == 2
    assert actual["equal_condition_macro"]["v1"]["iou"] == 2.
    assert sum(actual["pixel_counts"]["oracle_gain_counts"].values()) == 288
    assert json.loads(json.dumps(actual, allow_nan=False)) == actual


def test_aggregate_empty_denominators_are_unknown_not_infinity():
    record, cell, _, _ = _fixture()
    for key in ("v1_fn", "v1_fp", "unrepairable_fn", "unrepairable_fp"):
        record["counts"][key] = 0
    actual = runner.aggregate([record], [cell])
    assert actual["v1_fn_unrepairable_fraction"] is None
    assert actual["v1_fp_unrepairable_fraction"] is None


@pytest.mark.parametrize("records,cells", [([], []), ([{}], []), ([], [{}])])
def test_empty_aggregate_rejected(records, cells):
    with pytest.raises(ValueError):
        runner.aggregate(records, cells)


def test_synthetic_oracle_counts_agree_with_native_evaluator():
    from scripts.run_p3_stage_b_screen_v1 import _evaluation_result, _endpoint_summary
    record, _, result, target = _fixture()
    prediction = result["arrays"]["oracle_mask"].astype(np.float32)
    fixed = _endpoint_summary(_evaluation_result(prediction[None, None], target[None, None], ["fixture"]))
    assert fixed["intersection_pixels"] == result["counts"]["oracle_tp"]
    assert fixed["false_positive_pixels"] == result["counts"]["oracle_fp"]
    assert fixed["false_negative_pixels"] == result["counts"]["oracle_fn"]
    assert fixed["detected_targets"] == record["objects"]["diagnostic"]["detected_targets"]
    assert fixed["false_alarm_pixels"] == record["objects"]["diagnostic"]["false_alarm_pixels"]


def test_array_loader_is_readonly_and_pickle_disabled(tmp_path):
    path = tmp_path / "array.npy"
    np.save(path, np.zeros((2, 3), dtype=np.float32), allow_pickle=False)
    result = runner.checked_array(path, (2, 3))
    assert not result.flags.writeable
    with pytest.raises(ValueError): runner.checked_array(path, (3, 2))


def test_json_does_not_overwrite(tmp_path):
    path = tmp_path / "once.json"
    runner.write_json(path, {"fixed": 1})
    with pytest.raises(FileExistsError): runner.write_json(path, {"fixed": 2})
    assert json.loads(path.read_text()) == {"fixed": 1}


def test_config_mismatch_stops_before_parent_or_payload_access(tmp_path, monkeypatch):
    path = tmp_path / "config.yaml"
    path.write_text("fixture: true\n")
    monkeypatch.setattr(runner, "CONFIG", path)
    with pytest.raises(ValueError, match="configuration hash"):
        runner.prepare()
