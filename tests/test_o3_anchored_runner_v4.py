"""Synthetic integration checks; no real dataset or model checkpoint access."""
import copy
import json

import numpy as np
import pytest
import torch

from scripts import run_o3_anchored_residual_v4 as runner
from scripts.run_o3_multiscale_train8_v1 import supervised_loss, training_order
from model.o3_anchored_residual_v4 import O3AnchoredResidualV4
from model.o3_multiscale_residual_v1 import O3MultiScaleResidual


def test_box_bound_matches_independent_channel_extrema():
    features = np.ones((16, 2, 2), dtype=np.float32)
    weights = np.ones(16, dtype=np.float32)
    z0 = np.array([[-1., -.4], [1., .4]], dtype=np.float32)
    z1 = np.array([[-.8, -.2], [.8, .6]], dtype=np.float32)
    targets = np.array([[1, 1], [0, 0]])
    out = runner.box_capacity(features, z0, z1, targets, weights)
    assert out["maximum_logit_increment"] == .8
    assert out["v1_fn"] == out["v1_fp"] == 2
    assert out["fn_upper_le_zero"] == out["fp_lower_gt_zero"] == 1
    # A signed head still has the same l1 box capacity.
    weights[::2] *= -1
    assert runner.box_capacity(features, z0, z1, targets, weights) == out


def test_box_strict_zero_boundary_and_rms_floor():
    h = np.zeros((16, 1, 2), dtype=np.float64)
    margin = .05 * 1e-6 * 16
    z0 = np.array([[-margin, margin]])
    z1 = np.array([[-1., 1.]])
    out = runner.box_capacity(h, z0, z1, np.array([[1, 0]]), np.ones(16))
    assert out["fn_upper_le_zero"] == 1
    assert out["fp_lower_gt_zero"] == 0


@pytest.mark.parametrize("kind", ["features_shape", "z_shape", "weights", "nan", "gt"])
def test_bad_box_inputs(kind):
    h, z0, z1, y, w = np.ones((16, 2, 2)), np.zeros((2, 2)), np.zeros((2, 2)), np.zeros((2, 2)), np.ones(16)
    if kind == "features_shape": h = h[:8]
    elif kind == "z_shape": z1 = z1[:1]
    elif kind == "weights": w = w[:8]
    elif kind == "nan": h[0, 0, 0] = np.nan
    elif kind == "gt": y[0, 0] = .5
    with pytest.raises(ValueError): runner.box_capacity(h, z0, z1, y, w)


def test_empty_gt_has_no_fabricated_maximum():
    result = runner.box_capacity(np.ones((16, 2, 2)), np.ones((2, 2)), np.ones((2, 2)), np.zeros((2, 2)), np.ones(16))
    assert result["maximum_upper_logit_on_gt"] is None
    json.dumps(result, allow_nan=False)


def test_schedule_matches_parent_not_a_new_random_stream():
    expected = training_order(104)
    record = {"sample_order": "condition_major_then_image_id", "indices": expected}
    assert runner.checked_schedule(record) == expected
    changed = copy.deepcopy(record)
    changed["indices"][0][0] = (changed["indices"][0][0] + 1) % 104
    with pytest.raises(ValueError): runner.checked_schedule(changed)


@pytest.mark.parametrize("record", [None, {}, {"sample_order": "wrong", "indices": []}])
def test_bad_schedule(record):
    with pytest.raises(ValueError): runner.checked_schedule(record)


def test_two_arm_source_fit_starts_identical_preserves_anchor_and_head():
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(42)
        control = O3MultiScaleResidual().double()
        with torch.no_grad(): control.project.weight.normal_(std=.2)
        candidate = O3AnchoredResidualV4(control.state_dict())
        head = torch.nn.Conv2d(16, 1, 1).double().requires_grad_(False)
        h = torch.randn(4, 16, 7, 8, dtype=torch.float64)
        gt = (torch.rand(4, 1, 7, 8, dtype=torch.float64) > .94).double()
        before_head = copy.deepcopy(head.state_dict())
        before_anchor = copy.deepcopy(candidate.anchor.state_dict())
        assert torch.equal(control(h), candidate(h))
        assert torch.equal(supervised_loss(head(control(h)), gt), supervised_loss(head(candidate(h)), gt))
        for branch, parameters in ((control, control.parameters()), (candidate, candidate.learnable_parameters())):
            optimizer = torch.optim.Adam(parameters, lr=.001)
            assert not optimizer.state
            for _ in range(2):
                optimizer.zero_grad(set_to_none=True)
                loss = supervised_loss(head(branch(h)), gt)
                loss.backward()
                optimizer.step()
            assert all(float(state["step"]) == 2 for state in optimizer.state.values())
        assert all(torch.equal(value, head.state_dict()[key]) for key, value in before_head.items())
        assert all(torch.equal(value, candidate.anchor.state_dict()[key]) for key, value in before_anchor.items())
        assert all(parameter.grad is None for parameter in candidate.anchor.parameters())
        assert all(parameter.grad is None for parameter in head.parameters())
        assert not torch.equal(candidate(h), control(h))


@pytest.mark.parametrize("arm", ["control", "candidate"])
def test_trained_predictions_not_mislabeled_as_gt_oracle(arm):
    mask = np.zeros((8, 8), dtype=bool)
    mask[3, 3] = True
    result = runner.trained_transitions(mask, mask, mask, arm)
    assert result["comparison"] == f"v1→{arm}"
    assert result["interpretation"]["oracle_mask"] is False
    assert result["interpretation"]["inference_uses_gt"] is False
    assert result["transition_counts"]["TP→TP"] == 1
    assert result["protocol"]["distance_boundary"] == "strict_less_than"


def test_unknown_trained_arm_rejected():
    mask = np.zeros((8, 8), dtype=bool)
    with pytest.raises(ValueError): runner.trained_transitions(mask, mask, mask, "oracle")


def test_config_mismatch_stops_before_payload_loading(tmp_path, monkeypatch):
    path = tmp_path / "config.yaml"
    path.write_text("fixture: true\n")
    monkeypatch.setattr(runner, "CONFIG", path)
    with pytest.raises(ValueError, match="configuration hash"):
        runner.prepare()
