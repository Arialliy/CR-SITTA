from __future__ import annotations

import copy
import inspect

import pytest
import torch
from torch import nn

from analysis import ipma_engineering_checks_v1 as checks
from model.ipma_d0_adapter_v1 import IdentityMetaAdapter
from model.loss import SLSIoULoss


def engineering_config():
    return {
        "inner": {"steps": 1, "learning_rate": 0.05, "delta_l2_radius": 0.01},
        "engineering": {"proxy_gradient_norm_floor": 1e-8, "probe_logit_rms_floor": 1e-6,
            "post_logit_rms_floor": 1e-6, "minimum_informative_episodes": 12,
            "meta_gradient_norm_floor": 1e-8, "minimum_nonzero_meta_fit_episodes": 6,
            "require_all_finite": True, "require_all_identity_exact": True,
            "require_all_replays_exact": True, "require_all_fd_passed": True},
        "finite_difference": {"device": "cpu", "dtype": "float64", "epsilon": 1e-4,
            "absolute_tolerance": 1e-6, "relative_tolerance": 1e-3, "direction_seed": 20260907,
            "directions": ["normalized_meta_gradient", "normalized_seeded_random"],
            "teacher": "fixed_native_float32_cast_double"},
        "task_loss": {"name": "SLSIoULoss", "warm_epoch": 5, "epoch": 999, "with_shape": True, "input": "raw_logits"},
    }


def episode(seed=37):
    torch.set_num_threads(2)
    torch.manual_seed(seed)
    adapter = IdentityMetaAdapter().eval()
    head = nn.Conv2d(16, 1, 1).eval()
    for parameter in head.parameters():
        parameter.requires_grad_(False)
    observed = torch.randn(1, 16, 8, 9)
    probe = observed + 0.8 * torch.randn_like(observed)
    with torch.no_grad():
        teacher = head(observed).sigmoid().detach()
    target = torch.zeros_like(teacher)
    target[..., 2:4, 4:6] = 1
    return adapter, head, observed, probe, teacher, target


@pytest.fixture(scope="module")
def valid_receipts():
    adapter, head, observed, probe, teacher, target = episode()
    label, _ = checks.label_free_episode(adapter, head, observed, probe, teacher)
    fit, _ = checks.supervised_fit_diagnostic(adapter, head, observed, probe, teacher, target, engineering_config())
    assert label["label_free_signal_passed"]
    assert fit["meta_gradient_above_floor"]
    assert fit["finite_difference"]["all_directions_passed"]
    return label, fit


def receipt_lists(valid_receipts):
    label, fit = valid_receipts
    return [copy.deepcopy(label) for _ in range(16)], [copy.deepcopy(fit) for _ in range(8)]


def test_native_identity_replay_and_create_graph_numerics_are_bit_exact():
    adapter, head, observed, probe, teacher, _ = episode()
    before_adapter = checks.module_signature(adapter)
    before_head = checks.module_signature(head)
    receipt, tensors = checks.label_free_episode(adapter, head, observed, probe, teacher)
    for key in ("identity_exact", "repeat_exact", "create_graph_numeric_exact", "finite", "state_unchanged", "delta_within_radius"):
        assert receipt[key]
    assert torch.equal(tensors["source_logits"], tensors["identity_logits"])
    assert torch.count_nonzero(tensors["delta0"]) == 0
    assert all(not value.requires_grad and value.dtype == torch.float32 for value in tensors.values())
    assert checks.module_signature(adapter) == before_adapter
    assert checks.module_signature(head) == before_head
    assert not receipt["gt_accessed"]


@pytest.mark.parametrize("radius,active", [(1e-6, True), (1.0, False)])
def test_native_projection_branches_are_recorded_without_modifying_state(radius, active):
    adapter, head, observed, probe, teacher, _ = episode()
    receipt, _ = checks.label_free_episode(adapter, head, observed, probe, teacher, radius=radius)
    assert receipt["projection_active"] is active
    assert receipt["delta_within_radius"]


def test_identity_probe_is_not_claimed_informative():
    adapter, head, observed, _, teacher, _ = episode()
    receipt, _ = checks.label_free_episode(adapter, head, observed, observed.clone(), teacher)
    assert receipt["probe_observed_logit_rms"] == 0
    assert not receipt["label_free_signal_passed"]


def test_zero_host_feature_floor_is_reported_and_not_called_success():
    adapter, head, observed, _, _, _ = episode()
    observed = torch.zeros_like(observed)
    teacher = head(observed).sigmoid().detach()
    receipt, _ = checks.label_free_episode(adapter, head, observed, observed.clone(), teacher)
    assert receipt["host_feature_rms"] == 0
    assert receipt["host_feature_rms_floor_active"]
    assert receipt["residual_over_floored_host_rms"] == 0
    assert not receipt["label_free_signal_passed"]


def test_teacher_must_be_native_observed_source_not_arbitrary_probabilities():
    adapter, head, observed, probe, teacher, _ = episode()
    teacher = teacher * 0.9
    with pytest.raises(ValueError, match="observed-source"):
        checks.label_free_episode(adapter, head, observed, probe, teacher)


def test_native_path_rejects_float64_or_live_backbone_features():
    adapter, head, observed, probe, teacher, _ = episode()
    with pytest.raises(ValueError, match="float32"):
        checks.label_free_episode(adapter.double(), head.double(), observed.double(), probe.double(), teacher.double())
    adapter, head, observed, probe, teacher, _ = episode()
    with pytest.raises(ValueError, match="detached"):
        checks.label_free_episode(adapter, head, observed.requires_grad_(), probe, teacher)


def test_native_path_rejects_ambient_autocast():
    adapter, head, observed, probe, teacher, _ = episode()
    with torch.autocast("cpu", dtype=torch.bfloat16):
        with pytest.raises(ValueError, match="autocast"):
            checks.label_free_episode(adapter, head, observed, probe, teacher)


def test_fit_uses_raw_logits_exact_original_sls_and_true_meta_graph(valid_receipts):
    adapter, head, observed, probe, teacher, target = episode()
    receipt, tensors = checks.supervised_fit_diagnostic(adapter, head, observed, probe, teacher, target, engineering_config())
    expected = SLSIoULoss()(tensors["pre_logits"], target, 5, 999)
    assert receipt["pre_sls"] == float(expected)
    assert receipt["meta_gradient_norm"] > 1e-8
    assert set(receipt["parameter_meta_gradient_norms"]) == {"down.weight", "up.weight"}
    assert tensors["meta_gradient_vector"].numel() == sum(p.numel() for p in adapter.parameters())
    assert all(p.grad is None for p in adapter.parameters())
    assert not receipt["task_improvement_required_for_engineering_pass"]


@pytest.mark.parametrize("radius", [1e-5, 1.0])
def test_double_fd_passes_with_active_and_inactive_projection(radius):
    adapter, head, observed, probe, teacher, target = episode()
    native, _ = checks.label_free_episode(adapter, head, observed, probe, teacher, radius=radius)
    assert native["projection_active"] is (radius < 0.01)
    receipt = checks.finite_difference_replay(adapter, head, observed, probe, teacher, target, radius=radius)
    assert receipt["all_directions_passed"]
    assert receipt["dtype"] == "float64" and receipt["device"] == "cpu"
    assert not receipt["native_outputs_equivalence_claimed"]
    assert receipt["teacher_sha256"] == checks.tensor_sha256(teacher.double())
    assert {row["direction"] for row in receipt["directions"]} == {"normalized_meta_gradient", "normalized_seeded_random"}


def test_empty_target_is_kept_and_finite():
    adapter, head, observed, probe, teacher, target = episode()
    receipt, _ = checks.supervised_fit_diagnostic(adapter, head, observed, probe, teacher, torch.zeros_like(target), engineering_config())
    assert receipt["target_empty"]
    assert receipt["finite"]
    assert receipt["finite_difference"]["all_directions_passed"]


@pytest.mark.parametrize("section,key,value", [
    ("inner", "learning_rate", 0.1), ("inner", "delta_l2_radius", 0.1),
    ("finite_difference", "epsilon", 1e-3), ("finite_difference", "direction_seed", 7),
    ("engineering", "minimum_informative_episodes", 1), ("task_loss", "input", "sigmoid_logits"),
])
def test_real_fit_configuration_cannot_silently_change_frozen_numerics(section, key, value):
    cfg = engineering_config()
    cfg[section][key] = value
    with pytest.raises(ValueError, match="unregistered"):
        checks.validate_engineering_configuration(cfg)


def test_gate_accepts_technical_success_only_as_future_bounded_eligibility(valid_receipts):
    label, fit = receipt_lists(valid_receipts)
    result = checks.aggregate_engineering_gate(label, fit)
    assert result["engineering_passed"]
    assert result["bounded_meta_training_engineering_eligible"]
    assert result["maximum_future_outer_steps"] == 64
    assert result["requires_frozen_r5_contract"]
    for key in ("ipma_meta_training_allowed", "ipma_meta_training_executed", "execution_started",
                "full_source_training_allowed", "formal_test_allowed", "paper_result"):
        assert result[key] is False
    assert result["current_outer_optimizer_steps"] == 0


def test_gate_does_not_select_on_fit_sls_improvement(valid_receipts):
    label, fit = receipt_lists(valid_receipts)
    for row in fit:
        row["post_sls"] = row["pre_sls"] + 100
        row["post_minus_pre_sls"] = 100
    assert checks.aggregate_engineering_gate(label, fit)["engineering_passed"]


def test_gate_requires_twelve_cases_jointly_above_all_three_signal_floors(valid_receipts):
    label, fit = receipt_lists(valid_receipts)
    for row in label[:4]:
        row["post_observed_logit_rms"] = 1e-6
    assert checks.aggregate_engineering_gate(label, fit)["engineering_passed"]
    label[4]["inner_gradient_norm"] = 1e-8
    result = checks.aggregate_engineering_gate(label, fit)
    assert not result["engineering_passed"]
    assert result["informative_label_free_case_count"] == 11


def test_gate_requires_six_nonzero_meta_cases_but_every_fd_case_must_pass(valid_receipts):
    label, fit = receipt_lists(valid_receipts)
    for row in fit[:2]:
        row["meta_gradient_norm"] = 1e-8
    assert checks.aggregate_engineering_gate(label, fit)["engineering_passed"]
    fit[2]["meta_gradient_norm"] = 1e-8
    assert not checks.aggregate_engineering_gate(label, fit)["engineering_passed"]
    label, fit = receipt_lists(valid_receipts)
    fit[0]["finite_difference"]["directions"][0]["passed"] = False
    assert not checks.aggregate_engineering_gate(label, fit)["engineering_passed"]


@pytest.mark.parametrize("mutation", ["missing_label", "missing_fit", "missing_finite", "nonfinite", "identity", "replay", "update", "missing_fd_number", "fake_fd_flag", "missing_signal", "radius_drift"])
def test_engineering_gate_is_fail_closed_on_incomplete_or_inconsistent_receipts(valid_receipts, mutation):
    label, fit = receipt_lists(valid_receipts)
    if mutation == "missing_label":
        label.pop()
    elif mutation == "missing_fit":
        fit.pop()
    elif mutation == "missing_finite":
        del label[0]["finite"]
    elif mutation == "nonfinite":
        fit[0]["unexpected_nested"] = {"bad": float("nan")}
    elif mutation == "identity":
        label[0]["identity_exact"] = False
    elif mutation == "replay":
        label[0]["create_graph_numeric_exact"] = False
    elif mutation == "update":
        fit[0]["optimizer_steps_applied"] = 1
    elif mutation == "missing_fd_number":
        del fit[0]["finite_difference"]["directions"][0]["plus_task_loss"]
    elif mutation == "fake_fd_flag":
        fit[0]["finite_difference"]["directions"][0]["plus_task_loss"] += 1
    elif mutation == "missing_signal":
        del label[0]["probe_observed_probability_rms"]
    else:
        label[0]["radius"] = 0.1
    assert not checks.aggregate_engineering_gate(label, fit)["engineering_passed"]


def test_label_free_api_has_no_label_path_or_sample_selection_channel():
    fields = set(inspect.signature(checks.label_free_episode).parameters)
    assert fields == {"adapter", "head", "h_observed", "h_probe", "teacher", "lr", "radius"}
    source = inspect.getsource(checks)
    for forbidden in ("optimizer.step(", "Image.open", "torch.load", "test_split", "load_sample("):
        assert forbidden not in source
