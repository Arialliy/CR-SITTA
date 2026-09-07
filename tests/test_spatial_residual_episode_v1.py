"""CPU synthetic episodes; only frozen YAML metadata is read, never real data."""

from __future__ import annotations

import copy
from dataclasses import replace
import hashlib
import inspect

import pytest
import torch
from torch import nn
import yaml

from scripts import run_p3_stage_b4_full_pilot64_v1 as b4
from scripts import run_p3_stage_b_screen_v1 as b3
from tta.adapters.decoder_spatial_residual_v1 import DecoderSpatialResidual
from tta import spatial_residual_episode_v1 as episode


@pytest.fixture(autouse=True)
def cpu_execution():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(53)
        yield
    torch.set_num_threads(previous)


@pytest.fixture
def contract():
    path = b4.REPOSITORY / "configs/p3_stage_b4_full_pilot64_proposal_gate_v1.yaml"
    payload = path.read_bytes()
    return b4.FullPilotContract(
        repository=b4.REPOSITORY,
        config_path=path,
        config_sha256=hashlib.sha256(payload).hexdigest(),
        raw=yaml.safe_load(payload),
    )


@pytest.fixture
def inputs(contract):
    observed = torch.zeros(1, 16, 256, 256)
    observed[:, 0] = -6.0
    observed[:, 0, 100:104, 100:104] = 6.0
    head = nn.Conv2d(16, 1, 1, bias=False).eval().requires_grad_(False)
    head.weight.zero_()
    head.weight[0, 0] = 1.0
    logits = head(observed)
    return dict(
        contract=contract,
        module=DecoderSpatialResidual(),
        head=head,
        observed_features=observed,
        student_features=observed * 0.95,
        source_logits=logits,
        teacher=logits.sigmoid(),
        uncertainty=torch.zeros_like(logits),
    )


def assert_reset(module):
    assert torch.count_nonzero(module.kernel) == 0
    assert module.kernel.grad is None
    assert module.kernel._backward_hooks is None or not module.kernel._backward_hooks


def test_real_o3_accepted_episode_preserves_loss_and_gradient(inputs, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("real payload or outer target access is forbidden")

    monkeypatch.setattr(b3, "_load_outer_targets", forbidden)
    monkeypatch.setattr(torch, "load", forbidden)
    region = b3._build_region_weights(
        inputs["contract"], inputs["teacher"], inputs["uncertainty"]
    )
    loss, terms = b3._objective_loss(
        inputs["contract"], "O3",
        inputs["head"](inputs["module"](inputs["student_features"])),
        inputs["teacher"], region,
    )
    gradient, = torch.autograd.grad(loss, inputs["module"].kernel)
    state = {name: value.clone() for name, value in inputs["head"].state_dict().items()}
    before_rng = torch.random.get_rng_state()
    result = episode.run_episode(**inputs)
    diagnostics = result["diagnostics"]
    assert diagnostics["accepted_update"]
    assert diagnostics["finite"]
    assert diagnostics["proposal_loss_before"] == loss.item()
    assert diagnostics["loss_terms_before"] == terms
    assert torch.equal(result["proxy_gradient"], gradient.flatten().double())
    assert diagnostics["proposal_loss_after"] < diagnostics["proposal_loss_before"]
    assert diagnostics["absolute_radius"] == 0.25
    assert diagnostics["relative_radius"] is None
    assert result["endpoint_kernel"].norm().item() <= 0.25
    assert torch.count_nonzero(result["endpoint_kernel"]) > 0
    assert not torch.equal(result["post_probabilities"], result["source_probabilities"])
    assert torch.equal(result["source_probabilities"], inputs["teacher"])
    for name in ("proxy_gradient", "direction"):
        assert result[name].shape == (144,)
        assert result[name].dtype == torch.float64
        assert result[name].device.type == "cpu"
        assert not result[name].requires_grad
    for name in ("post_probabilities", "source_probabilities", "endpoint_kernel"):
        assert result[name].dtype == torch.float32
        assert result[name].device.type == "cpu"
        assert not result[name].requires_grad
    for name in ("episode_reset_exact", "head_state_unchanged", "cached_inputs_unchanged"):
        assert diagnostics[name]
    assert diagnostics["method_label_accesses"] == 0
    assert diagnostics["optimizer_objects_created"] == 0
    assert not diagnostics["sfs_backward_called"]
    assert not diagnostics["cuda_nondeterministic_backward_override"]
    assert torch.equal(before_rng, torch.random.get_rng_state())
    assert all(torch.equal(state[name], value) for name, value in inputs["head"].state_dict().items())
    assert_reset(inputs["module"])


def test_rejected_proposal_restores_exact_source_and_no_grad(inputs, monkeypatch):
    calls = []

    def reject(observed, proposal):
        calls.append(observed)
        return False, "synthetic_reject"

    monkeypatch.setattr(episode, "_safety_decision", reject)
    result = episode.run_episode(**inputs)
    assert calls
    assert not result["diagnostics"]["accepted_update"]
    assert result["diagnostics"]["source_post_probability_bit_exact"]
    assert torch.equal(result["source_probabilities"], result["post_probabilities"])
    assert torch.count_nonzero(result["endpoint_kernel"]) == 0
    assert_reset(inputs["module"])


def test_exception_after_candidate_mutation_still_clears_state(inputs, monkeypatch):
    def broken_proposal(**kwargs):
        with torch.no_grad():
            inputs["module"].kernel.fill_(0.1)
        inputs["module"].kernel.grad = torch.ones_like(inputs["module"].kernel)
        raise RuntimeError("synthetic proposal failure")

    monkeypatch.setattr(episode, "propose_and_backtrack", broken_proposal)
    with pytest.raises(RuntimeError, match="synthetic proposal failure"):
        episode.run_episode(**inputs)
    assert_reset(inputs["module"])


def test_safety_exception_during_real_backtracking_restores_state(inputs, monkeypatch):
    def fail(*args):
        assert torch.count_nonzero(inputs["module"].kernel) > 0
        raise RuntimeError("synthetic safety failure")

    monkeypatch.setattr(episode, "_safety_decision", fail)
    with pytest.raises(RuntimeError, match="synthetic safety failure"):
        episode.run_episode(**inputs)
    assert_reset(inputs["module"])


def test_head_snapshot_exception_still_resets_stale_module(inputs, monkeypatch):
    with torch.no_grad():
        inputs["module"].kernel.fill_(0.1)

    def fail(*args, **kwargs):
        raise RuntimeError("synthetic head snapshot failure")

    monkeypatch.setattr(inputs["head"], "state_dict", fail)
    with pytest.raises(RuntimeError, match="synthetic head snapshot failure"):
        episode.run_episode(**inputs)
    assert_reset(inputs["module"])


def test_nonfinite_head_is_rejected_without_falsely_reporting_mutation(inputs):
    inputs["head"].weight.flatten()[0] = float("nan")
    with pytest.raises(episode.SpatialEpisodeError, match="output head must be"):
        episode.run_episode(**inputs)
    assert_reset(inputs["module"])


def test_a_b_a_episodes_have_no_cross_image_state(inputs):
    a = episode.run_episode(**inputs)
    b_inputs = dict(inputs, student_features=inputs["observed_features"] * 0.90)
    b = episode.run_episode(**b_inputs)
    again = episode.run_episode(**inputs)
    assert not torch.equal(a["proxy_gradient"], b["proxy_gradient"])
    for name in ("post_probabilities", "proxy_gradient", "direction", "endpoint_kernel"):
        assert torch.equal(a[name], again[name])
    assert a["diagnostics"] == again["diagnostics"]
    assert_reset(inputs["module"])


@pytest.mark.parametrize("fault", ["weight", "grad"])
def test_stale_episode_state_is_rejected_and_cleared(inputs, fault):
    if fault == "weight":
        with torch.no_grad():
            inputs["module"].kernel.fill_(0.1)
    else:
        inputs["module"].kernel.grad = torch.ones_like(inputs["module"].kernel)
    with pytest.raises(episode.SpatialEpisodeError, match="fresh zero kernel"):
        episode.run_episode(**inputs)
    assert_reset(inputs["module"])


@pytest.mark.parametrize("name", ["observed_features", "student_features", "teacher", "uncertainty", "source_logits"])
@pytest.mark.parametrize("fault", ["dtype", "shape", "nan", "grad"])
def test_invalid_inputs_are_rejected_without_state_leak(inputs, name, fault):
    if fault == "dtype":
        inputs[name] = inputs[name].double()
    elif fault == "shape":
        inputs[name] = inputs[name][..., :255]
    elif fault == "nan":
        inputs[name].flatten()[0] = float("nan")
    else:
        inputs[name].requires_grad_(True)
    with pytest.raises(episode.SpatialEpisodeError, match=name):
        episode.run_episode(**inputs)
    assert_reset(inputs["module"])


@pytest.mark.parametrize("fault", ["teacher_range", "teacher_identity", "logit_identity", "negative_uncertainty", "radius", "ratio", "head_training", "head_grad"])
def test_frozen_semantic_guards(inputs, fault):
    if fault == "teacher_range":
        inputs["teacher"].flatten()[0] = 1.1
    elif fault == "teacher_identity":
        inputs["teacher"].flatten()[0] += 0.001
    elif fault == "logit_identity":
        inputs["source_logits"].flatten()[0] += 0.001
    elif fault == "negative_uncertainty":
        inputs["uncertainty"].flatten()[0] = -0.1
    elif fault == "radius":
        inputs["absolute_radius"] = 0.5
    elif fault == "ratio":
        inputs["module"].max_residual_ratio = 0.1
    elif fault == "head_training":
        inputs["head"].train()
    else:
        inputs["head"].weight.requires_grad_(True)
    with pytest.raises(episode.SpatialEpisodeError):
        episode.run_episode(**inputs)
    assert_reset(inputs["module"])


def test_contract_values_cannot_be_overridden_after_hash_validation(inputs):
    raw = copy.deepcopy(inputs["contract"].raw)
    raw["objectives"]["O3"]["lambda_foreground"] = 2.0
    inputs["contract"] = replace(inputs["contract"], raw=raw)
    with pytest.raises(episode.SpatialEpisodeError, match="contract values differ"):
        episode.run_episode(**inputs)
    assert_reset(inputs["module"])


def test_all_foreground_has_no_reliable_background_and_fails_closed(inputs):
    inputs["observed_features"][:, 0] = 6.0
    inputs["source_logits"] = inputs["head"](inputs["observed_features"])
    inputs["teacher"] = inputs["source_logits"].sigmoid()
    with pytest.raises(episode.SpatialEpisodeError, match="background region is empty"):
        episode.run_episode(**inputs)
    assert_reset(inputs["module"])


@pytest.mark.parametrize("failure", [None, "background", "fraction", "components", "all"])
def test_original_three_safety_comparisons_and_boundary(contract, failure):
    proposal = contract.raw["proposal"]
    limits = proposal["per_attempt_label_free_safety"]
    observed = dict(
        source_reliable_background_mass=0.0,
        post_reliable_background_mass=limits["reliable_background_mass_delta_maximum"] + proposal["epsilon"],
        source_foreground_fraction=0.0,
        post_foreground_fraction=limits["predicted_positive_fraction_delta_maximum"] + proposal["epsilon"],
        post_connected_component_count=4,
        connected_component_count_maximum=4,
    )
    if failure in ("background", "all"):
        observed["post_reliable_background_mass"] += 1e-8
    if failure in ("fraction", "all"):
        observed["post_foreground_fraction"] += 1e-8
    if failure in ("components", "all"):
        observed["post_connected_component_count"] += 1
    accepted, reason = episode._safety_decision(observed, proposal)
    assert accepted is (failure is None)
    if failure == "all":
        assert reason == "reliable_background_mass+predicted_positive_fraction+connected_component_count"


def test_method_api_has_no_targets_ids_dataset_or_output_paths():
    assert set(inspect.signature(episode.run_episode).parameters) == {
        "contract", "module", "head", "observed_features", "student_features",
        "teacher", "uncertainty", "source_logits", "absolute_radius",
    }
