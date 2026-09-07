"""Synthetic bridge tests: stub original O3, no checkpoint/data/GT/GPU."""
from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from tta import o3_endpoint_features_v1 as bridge
from tta.state_manager import EpisodicStateManager


class Host(nn.Module):
    def __init__(self):
        super().__init__()
        self.offset = nn.Parameter(torch.zeros(1, 16, 1, 1))
        self.output_0 = nn.Conv2d(16, 1, 1, bias=False)
        with torch.no_grad():
            self.output_0.weight.fill_(.05)

    def forward(self, image):
        features = image[:, :1].repeat(1, 16, 1, 1) + self.offset
        return self.output_0(features)


class Adapter:
    def __init__(self, model):
        self.model = model
        self.capture_error = False
        self.bad_head_replay = False
        self.skip_capture = False
        self.double_capture = False

    def set_source_eval_mode(self):
        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

    def forward_logits(self, image):
        active_capture = bool(self.model.output_0._forward_pre_hooks)
        if active_capture and self.skip_capture:
            return image[:, :1] * 0
        result = self.model(image)
        if active_capture and self.capture_error:
            raise RuntimeError("synthetic capture failure")
        if active_capture and self.double_capture:
            self.model(image)
        if active_capture and self.bad_head_replay:
            result = result + .25
        return result


@pytest.fixture
def setup(monkeypatch):
    old_threads = torch.get_num_threads()
    torch.set_num_threads(2)
    model = Host()
    adapter = Adapter(model)
    adapter.set_source_eval_mode()
    manager = EpisodicStateManager(model)
    image = torch.full((1, 3, 256, 256), .2)
    with torch.no_grad():
        teacher = torch.sigmoid(adapter.forward_logits(image))
    kwargs = dict(contract=SimpleNamespace(config_sha256=bridge.b4.FROZEN_CONFIG_SHA256,
                                          raw={"protocol_id": bridge.b4.PROTOCOL_ID}),
                  model=model, adapter=adapter, film=nn.Identity(), state_manager=manager,
                  image=image, student_image=image * .95, teacher=teacher,
                  uncertainty=torch.zeros_like(teacher))
    state = {"calls": 0, "accepted": True, "raise": False, "bad_post": False,
             "source_snapshot": manager.source_fingerprint.full_sha256}

    def original(**arguments):
        state["calls"] += 1
        assert arguments["candidate_id"] == "O3_P2"
        assert not model.output_0._forward_pre_hooks
        assert torch.is_grad_enabled()
        assert arguments["teacher"] is kwargs["teacher"]
        # Simulate old loss/safety forwards. The bridge must not capture them.
        with torch.no_grad():
            adapter.forward_logits(arguments["student_image"])
            adapter.forward_logits(arguments["image"])
            if state["accepted"]:
                model.offset.add_(.3)
        model.offset.requires_grad_(True)
        if state["raise"]:
            raise RuntimeError("synthetic old O3 failure")
        with torch.no_grad():
            post = torch.sigmoid(adapter.forward_logits(arguments["image"])).cpu()
        if state["bad_post"]:
            post = post + .01
        return post, torch.zeros(416), torch.zeros(416), {
            "candidate_id": "O3_P2", "finite": True, "accepted_update": state["accepted"],
            "method_label_accesses": 0, "original_marker": "preserved",
        }

    monkeypatch.setattr(bridge.b4, "_candidate_episode_from_source", original)
    yield kwargs, state
    assert not model.output_0._forward_pre_hooks
    assert manager.assert_source_state().full_sha256 == state["source_snapshot"]
    torch.set_num_threads(old_threads)


def test_accepted_endpoint_is_captured_before_exact_reset(setup):
    kwargs, state = setup
    output = bridge.capture_o3_endpoint(**kwargs)
    assert state["calls"] == 1
    assert torch.equal(output["features"], torch.full((1, 16, 256, 256), .5))
    assert torch.equal(output["source_probabilities"], kwargs["teacher"])
    assert not torch.equal(output["o3_probabilities"], kwargs["teacher"])
    for key in ("features", "source_probabilities", "o3_probabilities"):
        value = output[key]
        assert value.dtype == torch.float32 and value.device.type == "cpu"
        assert not value.requires_grad and value.grad_fn is None
    assert output["diagnostics"]["original_marker"] == "preserved"
    assert output["diagnostics"]["source_state_restored"] is True
    assert output["diagnostics"]["bridge_capture_forward_count"] == 1


def test_rejected_endpoint_remains_exact_source(setup):
    kwargs, state = setup
    state["accepted"] = False
    result = bridge.capture_o3_endpoint(**kwargs)
    assert torch.equal(result["source_probabilities"], result["o3_probabilities"])
    assert result["diagnostics"]["accepted_update"] is False


def test_consecutive_episodes_do_not_accumulate_or_alias(setup):
    kwargs, _ = setup
    first = bridge.capture_o3_endpoint(**kwargs)
    second = bridge.capture_o3_endpoint(**kwargs)
    assert torch.equal(first["features"], second["features"])
    first["features"].zero_()
    first["source_probabilities"].zero_()
    assert torch.count_nonzero(second["features"])
    assert torch.count_nonzero(kwargs["teacher"])


def test_old_o3_exception_restores_host(setup):
    kwargs, state = setup
    state["raise"] = True
    with pytest.raises(RuntimeError, match="synthetic old O3 failure"):
        bridge.capture_o3_endpoint(**kwargs)


@pytest.mark.parametrize("fault,match", [("capture_error", "synthetic capture failure"),
                                        ("bad_head_replay", "head replay"),
                                        ("skip_capture", "exactly one"),
                                        ("double_capture", "exactly one")])
def test_capture_errors_remove_hook_and_restore_host(setup, fault, match):
    kwargs, _ = setup
    setattr(kwargs["adapter"], fault, True)
    with pytest.raises(RuntimeError, match=match):
        bridge.capture_o3_endpoint(**kwargs)


def test_old_post_probability_mismatch_rejected(setup):
    kwargs, state = setup
    state["bad_post"] = True
    with pytest.raises(bridge.O3EndpointFeatureError, match="original O3 probabilities"):
        bridge.capture_o3_endpoint(**kwargs)


def test_teacher_mismatch_rejected_before_old_o3(setup):
    kwargs, state = setup
    kwargs["teacher"] = kwargs["teacher"] + .01
    with pytest.raises(bridge.O3EndpointFeatureError, match="bit-exact to teacher"):
        bridge.capture_o3_endpoint(**kwargs)
    assert state["calls"] == 0


@pytest.mark.parametrize("key,fault", [("image", "shape"), ("student_image", "dtype"),
                                      ("teacher", "grad"), ("uncertainty", "nan")])
def test_invalid_inputs_fail_without_old_o3(setup, key, fault):
    kwargs, state = setup
    value = kwargs[key]
    if fault == "shape":
        kwargs[key] = value[..., :-1]
    elif fault == "dtype":
        kwargs[key] = value.double()
    elif fault == "grad":
        kwargs[key] = value.clone().requires_grad_(True)
    else:
        kwargs[key] = torch.full_like(value, float("nan"))
    with pytest.raises(bridge.O3EndpointFeatureError, match="invalid detached"):
        bridge.capture_o3_endpoint(**kwargs)
    assert state["calls"] == 0


def test_inference_mode_rejected_but_no_grad_caller_is_supported(setup):
    kwargs, state = setup
    with torch.inference_mode(), pytest.raises(bridge.O3EndpointFeatureError, match="inference_mode"):
        bridge.capture_o3_endpoint(**kwargs)
    assert state["calls"] == 0
    with torch.no_grad():
        output = bridge.capture_o3_endpoint(**kwargs)
    assert output["diagnostics"]["source_state_restored"] is True


def test_wrong_contract_rejected_before_old_o3(setup):
    kwargs, state = setup
    kwargs["contract"].config_sha256 = "0" * 64
    with pytest.raises(bridge.O3EndpointFeatureError, match="original frozen B4"):
        bridge.capture_o3_endpoint(**kwargs)
    assert state["calls"] == 0


def test_primary_and_cleanup_failures_preserve_primary_on_python310(setup, monkeypatch):
    kwargs, _ = setup

    class LegacyError(RuntimeError):
        add_note = None  # Exercise the Python 3.10 branch on any Python version.

    original_error = LegacyError("original O3 failure")

    def failing_original(**arguments):
        with torch.no_grad():
            arguments["model"].offset.add_(.3)
        raise original_error

    original_reset = kwargs["state_manager"].reset_to_source
    reset_count = 0

    def reset_then_fail():
        nonlocal reset_count
        reset_count += 1
        original_reset()
        if reset_count == 2:
            raise RuntimeError("synthetic cleanup failure")

    monkeypatch.setattr(bridge.b4, "_candidate_episode_from_source", failing_original)
    monkeypatch.setattr(kwargs["state_manager"], "reset_to_source", reset_then_fail)
    with pytest.raises(LegacyError, match="original O3 failure") as caught:
        bridge.capture_o3_endpoint(**kwargs)
    assert caught.value is original_error
    assert any("synthetic cleanup failure" in note for note in caught.value.__notes__)


def test_cleanup_failure_without_primary_is_reported(setup, monkeypatch):
    kwargs, _ = setup
    original_reset = kwargs["state_manager"].reset_to_source
    reset_count = 0

    def reset_then_fail():
        nonlocal reset_count
        reset_count += 1
        original_reset()
        if reset_count == 2:
            raise RuntimeError("synthetic cleanup failure")

    monkeypatch.setattr(kwargs["state_manager"], "reset_to_source", reset_then_fail)
    with pytest.raises(bridge.O3EndpointFeatureError, match="failed exact Source cleanup") as caught:
        bridge.capture_o3_endpoint(**kwargs)
    assert isinstance(caught.value.__cause__, RuntimeError)
    assert "synthetic cleanup failure" in str(caught.value.__cause__)
