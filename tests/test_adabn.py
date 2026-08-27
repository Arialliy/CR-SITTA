from __future__ import annotations

from copy import deepcopy

import torch
from torch import nn

from tta.adabn import AdaBNMethod
from tta.episodic_runner import EpisodicRunner
from tta.model_adapter import IRSTDModelAdapter
from tta.state_manager import EpisodicStateManager


class DeterministicAdaBNModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(1, 1, 1, bias=False)
        self.bn = nn.BatchNorm2d(1)
        self.dropout = nn.Dropout2d(p=0.95)
        self.head = nn.Conv2d(1, 1, 1, bias=False)
        with torch.no_grad():
            self.conv.weight.fill_(1.0)
            self.bn.weight.fill_(1.0)
            self.bn.bias.zero_()
            self.bn.running_mean.fill_(5.0)
            self.bn.running_var.fill_(4.0)
            self.head.weight.fill_(1.0)

    def forward(self, image: torch.Tensor, warm_flag: bool):
        features = self.dropout(self.bn(self.conv(image)))
        return ([features] if warm_flag else []), self.head(features)


def _metadata(image_id: str) -> dict:
    return {
        "image_id": image_id,
        "original_size": [4, 4],
        "dataset": "source-pilot",
        "corruption": "gaussian_noise",
        "severity": 3,
        "seed": 42,
    }


def _runner():
    model = DeterministicAdaBNModel()
    adapter = IRSTDModelAdapter(model)
    adapter.set_source_eval_mode()
    state = EpisodicStateManager(model, optimizer=None)
    runner = EpisodicRunner(adapter, state)
    return model, adapter, state, runner


def _assert_nested_exact(actual, expected) -> None:
    assert actual.keys() == expected.keys()
    for name in expected:
        assert torch.equal(actual[name], expected[name])


def test_adabn_changes_statistics_not_parameters_or_running_buffers() -> None:
    model, adapter, state, runner = _runner()
    source_state = deepcopy(model.state_dict())
    image = torch.arange(16, dtype=torch.float32).reshape(1, 1, 4, 4) / 4.0

    result = runner.run_one_image(
        image=image,
        metadata=_metadata("statistics"),
        method=AdaBNMethod(),
    )

    assert result.outcome.decision == "adapted"
    assert result.outcome.optimizer_steps == 0
    assert result.outcome.diagnostics["bn_protocol"] == (
        "single_image_spatial_batch_stats"
    )
    assert result.outcome.diagnostics["learnable_update"] is False
    assert not result.pre_post_bit_exact
    probability_change = (
        adapter.logits_to_prob(result.logits_post)
        - adapter.logits_to_prob(result.logits_pre)
    ).abs()
    assert float(probability_change.max()) > 0.0

    source = result.source_fingerprint
    after_prepare = result.state_after_prepare_fingerprint
    after_adapt = result.state_after_adapt_fingerprint
    after_post = result.state_after_post_fingerprint
    assert after_prepare.model_sha256 == source.model_sha256
    assert after_adapt.model_sha256 == source.model_sha256
    assert after_post.model_sha256 == source.model_sha256
    assert after_prepare.gradients_sha256 == source.gradients_sha256
    assert after_adapt.gradients_sha256 == source.gradients_sha256
    assert after_post.gradients_sha256 == source.gradients_sha256
    assert after_prepare.topology_sha256 == source.topology_sha256
    assert after_adapt.topology_sha256 == source.topology_sha256
    assert after_post.topology_sha256 == source.topology_sha256
    assert after_prepare.runtime_sha256 != source.runtime_sha256
    assert after_adapt.runtime_sha256 != source.runtime_sha256
    assert after_post.runtime_sha256 != source.runtime_sha256
    assert after_prepare == after_adapt == after_post
    assert result.reset_fingerprint == source
    _assert_nested_exact(model.state_dict(), source_state)
    assert state.assert_source_state() == source


def test_adabn_forward_does_not_accumulate_any_batchnorm_buffer() -> None:
    model = DeterministicAdaBNModel()
    adapter = IRSTDModelAdapter(model)
    adapter.set_source_eval_mode()
    buffers = {
        "running_mean": model.bn.running_mean.clone(),
        "running_var": model.bn.running_var.clone(),
        "num_batches_tracked": model.bn.num_batches_tracked.clone(),
    }

    adapter.set_adabn_mode()
    with torch.no_grad():
        adapter.forward_logits(torch.randn(1, 1, 4, 4))

    assert torch.equal(model.bn.running_mean, buffers["running_mean"])
    assert torch.equal(model.bn.running_var, buffers["running_var"])
    assert torch.equal(
        model.bn.num_batches_tracked,
        buffers["num_batches_tracked"],
    )
    assert not any(parameter.requires_grad for parameter in model.parameters())
    assert all(parameter.grad is None for parameter in model.parameters())
    assert not model.dropout.training


def test_adabn_a_b_and_b_a_are_bit_exact_and_reset_each_image() -> None:
    _model, _adapter, state, runner = _runner()
    method = AdaBNMethod()
    images = {
        "a": torch.linspace(-1.0, 1.0, 16).reshape(1, 1, 4, 4),
        "b": torch.linspace(2.0, 6.0, 16).reshape(1, 1, 4, 4),
    }

    forward = {
        key: runner.run_one_image(
            image=images[key], metadata=_metadata(key), method=method
        )
        for key in ("a", "b")
    }
    reverse = {
        key: runner.run_one_image(
            image=images[key], metadata=_metadata(key), method=method
        )
        for key in ("b", "a")
    }

    for key in images:
        assert torch.equal(forward[key].logits_pre, reverse[key].logits_pre)
        assert torch.equal(forward[key].logits_post, reverse[key].logits_post)
        assert (
            forward[key].state_after_post_fingerprint
            == reverse[key].state_after_post_fingerprint
        )
        assert forward[key].reset_fingerprint == state.source_fingerprint
        assert reverse[key].reset_fingerprint == state.source_fingerprint
    state.assert_source_state()
