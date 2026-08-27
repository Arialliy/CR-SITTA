from __future__ import annotations

from dataclasses import dataclass
import random

import numpy as np
import pytest
import torch
from torch import nn

from metrics.irstd_metrics import IRSTDEvaluationProtocol, UnifiedResearchEvaluator
from metrics.official_metric_adapter import OfficialMetricAdapter
from tta.episodic_runner import (
    AdaptationOutcome,
    EpisodeProtocolError,
    EpisodicRunner,
    NoUpdateMethod,
)
from tta.model_adapter import IRSTDModelAdapter
from tta.state_manager import EpisodicStateManager


class TinyNSFPNLike(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(3, 4, 1, bias=False)
        self.bn = nn.BatchNorm2d(4)
        self.dropout = nn.Dropout2d(p=0.9)
        self.head = nn.Conv2d(4, 1, 1)

    def forward(self, image: torch.Tensor, warm_flag: bool):
        features = self.dropout(self.bn(self.conv(image)))
        logits = self.head(features)
        return ([features] if warm_flag else []), logits


def _metadata(image_id: str) -> dict:
    return {
        "image_id": image_id,
        "original_size": [8, 8],
        "dataset": "toy",
        "corruption": "clean",
        "severity": 0,
        "seed": 42,
    }


def _source_runner(
    *, optimizer: torch.optim.Optimizer | None = None
) -> tuple[TinyNSFPNLike, IRSTDModelAdapter, EpisodicStateManager, EpisodicRunner]:
    torch.manual_seed(4)
    model = TinyNSFPNLike()
    adapter = IRSTDModelAdapter(model)
    adapter.set_source_eval_mode()
    state = EpisodicStateManager(model, optimizer)
    return model, adapter, state, EpisodicRunner(adapter, state)


def test_no_update_is_bit_exact_and_leaves_complete_source_state() -> None:
    model, _adapter, state, runner = _source_runner()
    image = torch.randn(1, 3, 8, 8)
    original = image.clone()

    result = runner.run_one_image(
        image=image, metadata=_metadata("a"), method=NoUpdateMethod()
    )

    assert torch.equal(result.logits_pre, result.logits_post)
    assert result.pre_post_bit_exact is True
    assert result.input_unchanged is True
    assert torch.equal(image, original)
    assert result.source_state_sha256 == result.state_after_adapt_sha256
    assert result.source_state_sha256 == result.state_after_post_sha256
    assert result.source_state_sha256 == result.reset_state_sha256
    assert result.outcome.optimizer_steps == 0
    assert result.outcome.diagnostics["reason"] == "forced_no_update"
    assert state.assert_source_state().full_sha256 == result.source_state_sha256
    assert not model.training


@dataclass
class SyntheticOneStepMethod:
    optimizer: torch.optim.Optimizer
    name: str = "synthetic_test_step"
    requires_grad: bool = True
    allowed_state_changes: frozenset[str] = frozenset(
        {"model", "optimizer", "runtime", "gradients"}
    )

    def prepare_episode(self, adapter: IRSTDModelAdapter) -> None:
        adapter.set_tent_mode(use_batch_stats=False)

    def adapt_one_image(self, *, adapter, image, logits_pre, metadata):
        del logits_pre, metadata
        self.optimizer.zero_grad(set_to_none=True)
        loss = adapter.forward_logits(image).square().mean()
        loss.backward()
        self.optimizer.step()
        return AdaptationOutcome(
            decision="adapted",
            optimizer_steps=1,
            diagnostics={"test_loss": float(loss.detach())},
        )


def test_a_b_and_b_a_are_order_independent_with_optimizer_reset() -> None:
    torch.manual_seed(7)
    model = TinyNSFPNLike()
    adapter = IRSTDModelAdapter(model)
    adapter.set_source_eval_mode()
    params, _names = adapter.collect_adaptable_params()
    optimizer = torch.optim.SGD(params, lr=0.05, momentum=0.9)
    state = EpisodicStateManager(model, optimizer)
    runner = EpisodicRunner(adapter, state)
    method = SyntheticOneStepMethod(optimizer)
    images = {
        "a": torch.randn(1, 3, 8, 8),
        "b": torch.randn(1, 3, 8, 8) + 0.5,
    }

    forward_order = {
        key: runner.run_one_image(
            image=images[key], metadata=_metadata(key), method=method
        )
        for key in ("a", "b")
    }
    reverse_order = {
        key: runner.run_one_image(
            image=images[key], metadata=_metadata(key), method=method
        )
        for key in ("b", "a")
    }

    for key in images:
        assert torch.equal(
            forward_order[key].logits_pre, reverse_order[key].logits_pre
        )
        assert torch.equal(
            forward_order[key].logits_post, reverse_order[key].logits_post
        )
        assert (
            forward_order[key].state_after_adapt_sha256
            == reverse_order[key].state_after_adapt_sha256
        )
        assert (
            forward_order[key].state_after_post_sha256
            == reverse_order[key].state_after_post_sha256
        )
        assert forward_order[key].reset_state_sha256 == state.source_fingerprint.full_sha256
        assert forward_order[key].state_after_adapt_sha256 != (
            state.source_fingerprint.full_sha256
        )
    assert optimizer.state_dict()["state"] == {}
    state.assert_source_state()


class FailingMethod:
    name = "failing"
    optimizer = None
    requires_grad = False
    allowed_state_changes: frozenset[str] = frozenset()

    def prepare_episode(self, adapter: IRSTDModelAdapter) -> None:
        adapter.set_source_eval_mode()

    def adapt_one_image(self, *, adapter, image, logits_pre, metadata):
        del image, logits_pre, metadata
        with torch.no_grad():
            next(adapter.model.parameters()).add_(1.0)
        raise RuntimeError("intentional failure")


def test_exception_still_restores_everything() -> None:
    _model, _adapter, state, runner = _source_runner()

    with pytest.raises(RuntimeError, match="intentional failure"):
        runner.run_one_image(
            image=torch.randn(1, 3, 8, 8),
            metadata=_metadata("failure"),
            method=FailingMethod(),
        )

    state.assert_source_state()


class SpyMethod(NoUpdateMethod):
    name = "spy"

    def __init__(self) -> None:
        self.seen_keys: set[str] | None = None

    def adapt_one_image(self, *, adapter, image, logits_pre, metadata):
        self.seen_keys = set(metadata)
        image.fill_(999.0)
        logits_pre.fill_(999.0)
        return AdaptationOutcome("no_update", 0, {"seen": True})


def test_label_firewall_and_private_method_copies() -> None:
    _model, _adapter, state, runner = _source_runner()
    image = torch.randn(1, 3, 8, 8)
    original = image.clone()
    spy = SpyMethod()

    result = runner.run_one_image(
        image=image, metadata=_metadata("safe"), method=spy
    )

    assert spy.seen_keys == set(_metadata("safe"))
    assert torch.equal(image, original)
    assert result.pre_post_bit_exact
    assert result.metadata["original_size"] == (8, 8)
    with pytest.raises(TypeError):
        result.metadata["original_size"][0] = 99
    state.assert_source_state()

    unsafe = {**_metadata("unsafe"), "mask": torch.ones(1, 1, 8, 8)}
    with pytest.raises(EpisodeProtocolError, match="outside the label-free allowlist"):
        runner.run_one_image(image=image, metadata=unsafe, method=spy)
    state.assert_source_state()

    smuggled = {**_metadata("smuggled"), "original_size": torch.ones(8, 8)}
    with pytest.raises(EpisodeProtocolError, match="original_size"):
        runner.run_one_image(image=image, metadata=smuggled, method=spy)
    state.assert_source_state()


class LyingNoUpdateMethod(NoUpdateMethod):
    name = "lying_no_update"

    def adapt_one_image(self, *, adapter, image, logits_pre, metadata):
        del image, logits_pre, metadata
        with torch.no_grad():
            next(adapter.model.parameters()).add_(0.25)
        return AdaptationOutcome("no_update", 0, {})


def test_no_update_cannot_hide_a_state_change() -> None:
    _model, _adapter, state, runner = _source_runner()

    with pytest.raises(EpisodeProtocolError, match="forbidden state components"):
        runner.run_one_image(
            image=torch.randn(1, 3, 8, 8),
            metadata=_metadata("liar"),
            method=LyingNoUpdateMethod(),
        )

    state.assert_source_state()


class DropoutTrapMethod(NoUpdateMethod):
    name = "dropout_trap"

    def prepare_episode(self, adapter: IRSTDModelAdapter) -> None:
        adapter.model.dropout.train()


def test_non_bn_training_mode_is_rejected_and_reset() -> None:
    model, _adapter, state, runner = _source_runner()

    with pytest.raises(EpisodeProtocolError, match="non-BatchNorm modules"):
        runner.run_one_image(
            image=torch.randn(1, 3, 8, 8),
            metadata=_metadata("dropout"),
            method=DropoutTrapMethod(),
        )

    assert not model.dropout.training
    state.assert_source_state()


def test_adaptation_enables_grad_inside_outer_no_grad() -> None:
    torch.manual_seed(71)
    model = TinyNSFPNLike()
    adapter = IRSTDModelAdapter(model)
    adapter.set_source_eval_mode()
    params, _names = adapter.collect_adaptable_params()
    optimizer = torch.optim.SGD(params, lr=0.05, momentum=0.9)
    state = EpisodicStateManager(model, optimizer)
    runner = EpisodicRunner(adapter, state)

    with torch.no_grad():
        result = runner.run_one_image(
            image=torch.randn(1, 3, 8, 8),
            metadata=_metadata("outer-no-grad"),
            method=SyntheticOneStepMethod(optimizer),
        )

    assert result.outcome.optimizer_steps == 1
    assert result.state_after_adapt_sha256 != result.source_state_sha256
    state.assert_source_state()


def test_inference_mode_fails_closed_and_restores_source() -> None:
    model = TinyNSFPNLike()
    adapter = IRSTDModelAdapter(model)
    adapter.set_source_eval_mode()
    params, _names = adapter.collect_adaptable_params()
    optimizer = torch.optim.SGD(params, lr=0.05)
    state = EpisodicStateManager(model, optimizer)
    runner = EpisodicRunner(adapter, state)
    image = torch.randn(1, 3, 8, 8)
    with torch.inference_mode():
        with pytest.raises(EpisodeProtocolError, match="torch.inference_mode"):
            runner.run_one_image(
                image=image,
                metadata=_metadata("inference-mode"),
                method=SyntheticOneStepMethod(optimizer),
            )
    state.assert_source_state()


def test_invalid_protocol_inputs_still_restore_preexisting_state_drift() -> None:
    model, _adapter, state, runner = _source_runner()
    image = torch.randn(1, 3, 8, 8)

    with torch.no_grad():
        model.head.bias.add_(3.0)
    unsafe = {**_metadata("unsafe"), "mask": torch.ones(1, 1, 8, 8)}
    with pytest.raises(EpisodeProtocolError, match="outside the label-free allowlist"):
        runner.run_one_image(image=image, metadata=unsafe, method=NoUpdateMethod())
    state.assert_source_state()

    with torch.no_grad():
        model.head.bias.sub_(2.0)
    with pytest.raises(TypeError, match="method must expose"):
        runner.run_one_image(image=image, metadata=_metadata("bad-method"), method=object())
    state.assert_source_state()


@pytest.mark.parametrize("source_drift", ["training", "trainable", "bn_stats"])
def test_runner_rejects_a_non_source_snapshot(source_drift: str) -> None:
    model = TinyNSFPNLike()
    adapter = IRSTDModelAdapter(model)
    adapter.set_source_eval_mode()
    if source_drift == "training":
        model.bn.train()
    elif source_drift == "trainable":
        model.head.weight.requires_grad_(True)
    else:
        model.bn.track_running_stats = False
    state = EpisodicStateManager(model)

    with pytest.raises(EpisodeProtocolError, match="Source"):
        EpisodicRunner(adapter, state)


class ConvGradientTrapMethod(NoUpdateMethod):
    name = "conv-gradient-trap"

    def prepare_episode(self, adapter: IRSTDModelAdapter) -> None:
        adapter.set_source_eval_mode()
        adapter.model.conv.weight.requires_grad_(True)


def test_non_bn_trainable_parameter_is_rejected_and_reset() -> None:
    _model, _adapter, state, runner = _source_runner()
    with pytest.raises(EpisodeProtocolError, match="only BatchNorm2d affine"):
        runner.run_one_image(
            image=torch.randn(1, 3, 8, 8),
            metadata=_metadata("conv-grad"),
            method=ConvGradientTrapMethod(),
        )
    state.assert_source_state()


def test_unmanaged_optimizer_is_rejected_and_reset() -> None:
    torch.manual_seed(73)
    model = TinyNSFPNLike()
    adapter = IRSTDModelAdapter(model)
    adapter.set_source_eval_mode()
    params, _names = adapter.collect_adaptable_params()
    managed = torch.optim.SGD(params, lr=0.05)
    unmanaged = torch.optim.Adam(params, lr=1e-3)
    state = EpisodicStateManager(model, managed)
    runner = EpisodicRunner(adapter, state)

    with pytest.raises(EpisodeProtocolError, match="must be identical"):
        runner.run_one_image(
            image=torch.randn(1, 3, 8, 8),
            metadata=_metadata("wrong-optimizer"),
            method=SyntheticOneStepMethod(unmanaged),
        )
    state.assert_source_state()


class GlobalRandomConsumer(NoUpdateMethod):
    name = "global-random-consumer"

    def adapt_one_image(self, *, adapter, image, logits_pre, metadata):
        del adapter, image, logits_pre, metadata
        random.random()
        np.random.random()
        torch.rand(3)
        return AdaptationOutcome("no_update", 0, {})


def test_python_numpy_and_torch_rng_do_not_leak_across_episodes() -> None:
    _model, _adapter, state, runner = _source_runner()
    image = torch.randn(1, 3, 8, 8)
    python_before = random.getstate()
    numpy_before = np.random.get_state()
    torch_before = torch.get_rng_state().clone()

    runner.run_one_image(
        image=image,
        metadata=_metadata("rng"),
        method=GlobalRandomConsumer(),
    )

    assert random.getstate() == python_before
    numpy_after = np.random.get_state()
    assert numpy_after[0] == numpy_before[0]
    assert np.array_equal(numpy_after[1], numpy_before[1])
    assert numpy_after[2:] == numpy_before[2:]
    assert torch.equal(torch.get_rng_state(), torch_before)
    state.assert_source_state()


class StatefulCounterMethod:
    name = "stateful-counter"
    optimizer = None
    requires_grad = False
    allowed_state_changes = frozenset({"extras"})

    def __init__(self) -> None:
        self.counter = 0

    def state_dict(self):
        return {"counter": self.counter}

    def load_state_dict(self, state):
        self.counter = state["counter"]

    def prepare_episode(self, adapter: IRSTDModelAdapter) -> None:
        adapter.set_source_eval_mode()

    def adapt_one_image(self, *, adapter, image, logits_pre, metadata):
        del adapter, image, logits_pre, metadata
        self.counter += 1
        return AdaptationOutcome("adapted", 0, {"counter": self.counter})


def test_registered_method_state_is_reset_between_images() -> None:
    torch.manual_seed(79)
    model = TinyNSFPNLike()
    adapter = IRSTDModelAdapter(model)
    adapter.set_source_eval_mode()
    method = StatefulCounterMethod()
    state = EpisodicStateManager(model, extra_stateful={"method": method})
    runner = EpisodicRunner(adapter, state)

    results = [
        runner.run_one_image(
            image=torch.randn(1, 3, 8, 8),
            metadata=_metadata(image_id),
            method=method,
        )
        for image_id in ("a", "b")
    ]

    assert [result.outcome.diagnostics["counter"] for result in results] == [1, 1]
    assert method.counter == 0
    assert all(
        result.state_after_adapt_sha256 != result.source_state_sha256
        for result in results
    )
    state.assert_source_state()


def test_no_update_probability_mask_and_aggregate_metrics_are_exact() -> None:
    _model, adapter, state, runner = _source_runner()
    protocol = IRSTDEvaluationProtocol(
        fixed_probability_threshold=0.5,
        froc_probability_thresholds=(0.0, 0.5, 1.0),
    )
    unified_pre = UnifiedResearchEvaluator(protocol)
    unified_post = UnifiedResearchEvaluator(protocol)
    official_pre = OfficialMetricAdapter(image_size=8)
    official_post = OfficialMetricAdapter(image_size=8)

    for index in range(2):
        image = torch.randn(1, 3, 8, 8) + index * 0.1
        target = ((torch.arange(64).reshape(1, 1, 8, 8) + index) % 7 == 0).float()
        result = runner.run_one_image(
            image=image,
            metadata=_metadata(f"metric-{index}"),
            method=NoUpdateMethod(),
        )
        probability_pre = adapter.logits_to_prob(result.logits_pre)
        probability_post = adapter.logits_to_prob(result.logits_post)
        assert torch.equal(probability_pre, probability_post)
        assert torch.equal(probability_pre > 0.5, probability_post > 0.5)
        unified_pre.update_logits(result.logits_pre, target)
        unified_post.update_logits(result.logits_post, target)
        official_pre.update(result.logits_pre, target)
        official_post.update(result.logits_post, target)

    assert unified_pre.compute().to_dict() == unified_post.compute().to_dict()
    assert official_pre.compute().to_dict() == official_post.compute().to_dict()
    assert not bool((torch.sigmoid(torch.tensor(0.0)) > 0.5).item())
    state.assert_source_state()
