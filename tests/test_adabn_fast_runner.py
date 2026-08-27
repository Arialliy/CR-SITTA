from __future__ import annotations

import random

import numpy as np
import pytest
import torch
from torch import nn

import tta.adabn_fast_runner as fast_runner_module
from tta.adabn_fast_runner import (
    CHECK_KIND,
    AdaBNFastProtocolError,
    AdaBNFastRecoveryError,
    AdaBNFastRunner,
    full_audit_due,
)
from tta.model_adapter import IRSTDModelAdapter
from tta.state_manager import EpisodicStateManager


class FastAdaBNToy(nn.Module):
    def __init__(self, *, consume_rng: bool = False) -> None:
        super().__init__()
        self.conv = nn.Conv2d(1, 2, 1, bias=False)
        self.bn = nn.BatchNorm2d(2)
        self.dropout = nn.Dropout2d(p=0.95)
        self.head = nn.Conv2d(2, 1, 1)
        self.register_buffer("persistent_cache", torch.tensor([3.0]))
        self.register_buffer(
            "nonpersistent_cache", torch.tensor([4.0]), persistent=False
        )
        self.register_buffer("optional_cache", None, persistent=False)
        self.tamper: str | None = None
        self.tamper_done = False
        self.consume_rng = consume_rng
        with torch.no_grad():
            self.conv.weight.copy_(torch.tensor([[[[1.0]]], [[[0.5]]]]))
            self.bn.weight.fill_(1.0)
            self.bn.bias.zero_()
            self.bn.running_mean.copy_(torch.tensor([4.0, -3.0]))
            self.bn.running_var.copy_(torch.tensor([2.0, 5.0]))
            self.head.weight.fill_(0.75)
            self.head.bias.fill_(0.1)

    def _tamper_once(self) -> None:
        if self.tamper_done or self.tamper is None or not self.bn.training:
            return
        self.tamper_done = True
        if self.tamper == "parameter":
            with torch.no_grad():
                self.head.bias.add_(1.0)
        elif self.tamper == "persistent_buffer":
            self.persistent_cache.add_(1.0)
        elif self.tamper == "nonpersistent_buffer":
            self.nonpersistent_cache.add_(1.0)
        elif self.tamper == "none_buffer":
            self.optional_cache = torch.tensor([9.0])
        elif self.tamper == "gradient":
            self.head.bias.grad = torch.ones_like(self.head.bias)
        elif self.tamper == "requires_grad":
            self.head.bias.requires_grad_(True)
        elif self.tamper == "runtime":
            self.dropout.train()
        elif self.tamper == "module_topology":
            self.add_module("injected", nn.Identity())
        else:  # pragma: no cover - helper misuse
            raise AssertionError(self.tamper)

    def forward(self, image: torch.Tensor, warm_flag: bool):
        if self.consume_rng:
            random.random()
            np.random.random()
            torch.rand(2)
        features = self.dropout(self.bn(self.conv(image)))
        logits = self.head(features)
        self._tamper_once()
        return ([features] if warm_flag else []), logits


def _metadata(image_id: str) -> dict:
    return {
        "image_id": image_id,
        "original_size": [4, 4],
        "dataset": "toy",
        "corruption": "gaussian_noise",
        "severity": 3,
        "seed": 42,
    }


def _build(
    *, cadence: int | None = None, consume_rng: bool = False
) -> tuple[
    FastAdaBNToy,
    IRSTDModelAdapter,
    EpisodicStateManager,
    AdaBNFastRunner,
]:
    torch.manual_seed(17)
    model = FastAdaBNToy(consume_rng=consume_rng)
    adapter = IRSTDModelAdapter(model)
    adapter.set_source_eval_mode()
    state = EpisodicStateManager(model, optimizer=None)
    runner = AdaBNFastRunner(
        adapter,
        state,
        full_audit_cadence=cadence,
    )
    return model, adapter, state, runner


def _image(offset: float = 0.0) -> torch.Tensor:
    return torch.linspace(-1.0, 2.0, 16).reshape(1, 1, 4, 4) + offset


def test_fast_adabn_runs_source_pre_and_adabn_post_with_complete_exact_gates() -> None:
    model, _adapter, state, runner = _build()
    image = _image()
    original = image.clone()

    result = runner.run_one_image(image=image, metadata=_metadata("sample"))

    assert result.method == "adabn"
    assert result.check_kind == CHECK_KIND
    assert result.episode_number == 1
    assert not torch.equal(result.logits_pre, result.logits_post)
    assert not result.logits_pre.requires_grad
    assert not result.logits_post.requires_grad
    assert result.input_unchanged
    assert torch.equal(image, original)
    assert set(result.exact_checks) == {
        "episode_start",
        "source_forward",
        "prepare_adabn",
        "adabn_forward",
        "runtime_reset",
    }
    for check in result.exact_checks.values():
        assert check.check_kind == CHECK_KIND
        assert check.parameter_bytes_exact
        assert check.all_registered_buffer_bytes_exact
        assert check.buffer_none_and_persistence_topology_exact
        assert check.nonpersistent_buffer_tensors == 1
        assert check.none_buffers == 1
        assert check.batchnorm_running_mean_buffers == 1
        assert check.batchnorm_running_var_buffers == 1
        assert check.batchnorm_num_batches_tracked_buffers == 1
    assert all(result.checks.values())
    assert result.full_audit_performed is False
    assert result.full_audit_reason is None
    assert result.post_full_fingerprint is None
    assert result.reset_full_fingerprint is None
    assert result.post_full_state_differences is None
    assert not model.training
    assert not model.bn.training
    assert model.bn.track_running_stats
    assert state.assert_source_state().full_sha256 == result.source_state_sha256


@pytest.mark.parametrize(
    ("episode_number", "cadence", "force", "expected"),
    [
        (1, None, False, (False, None)),
        (1, 2, False, (False, None)),
        (2, 2, False, (True, "cadence")),
        (3, 2, False, (False, None)),
        (3, None, True, (True, "forced")),
        (4, 2, True, (True, "forced")),
    ],
)
def test_full_audit_cadence_formula(
    episode_number: int,
    cadence: int | None,
    force: bool,
    expected: tuple[bool, str | None],
) -> None:
    assert full_audit_due(episode_number, cadence, force=force) == expected


@pytest.mark.parametrize("cadence", [0, -1, True, 1.5])
def test_invalid_full_audit_cadence_is_rejected(cadence) -> None:
    with pytest.raises((TypeError, ValueError)):
        full_audit_due(1, cadence)


def test_full_audit_fields_exist_only_when_cadence_or_force_computes_them() -> None:
    _model, _adapter, state, runner = _build(cadence=2)

    first = runner.run_one_image(image=_image(), metadata=_metadata("one"))
    second = runner.run_one_image(image=_image(1.0), metadata=_metadata("two"))
    third = runner.run_one_image(
        image=_image(2.0),
        metadata=_metadata("three"),
        force_full_audit=True,
    )

    assert not first.full_audit_performed
    assert first.post_full_fingerprint is None
    assert first.reset_full_fingerprint is None
    for result, reason in ((second, "cadence"), (third, "forced")):
        assert result.full_audit_performed
        assert result.full_audit_reason == reason
        assert result.post_full_fingerprint is not None
        assert result.post_full_state_differences == ("runtime",)
        assert result.reset_full_fingerprint == state.source_fingerprint


def test_a_b_and_b_a_are_bit_exact_with_per_image_runtime_reset() -> None:
    _model, _adapter, state, runner = _build()
    images = {"a": _image(), "b": _image(3.0)}
    forward = {
        key: runner.run_one_image(image=images[key], metadata=_metadata(key))
        for key in ("a", "b")
    }
    reverse = {
        key: runner.run_one_image(image=images[key], metadata=_metadata(key))
        for key in ("b", "a")
    }

    for key in images:
        assert torch.equal(forward[key].logits_pre, reverse[key].logits_pre)
        assert torch.equal(forward[key].logits_post, reverse[key].logits_post)
        assert forward[key].source_state_sha256 == reverse[key].source_state_sha256
    assert runner.completed_episodes == 4
    state.assert_source_state()


@pytest.mark.parametrize(
    "tamper",
    [
        "parameter",
        "persistent_buffer",
        "nonpersistent_buffer",
        "none_buffer",
        "gradient",
        "requires_grad",
        "runtime",
    ],
)
def test_value_gradient_and_runtime_tampering_full_resets_then_aborts(
    tamper: str,
) -> None:
    model, _adapter, state, runner = _build()
    model.tamper = tamper

    with pytest.raises(AdaBNFastProtocolError):
        runner.run_one_image(image=_image(), metadata=_metadata(tamper))

    assert runner.aborted
    assert runner.completed_episodes == 0
    state.assert_source_state()
    with pytest.raises(AdaBNFastProtocolError, match="previous failed episode"):
        runner.run_one_image(image=_image(), metadata=_metadata("again"))


def test_module_topology_tampering_attempts_full_reset_and_fails_closed() -> None:
    model, _adapter, _state, runner = _build()
    model.tamper = "module_topology"

    with pytest.raises(AdaBNFastRecoveryError) as captured:
        runner.run_one_image(image=_image(), metadata=_metadata("topology"))

    assert "module topology" in str(captured.value.original_error)
    assert "topology changed" in str(captured.value.recovery_error)
    assert runner.aborted
    with pytest.raises(AdaBNFastProtocolError, match="previous failed episode"):
        runner.run_one_image(image=_image(), metadata=_metadata("again"))


def test_unsafe_metadata_full_resets_and_aborts_without_exposing_a_label() -> None:
    _model, _adapter, state, runner = _build()
    unsafe = {**_metadata("unsafe"), "mask": torch.ones(1, 1, 4, 4)}

    with pytest.raises(AdaBNFastProtocolError, match="outside the label-free"):
        runner.run_one_image(image=_image(), metadata=unsafe)

    assert runner.aborted
    state.assert_source_state()


def test_python_numpy_and_torch_rng_are_restored() -> None:
    _model, _adapter, state, runner = _build(consume_rng=True)
    python_before = random.getstate()
    numpy_before = np.random.get_state()
    torch_before = torch.get_rng_state().clone()

    runner.run_one_image(image=_image(), metadata=_metadata("rng"))

    assert random.getstate() == python_before
    numpy_after = np.random.get_state()
    assert numpy_after[0] == numpy_before[0]
    assert np.array_equal(numpy_after[1], numpy_before[1])
    assert numpy_after[2:] == numpy_before[2:]
    assert torch.equal(torch.get_rng_state(), torch_before)
    state.assert_source_state()


def test_optimizer_is_forbidden_at_construction() -> None:
    model = FastAdaBNToy()
    adapter = IRSTDModelAdapter(model)
    adapter.set_source_eval_mode()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    state = EpisodicStateManager(model, optimizer=optimizer)

    with pytest.raises(AdaBNFastProtocolError, match="forbids an optimizer"):
        AdaBNFastRunner(adapter, state)


def test_noncanonical_source_runtime_is_forbidden_at_construction() -> None:
    model = FastAdaBNToy()
    adapter = IRSTDModelAdapter(model)
    adapter.set_source_eval_mode()
    model.dropout.train()
    state = EpisodicStateManager(model, optimizer=None)

    with pytest.raises(AdaBNFastProtocolError, match="every module in eval mode"):
        AdaBNFastRunner(adapter, state)


def test_method_specific_extra_state_is_forbidden_at_construction() -> None:
    class ExtraState:
        def state_dict(self):
            return {"value": 1}

        def load_state_dict(self, _state) -> None:
            return None

    model = FastAdaBNToy()
    adapter = IRSTDModelAdapter(model)
    adapter.set_source_eval_mode()
    state = EpisodicStateManager(
        model,
        optimizer=None,
        extra_stateful={"extra": ExtraState()},
    )

    with pytest.raises(AdaBNFastProtocolError, match="extra state"):
        AdaBNFastRunner(adapter, state)


def test_duplicate_module_alias_is_detected_and_runner_aborts() -> None:
    model, _adapter, _state, runner = _build()
    model.alias_of_bn = model.bn

    with pytest.raises((AdaBNFastProtocolError, AdaBNFastRecoveryError)):
        runner.run_one_image(image=_image(), metadata=_metadata("alias"))

    assert runner.aborted


def test_rng_restore_failure_aborts_runner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _model, _adapter, state, runner = _build()

    def fail_restore(_state) -> None:
        raise RuntimeError("rng restore failed")

    monkeypatch.setattr(fast_runner_module, "_restore_rng_state", fail_restore)
    with pytest.raises(RuntimeError, match="rng restore failed"):
        runner.run_one_image(image=_image(), metadata=_metadata("rng-failure"))

    assert runner.aborted
    state.assert_source_state()
