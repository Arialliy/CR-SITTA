from __future__ import annotations

from copy import deepcopy
from typing import Any

import pytest
import torch
from torch import nn

from tta.state_manager import (
    EpisodicStateManager,
    SourceSnapshotError,
    SourceStateMismatchError,
    StatefulHooks,
)


class TinyEpisodicModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(3, 4, kernel_size=1)
        self.bn = nn.BatchNorm2d(4)
        self.dropout = nn.Dropout2d(p=0.5)
        self.head = nn.Conv2d(4, 1, kernel_size=1)
        self.register_buffer(
            "nonpersistent_cache",
            torch.tensor([1.0, 2.0]),
            persistent=False,
        )
        self.register_buffer("optional_cache", None, persistent=False)

    def forward(self, image):
        return self.head(self.dropout(self.bn(self.conv(image))))


class ExtraMethodState:
    def __init__(self) -> None:
        self.step = 3
        self.scores = torch.tensor([0.25, 0.75])

    def state_dict(self):
        return {"step": self.step, "scores": self.scores.clone()}

    def load_state_dict(self, state):
        self.step = state["step"]
        self.scores = state["scores"].clone()


def assert_nested_exact(actual: Any, expected: Any) -> None:
    if isinstance(expected, torch.Tensor):
        assert isinstance(actual, torch.Tensor)
        assert actual.dtype == expected.dtype
        assert actual.device == expected.device
        assert actual.shape == expected.shape
        assert torch.equal(actual, expected)
    elif isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for key in expected:
            assert_nested_exact(actual[key], expected[key])
    elif isinstance(expected, (list, tuple)):
        assert type(actual) is type(expected)
        assert len(actual) == len(expected)
        for actual_item, expected_item in zip(actual, expected, strict=True):
            assert_nested_exact(actual_item, expected_item)
    else:
        assert actual == expected


def make_initialized_optimizer(kind: str):
    torch.manual_seed(17)
    model = TinyEpisodicModel()
    if kind == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    elif kind == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=0.02, momentum=0.9)
    else:  # pragma: no cover - helper misuse
        raise ValueError(kind)

    model.train()
    image = torch.randn(3, 3, 7, 9)
    loss = model(image).square().mean()
    loss.backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)

    # Preserve a deliberately mixed runtime state to prove reset does not just
    # call model.eval() or model.train() globally.
    model.eval()
    model.bn.train()
    model.conv.weight.requires_grad_(False)
    return model, optimizer


def test_parameter_and_all_batchnorm_buffers_restore_exactly() -> None:
    model, optimizer = make_initialized_optimizer("adam")
    source_state = deepcopy(model.state_dict())
    source_hash = EpisodicStateManager(model, optimizer).source_fingerprint
    manager = EpisodicStateManager(model, optimizer)

    with torch.no_grad():
        model.bn.weight.add_(0.125)
        model.bn.running_mean.add_(2.0)
        model.bn.running_var.mul_(0.5)
        model.bn.num_batches_tracked.add_(11)

    current = manager.current_fingerprint()
    assert current.model_sha256 != source_hash.model_sha256
    with pytest.raises(SourceStateMismatchError, match="model"):
        manager.assert_source_state()

    restored = manager.reset_to_source()

    assert restored == manager.source_fingerprint
    assert manager.state_hash() == manager.source_fingerprint.full_sha256
    assert_nested_exact(model.state_dict(), source_state)


@pytest.mark.parametrize("optimizer_kind", ["adam", "sgd"])
def test_optimizer_moments_momentum_and_hyperparameters_restore(optimizer_kind) -> None:
    model, optimizer = make_initialized_optimizer(optimizer_kind)
    source_optimizer_state = deepcopy(optimizer.state_dict())
    manager = EpisodicStateManager(model, optimizer)

    optimizer.param_groups[0]["lr"] = 0.9
    for state in optimizer.state.values():
        for name, value in state.items():
            if isinstance(value, torch.Tensor):
                value.add_(7)
            elif isinstance(value, (int, float)):
                state[name] = value + 7

    current = manager.current_fingerprint()
    assert current.optimizer_sha256 != manager.source_fingerprint.optimizer_sha256
    with pytest.raises(SourceStateMismatchError, match="optimizer"):
        manager.assert_source_state()

    manager.reset_to_source()

    assert_nested_exact(optimizer.state_dict(), source_optimizer_state)
    if optimizer_kind == "adam":
        assert all(
            "exp_avg" in state and "exp_avg_sq" in state
            for state in optimizer.state.values()
        )
    else:
        assert all("momentum_buffer" in state for state in optimizer.state.values())


def test_runtime_flags_gradients_and_mixed_module_modes_restore() -> None:
    model, optimizer = make_initialized_optimizer("adam")
    source_modes = {name: module.training for name, module in model.named_modules()}
    source_requires_grad = {
        name: parameter.requires_grad for name, parameter in model.named_parameters()
    }
    source_track_running_stats = model.bn.track_running_stats
    manager = EpisodicStateManager(model, optimizer)

    model.train()
    model.bn.track_running_stats = False
    for parameter in model.parameters():
        parameter.requires_grad_(True)
        parameter.grad = torch.ones_like(parameter)

    current = manager.current_fingerprint()
    assert current.runtime_sha256 != manager.source_fingerprint.runtime_sha256
    assert current.gradients_sha256 != manager.source_fingerprint.gradients_sha256
    with pytest.raises(SourceStateMismatchError, match="runtime, gradients"):
        manager.assert_source_state()

    manager.reset_to_source()

    restored_modes = {
        name: module.training for name, module in model.named_modules()
    }
    assert restored_modes == source_modes
    assert {
        name: parameter.requires_grad for name, parameter in model.named_parameters()
    } == source_requires_grad
    assert model.bn.track_running_stats is source_track_running_stats
    assert all(parameter.grad is None for parameter in model.parameters())


def test_batchnorm_momentum_and_eps_are_fingerprinted_and_restored() -> None:
    model, optimizer = make_initialized_optimizer("adam")
    source_momentum = model.bn.momentum
    source_eps = model.bn.eps
    manager = EpisodicStateManager(model, optimizer)

    model.bn.momentum = 0.93
    model.bn.eps = 0.125

    changed = manager.current_fingerprint()
    assert changed.differing_components(manager.source_fingerprint) == ("runtime",)
    manager.reset_to_source()
    assert model.bn.momentum == source_momentum
    assert model.bn.eps == source_eps


def test_nonpersistent_buffer_values_and_none_topology_restore() -> None:
    model, optimizer = make_initialized_optimizer("adam")
    source_cache = model.nonpersistent_cache.clone()
    manager = EpisodicStateManager(model, optimizer)

    model.nonpersistent_cache.add_(100)
    model.optional_cache = torch.tensor([-5.0])

    changed = manager.current_fingerprint()
    assert changed.differing_components(manager.source_fingerprint) == ("model",)
    manager.reset_to_source()
    assert torch.equal(model.nonpersistent_cache, source_cache)
    assert model.optional_cache is None


def test_official_tent_style_none_batchnorm_buffers_restore_safely() -> None:
    model, optimizer = make_initialized_optimizer("adam")
    source_state = deepcopy(model.state_dict())
    manager = EpisodicStateManager(model, optimizer)

    with torch.no_grad():
        model.head.bias.add_(8)
    model.bn.running_mean = None
    model.bn.running_var = None
    model.bn.num_batches_tracked = None

    manager.reset_to_source()

    assert_nested_exact(model.state_dict(), source_state)
    assert isinstance(model.bn.running_mean, torch.Tensor)
    assert isinstance(model.bn.running_var, torch.Tensor)
    assert isinstance(model.bn.num_batches_tracked, torch.Tensor)


@pytest.mark.parametrize("drift", ["persistence", "registration"])
def test_buffer_topology_drift_fails_before_any_partial_restore(drift) -> None:
    model, optimizer = make_initialized_optimizer("adam")
    manager = EpisodicStateManager(model, optimizer)
    with torch.no_grad():
        model.head.bias.add_(4)
    tampered_bias = model.head.bias.detach().clone()

    if drift == "persistence":
        model._non_persistent_buffers_set.remove("nonpersistent_cache")
        expected_message = "persistence topology changed"
    else:
        model._buffers.pop("nonpersistent_cache")
        expected_message = "registered buffer topology changed"

    with pytest.raises(SourceSnapshotError, match=expected_message):
        manager.reset_to_source()

    # The parameter mutation remains, proving structural validation happened
    # before model.load_state_dict wrote any source parameter.
    assert torch.equal(model.head.bias, tampered_bias)


def test_fresh_optimizer_parameter_reorder_is_detected_and_restored() -> None:
    model = nn.Sequential(nn.Linear(3, 4), nn.Linear(4, 1))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    source_order = tuple(optimizer.param_groups[0]["params"])
    manager = EpisodicStateManager(model, optimizer)

    parameters = optimizer.param_groups[0]["params"]
    parameters[0], parameters[1] = parameters[1], parameters[0]

    changed = manager.current_fingerprint()
    assert changed.differing_components(manager.source_fingerprint) == ("optimizer",)
    with pytest.raises(SourceStateMismatchError, match="optimizer"):
        manager.assert_source_state()

    manager.reset_to_source()
    restored_order = tuple(optimizer.param_groups[0]["params"])
    assert all(
        restored is source
        for restored, source in zip(restored_order, source_order, strict=True)
    )


def test_component_fingerprint_localizes_independent_tampering() -> None:
    model, optimizer = make_initialized_optimizer("adam")
    extra = ExtraMethodState()
    manager = EpisodicStateManager(
        model,
        optimizer,
        extra_stateful={"method": extra},
    )
    source = manager.source_fingerprint

    with torch.no_grad():
        model.head.bias.add_(1)
    changed = manager.current_fingerprint()
    assert changed.differing_components(source) == ("model",)
    manager.reset_to_source()

    next(iter(optimizer.state.values()))["exp_avg"].add_(1)
    changed = manager.current_fingerprint()
    assert changed.differing_components(source) == ("optimizer",)
    manager.reset_to_source()

    model.dropout.train(not model.dropout.training)
    changed = manager.current_fingerprint()
    assert changed.differing_components(source) == ("runtime",)
    manager.reset_to_source()

    model.head.bias.grad = torch.ones_like(model.head.bias)
    changed = manager.current_fingerprint()
    assert changed.differing_components(source) == ("gradients",)
    manager.reset_to_source()

    extra.step += 1
    changed = manager.current_fingerprint()
    assert changed.differing_components(source) == ("extras",)
    manager.reset_to_source()

    assert manager.assert_source_state() == source
    assert extra.step == 3
    assert torch.equal(extra.scores, torch.tensor([0.25, 0.75]))


def test_explicit_callbacks_restore_defensively_copied_extra_state() -> None:
    live = {"counter": 4, "history": [1, 2]}

    def snapshot():
        return live

    def restore(state):
        live.clear()
        live.update(state)

    manager = EpisodicStateManager(
        nn.Linear(2, 1),
        extra_stateful={"callback": StatefulHooks(snapshot, restore)},
    )
    live["counter"] = 99
    live["history"].append(3)

    manager.reset_to_source()
    assert live == {"counter": 4, "history": [1, 2]}

    # Mutating the restored live object must not mutate the private source copy.
    live["history"].append(8)
    manager.reset_to_source()
    assert live == {"counter": 4, "history": [1, 2]}


def test_source_snapshot_is_one_time_and_cannot_be_overwritten() -> None:
    model = nn.Linear(3, 2)
    manager = EpisodicStateManager(model, capture_on_init=False)

    with pytest.raises(SourceSnapshotError, match="has not been saved"):
        manager.assert_source_state()

    source = manager.save_source_state()
    with torch.no_grad():
        model.weight.add_(10)
    with pytest.raises(SourceSnapshotError, match="cannot be overwritten"):
        manager.save_source_state()

    manager.reset_to_source()
    assert manager.source_fingerprint == source


def test_parameter_object_replacement_fails_closed() -> None:
    model = nn.Linear(3, 2)
    manager = EpisodicStateManager(model)
    model.weight = nn.Parameter(model.weight.detach().clone())

    changed = manager.current_fingerprint()
    assert changed.differing_components(manager.source_fingerprint) == ("topology",)
    with pytest.raises(SourceStateMismatchError, match="topology"):
        manager.assert_source_state()
    with pytest.raises(SourceSnapshotError, match="parameter objects changed"):
        manager.reset_to_source()
