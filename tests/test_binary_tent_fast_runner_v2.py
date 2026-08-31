from __future__ import annotations

import math

import pytest
import torch
from torch import Tensor, nn
from torch.optim import Adam

from tta.binary_tent import (
    BN_PROTOCOL_SOURCE_STATS,
    BinaryTentMethod,
)
from tta.binary_tent_fast_runner import (
    BinaryTentFastProtocolError,
    BinaryTentFastRunner,
)
from tta.binary_tent_fast_runner_v2 import (
    RELATIVE_STEP_NORM_EPSILON,
    ZERO_UPDATE_POLICY,
    BinaryTentFastRunnerV2,
)
from tta.model_adapter import IRSTDModelAdapter
from tta.state_manager import EpisodicStateManager


class _Toy(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.stem = nn.Conv2d(3, 4, 1)
        self.bn = nn.BatchNorm2d(4)
        self.head = nn.Conv2d(4, 1, 1)
        with torch.no_grad():
            self.stem.weight.copy_(
                torch.tensor(
                    [
                        [[[0.20]], [[-0.10]], [[0.05]]],
                        [[[0.05]], [[0.25]], [[-0.15]]],
                        [[[-0.20]], [[0.10]], [[0.30]]],
                        [[[0.15]], [[0.05]], [[0.10]]],
                    ]
                )
            )
            self.stem.bias.copy_(torch.tensor([0.1, -0.2, 0.05, 0.3]))
            self.bn.weight.copy_(torch.tensor([1.0, 0.8, 1.2, 0.9]))
            self.bn.bias.copy_(torch.tensor([0.1, -0.1, 0.05, 0.2]))
            self.bn.running_mean.copy_(torch.tensor([0.3, -0.4, 0.2, 0.1]))
            self.bn.running_var.copy_(torch.tensor([0.7, 1.3, 0.9, 1.1]))
            self.head.weight.copy_(
                torch.tensor([[[[0.9]], [[-0.4]], [[0.6]], [[0.3]]]])
            )
            self.head.bias.fill_(-0.35)
        self.tamper_post = False
        self.tamper_done = False

    def forward(self, image: Tensor, warm_flag: bool = False):
        del warm_flag
        logits = self.head(torch.relu(self.bn(self.stem(image))))
        if (
            self.tamper_post
            and not self.tamper_done
            and self.bn.weight.requires_grad
            and not torch.is_grad_enabled()
        ):
            self.tamper_done = True
            with torch.no_grad():
                self.head.bias.add_(1.0)
        return (), logits


class _NonFiniteStateAdam(Adam):
    @torch.no_grad()
    def step(self, closure=None):
        result = super().step(closure)
        first_state = next(iter(self.state.values()))
        first_state["exp_avg"].fill_(math.nan)
        return result


def _image() -> Tensor:
    return torch.linspace(-1.2, 1.4, 3 * 8 * 8).reshape(1, 3, 8, 8)


def _metadata(label: str) -> dict[str, object]:
    return {
        "image_id": label,
        "original_size": (8, 8),
        "dataset": "toy",
        "corruption": "gaussian_blur",
        "severity": 1,
        "seed": 42,
    }


def _build(
    *, optimizer_name: str, learning_rate: float, strict: bool = False
):
    model = _Toy()
    adapter = IRSTDModelAdapter(model)
    adapter.set_source_eval_mode()
    method = BinaryTentMethod.from_adapter(
        adapter,
        optimizer_name=optimizer_name,
        learning_rate=learning_rate,
        bn_protocol=BN_PROTOCOL_SOURCE_STATS,
    )
    state = EpisodicStateManager(model, method.optimizer)
    runner_type = BinaryTentFastRunner if strict else BinaryTentFastRunnerV2
    runner = runner_type(adapter, state, method, full_audit_cadence=1)
    return method, state, runner


@pytest.mark.parametrize("optimizer_name", ["Adam", "SGD"])
def test_v2_accepts_truthful_zero_effect_and_restores_exactly(
    optimizer_name: str,
) -> None:
    method, state, runner = _build(
        optimizer_name=optimizer_name,
        learning_rate=1e-45,
    )
    result = runner.run_one_image(
        image=_image(), metadata=_metadata(f"zero-{optimizer_name}")
    )

    assert result.diagnostics["optimizer_steps"] == 1
    assert result.diagnostics["temporary_optimizer_state_parameter_count"] > 0
    assert result.diagnostics["changed_bn_affine_tensors_fast_gate"] == 0
    assert result.diagnostics["number_updated_bn_affine_tensors"] == 0
    assert result.diagnostics["actual_parameter_delta_nonzero"] is False
    assert result.diagnostics["numerically_zero_parameter_delta"] is True
    assert result.diagnostics["zero_parameter_update_observed"] is True
    assert result.diagnostics["zero_parameter_update_allowed"] is True
    assert result.diagnostics["require_nonzero_parameter_update"] is False
    assert result.diagnostics["zero_parameter_update_policy"] == ZERO_UPDATE_POLICY
    assert result.diagnostics["source_parameter_norm"] > 0.0
    assert result.diagnostics["relative_step_norm"] == 0.0
    assert (
        result.diagnostics["relative_step_norm_epsilon"]
        == RELATIVE_STEP_NORM_EPSILON
        == 1e-12
    )
    assert result.checks["bn_affine_changed_by_one_step"] is False
    assert result.checks["optimizer_changed_by_one_step"] is True
    assert result.adapt_full_state_differences == ("optimizer", "runtime")
    assert result.adapt_full_fingerprint == result.post_full_fingerprint
    assert result.reset_full_fingerprint == state.source_fingerprint
    assert runner.completed_episodes == 1
    assert not runner.aborted
    assert method.optimizer.state == {}
    state.assert_source_state()


@pytest.mark.parametrize("optimizer_name", ["Adam", "SGD"])
def test_v2_preserves_nonzero_update_evidence(optimizer_name: str) -> None:
    method, state, runner = _build(
        optimizer_name=optimizer_name,
        learning_rate=1e-3,
    )
    calls = 0
    original_step = method.optimizer.step

    def counted_step(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original_step(*args, **kwargs)

    method.optimizer.step = counted_step
    result = runner.run_one_image(
        image=_image(), metadata=_metadata(f"nonzero-{optimizer_name}")
    )

    assert calls == 1
    assert result.diagnostics["optimizer_steps"] == 1
    assert result.diagnostics["changed_bn_affine_tensors_fast_gate"] > 0
    assert result.diagnostics["number_updated_bn_affine_tensors"] > 0
    assert result.diagnostics["actual_parameter_delta_nonzero"] is True
    assert result.diagnostics["numerically_zero_parameter_delta"] is False
    assert result.diagnostics["zero_parameter_update_observed"] is False
    assert result.diagnostics["relative_step_norm"] > 0.0
    assert result.adapt_full_state_differences == (
        "model",
        "optimizer",
        "runtime",
    )
    assert result.reset_full_fingerprint == state.source_fingerprint
    assert method.optimizer.state == {}
    state.assert_source_state()


def test_historical_runner_remains_strict_for_zero_effect() -> None:
    method, state, runner = _build(
        optimizer_name="Adam",
        learning_rate=1e-45,
        strict=True,
    )
    with pytest.raises(
        BinaryTentFastProtocolError,
        match="did not change any BN affine tensor",
    ):
        runner.run_one_image(image=_image(), metadata=_metadata("strict-zero"))
    assert runner.aborted
    assert runner.completed_episodes == 0
    assert method.optimizer.state == {}
    state.assert_source_state()


def test_v2_rejects_nonfinite_temporary_optimizer_state_and_resets() -> None:
    model = _Toy()
    adapter = IRSTDModelAdapter(model)
    adapter.set_source_eval_mode()
    parameters, names = adapter.collect_adaptable_params()
    optimizer = _NonFiniteStateAdam(parameters, lr=1e-3, foreach=False)
    method = BinaryTentMethod(
        optimizer,
        parameter_names=names,
        bn_protocol=BN_PROTOCOL_SOURCE_STATS,
    )
    state = EpisodicStateManager(model, optimizer)
    runner = BinaryTentFastRunnerV2(
        adapter, state, method, full_audit_cadence=1
    )
    with pytest.raises(
        BinaryTentFastProtocolError,
        match="optimizer tensors contain NaN/Inf",
    ):
        runner.run_one_image(image=_image(), metadata=_metadata("nonfinite"))
    assert runner.aborted
    assert runner.completed_episodes == 0
    assert method.optimizer.state == {}
    state.assert_source_state()


def test_v2_policy_argument_is_fixed_to_allow_zero() -> None:
    model = _Toy()
    adapter = IRSTDModelAdapter(model)
    adapter.set_source_eval_mode()
    method = BinaryTentMethod.from_adapter(
        adapter,
        optimizer_name="Adam",
        learning_rate=1e-3,
        bn_protocol=BN_PROTOCOL_SOURCE_STATS,
    )
    state = EpisodicStateManager(model, method.optimizer)
    with pytest.raises(ValueError, match="allow-zero SS-v2 runner"):
        BinaryTentFastRunnerV2(
            adapter,
            state,
            method,
            require_nonzero_parameter_update=True,
        )
    with pytest.raises(TypeError, match="must be boolean"):
        BinaryTentFastRunnerV2(
            adapter,
            state,
            method,
            require_nonzero_parameter_update=0,  # type: ignore[arg-type]
        )


def test_v2_post_forward_mutation_fails_closed_and_full_resets() -> None:
    model = _Toy()
    model.tamper_post = True
    adapter = IRSTDModelAdapter(model)
    adapter.set_source_eval_mode()
    method = BinaryTentMethod.from_adapter(
        adapter,
        optimizer_name="Adam",
        learning_rate=1e-3,
        bn_protocol=BN_PROTOCOL_SOURCE_STATS,
    )
    state = EpisodicStateManager(model, method.optimizer)
    runner = BinaryTentFastRunnerV2(adapter, state, method)
    with pytest.raises(BinaryTentFastProtocolError):
        runner.run_one_image(image=_image(), metadata=_metadata("post-mutation"))
    assert runner.aborted
    assert runner.completed_episodes == 0
    assert method.optimizer.state == {}
    state.assert_source_state()
