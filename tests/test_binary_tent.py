from __future__ import annotations

from copy import deepcopy

import pytest
import torch
from torch import Tensor, nn
from torch.optim import Adam

from tta.adabn import AdaBNMethod
from tta.binary_tent import (
    BN_PROTOCOL_BATCH_STATS,
    BN_PROTOCOL_SOURCE_STATS,
    CUDA_BACKWARD_TEMPORARILY_DISABLE,
    BinaryTentMethod,
    BinaryTentOutcome,
    BinaryTentProtocolError,
    binary_entropy_map,
    build_binary_tent_optimizer,
)
from tta.binary_tent_runner import BinaryTentEpisodicRunner
from tta.episodic_runner import EpisodicRunner
from tta.model_adapter import IRSTDModelAdapter
from tta.state_manager import EpisodicStateManager


class ToyIRSTD(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.stem = nn.Conv2d(3, 4, kernel_size=1, bias=True)
        self.bn = nn.BatchNorm2d(4)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout2d(p=0.8)
        self.head = nn.Conv2d(4, 1, kernel_size=1, bias=True)

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

    def forward(self, image: Tensor, warm_flag: bool = False):
        del warm_flag
        feature = self.dropout(self.relu(self.bn(self.stem(image))))
        return (), self.head(feature)


class _DeterminismProbe(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value: Tensor, events: list[tuple[str, bool, bool]], fail: bool):
        ctx.events = events
        ctx.fail = fail
        events.append(
            (
                "forward",
                torch.are_deterministic_algorithms_enabled(),
                torch.is_deterministic_algorithms_warn_only_enabled(),
            )
        )
        return value.clone()

    @staticmethod
    def backward(ctx, gradient: Tensor):
        ctx.events.append(
            (
                "backward",
                torch.are_deterministic_algorithms_enabled(),
                torch.is_deterministic_algorithms_warn_only_enabled(),
            )
        )
        if ctx.fail:
            raise RuntimeError("synthetic backward failure")
        return gradient, None, None


class DeterminismProbeToy(ToyIRSTD):
    def __init__(self, *, fail_backward: bool = False) -> None:
        super().__init__()
        self.fail_backward = fail_backward
        self.determinism_events: list[tuple[str, bool, bool]] = []

    def forward(self, image: Tensor, warm_flag: bool = False):
        auxiliary, logits = super().forward(image, warm_flag)
        logits = _DeterminismProbe.apply(
            logits, self.determinism_events, self.fail_backward
        )
        return auxiliary, logits


def _image(offset: float = 0.0) -> Tensor:
    values = torch.linspace(-1.2, 1.4, 3 * 8 * 8, dtype=torch.float32)
    return values.reshape(1, 3, 8, 8) + offset


def _metadata(image_id: str) -> dict[str, object]:
    return {
        "image_id": image_id,
        "original_size": (8, 8),
        "dataset": "toy",
        "corruption": "gaussian_noise",
        "severity": 3,
        "seed": 42,
    }


def _runner(
    model: nn.Module,
    *,
    protocol: str,
    optimizer_name: str = "Adam",
    learning_rate: float = 1e-3,
    cuda_backward_determinism_policy: str = "require_disabled",
) -> tuple[BinaryTentMethod, EpisodicStateManager, BinaryTentEpisodicRunner]:
    adapter = IRSTDModelAdapter(model)
    adapter.set_source_eval_mode()
    method = BinaryTentMethod.from_adapter(
        adapter,
        optimizer_name=optimizer_name,
        learning_rate=learning_rate,
        bn_protocol=protocol,
        cuda_backward_determinism_policy=cuda_backward_determinism_policy,
    )
    state = EpisodicStateManager(model, method.optimizer)
    return method, state, BinaryTentEpisodicRunner(adapter, state)


def test_binary_entropy_numeric_symmetry_and_gradients() -> None:
    logits = torch.tensor([[[[-2.0, 0.0, 2.0]]]], requires_grad=True)
    entropy = binary_entropy_map(logits)
    assert entropy.shape == logits.shape
    assert torch.allclose(entropy[..., 0], entropy[..., 2], atol=1e-6, rtol=0.0)
    assert entropy[0, 0, 0, 1].item() == pytest.approx(torch.log(torch.tensor(2.0)).item())
    entropy.sum().backward()
    assert logits.grad is not None
    assert logits.grad[0, 0, 0, 0].item() > 0.0
    assert logits.grad[0, 0, 0, 1].item() == 0.0
    assert logits.grad[0, 0, 0, 2].item() < 0.0

    extreme = binary_entropy_map(
        torch.tensor([[[[-1.0e6, 1.0e6]]]], dtype=torch.float32)
    )
    assert torch.isfinite(extreme).all()


def test_one_channel_softmax_entropy_is_invalid_zero_control() -> None:
    logits = torch.tensor([[[[-2.0, 0.5, 4.0]]]], requires_grad=True)
    probability = logits.softmax(dim=1)
    entropy = -(probability * logits.log_softmax(dim=1)).sum(dim=1).mean()
    assert entropy.item() == 0.0
    entropy.backward()
    assert logits.grad is not None
    assert torch.count_nonzero(logits.grad).item() == 0


@pytest.mark.parametrize("name", ["Adam", "SGD"])
def test_optimizer_is_fully_specified_and_rejects_bad_inputs(name: str) -> None:
    parameter = nn.Parameter(torch.ones(2))
    optimizer = build_binary_tent_optimizer(
        [parameter], name=name, learning_rate=3e-4
    )
    group = optimizer.param_groups[0]
    assert group["lr"] == pytest.approx(3e-4)
    assert group["weight_decay"] == 0.0
    assert group["foreach"] is False
    if name == "Adam":
        assert group["fused"] is False
        assert group["betas"] == (0.9, 0.999)
        assert group["eps"] == 1e-8
        assert group["amsgrad"] is False
    else:
        assert group["momentum"] == 0.9
        assert group["dampening"] == 0.0
        assert group["nesterov"] is True

    with pytest.raises(ValueError, match="duplicates"):
        build_binary_tent_optimizer(
            [parameter, parameter], name=name, learning_rate=1e-3
        )
    with pytest.raises(ValueError, match="positive"):
        build_binary_tent_optimizer([parameter], name=name, learning_rate=0.0)


def test_batch_stats_one_step_updates_only_bn_affine_and_resets_exactly() -> None:
    model = ToyIRSTD()
    adapter = IRSTDModelAdapter(model)
    adapter.set_source_eval_mode()
    method = BinaryTentMethod.from_adapter(
        adapter,
        optimizer_name="Adam",
        learning_rate=1e-3,
        bn_protocol=BN_PROTOCOL_BATCH_STATS,
    )
    state = EpisodicStateManager(model, method.optimizer)
    source_parameters = {
        name: parameter.detach().clone()
        for name, parameter in model.named_parameters()
    }
    source_buffers = {
        name: buffer.detach().clone() for name, buffer in model.named_buffers()
    }
    with torch.no_grad():
        logits_source = adapter.forward_logits(_image())
    method.prepare_episode(adapter)
    outcome = method.adapt_one_image(
        adapter=adapter,
        image=_image(),
        logits_pre=logits_source,
        metadata=_metadata("A"),
    )
    assert isinstance(outcome, BinaryTentOutcome)
    assert outcome.optimizer_steps == 1
    assert outcome.diagnostics["gradient_norm"] > 0.0
    assert outcome.diagnostics["step_norm"] > 0.0
    assert outcome.diagnostics["number_trainable_tensors"] == 2
    assert outcome.diagnostics["number_trainable_scalars"] == 8
    assert outcome.diagnostics["actual_parameter_delta_nonzero"] is True

    changed = {
        name
        for name, parameter in model.named_parameters()
        if not torch.equal(parameter, source_parameters[name])
    }
    assert changed
    assert changed <= {"bn.weight", "bn.bias"}
    for name, buffer in model.named_buffers():
        assert torch.equal(buffer, source_buffers[name]), name
    assert model.dropout.training is False
    assert all(parameter.grad is None for parameter in model.parameters())
    assert method.optimizer.state

    state.reset_to_source()
    state.assert_source_state()
    assert method.optimizer.state == {}


@pytest.mark.parametrize(
    "protocol", [BN_PROTOCOL_BATCH_STATS, BN_PROTOCOL_SOURCE_STATS]
)
def test_three_prediction_runner_and_full_state_reset(protocol: str) -> None:
    model = ToyIRSTD()
    method, state, runner = _runner(model, protocol=protocol)
    result = runner.run_one_image(
        image=_image(), metadata=_metadata("A"), method=method
    )

    assert result.logits_source_pre.shape == (1, 1, 8, 8)
    assert result.logits_tent_pre.shape == (1, 1, 8, 8)
    assert result.logits_tent_post.shape == (1, 1, 8, 8)
    assert result.diagnostics["optimizer_steps"] == 1
    assert result.diagnostics["gradient_norm"] > 0.0
    assert result.diagnostics["step_norm"] > 0.0
    assert result.diagnostics["entropy_pre"] == result.entropy_tent_pre
    assert result.diagnostics["entropy_post"] == result.entropy_tent_post
    assert result.diagnostics["entropy_delta"] == (
        result.diagnostics["entropy_post"] - result.diagnostics["entropy_pre"]
    )
    assert "optimization_entropy_pre_device" in result.diagnostics
    assert result.diagnostics["entropy_pre_reduction_abs_error"] >= 0.0
    assert result.diagnostics["post_forward_state_unchanged"] is True
    assert result.diagnostics[
        "non_adaptable_parameters_resident_bit_exact"
    ] is True
    assert result.diagnostics["registered_buffers_resident_bit_exact"] is True
    assert result.entropy_tent_post <= result.entropy_tent_pre + 1e-6
    assert result.episode.state_changes_after_prepare == ("runtime",)
    assert set(result.episode.state_changes_after_adapt) == {
        "model",
        "optimizer",
        "runtime",
    }
    assert result.episode.reset_state_sha256 == result.episode.source_state_sha256
    state.assert_source_state()
    assert method.optimizer.state == {}
    if protocol == BN_PROTOCOL_SOURCE_STATS:
        assert result.source_tent_pre_bit_exact is True
    else:
        assert result.source_tent_pre_bit_exact is False


def test_batch_stats_tent_pre_is_bit_exact_with_adabn() -> None:
    source_model = ToyIRSTD()
    adabn_model = deepcopy(source_model)
    tent_model = deepcopy(source_model)
    image = _image(0.17)

    adabn_adapter = IRSTDModelAdapter(adabn_model)
    adabn_adapter.set_source_eval_mode()
    adabn_state = EpisodicStateManager(adabn_model)
    adabn_runner = EpisodicRunner(adabn_adapter, adabn_state)
    adabn_result = adabn_runner.run_one_image(
        image=image,
        metadata=_metadata("parity"),
        method=AdaBNMethod(),
    )

    tent_method, _tent_state, tent_runner = _runner(
        tent_model,
        protocol=BN_PROTOCOL_BATCH_STATS,
        learning_rate=1e-4,
    )
    tent_result = tent_runner.run_one_image(
        image=image,
        metadata=_metadata("parity"),
        method=tent_method,
    )
    assert torch.equal(
        tent_result.logits_source_pre, adabn_result.logits_pre
    )
    assert torch.equal(
        tent_result.logits_tent_pre, adabn_result.logits_post
    )


@pytest.mark.parametrize(
    "protocol", [BN_PROTOCOL_BATCH_STATS, BN_PROTOCOL_SOURCE_STATS]
)
def test_episode_order_isolation(protocol: str) -> None:
    method, state, runner = _runner(ToyIRSTD(), protocol=protocol)
    image_a = _image(0.0)
    image_b = _image(0.41)

    a_first = runner.run_one_image(
        image=image_a, metadata=_metadata("A"), method=method
    )
    b_second = runner.run_one_image(
        image=image_b, metadata=_metadata("B"), method=method
    )
    b_first = runner.run_one_image(
        image=image_b, metadata=_metadata("B"), method=method
    )
    a_second = runner.run_one_image(
        image=image_a, metadata=_metadata("A"), method=method
    )

    for left, right in ((a_first, a_second), (b_second, b_first)):
        assert torch.equal(left.logits_source_pre, right.logits_source_pre)
        assert torch.equal(left.logits_tent_pre, right.logits_tent_pre)
        assert torch.equal(left.logits_tent_post, right.logits_tent_post)
        assert left.diagnostics["gradient_norm"] == right.diagnostics["gradient_norm"]
        assert left.diagnostics["step_norm"] == right.diagnostics["step_norm"]
    state.assert_source_state()


class RaisingAdam(Adam):
    def step(self, closure=None):
        super().step(closure=closure)
        raise RuntimeError("synthetic optimizer failure")


def test_optimizer_exception_still_restores_model_optimizer_and_runtime() -> None:
    model = ToyIRSTD()
    adapter = IRSTDModelAdapter(model)
    adapter.set_source_eval_mode()
    parameters, names = adapter.collect_adaptable_params()
    optimizer = RaisingAdam(parameters, lr=1e-3, foreach=False, fused=False)
    method = BinaryTentMethod(
        optimizer,
        parameter_names=names,
        bn_protocol=BN_PROTOCOL_BATCH_STATS,
    )
    state = EpisodicStateManager(model, optimizer)
    runner = BinaryTentEpisodicRunner(adapter, state)

    with pytest.raises(RuntimeError, match="synthetic optimizer failure"):
        runner.run_one_image(
            image=_image(), metadata=_metadata("failure"), method=method
        )
    state.assert_source_state()
    assert optimizer.state == {}
    assert all(parameter.grad is None for parameter in model.parameters())


def test_parameter_names_and_optimizer_order_are_fail_closed() -> None:
    model = ToyIRSTD()
    adapter = IRSTDModelAdapter(model)
    adapter.set_source_eval_mode()
    parameters, names = adapter.collect_adaptable_params()
    optimizer = build_binary_tent_optimizer(
        parameters, name="Adam", learning_rate=1e-3
    )
    method = BinaryTentMethod(
        optimizer,
        parameter_names=tuple(reversed(names)),
        bn_protocol=BN_PROTOCOL_BATCH_STATS,
    )
    state = EpisodicStateManager(model, optimizer)
    runner = BinaryTentEpisodicRunner(adapter, state)

    with pytest.raises(BinaryTentProtocolError, match="parameter_names"):
        runner.run_one_image(
            image=_image(), metadata=_metadata("bad-order"), method=method
        )
    state.assert_source_state()
    assert optimizer.state == {}


def test_non_float32_episode_is_rejected_and_reset() -> None:
    model = ToyIRSTD().double()
    method, state, runner = _runner(model, protocol=BN_PROTOCOL_SOURCE_STATS)

    with pytest.raises(BinaryTentProtocolError, match="requires float32"):
        runner.run_one_image(
            image=_image().double(), metadata=_metadata("float64"), method=method
        )
    state.assert_source_state()
    assert method.optimizer.state == {}


def test_outer_cpu_autocast_is_rejected_and_reset() -> None:
    method, state, runner = _runner(
        ToyIRSTD(), protocol=BN_PROTOCOL_SOURCE_STATS
    )

    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        with pytest.raises(BinaryTentProtocolError, match="float32|autocast"):
            runner.run_one_image(
                image=_image(), metadata=_metadata("autocast"), method=method
            )
    state.assert_source_state()
    assert method.optimizer.state == {}


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_cuda_strict_deterministic_policy_is_rejected_and_reset() -> None:
    device = torch.device("cuda:0")
    method, state, runner = _runner(
        ToyIRSTD().to(device), protocol=BN_PROTOCOL_SOURCE_STATS
    )
    was_enabled = torch.are_deterministic_algorithms_enabled()
    was_warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    try:
        torch.use_deterministic_algorithms(True, warn_only=False)
        with pytest.raises(
            BinaryTentProtocolError, match="deterministic algorithms to be disabled"
        ):
            runner.run_one_image(
                image=_image().to(device),
                metadata=_metadata("strict-deterministic"),
                method=method,
            )
        state.assert_source_state()
        assert method.optimizer.state == {}
    finally:
        torch.use_deterministic_algorithms(was_enabled, warn_only=was_warn_only)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_cuda_temporary_backward_policy_preserves_strict_forwards_and_restores() -> None:
    device = torch.device("cuda:0")
    model = DeterminismProbeToy().to(device)
    method, state, runner = _runner(
        model,
        protocol=BN_PROTOCOL_SOURCE_STATS,
        cuda_backward_determinism_policy=CUDA_BACKWARD_TEMPORARILY_DISABLE,
    )
    was_enabled = torch.are_deterministic_algorithms_enabled()
    was_warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    try:
        torch.use_deterministic_algorithms(True, warn_only=False)
        result = runner.run_one_image(
            image=_image().to(device),
            metadata=_metadata("temporary-backward-disable"),
            method=method,
        )
        diagnostics = result.diagnostics
        assert diagnostics["forward_deterministic_algorithms_enabled"] is True
        assert diagnostics["forward_deterministic_algorithms_warn_only"] is False
        assert diagnostics["backward_deterministic_algorithms_enabled"] is False
        assert diagnostics["cuda_backward_determinism_policy"] == (
            CUDA_BACKWARD_TEMPORARILY_DISABLE
        )
        assert diagnostics["deterministic_policy_restored_after_backward"] is True
        assert diagnostics["deterministic_algorithms_enabled"] is True
        assert torch.are_deterministic_algorithms_enabled() is True
        assert torch.is_deterministic_algorithms_warn_only_enabled() is False
        assert model.determinism_events == [
            ("forward", True, False),
            ("forward", True, False),
            ("backward", False, False),
            ("forward", True, False),
        ]
        state.assert_source_state()
        assert method.optimizer.state == {}
    finally:
        torch.use_deterministic_algorithms(was_enabled, warn_only=was_warn_only)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_cuda_temporary_backward_policy_restores_after_backward_exception() -> None:
    device = torch.device("cuda:0")
    model = DeterminismProbeToy(fail_backward=True).to(device)
    method, state, runner = _runner(
        model,
        protocol=BN_PROTOCOL_SOURCE_STATS,
        cuda_backward_determinism_policy=CUDA_BACKWARD_TEMPORARILY_DISABLE,
    )
    was_enabled = torch.are_deterministic_algorithms_enabled()
    was_warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    try:
        torch.use_deterministic_algorithms(True, warn_only=False)
        with pytest.raises(RuntimeError, match="synthetic backward failure"):
            runner.run_one_image(
                image=_image().to(device),
                metadata=_metadata("temporary-policy-backward-failure"),
                method=method,
            )
        assert model.determinism_events == [
            ("forward", True, False),
            ("forward", True, False),
            ("backward", False, False),
        ]
        assert torch.are_deterministic_algorithms_enabled() is True
        assert torch.is_deterministic_algorithms_warn_only_enabled() is False
        state.assert_source_state()
        assert method.optimizer.state == {}
        assert all(parameter.grad is None for parameter in model.parameters())
    finally:
        torch.use_deterministic_algorithms(was_enabled, warn_only=was_warn_only)


class MutatingToyIRSTD(ToyIRSTD):
    def __init__(self, mutation: str) -> None:
        super().__init__()
        self.mutation = mutation
        self.register_buffer("persistent_guard", torch.tensor([1.0]))
        self.register_buffer(
            "nonpersistent_guard", torch.tensor([2.0]), persistent=False
        )
        self.register_buffer("none_guard", None)

    def forward(self, image: Tensor, warm_flag: bool = False):
        if self.bn.training and torch.is_grad_enabled():
            with torch.no_grad():
                if self.mutation == "parameter":
                    self.head.bias.add_(1.0)
                elif self.mutation == "persistent_buffer":
                    self.persistent_guard.add_(1.0)
                elif self.mutation == "nonpersistent_buffer":
                    self.nonpersistent_guard.add_(1.0)
                elif self.mutation == "none_buffer":
                    self.none_guard = torch.tensor([3.0], device=image.device)
        return super().forward(image, warm_flag)


@pytest.mark.parametrize(
    "mutation,match",
    [
        ("parameter", "non-adaptable parameters changed"),
        ("persistent_buffer", "registered buffers changed"),
        ("nonpersistent_buffer", "registered buffers changed"),
        ("none_buffer", "registered buffers topology/value kind changed"),
    ],
)
def test_resident_gate_catches_forbidden_value_changes(
    mutation: str, match: str
) -> None:
    method, state, runner = _runner(
        MutatingToyIRSTD(mutation), protocol=BN_PROTOCOL_BATCH_STATS
    )
    with pytest.raises(BinaryTentProtocolError, match=match):
        runner.run_one_image(
            image=_image(), metadata=_metadata(mutation), method=method
        )
    state.assert_source_state()
    assert method.optimizer.state == {}


class PostForwardMutatingToyIRSTD(ToyIRSTD):
    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("post_guard", torch.tensor([0.0]))

    def forward(self, image: Tensor, warm_flag: bool = False):
        if self.bn.training and not torch.is_grad_enabled():
            self.post_guard.add_(1.0)
        return super().forward(image, warm_flag)


def test_post_forward_state_mutation_is_rejected_and_reset() -> None:
    method, state, runner = _runner(
        PostForwardMutatingToyIRSTD(), protocol=BN_PROTOCOL_BATCH_STATS
    )
    with pytest.raises(BinaryTentProtocolError, match="post-update inference"):
        runner.run_one_image(
            image=_image(), metadata=_metadata("post-mutation"), method=method
        )
    state.assert_source_state()
    assert method.optimizer.state == {}
