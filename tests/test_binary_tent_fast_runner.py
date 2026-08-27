from __future__ import annotations

import random

import numpy as np
import pytest
import torch
from torch import Tensor, nn
from torch.optim import Adam, Optimizer

import tta.binary_tent_fast_runner as fast_runner_module
from tta.binary_tent import (
    BN_PROTOCOL_BATCH_STATS,
    BN_PROTOCOL_SOURCE_STATS,
    CUDA_BACKWARD_TEMPORARILY_DISABLE,
    BinaryTentMethod,
)
from tta.binary_tent_fast_runner import (
    CHECK_KIND,
    BinaryTentFastProtocolError,
    BinaryTentFastRecoveryError,
    BinaryTentFastRunner,
)
from tta.binary_tent_runner import BinaryTentEpisodicRunner
from tta.model_adapter import IRSTDModelAdapter
from tta.state_manager import EpisodicStateManager


class FastTentToy(nn.Module):
    def __init__(self, *, consume_rng: bool = False) -> None:
        super().__init__()
        self.stem = nn.Conv2d(3, 4, 1)
        self.bn = nn.BatchNorm2d(4)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout2d(0.9)
        self.head = nn.Conv2d(4, 1, 1)
        self.register_buffer("persistent_guard", torch.tensor([3.0]))
        self.register_buffer(
            "nonpersistent_guard", torch.tensor([4.0]), persistent=False
        )
        self.register_buffer("none_guard", None, persistent=False)
        self.consume_rng = consume_rng
        self.tamper: str | None = None
        self.tamper_done = False
        self.optimizer_for_tamper: Optimizer | None = None

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

    def _tamper_if_requested(self, image: Tensor) -> None:
        if self.tamper_done or self.tamper is None:
            return
        adapting = self.bn.weight.requires_grad
        if not adapting:
            return
        in_update_forward = torch.is_grad_enabled()
        in_post_forward = not in_update_forward
        if self.tamper.startswith("adapt_") and not in_update_forward:
            return
        if self.tamper.startswith("post_") and not in_post_forward:
            return

        self.tamper_done = True
        with torch.no_grad():
            if self.tamper == "adapt_parameter":
                self.head.bias.add_(1.0)
            elif self.tamper == "adapt_persistent_buffer":
                self.persistent_guard.add_(1.0)
            elif self.tamper == "adapt_nonpersistent_buffer":
                self.nonpersistent_guard.add_(1.0)
            elif self.tamper == "adapt_none_buffer":
                self.none_guard = torch.tensor([8.0], device=image.device)
            elif self.tamper == "post_bn_affine":
                self.bn.weight.add_(1.0)
            elif self.tamper == "post_buffer":
                self.persistent_guard.add_(1.0)
            elif self.tamper == "post_optimizer":
                assert self.optimizer_for_tamper is not None
                for state in self.optimizer_for_tamper.state.values():
                    step = state.get("step")
                    if isinstance(step, Tensor):
                        step.add_(1)
            else:  # pragma: no cover - helper misuse
                raise AssertionError(self.tamper)

    def forward(self, image: Tensor, warm_flag: bool = False):
        del warm_flag
        if self.consume_rng:
            random.random()
            np.random.random()
            torch.rand(2, device=image.device)
        feature = self.dropout(self.relu(self.bn(self.stem(image))))
        logits = self.head(feature)
        self._tamper_if_requested(image)
        return (), logits


class StatelessStep(Optimizer):
    """Changes parameters but deliberately creates no optimizer state."""

    def __init__(self, parameters, lr: float = 1e-3) -> None:
        super().__init__(parameters, {"lr": lr})

    @torch.no_grad()
    def step(self, closure=None):
        del closure
        for group in self.param_groups:
            for parameter in group["params"]:
                if parameter.grad is not None:
                    parameter.add_(parameter.grad, alpha=-group["lr"])


class ReorderingAdam(Adam):
    @torch.no_grad()
    def step(self, closure=None):
        result = super().step(closure)
        self.param_groups[0]["params"].reverse()
        return result


class RaisingAdam(Adam):
    @torch.no_grad()
    def step(self, closure=None):
        super().step(closure)
        raise RuntimeError("synthetic optimizer failure")


def _image(offset: float = 0.0, *, device: torch.device | None = None) -> Tensor:
    values = torch.linspace(-1.2, 1.4, 3 * 8 * 8, dtype=torch.float32)
    value = values.reshape(1, 3, 8, 8) + offset
    return value if device is None else value.to(device)


def _metadata(image_id: str) -> dict[str, object]:
    return {
        "image_id": image_id,
        "original_size": (8, 8),
        "dataset": "toy",
        "corruption": "gaussian_noise",
        "severity": 3,
        "seed": 42,
    }


def _build_fast(
    *,
    protocol: str = BN_PROTOCOL_BATCH_STATS,
    optimizer_name: str = "Adam",
    cadence: int | None = None,
    consume_rng: bool = False,
    device: torch.device | None = None,
    cuda_backward_determinism_policy: str = "require_disabled",
) -> tuple[
    FastTentToy,
    BinaryTentMethod,
    EpisodicStateManager,
    BinaryTentFastRunner,
]:
    model = FastTentToy(consume_rng=consume_rng)
    if device is not None:
        model = model.to(device)
    adapter = IRSTDModelAdapter(model)
    adapter.set_source_eval_mode()
    method = BinaryTentMethod.from_adapter(
        adapter,
        optimizer_name=optimizer_name,
        learning_rate=1e-3,
        bn_protocol=protocol,
        cuda_backward_determinism_policy=cuda_backward_determinism_policy,
    )
    state = EpisodicStateManager(model, method.optimizer)
    runner = BinaryTentFastRunner(
        adapter,
        state,
        method,
        full_audit_cadence=cadence,
    )
    return model, method, state, runner


def _build_with_optimizer(
    optimizer_type: type[Optimizer],
) -> tuple[FastTentToy, BinaryTentMethod, EpisodicStateManager, BinaryTentFastRunner]:
    model = FastTentToy()
    adapter = IRSTDModelAdapter(model)
    adapter.set_source_eval_mode()
    parameters, names = adapter.collect_adaptable_params()
    if optimizer_type is StatelessStep:
        optimizer = StatelessStep(parameters, lr=1e-3)
    else:
        optimizer = optimizer_type(
            parameters,
            lr=1e-3,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=0.0,
            amsgrad=False,
            foreach=False,
        )
    method = BinaryTentMethod(
        optimizer,
        parameter_names=names,
        bn_protocol=BN_PROTOCOL_BATCH_STATS,
    )
    state = EpisodicStateManager(model, optimizer)
    runner = BinaryTentFastRunner(adapter, state, method)
    return model, method, state, runner


@pytest.mark.parametrize(
    "protocol", [BN_PROTOCOL_BATCH_STATS, BN_PROTOCOL_SOURCE_STATS]
)
def test_fast_runner_returns_three_predictions_and_restores_complete_source(
    protocol: str,
) -> None:
    model, method, state, runner = _build_fast(protocol=protocol)
    image = _image()
    original = image.clone()

    result = runner.run_one_image(image=image, metadata=_metadata("sample"))

    assert result.method == "binary_episodic_tent"
    assert result.episode_number == 1
    assert result.check_kind == CHECK_KIND
    assert result.logits_source_pre.shape == (1, 1, 8, 8)
    assert result.logits_tent_pre.shape == (1, 1, 8, 8)
    assert result.logits_tent_post.shape == (1, 1, 8, 8)
    assert result.entropy_tent_pre == pytest.approx(
        result.diagnostics["entropy_pre"]
    )
    assert result.entropy_tent_post == pytest.approx(
        result.diagnostics["entropy_post"]
    )
    if protocol == BN_PROTOCOL_SOURCE_STATS:
        assert result.source_tent_pre_bit_exact
    else:
        assert not result.source_tent_pre_bit_exact
    assert result.diagnostics["changed_bn_affine_tensors_fast_gate"] > 0
    assert result.diagnostics["temporary_optimizer_state_parameter_count"] > 0
    assert result.input_unchanged
    assert torch.equal(image, original)
    assert set(result.resident_checks) == {
        "episode_start",
        "source_forward",
        "prepare_episode",
        "adapt_one_image",
        "post_forward",
        "fast_reset",
    }
    assert all(result.checks.values())
    assert not result.full_audit_performed
    assert result.adapt_full_fingerprint is None
    assert result.post_full_fingerprint is None
    assert result.reset_full_fingerprint is None
    assert runner.completed_episodes == 1
    assert not runner.aborted
    assert method.optimizer.state == {}
    assert all(parameter.grad is None for parameter in model.parameters())
    state.assert_source_state()


def test_forced_and_cadence_full_sha_audits_cover_adapt_post_and_reset() -> None:
    _model, _method, state, runner = _build_fast(cadence=2)

    first = runner.run_one_image(image=_image(), metadata=_metadata("one"))
    second = runner.run_one_image(image=_image(0.2), metadata=_metadata("two"))
    third = runner.run_one_image(
        image=_image(0.4),
        metadata=_metadata("three"),
        force_full_audit=True,
    )

    assert not first.full_audit_performed
    for result, reason in ((second, "cadence"), (third, "forced")):
        assert result.full_audit_performed
        assert result.full_audit_reason == reason
        assert result.adapt_full_state_differences == (
            "model",
            "optimizer",
            "runtime",
        )
        assert result.adapt_full_fingerprint == result.post_full_fingerprint
        assert result.reset_full_fingerprint == state.source_fingerprint


@pytest.mark.parametrize(
    "protocol", [BN_PROTOCOL_BATCH_STATS, BN_PROTOCOL_SOURCE_STATS]
)
@pytest.mark.parametrize("optimizer_name", ["Adam", "SGD"])
def test_fast_and_generic_runner_three_output_parity(
    protocol: str, optimizer_name: str
) -> None:
    fast_model, fast_method, fast_state, fast = _build_fast(
        protocol=protocol, optimizer_name=optimizer_name
    )
    generic_model = FastTentToy()
    generic_adapter = IRSTDModelAdapter(generic_model)
    generic_adapter.set_source_eval_mode()
    generic_method = BinaryTentMethod.from_adapter(
        generic_adapter,
        optimizer_name=optimizer_name,
        learning_rate=1e-3,
        bn_protocol=protocol,
    )
    generic_state = EpisodicStateManager(generic_model, generic_method.optimizer)
    generic = BinaryTentEpisodicRunner(generic_adapter, generic_state)

    fast_result = fast.run_one_image(
        image=_image(), metadata=_metadata("parity")
    )
    generic_result = generic.run_one_image(
        image=_image(), metadata=_metadata("parity"), method=generic_method
    )

    assert torch.equal(
        fast_result.logits_source_pre, generic_result.logits_source_pre
    )
    assert torch.equal(fast_result.logits_tent_pre, generic_result.logits_tent_pre)
    assert torch.equal(
        fast_result.logits_tent_post, generic_result.logits_tent_post
    )
    assert fast_result.entropy_tent_pre == generic_result.entropy_tent_pre
    assert fast_result.entropy_tent_post == generic_result.entropy_tent_post
    assert fast_result.diagnostics["gradient_norm"] == (
        generic_result.diagnostics["gradient_norm"]
    )
    assert fast_result.diagnostics["step_norm"] == (
        generic_result.diagnostics["step_norm"]
    )
    fast_state.assert_source_state()
    generic_state.assert_source_state()
    assert fast_method.optimizer.state == {}
    assert generic_method.optimizer.state == {}
    assert tuple(fast_model.state_dict()) == tuple(generic_model.state_dict())


def test_a_b_and_b_a_are_bit_exact_with_fast_per_image_reset() -> None:
    _model, _method, state, runner = _build_fast()
    images = {"a": _image(), "b": _image(2.0)}
    forward = {
        key: runner.run_one_image(image=images[key], metadata=_metadata(key))
        for key in ("a", "b")
    }
    reverse = {
        key: runner.run_one_image(image=images[key], metadata=_metadata(key))
        for key in ("b", "a")
    }

    for key in images:
        for field in (
            "logits_source_pre",
            "logits_tent_pre",
            "logits_tent_post",
        ):
            assert torch.equal(getattr(forward[key], field), getattr(reverse[key], field))
        assert forward[key].source_state_sha256 == reverse[key].source_state_sha256
    assert runner.completed_episodes == 4
    state.assert_source_state()


@pytest.mark.parametrize(
    "tamper",
    [
        "adapt_parameter",
        "adapt_persistent_buffer",
        "adapt_nonpersistent_buffer",
        "adapt_none_buffer",
        "post_bn_affine",
        "post_buffer",
        "post_optimizer",
    ],
)
def test_malicious_model_or_post_state_mutation_emergency_resets_and_aborts(
    tamper: str,
) -> None:
    model, method, state, runner = _build_fast()
    model.tamper = tamper
    model.optimizer_for_tamper = method.optimizer

    with pytest.raises(BinaryTentFastProtocolError):
        runner.run_one_image(image=_image(), metadata=_metadata(tamper))

    assert runner.aborted
    assert runner.completed_episodes == 0
    state.assert_source_state()
    assert method.optimizer.state == {}
    with pytest.raises(BinaryTentFastProtocolError, match="previous failed episode"):
        runner.run_one_image(image=_image(), metadata=_metadata("again"))


def test_stateless_optimizer_is_rejected_after_step_then_runner_aborts() -> None:
    _model, method, state, runner = _build_with_optimizer(StatelessStep)

    with pytest.raises(
        BinaryTentFastProtocolError, match="did not create temporary optimizer state"
    ):
        runner.run_one_image(image=_image(), metadata=_metadata("stateless"))

    assert runner.aborted
    assert method.optimizer.state == {}
    state.assert_source_state()


def test_optimizer_parameter_reordering_is_rejected_and_reset() -> None:
    _model, method, state, runner = _build_with_optimizer(ReorderingAdam)

    with pytest.raises(BinaryTentFastProtocolError, match="order/binding"):
        runner.run_one_image(image=_image(), metadata=_metadata("reorder"))

    assert runner.aborted
    assert method.optimizer.state == {}
    state.assert_source_state()


def test_optimizer_exception_after_partial_step_is_reset_and_aborts() -> None:
    _model, method, state, runner = _build_with_optimizer(RaisingAdam)

    with pytest.raises(RuntimeError, match="synthetic optimizer failure"):
        runner.run_one_image(image=_image(), metadata=_metadata("raise"))

    assert runner.aborted
    assert method.optimizer.state == {}
    state.assert_source_state()


def test_direct_module_alias_topology_mutation_fails_closed_and_aborts() -> None:
    model, _method, _state, runner = _build_fast()
    model.alias_of_bn = model.bn

    with pytest.raises(BinaryTentFastRecoveryError) as captured:
        runner.run_one_image(image=_image(), metadata=_metadata("alias"))

    assert "topology" in str(captured.value.original_error)
    assert runner.aborted
    with pytest.raises(BinaryTentFastProtocolError, match="previous failed episode"):
        runner.run_one_image(image=_image(), metadata=_metadata("again"))


def test_nonempty_source_optimizer_state_is_rejected_at_construction() -> None:
    model = FastTentToy()
    adapter = IRSTDModelAdapter(model)
    adapter.set_source_eval_mode()
    method = BinaryTentMethod.from_adapter(
        adapter,
        optimizer_name="Adam",
        learning_rate=1e-3,
        bn_protocol=BN_PROTOCOL_BATCH_STATS,
    )
    parameter = method.optimizer.param_groups[0]["params"][0]
    method.optimizer.state[parameter]["synthetic_source_state"] = torch.tensor(1.0)
    state = EpisodicStateManager(model, method.optimizer)

    with pytest.raises(BinaryTentFastProtocolError, match="empty Source optimizer"):
        BinaryTentFastRunner(adapter, state, method)


def test_python_numpy_torch_rng_and_caller_input_are_restored() -> None:
    _model, _method, state, runner = _build_fast(consume_rng=True)
    image = _image()
    image_before = image.clone()
    python_before = random.getstate()
    numpy_before = np.random.get_state()
    torch_before = torch.get_rng_state().clone()

    runner.run_one_image(image=image, metadata=_metadata("rng"))

    assert torch.equal(image, image_before)
    assert random.getstate() == python_before
    numpy_after = np.random.get_state()
    assert numpy_after[0] == numpy_before[0]
    assert np.array_equal(numpy_after[1], numpy_before[1])
    assert numpy_after[2:] == numpy_before[2:]
    assert torch.equal(torch.get_rng_state(), torch_before)
    state.assert_source_state()


def test_rng_restore_failure_emergency_resets_and_aborts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _model, _method, state, runner = _build_fast()

    def fail_restore(_state) -> None:
        raise RuntimeError("synthetic RNG restore failure")

    monkeypatch.setattr(fast_runner_module, "_restore_rng_state", fail_restore)
    with pytest.raises(RuntimeError, match="synthetic RNG restore failure"):
        runner.run_one_image(image=_image(), metadata=_metadata("rng-fail"))
    assert runner.aborted
    state.assert_source_state()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_cuda_toy_one_image_runs_and_resets_without_long_calibration() -> None:
    device = torch.device("cuda:0")
    was_enabled = torch.are_deterministic_algorithms_enabled()
    was_warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    try:
        torch.use_deterministic_algorithms(True, warn_only=False)
        _model, method, state, runner = _build_fast(
            protocol=BN_PROTOCOL_SOURCE_STATS,
            device=device,
            cuda_backward_determinism_policy=(
                CUDA_BACKWARD_TEMPORARILY_DISABLE
            ),
        )
        result = runner.run_one_image(
            image=_image(device=device),
            metadata=_metadata("cuda-toy"),
            force_full_audit=True,
        )
        assert result.logits_source_pre.device.type == "cpu"
        assert result.logits_tent_pre.device.type == "cpu"
        assert result.logits_tent_post.device.type == "cpu"
        assert all(result.checks.values())
        assert result.full_audit_performed
        assert result.diagnostics[
            "forward_deterministic_algorithms_enabled"
        ] is True
        assert result.diagnostics[
            "backward_deterministic_algorithms_enabled"
        ] is False
        assert result.diagnostics[
            "deterministic_policy_restored_after_backward"
        ] is True
        assert torch.are_deterministic_algorithms_enabled() is True
        assert method.optimizer.state == {}
        state.assert_source_state()
    finally:
        torch.use_deterministic_algorithms(was_enabled, warn_only=was_warn_only)
