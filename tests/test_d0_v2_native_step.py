from __future__ import annotations

import json

import pytest
import torch
from torch import nn

from analysis.d0_v2_independent_candidate_contract import (
    FROZEN_CANDIDATES,
    IndependentCandidateExecutionLedger,
)
from analysis.analyze_tent_optimizer_geometry import NativeFirstStepReference
import tta.d0_v2_native_step as native


def _parameters() -> tuple[nn.Parameter, nn.Parameter]:
    first = nn.Parameter(torch.tensor([2.1058686, -0.125], dtype=torch.float32))
    second = nn.Parameter(torch.tensor([-0.25], dtype=torch.float32))
    first.grad = torch.tensor([3.7e-11, -0.125], dtype=torch.float32)
    second.grad = torch.tensor([0.5], dtype=torch.float32)
    return first, second


def _optimizer(
    name: str,
    parameters: list[nn.Parameter],
    learning_rate: float = 3e-5,
) -> torch.optim.Optimizer:
    if name == "Adam":
        return torch.optim.Adam(
            parameters,
            lr=learning_rate,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=0.0,
            amsgrad=False,
            foreach=False,
            maximize=False,
            capturable=False,
            differentiable=False,
            fused=False,
        )
    assert name == "SGD"
    return torch.optim.SGD(
        parameters,
        lr=learning_rate,
        momentum=0.9,
        dampening=0.0,
        weight_decay=0.0,
        nesterov=True,
        maximize=False,
        foreach=False,
        differentiable=False,
    )


@pytest.mark.parametrize(("name", "state_count"), [("Adam", 6), ("SGD", 2)])
def test_actual_cpu_adam_sgd_first_step_is_bit_exact_and_auditable(
    name: str, state_count: int
) -> None:
    first, second = _parameters()
    optimizer = _optimizer(name, [first, second])
    callback_values: list[native.LiveNamedGradients] = []

    def callback(values: native.LiveNamedGradients) -> None:
        assert values[0][1] is first.grad
        assert values[1][1] is second.grad
        callback_values.append(values)

    observation = native.execute_observed_native_first_step(
        optimizer,
        named_parameters=(("first", first), ("second", second)),
        expected_spec=native.frozen_first_step_spec(name, 3e-5),
        gradient_callback=callback,
    )

    assert len(callback_values) == 1
    assert observation.optimizer_name == name
    assert observation.parameter_tensor_count == 2
    assert observation.gradient_tensor_count == 2
    assert observation.scalar_parameter_count == 3
    assert observation.optimizer_state_parameter_count == 2
    assert observation.optimizer_state_tensor_count == state_count
    assert observation.native_reference_optimizer_state_tensor_count == state_count
    assert observation.bit_exact_parameter_tensor_count == 2
    assert observation.bit_exact_optimizer_state_tensor_count == state_count
    assert observation.step_norm_l2 > 0.0
    assert (
        observation.actual_parameter_after_bundle_sha256
        == observation.reference_parameter_after_bundle_sha256
    )
    assert (
        observation.actual_optimizer_state_bundle_sha256
        == observation.reference_optimizer_state_bundle_sha256
    )
    value = observation.to_dict()
    assert value["gates"]["live_gradient_objects_captured_inside_actual_step"] is True
    assert value["gates"]["original_optimizer_step_called_once"] is True
    assert value["authorization"] == {
        "engineering_observation_only": True,
        "scientific_gate_status": "unresolved",
        "scientific_selection_performed": False,
        "stage2_authorized": False,
    }
    json.dumps(value, allow_nan=False)


def test_context_manager_restores_step_and_is_one_shot() -> None:
    first, second = _parameters()
    optimizer = _optimizer("SGD", [first, second])
    class_step = type(optimizer).step
    observer = native.NativeFirstStepObserver(
        optimizer,
        named_parameters=(("first", first), ("second", second)),
        expected_spec=native.frozen_first_step_spec("SGD", 3e-5),
    )
    with observer:
        optimizer.step()
        with pytest.raises(native.D0V2NativeStepError, match="exactly once"):
            optimizer.step()
    assert "step" not in optimizer.__dict__
    assert type(optimizer).step is class_step
    assert observer.observation.optimizer_name == "SGD"
    with pytest.raises(native.D0V2NativeStepError, match="cannot be reused"):
        with observer:
            pass


def test_observer_may_wrap_source_frozen_parameters_before_runner_prepares_them() -> None:
    first = nn.Parameter(torch.tensor([1.0], dtype=torch.float32))
    second = nn.Parameter(torch.tensor([2.0], dtype=torch.float32))
    optimizer = _optimizer("SGD", [first, second])
    first.requires_grad_(False)
    second.requires_grad_(False)
    observer = native.NativeFirstStepObserver(
        optimizer,
        named_parameters=(("first", first), ("second", second)),
        expected_spec=native.frozen_first_step_spec("SGD", 3e-5),
    )

    with observer:
        # BinaryTentFastRunner prepares the BN affine tensors after the outer
        # observer has been installed but before the actual optimizer step.
        first.requires_grad_(True)
        second.requires_grad_(True)
        first.grad = torch.tensor([0.25], dtype=torch.float32)
        second.grad = torch.tensor([-0.5], dtype=torch.float32)
        optimizer.step()

    assert observer.observation.gradient_tensor_count == 2


def test_actual_step_rejects_parameters_that_remain_source_frozen() -> None:
    first = nn.Parameter(torch.tensor([1.0], dtype=torch.float32))
    optimizer = _optimizer("SGD", [first])
    first.requires_grad_(False)
    first.grad = torch.tensor([0.25], dtype=torch.float32)
    observer = native.NativeFirstStepObserver(
        optimizer,
        named_parameters=(("first", first),),
        expected_spec=native.frozen_first_step_spec("SGD", 3e-5),
    )
    with pytest.raises(native.D0V2NativeStepError, match="actual step"):
        with observer:
            optimizer.step()


def test_context_fails_closed_when_no_step_occurs() -> None:
    first, second = _parameters()
    optimizer = _optimizer("Adam", [first, second])
    observer = native.NativeFirstStepObserver(
        optimizer,
        named_parameters=(("first", first), ("second", second)),
        expected_spec=native.frozen_first_step_spec("Adam", 3e-5),
    )
    with pytest.raises(native.D0V2NativeStepError, match="without one verified"):
        with observer:
            pass
    assert "step" not in optimizer.__dict__


def test_nested_observer_for_same_optimizer_is_rejected() -> None:
    first, second = _parameters()
    optimizer = _optimizer("Adam", [first, second])
    spec = native.frozen_first_step_spec("Adam", 3e-5)
    outer = native.NativeFirstStepObserver(
        optimizer,
        named_parameters=(("first", first), ("second", second)),
        expected_spec=spec,
    )
    inner = native.NativeFirstStepObserver(
        optimizer,
        named_parameters=(("first", first), ("second", second)),
        expected_spec=spec,
    )
    with pytest.raises(native.D0V2NativeStepError, match="already observed"):
        with outer:
            with inner:
                pass


def test_nonempty_optimizer_state_is_rejected_before_second_update() -> None:
    first, second = _parameters()
    optimizer = _optimizer("Adam", [first, second])
    optimizer.step()
    first.grad = torch.tensor([0.1, 0.2], dtype=torch.float32)
    second.grad = torch.tensor([0.3], dtype=torch.float32)
    before = (first.detach().clone(), second.detach().clone())
    with pytest.raises(native.D0V2NativeStepError, match="state must be empty"):
        native.execute_observed_native_first_step(
            optimizer,
            named_parameters=(("first", first), ("second", second)),
            expected_spec=native.frozen_first_step_spec("Adam", 3e-5),
        )
    assert torch.equal(first.detach(), before[0])
    assert torch.equal(second.detach(), before[1])


@pytest.mark.parametrize(
    "drift",
    [
        "param_group_lr",
        "defaults_lr",
        "foreach",
        "extra_field",
        "parameter_order",
        "second_group",
    ],
)
def test_exact_optimizer_contract_rejects_configuration_drift(drift: str) -> None:
    first, second = _parameters()
    optimizer = _optimizer("Adam", [first, second])
    named = (("first", first), ("second", second))
    if drift == "param_group_lr":
        optimizer.param_groups[0]["lr"] = 1e-4
    elif drift == "defaults_lr":
        optimizer.defaults["lr"] = 1e-4
    elif drift == "foreach":
        optimizer.param_groups[0]["foreach"] = True
    elif drift == "extra_field":
        optimizer.param_groups[0]["unsafe_extra"] = False
    elif drift == "parameter_order":
        named = (("second", second), ("first", first))
    else:
        optimizer.add_param_group(
            {"params": [nn.Parameter(torch.tensor([1.0]))]}
        )
    before = (first.detach().clone(), second.detach().clone())
    with pytest.raises(native.D0V2NativeStepError, match="optimizer"):
        native.execute_observed_native_first_step(
            optimizer,
            named_parameters=named,
            expected_spec=native.frozen_first_step_spec("Adam", 3e-5),
        )
    assert torch.equal(first.detach(), before[0])
    assert torch.equal(second.detach(), before[1])


@pytest.mark.parametrize("bad_gradient", ["missing", "nonfinite", "shared"])
def test_invalid_live_gradient_objects_fail_before_actual_step(
    bad_gradient: str,
) -> None:
    first, second = _parameters()
    if bad_gradient == "missing":
        second.grad = None
    elif bad_gradient == "nonfinite":
        second.grad = torch.tensor([float("nan")], dtype=torch.float32)
    else:
        first = nn.Parameter(torch.tensor([1.0], dtype=torch.float32))
        second = nn.Parameter(torch.tensor([2.0], dtype=torch.float32))
        shared = torch.tensor([0.5], dtype=torch.float32)
        first.grad = shared
        second.grad = shared
    optimizer = _optimizer("SGD", [first, second])
    before = (first.detach().clone(), second.detach().clone())
    with pytest.raises(native.D0V2NativeStepError, match="gradient"):
        native.execute_observed_native_first_step(
            optimizer,
            named_parameters=(("first", first), ("second", second)),
            expected_spec=native.frozen_first_step_spec("SGD", 3e-5),
        )
    assert torch.equal(first.detach(), before[0])
    assert torch.equal(second.detach(), before[1])


@pytest.mark.parametrize(
    "callback_failure",
    ["return_value", "replace", "mutate", "parameter", "optimizer"],
)
def test_callback_cannot_change_live_gradient_or_silently_return_metadata(
    callback_failure: str,
) -> None:
    first, second = _parameters()
    optimizer = _optimizer("SGD", [first, second])

    def callback(values: native.LiveNamedGradients):
        if callback_failure == "return_value":
            return {"not": "allowed"}
        if callback_failure == "replace":
            first.grad = first.grad.clone()  # type: ignore[union-attr]
        elif callback_failure == "mutate":
            values[0][1].add_(1.0)
        elif callback_failure == "parameter":
            first.data.add_(1.0)
        else:
            optimizer.param_groups[0]["lr"] = 1e-4
        return None

    before = (first.detach().clone(), second.detach().clone())
    with pytest.raises(
        native.D0V2NativeStepError, match="callback|gradient|optimizer"
    ):
        native.execute_observed_native_first_step(
            optimizer,
            named_parameters=(("first", first), ("second", second)),
            expected_spec=native.frozen_first_step_spec("SGD", 3e-5),
            gradient_callback=callback,
        )
    if callback_failure == "parameter":
        assert not torch.equal(first.detach(), before[0])
    else:
        assert torch.equal(first.detach(), before[0])
    assert torch.equal(second.detach(), before[1])
    assert not optimizer.state


def test_live_gradient_callback_registers_and_rejects_cross_candidate_reuse() -> None:
    ledger = IndependentCandidateExecutionLedger()
    first_candidate, second_candidate = FROZEN_CANDIDATES[:2]
    shared_gradient = torch.tensor([0.25], dtype=torch.float32)

    first_parameter = nn.Parameter(torch.tensor([1.0], dtype=torch.float32))
    first_parameter.grad = shared_gradient
    first_optimizer = _optimizer(
        "Adam", [first_parameter], first_candidate.learning_rate
    )
    first_graph = object()
    ledger.claim_candidate_objects(
        first_candidate,
        model=object(),
        method=object(),
        optimizer=first_optimizer,
        optimizer_state_entry_count=0,
    )

    def first_callback(values: native.LiveNamedGradients) -> None:
        assert values[0][1] is shared_gradient
        ledger.claim_candidate_backward(
            first_candidate,
            autograd_graph=first_graph,
            gradient_buffers=tuple(value for _, value in values),
        )

    native.execute_observed_native_first_step(
        first_optimizer,
        named_parameters=(("weight", first_parameter),),
        expected_spec=native.frozen_first_step_spec(
            "Adam", first_candidate.learning_rate
        ),
        gradient_callback=first_callback,
    )

    second_parameter = nn.Parameter(torch.tensor([1.0], dtype=torch.float32))
    second_parameter.grad = shared_gradient
    second_optimizer = _optimizer(
        "Adam", [second_parameter], second_candidate.learning_rate
    )
    ledger.claim_candidate_objects(
        second_candidate,
        model=object(),
        method=object(),
        optimizer=second_optimizer,
        optimizer_state_entry_count=0,
    )
    before = second_parameter.detach().clone()

    def second_callback(values: native.LiveNamedGradients) -> None:
        ledger.claim_candidate_backward(
            second_candidate,
            autograd_graph=object(),
            gradient_buffers=tuple(value for _, value in values),
        )

    with pytest.raises(ValueError, match="live object reuse"):
        native.execute_observed_native_first_step(
            second_optimizer,
            named_parameters=(("weight", second_parameter),),
            expected_spec=native.frozen_first_step_spec(
                "Adam", second_candidate.learning_rate
            ),
            gradient_callback=second_callback,
        )
    assert torch.equal(second_parameter.detach(), before)
    assert not second_optimizer.state


def test_single_bit_reference_endpoint_drift_hard_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first, second = _parameters()
    optimizer = _optimizer("Adam", [first, second])
    original = native.pytorch_first_step_reference

    def drifted_reference(**kwargs):
        reference = original(**kwargs)
        changed_parameters = dict(reference.parameters_after)
        value = changed_parameters["first"].clone()
        value.reshape(-1).view(torch.uint8)[0].bitwise_xor_(1)
        changed_parameters["first"] = value
        return NativeFirstStepReference(
            parameters_after=changed_parameters,
            optimizer_state=reference.optimizer_state,
        )

    monkeypatch.setattr(native, "pytorch_first_step_reference", drifted_reference)
    with pytest.raises(native.D0V2NativeStepError, match="endpoint.*bit-exact"):
        native.execute_observed_native_first_step(
            optimizer,
            named_parameters=(("first", first), ("second", second)),
            expected_spec=native.frozen_first_step_spec("Adam", 3e-5),
        )


@pytest.mark.parametrize("name", ["Adam", "SGD"])
def test_actual_optimizer_state_drift_hard_fails(name: str) -> None:
    first, second = _parameters()
    optimizer = _optimizer(name, [first, second])
    real_step = optimizer.step

    def tampered_step():
        result = real_step()
        field = "exp_avg" if name == "Adam" else "momentum_buffer"
        optimizer.state[first][field].reshape(-1)[0].add_(1.0)
        return result

    optimizer.step = tampered_step  # type: ignore[method-assign]
    with pytest.raises(native.D0V2NativeStepError, match="state.*bit-exact"):
        native.execute_observed_native_first_step(
            optimizer,
            named_parameters=(("first", first), ("second", second)),
            expected_spec=native.frozen_first_step_spec(name, 3e-5),
        )
    assert optimizer.step is tampered_step


def test_spec_and_parameter_schema_fail_closed() -> None:
    first, second = _parameters()
    optimizer = _optimizer("Adam", [first, second])
    wrong_spec = native.frozen_first_step_spec("SGD", 3e-5)
    with pytest.raises(native.D0V2NativeStepError, match="class"):
        native.execute_observed_native_first_step(
            optimizer,
            named_parameters=(("first", first), ("second", second)),
            expected_spec=wrong_spec,
        )
    with pytest.raises(native.D0V2NativeStepError, match="unique"):
        native.NativeFirstStepObserver(
            optimizer,
            named_parameters=(("same", first), ("same", second)),
            expected_spec=native.frozen_first_step_spec("Adam", 3e-5),
        )
