from __future__ import annotations

import pytest
import torch
from torch import nn

from tta.binary_tent import build_binary_tent_optimizer
from tta.d0_v2_native_step import NativeFirstStepObserver, frozen_first_step_spec
from tta.d0_v3_formal_capture import (
    D0V3FormalCaptureError,
    FormalFirstStepCapture,
)


def _parameter() -> nn.Parameter:
    return nn.Parameter(torch.tensor([1.0, -2.0], dtype=torch.float32))


def test_capture_records_exact_live_first_step_tensors() -> None:
    parameter = _parameter()
    optimizer = torch.optim.SGD(
        [parameter], lr=0.1, momentum=0.9, nesterov=True
    )
    parameter.grad = torch.tensor([0.25, -0.5], dtype=torch.float32)
    expected_before = parameter.detach().clone()
    expected_gradient = parameter.grad.detach().clone()

    with FormalFirstStepCapture(
        optimizer, named_parameters=(("bn.weight", parameter),)
    ) as capture:
        optimizer.step()

    tensors = capture.tensors
    assert tensors.parameter_names == ("bn.weight",)
    assert torch.equal(tensors.before_dict()["bn.weight"], expected_before)
    assert torch.equal(tensors.gradient_dict()["bn.weight"], expected_gradient)
    assert torch.equal(tensors.after_dict()["bn.weight"], parameter.detach())
    assert torch.equal(
        tensors.step_dict()["bn.weight"], parameter.detach() - expected_before
    )
    assert tensors.step_norm_l2 > 0.0
    assert len(tensors.parameter_live_storage) == 1
    assert len(tensors.gradient_live_storage) == 1
    assert tensors.optimizer_state_live_storage


def test_capture_can_wrap_an_existing_step_observer() -> None:
    parameter = _parameter()
    optimizer = torch.optim.SGD(
        [parameter], lr=0.1, momentum=0.9, nesterov=True
    )
    calls = 0
    original = optimizer.step

    def existing_observer(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    optimizer.step = existing_observer  # type: ignore[method-assign]
    parameter.grad = torch.tensor([0.25, -0.5], dtype=torch.float32)
    with FormalFirstStepCapture(
        optimizer, named_parameters=(("bn.weight", parameter),)
    ) as capture:
        optimizer.step()
    assert calls == 1
    assert capture.tensors.parameter_names == ("bn.weight",)
    assert optimizer.step is existing_observer


def test_capture_nests_outside_frozen_native_observer() -> None:
    parameter = _parameter()
    optimizer = build_binary_tent_optimizer(
        [parameter], name="SGD", learning_rate=1.0e-3
    )
    parameter.grad = torch.tensor([0.25, -0.5], dtype=torch.float32)
    named = (("bn.weight", parameter),)
    native = NativeFirstStepObserver(
        optimizer,
        named_parameters=named,
        expected_spec=frozen_first_step_spec("SGD", 1.0e-3),
    )
    with native:
        with FormalFirstStepCapture(
            optimizer, named_parameters=named
        ) as capture:
            optimizer.step()
    observation = native.observation
    assert observation.actual_parameter_after_bundle_sha256 == (
        observation.reference_parameter_after_bundle_sha256
    )
    assert capture.tensors.step_norm_l2 == pytest.approx(
        observation.step_norm_l2, rel=1.0e-12, abs=1.0e-12
    )


def test_capture_rejects_missing_gradient_before_calling_step() -> None:
    parameter = _parameter()
    optimizer = torch.optim.SGD(
        [parameter], lr=0.1, momentum=0.9, nesterov=True
    )
    with pytest.raises(D0V3FormalCaptureError, match="gradient is missing"):
        with FormalFirstStepCapture(
            optimizer, named_parameters=(("bn.weight", parameter),)
        ):
            optimizer.step()
    assert not hasattr(optimizer, FormalFirstStepCapture._MARKER)


def test_capture_rejects_zero_or_two_steps() -> None:
    parameter = _parameter()
    optimizer = torch.optim.SGD(
        [parameter], lr=0.1, momentum=0.9, nesterov=True
    )
    with pytest.raises(D0V3FormalCaptureError, match="without exactly one"):
        with FormalFirstStepCapture(
            optimizer, named_parameters=(("bn.weight", parameter),)
        ):
            pass

    parameter.grad = torch.tensor([0.25, -0.5], dtype=torch.float32)
    with pytest.raises(D0V3FormalCaptureError, match="exactly once"):
        with FormalFirstStepCapture(
            optimizer, named_parameters=(("bn.weight", parameter),)
        ):
            optimizer.step()
            optimizer.step()
