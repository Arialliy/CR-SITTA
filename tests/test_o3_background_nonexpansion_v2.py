"""CPU synthetic source-training checks; no research images, GT, or GPU."""

from __future__ import annotations

import copy
import inspect

import pytest
import torch
from torch import nn

from model.o3_multiscale_residual_v1 import O3MultiScaleResidual
from scripts.run_o3_multiscale_train8_v1 import supervised_loss
from training.o3_background_nonexpansion_v2 import (
    background_nonexpansion_loss,
    guarded_supervised_loss,
)


@pytest.fixture(autouse=True)
def deterministic_cpu():
    old_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(705)
        yield
    torch.set_num_threads(old_threads)


def inputs(dtype=torch.float32, shape=(2, 1, 5, 7)):
    reference = torch.randn(shape, dtype=dtype)
    targets = torch.zeros(shape, dtype=dtype)
    targets[:, :, 1:3, 2:4] = 1.0
    return reference.clone().requires_grad_(), reference, targets


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_exact_identity_has_zero_penalty_and_zero_gradient(dtype):
    logits, reference, targets = inputs(dtype)
    loss = background_nonexpansion_loss(logits, reference, targets)
    assert loss.shape == () and loss.dtype == dtype
    assert loss.requires_grad and loss.item() == 0.0
    gradient, = torch.autograd.grad(loss, logits)
    assert torch.count_nonzero(gradient) == 0


def test_only_background_increases_are_penalized_not_decreases_or_foreground():
    probability = torch.tensor([[[[0.75, 0.25, 0.75, 0.25]]]], dtype=torch.float64)
    logits = torch.logit(probability).requires_grad_()
    reference = torch.zeros_like(logits)
    targets = torch.tensor([[[[0.0, 0.0, 1.0, 1.0]]]], dtype=torch.float64)
    loss = background_nonexpansion_loss(logits, reference, targets)
    assert loss.item() == pytest.approx(0.25 / 2)
    gradient, = torch.autograd.grad(loss, logits)
    assert gradient.flatten()[0] > 0
    assert torch.count_nonzero(gradient.flatten()[1:]) == 0


def test_unchanged_existing_background_false_positive_is_not_penalized():
    logits = torch.full((1, 1, 2, 3), 4.0, requires_grad=True)
    reference = logits.detach().clone()
    targets = torch.zeros_like(reference)
    assert (logits.sigmoid() > 0.5).all()
    loss = background_nonexpansion_loss(logits, reference, targets)
    assert loss.item() == 0
    loss.backward()
    assert torch.count_nonzero(logits.grad) == 0


@pytest.mark.parametrize("foreground", [False, True])
def test_empty_foreground_or_background_remains_finite(foreground):
    logits = torch.full((2, 1, 3, 5), 1.0, dtype=torch.float64, requires_grad=True)
    reference = torch.zeros_like(logits)
    targets = torch.full_like(reference, float(foreground))
    loss = background_nonexpansion_loss(logits, reference, targets)
    expected = 0.0 if foreground else 15 * (torch.sigmoid(torch.tensor(1.0, dtype=torch.float64)).item() - 0.5)
    assert loss.item() == pytest.approx(expected)
    loss.backward()
    assert torch.isfinite(logits.grad).all()
    assert (torch.count_nonzero(logits.grad).item() == 0) is foreground


def test_per_image_foreground_denominator_not_global_or_background_mean():
    logits = torch.full((2, 1, 2, 3), torch.logit(torch.tensor(0.75, dtype=torch.float64)).item(), dtype=torch.float64)
    reference = torch.zeros_like(logits)
    targets = torch.zeros_like(logits)
    targets[0].flatten()[:1] = 1
    targets[1].flatten()[:3] = 1
    result = background_nonexpansion_loss(logits, reference, targets)
    assert result.item() == pytest.approx((5 * 0.25 / 1 + 3 * 0.25 / 3) / 2)
    separate = torch.stack([background_nonexpansion_loss(logits[i:i+1], reference[i:i+1], targets[i:i+1]) for i in range(2)]).mean()
    assert torch.equal(result, separate)


def test_other_image_changes_do_not_change_first_image_gradient():
    logits, reference, targets = inputs(torch.float64)
    with torch.no_grad():
        logits.add_(0.2)
    initial_gradient, = torch.autograd.grad(background_nonexpansion_loss(logits, reference, targets), logits)
    altered = logits.detach().clone()
    altered[1].add_(2.0)
    altered.requires_grad_(True)
    changed_gradient, = torch.autograd.grad(background_nonexpansion_loss(altered, reference, targets), altered)
    assert torch.equal(initial_gradient[0], changed_gradient[0])


def test_hand_computed_logit_gradient():
    probability = torch.tensor([[[[0.2, 0.8, 0.7], [0.3, 0.5, 0.9]]],
                                [[[0.6, 0.4, 0.8], [0.2, 0.5, 0.7]]]], dtype=torch.float64)
    logits = torch.logit(probability).requires_grad_()
    reference = torch.zeros_like(logits)
    targets = torch.zeros_like(logits)
    targets[0, 0, 0, 1] = 1
    targets[1, 0, 0, :2] = 1
    loss = background_nonexpansion_loss(logits, reference, targets)
    gradient, = torch.autograd.grad(loss, logits)
    predicted = logits.detach().sigmoid()
    denominator = targets.sum((1, 2, 3), keepdim=True).clamp_min(1)
    expected = (1 - targets) * (predicted > 0.5) * predicted * (1 - predicted) / denominator / 2
    assert torch.allclose(gradient, expected, rtol=1e-14, atol=1e-15)


def test_detached_teacher_never_receives_gradient():
    logits, reference, targets = inputs()
    teacher_leaf = reference.clone().requires_grad_(True)
    with torch.no_grad():
        logits.add_(0.1)
    background_nonexpansion_loss(logits, teacher_leaf.detach(), targets).backward()
    assert logits.grad.norm() > 0
    assert teacher_leaf.grad is None
    assert targets.grad is None


def test_augmented_loss_reuses_original_loss_and_fixed_unit_weight():
    logits, reference, targets = inputs(torch.float64)
    with torch.no_grad():
        logits.add_(0.2)
    parts = guarded_supervised_loss(logits, reference, targets)
    assert set(parts) == {"segmentation_full_fit_loss", "background_guard_full_fit_loss", "augmented_full_fit_loss"}
    assert all(value.shape == () and value.requires_grad for value in parts.values())
    assert torch.equal(parts["segmentation_full_fit_loss"], supervised_loss(logits, targets))
    assert torch.equal(parts["background_guard_full_fit_loss"], background_nonexpansion_loss(logits, reference, targets))
    assert torch.equal(parts["augmented_full_fit_loss"], parts["segmentation_full_fit_loss"] + parts["background_guard_full_fit_loss"])
    assert parts["background_guard_full_fit_loss"] > 0


def test_identity_first_step_matches_original_branch_loss_and_all_gradients():
    branch = O3MultiScaleResidual()
    replica = copy.deepcopy(branch)
    head = nn.Conv2d(16, 1, 1).eval().requires_grad_(False)
    h = torch.randn(2, 16, 9, 11)
    targets = torch.zeros(2, 1, 9, 11)
    targets[:, :, 3:5, 4:6] = 1
    with torch.no_grad():
        reference = head(h)
    original = supervised_loss(head(branch(h)), targets)
    parts = guarded_supervised_loss(head(replica(h)), reference, targets)
    assert parts["background_guard_full_fit_loss"].item() == 0
    assert torch.equal(original, parts["augmented_full_fit_loss"])
    original.backward()
    parts["augmented_full_fit_loss"].backward()
    for p, q in zip(branch.parameters(), replica.parameters(), strict=True):
        assert p.grad is not None and q.grad is not None
        assert torch.equal(p.grad, q.grad)
    assert head.weight.grad is None and head.bias.grad is None


def test_extreme_finite_logits_do_not_nan_and_saturation_is_not_nonzero_gradient_guarantee():
    logits = torch.tensor([[[[100.0, -100.0]]]], requires_grad=True)
    reference = torch.zeros_like(logits)
    targets = torch.zeros_like(logits)
    parts = guarded_supervised_loss(logits, reference, targets)
    assert all(torch.isfinite(value) for value in parts.values())
    assert parts["background_guard_full_fit_loss"] > 0
    guard_gradient, = torch.autograd.grad(parts["background_guard_full_fit_loss"], logits, retain_graph=True)
    assert guard_gradient[0, 0, 0, 0] == 0  # float32 sigmoid has saturated at 1.
    parts["augmented_full_fit_loss"].backward()
    assert torch.isfinite(logits.grad).all()


def test_inputs_not_mutated_and_noncontiguous_tensors_supported():
    values = tuple(value.transpose(2, 3) for value in inputs())
    before = tuple(value.detach().clone() for value in values)
    parts = guarded_supervised_loss(*values)
    assert torch.isfinite(parts["augmented_full_fit_loss"])
    assert all(torch.equal(value, snapshot) for value, snapshot in zip(values, before))


@pytest.mark.parametrize("name", ["reference_logits", "targets"])
def test_reference_and_gt_gradient_connections_rejected(name):
    values = dict(zip(("logits", "reference_logits", "targets"), inputs()))
    values[name].requires_grad_(True)
    with pytest.raises(ValueError, match=f"{name} must be detached"):
        guarded_supervised_loss(**values)


@pytest.mark.parametrize("index", [0, 1, 2])
@pytest.mark.parametrize("fault", ["nontensor", "shape", "dtype", "nan", "inf", "sparse"])
def test_invalid_tensors_fail_closed(index, fault):
    values = list(inputs())
    if fault == "nontensor":
        values[index] = None
    elif fault == "shape":
        values[index] = values[index][..., :4]
    elif fault == "dtype":
        values[index] = values[index].to(torch.float16)
    elif fault in ("nan", "inf"):
        with torch.no_grad():
            values[index].flatten()[0] = float(fault)
    else:
        values[index] = values[index].to_sparse()
    with pytest.raises((TypeError, ValueError)):
        guarded_supervised_loss(*values)


@pytest.mark.parametrize("shape", [(0, 1, 5, 7), (1, 2, 5, 7), (1, 1, 0, 7), (1, 1, 5)])
def test_invalid_binary_output_shapes(shape):
    values = [torch.zeros(shape) for _ in range(3)]
    with pytest.raises(ValueError, match="shape"):
        background_nonexpansion_loss(*values)


@pytest.mark.parametrize("value", [-1.0, 0.5, 2.0])
def test_nonbinary_gt_rejected(value):
    logits, reference, targets = inputs()
    targets.flatten()[0] = value
    with pytest.raises(ValueError, match="binary"):
        background_nonexpansion_loss(logits, reference, targets)


def test_matching_dtype_required():
    logits, reference, targets = inputs()
    with pytest.raises(ValueError, match="must match"):
        background_nonexpansion_loss(logits, reference.double(), targets)


def test_meta_device_rejected_without_accelerator():
    values = [torch.empty(1, 1, 5, 7, device="meta") for _ in range(3)]
    with pytest.raises(ValueError, match="materialized"):
        background_nonexpansion_loss(*values)


def test_no_hyperparameter_or_inference_update_api():
    for function in (background_nonexpansion_loss, guarded_supervised_loss):
        assert tuple(inspect.signature(function).parameters) == ("logits", "reference_logits", "targets")
        with pytest.raises(TypeError):
            function(*inputs(), weight=0.5)
