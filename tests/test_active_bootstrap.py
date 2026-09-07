from __future__ import annotations

import inspect

import pytest
import torch
import torch.nn.functional as F

from tta.objectives import (
    StageBObjectiveError,
    active_bootstrap_binary,
)


def _layout() -> tuple[torch.Tensor, ...]:
    teacher = torch.tensor([[[[0.95, 0.05], [0.90, 0.10]]]])
    target = torch.tensor([[[[1.0, 0.0], [0.5, 0.0]]]])
    background = torch.tensor([[[[0.0, 0.5], [0.0, 1.0]]]])
    stable = torch.ones_like(teacher, dtype=torch.bool)
    return teacher, target, background, stable


def test_equal_teacher_student_has_empty_support_and_graph_zero() -> None:
    teacher, target, background, stable = _layout()
    logits = torch.logit(teacher).requires_grad_()
    output = active_bootstrap_binary(
        logits,
        teacher,
        target,
        background,
        stable,
        entropy_margin=0.0,
    )
    assert output.has_active_support is False
    assert output.active_pixel_count == 0
    assert output.active_fraction.item() == 0.0
    assert output.total.item() == 0.0
    output.total.backward()
    assert torch.equal(logits.grad, torch.zeros_like(logits))


def test_less_certain_student_activates_and_has_nonzero_gradient() -> None:
    teacher, target, background, stable = _layout()
    logits = torch.zeros_like(teacher, requires_grad=True)
    output = active_bootstrap_binary(
        logits,
        teacher,
        target,
        background,
        stable,
        entropy_margin=0.01,
    )
    assert output.has_active_support is True
    assert output.active_pixel_count == teacher.numel()
    assert output.active_fraction.item() == 1.0
    output.total.backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()
    assert torch.count_nonzero(logits.grad) > 0


def test_foreground_and_background_are_normalized_separately() -> None:
    teacher, target, background, stable = _layout()
    logits = torch.zeros_like(teacher, requires_grad=True)
    output = active_bootstrap_binary(
        logits,
        teacher,
        target,
        background,
        stable,
        entropy_margin=0.0,
    )
    per_pixel = F.binary_cross_entropy_with_logits(
        logits, teacher, reduction="none"
    )
    expected_fg = (per_pixel * target).sum() / (target.sum() + 1.0e-6)
    expected_bg = (per_pixel * background).sum() / (
        background.sum() + 1.0e-6
    )
    assert torch.allclose(output.foreground, expected_fg)
    assert torch.allclose(output.background, expected_bg)
    assert torch.allclose(output.total, expected_fg + expected_bg)


def test_empty_foreground_is_exact_zero_without_nan() -> None:
    teacher, _, background, stable = _layout()
    logits = torch.zeros_like(teacher, requires_grad=True)
    output = active_bootstrap_binary(
        logits,
        teacher,
        torch.zeros_like(teacher),
        background,
        stable,
        entropy_margin=0.0,
    )
    assert output.foreground.item() == 0.0
    assert output.has_active_support is True
    assert torch.isfinite(output.total)
    output.total.backward()
    assert logits.grad is not None and torch.isfinite(logits.grad).all()


def test_tiny_active_set_fails_closed() -> None:
    teacher, target, background, stable = _layout()
    stable = torch.zeros_like(stable)
    stable[..., 0, 0] = True
    logits = torch.zeros_like(teacher, requires_grad=True)
    output = active_bootstrap_binary(
        logits,
        teacher,
        target,
        background,
        stable,
        entropy_margin=0.0,
        min_active_pixels=2,
    )
    assert output.has_active_support is False
    assert output.active_pixel_count == 0
    assert output.total.item() == 0.0


def test_teacher_weights_stability_and_activity_are_detached() -> None:
    teacher, target, background, stable = _layout()
    logits = torch.zeros_like(teacher, requires_grad=True)
    output = active_bootstrap_binary(
        logits,
        teacher,
        target,
        background,
        stable,
        entropy_margin=0.0,
    )
    assert not output.active_mask.requires_grad
    assert not output.active_fraction.requires_grad
    assert not output.active_foreground_weight.requires_grad
    assert not output.active_background_weight.requires_grad

    for name, value in (
        ("teacher", teacher),
        ("target", target),
        ("background", background),
    ):
        arguments = {
            "student_logits": logits,
            "teacher_probability": teacher,
            "target_weight": target,
            "background_weight": background,
            "stable_mask": stable,
            "entropy_margin": 0.0,
        }
        parameter_name = {
            "teacher": "teacher_probability",
            "target": "target_weight",
            "background": "background_weight",
        }[name]
        arguments[parameter_name] = value.clone().requires_grad_()
        with pytest.raises(StageBObjectiveError, match="detached"):
            active_bootstrap_binary(**arguments)

    with pytest.raises(StageBObjectiveError, match="stable_mask.*detached"):
        active_bootstrap_binary(
            logits,
            teacher,
            target,
            background,
            torch.ones_like(teacher, requires_grad=True),
            entropy_margin=0.0,
        )


def test_active_bootstrap_public_signature_has_label_firewall() -> None:
    names = set(inspect.signature(active_bootstrap_binary).parameters)
    assert names.isdisjoint(
        {"gt", "ground_truth", "outer_mask", "mask", "target", "condition",
         "corruption", "severity", "dataset_name"}
    )
