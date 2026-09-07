from __future__ import annotations

import torch

from tta.objectives import (
    foreground_mass_guard_components,
    region_balanced_loss,
)


def _teacher_and_background() -> tuple[torch.Tensor, torch.Tensor]:
    teacher = torch.tensor([[[[0.10, 0.20], [0.05, 0.15]]]])
    background = torch.ones_like(teacher)
    return teacher, background


def test_o4_forced_positive_bias_activates_with_nonzero_gradient() -> None:
    teacher, background = _teacher_and_background()
    probability = (teacher + 0.30).detach().requires_grad_()
    output = foreground_mass_guard_components(
        probability,
        teacher,
        background,
        margin=0.05,
    )
    assert output.loss.item() > 0.0
    output.loss.backward()
    assert probability.grad is not None
    assert torch.count_nonzero(probability.grad) == probability.numel()
    assert torch.all(probability.grad > 0.0)


def test_o4_negative_bias_does_not_activate() -> None:
    teacher, background = _teacher_and_background()
    probability = (teacher * 0.5).detach().requires_grad_()
    output = foreground_mass_guard_components(
        probability,
        teacher,
        background,
        margin=0.0,
    )
    assert output.loss.item() == 0.0
    output.loss.backward()
    assert torch.equal(probability.grad, torch.zeros_like(probability))


def test_inactive_o4_is_value_and_gradient_identical_to_o3() -> None:
    teacher = torch.tensor([[[[0.8, 0.1], [0.7, 0.2]]]])
    foreground = torch.tensor([[[[1.0, 0.0], [1.0, 0.0]]]])
    background = 1.0 - foreground
    # Reliable-background probabilities exactly match the teacher, so a
    # positive margin forces the one-sided O4 term to remain inactive.
    initial_logits = torch.logit(teacher)
    logits_o3 = initial_logits.clone().requires_grad_()
    logits_o4 = initial_logits.clone().requires_grad_()
    output_o3 = region_balanced_loss(
        logits_o3,
        teacher,
        foreground,
        background,
        mass_margin=0.10,
        foreground_overlap="iou",
    )
    output_o4 = region_balanced_loss(
        logits_o4,
        teacher,
        foreground,
        background,
        mass_margin=0.10,
        foreground_overlap="iou",
    )
    o3_loss = output_o3.foreground + output_o3.background
    o4_loss = output_o4.total
    assert output_o4.mass_guard.item() == 0.0
    assert torch.equal(o3_loss.detach(), o4_loss.detach())

    o3_loss.backward()
    o4_loss.backward()
    assert torch.equal(logits_o3.grad, logits_o4.grad)


def test_active_o4_changes_the_o3_gradient() -> None:
    teacher = torch.tensor([[[[0.8, 0.1], [0.7, 0.2]]]])
    foreground = torch.tensor([[[[1.0, 0.0], [1.0, 0.0]]]])
    background = 1.0 - foreground
    logits_o3 = torch.tensor([[[[1.4, 1.2], [1.0, 1.1]]]], requires_grad=True)
    logits_o4 = logits_o3.detach().clone().requires_grad_()
    output_o3 = region_balanced_loss(
        logits_o3,
        teacher,
        foreground,
        background,
        mass_margin=0.0,
        foreground_overlap="iou",
    )
    output_o4 = region_balanced_loss(
        logits_o4,
        teacher,
        foreground,
        background,
        mass_margin=0.0,
        foreground_overlap="iou",
    )
    assert output_o4.mass_guard.item() > 0.0
    (output_o3.foreground + output_o3.background).backward()
    output_o4.total.backward()
    assert not torch.equal(logits_o3.grad, logits_o4.grad)
