from __future__ import annotations

import inspect

import pytest
import torch

from tta.objectives import (
    StageBObjectiveError,
    candidate_local_contrast_consistency,
)


def _masks() -> tuple[torch.Tensor, torch.Tensor]:
    core = torch.zeros(1, 1, 5, 5, dtype=torch.bool)
    core[..., 2, 2] = True
    ring = torch.zeros_like(core)
    ring[..., 1:4, 1:4] = True
    ring &= ~core
    return core, ring


def test_matching_candidate_contrast_is_zero_with_live_graph() -> None:
    core, ring = _masks()
    teacher = torch.zeros(1, 1, 5, 5)
    teacher[core] = 2.0
    student = teacher.clone().requires_grad_()
    output = candidate_local_contrast_consistency(
        student, teacher, (core,), (ring,)
    )
    assert output.candidate_count == 1
    assert output.total.item() == 0.0
    output.total.backward()
    assert torch.equal(student.grad, torch.zeros_like(student))


def test_eroded_candidate_contrast_produces_restorative_gradient() -> None:
    core, ring = _masks()
    teacher = torch.zeros(1, 1, 5, 5)
    teacher[core] = 2.0
    student = torch.zeros_like(teacher, requires_grad=True)
    output = candidate_local_contrast_consistency(
        student, teacher, (core,), (ring,), temperature=0.2
    )
    assert output.total.item() > 0.0
    output.total.backward()
    assert student.grad is not None and torch.isfinite(student.grad).all()
    assert student.grad[core].mean() < 0.0
    assert student.grad[ring].mean() > 0.0


def test_empty_candidate_collection_is_graph_zero() -> None:
    student = torch.zeros(1, 1, 4, 4, requires_grad=True)
    output = candidate_local_contrast_consistency(
        student, student.detach(), (), ()
    )
    assert output.candidate_count == 0
    assert output.total.item() == 0.0
    output.total.backward()
    assert torch.equal(student.grad, torch.zeros_like(student))


def test_empty_core_or_ring_and_attached_metadata_fail_closed() -> None:
    student = torch.zeros(1, 1, 4, 4, requires_grad=True)
    teacher = student.detach().clone()
    empty = torch.zeros_like(student, dtype=torch.bool)
    full = torch.ones_like(empty)
    with pytest.raises(StageBObjectiveError, match="core.*empty"):
        candidate_local_contrast_consistency(
            student, teacher, (empty,), (full,)
        )
    with pytest.raises(StageBObjectiveError, match="ring.*empty"):
        candidate_local_contrast_consistency(
            student, teacher, (full,), (empty,)
        )
    with pytest.raises(StageBObjectiveError, match="teacher_logits.*detached"):
        candidate_local_contrast_consistency(
            student, teacher.requires_grad_(), (full,), (full,)
        )


def test_local_contrast_signature_has_label_firewall() -> None:
    names = set(inspect.signature(candidate_local_contrast_consistency).parameters)
    assert names.isdisjoint(
        {"gt", "ground_truth", "outer_mask", "target", "label", "condition",
         "corruption", "severity", "dataset_name", "split"}
    )
