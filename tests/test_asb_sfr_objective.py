from __future__ import annotations

import inspect
from collections import OrderedDict

import pytest
import torch
import torch.nn.functional as F

from tta.objectives._validation import StageBObjectiveError
from tta.objectives.asb_sfr import asb_sfr_proposal_objective
from tta.objectives.router_regularization import (
    router_coefficient_l2,
    router_coefficient_total_variation,
    router_regularization,
)


def _teacher_metadata() -> tuple[torch.Tensor, ...]:
    teacher_logits = torch.tensor([[[[2.0, -2.0], [2.0, -2.0]]]])
    teacher_probability = torch.sigmoid(teacher_logits)
    target_weight = torch.tensor([[[[1.0, 0.0], [1.0, 0.0]]]])
    background_weight = 1.0 - target_weight
    stable_mask = torch.ones_like(teacher_probability, dtype=torch.bool)
    core = torch.zeros_like(stable_mask)
    core[..., 0, 0] = True
    ring = torch.zeros_like(stable_mask)
    ring[..., 0, 1] = True
    return (
        teacher_probability,
        teacher_logits,
        target_weight,
        background_weight,
        stable_mask,
        core,
        ring,
    )


def _router_parameters(*, zero: bool = False) -> OrderedDict[str, torch.Tensor]:
    scale_values = (
        torch.zeros(1, 1, 2, 2)
        if zero
        else torch.tensor([[[[0.0, 1.0], [0.0, 1.0]]]])
    )
    return OrderedDict(
        (
            ("router_d0.scale_coeff", scale_values.requires_grad_()),
            (
                "router_d0.bias_coeff",
                torch.zeros(1, 1, 2, 2, requires_grad=True),
            ),
        )
    )


def _objective_arguments() -> dict[str, object]:
    teacher, teacher_logits, target, background, stable, core, ring = (
        _teacher_metadata()
    )
    return {
        "student_logits": torch.zeros_like(teacher, requires_grad=True),
        "teacher_probability": teacher,
        "teacher_logits": teacher_logits,
        "target_weight": target,
        "background_weight": background,
        "stable_mask": stable,
        "candidate_cores": (core,),
        "candidate_rings": (ring,),
        "candidate_weights": torch.tensor([0.25]),
        "router_parameters": _router_parameters(),
        "entropy_margin": 0.0,
        "foreground_loss_weight": 2.0,
        "background_loss_weight": 3.0,
        "candidate_contrast_loss_weight": 4.0,
        "adapter_l2_weight": 5.0,
        "spatial_tv_weight": 6.0,
        "contrast_temperature": 0.25,
        "contrast_ring_weight": 1.0,
        "contrast_huber_delta": 1.0,
    }


def test_composite_formula_components_and_activity_audit() -> None:
    arguments = _objective_arguments()
    output = asb_sfr_proposal_objective(**arguments)

    student = arguments["student_logits"]
    teacher = arguments["teacher_probability"]
    pixel_bce = F.binary_cross_entropy_with_logits(
        student, teacher, reduction="none"
    )
    target = arguments["target_weight"]
    background = arguments["background_weight"]
    expected_foreground = (pixel_bce * target).sum() / (
        target.sum() + 1.0e-6
    )
    expected_background = (pixel_bce * background).sum() / (
        background.sum() + 1.0e-6
    )
    # One-pixel core/ring: student contrast=0, teacher contrast=4.  Huber
    # delta=1 gives 3.5; the detached candidate weight is 0.25.
    expected_candidate = student.new_tensor(0.25 * 3.5)
    expected_l2 = student.new_tensor(2.0)
    expected_tv = student.new_tensor(1.0)
    expected_total = (
        2.0 * expected_foreground
        + 3.0 * expected_background
        + 4.0 * expected_candidate
        + 5.0 * expected_l2
        + 6.0 * expected_tv
    )

    assert torch.allclose(output.foreground, expected_foreground)
    assert torch.allclose(output.background, expected_background)
    assert torch.allclose(output.candidate_contrast, expected_candidate)
    assert torch.allclose(output.adapter_l2, expected_l2)
    assert torch.allclose(output.spatial_tv, expected_tv)
    assert torch.allclose(output.total, expected_total)
    assert torch.allclose(
        output.total,
        output.weighted_foreground
        + output.weighted_background
        + output.weighted_candidate_contrast
        + output.weighted_adapter_l2
        + output.weighted_spatial_tv,
    )
    assert output.has_active_support is True
    assert output.active_pixel_count == 4
    assert output.active_fraction.item() == 1.0
    assert output.active_foreground_weight.item() == 2.0
    assert output.active_background_weight.item() == 2.0
    assert output.candidate_count == 1
    assert output.candidate_weight_sum.item() == pytest.approx(0.25)
    for evidence in (
        output.active_fraction,
        output.active_foreground_weight,
        output.active_background_weight,
        output.candidate_weight_sum,
        output.active_mask,
    ):
        assert not evidence.requires_grad


def test_composite_has_student_and_router_gradients() -> None:
    arguments = _objective_arguments()
    output = asb_sfr_proposal_objective(**arguments)
    output.total.backward()

    student = arguments["student_logits"]
    assert student.grad is not None
    assert torch.isfinite(student.grad).all()
    assert torch.count_nonzero(student.grad) > 0
    for parameter in arguments["router_parameters"].values():
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
    assert torch.count_nonzero(
        arguments["router_parameters"]["router_d0.scale_coeff"].grad
    ) > 0


def test_empty_active_and_candidates_are_safe_graph_zero() -> None:
    teacher, teacher_logits, target, background, stable, _, _ = (
        _teacher_metadata()
    )
    student = teacher_logits.clone().requires_grad_()
    router_parameters = _router_parameters(zero=True)
    output = asb_sfr_proposal_objective(
        student,
        teacher,
        teacher_logits,
        target,
        background,
        stable,
        (),
        (),
        torch.empty(0),
        router_parameters,
        entropy_margin=0.0,
        foreground_loss_weight=1.0,
        background_loss_weight=1.0,
        candidate_contrast_loss_weight=1.0,
        adapter_l2_weight=1.0,
        spatial_tv_weight=1.0,
    )
    assert output.has_active_support is False
    assert output.active_pixel_count == 0
    assert output.candidate_count == 0
    assert output.foreground.item() == 0.0
    assert output.background.item() == 0.0
    assert output.candidate_contrast.item() == 0.0
    assert output.total.item() == 0.0
    output.total.backward()
    assert torch.equal(student.grad, torch.zeros_like(student))
    for parameter in router_parameters.values():
        assert parameter.grad is not None
        assert torch.equal(parameter.grad, torch.zeros_like(parameter))


def test_empty_active_support_gates_candidate_and_regularizer_proposals() -> None:
    teacher, teacher_logits, target, background, stable, core, ring = (
        _teacher_metadata()
    )
    # Matching teacher/student uncertainty makes active support empty, while
    # deliberately changing candidate contrast and router coefficients would
    # otherwise create a nonzero proposal.
    student = teacher_logits.clone()
    student[core] -= 1.0
    student = student.requires_grad_()
    router_parameters = _router_parameters(zero=False)
    output = asb_sfr_proposal_objective(
        student,
        teacher,
        teacher_logits,
        target,
        background,
        stable,
        (core,),
        (ring,),
        torch.ones(1),
        router_parameters,
        entropy_margin=1.0,
        foreground_loss_weight=1.0,
        background_loss_weight=1.0,
        candidate_contrast_loss_weight=1.0,
        adapter_l2_weight=1.0,
        spatial_tv_weight=1.0,
    )
    assert output.has_active_support is False
    assert output.candidate_contrast.item() > 0.0
    assert output.adapter_l2.item() > 0.0
    assert output.total.item() == 0.0
    output.total.backward()
    assert torch.equal(student.grad, torch.zeros_like(student))
    for parameter in router_parameters.values():
        assert parameter.grad is not None
        assert torch.equal(parameter.grad, torch.zeros_like(parameter))


def test_router_regularization_formula_zero_init_and_1x1_safety() -> None:
    parameters = _router_parameters()
    output = router_regularization(parameters)
    assert output.adapter_l2.item() == pytest.approx(2.0)
    assert output.spatial_tv.item() == pytest.approx(1.0)
    assert output.coefficient_tensor_count == 2
    assert output.coefficient_scalar_count == 8
    assert output.spatial_axis_count == 4

    zero_parameters = _router_parameters(zero=True)
    assert router_coefficient_l2(zero_parameters).item() == 0.0
    assert router_coefficient_total_variation(zero_parameters).item() == 0.0

    one_by_one = OrderedDict(
        (
            ("scale_coeff", torch.zeros(1, 2, 1, 1, requires_grad=True)),
            ("bias_coeff", torch.zeros(1, 2, 1, 1, requires_grad=True)),
        )
    )
    tv = router_coefficient_total_variation(one_by_one)
    assert tv.item() == 0.0
    tv.backward()
    for parameter in one_by_one.values():
        assert parameter.grad is not None
        assert torch.equal(parameter.grad, torch.zeros_like(parameter))


@pytest.mark.parametrize(
    ("argument_name", "invalid_value"),
    (
        ("foreground_loss_weight", -1.0),
        ("background_loss_weight", float("nan")),
        ("candidate_contrast_loss_weight", float("inf")),
        ("adapter_l2_weight", True),
        ("spatial_tv_weight", -0.01),
    ),
)
def test_invalid_objective_weights_fail_closed(
    argument_name: str,
    invalid_value: object,
) -> None:
    arguments = _objective_arguments()
    arguments[argument_name] = invalid_value
    with pytest.raises((StageBObjectiveError, TypeError)):
        asb_sfr_proposal_objective(**arguments)


def test_attached_or_invalid_teacher_and_candidate_metadata_fail_closed() -> None:
    for argument_name in (
        "teacher_probability",
        "teacher_logits",
        "target_weight",
        "background_weight",
        "candidate_weights",
    ):
        arguments = _objective_arguments()
        arguments[argument_name] = arguments[argument_name].clone().requires_grad_()
        with pytest.raises(StageBObjectiveError, match="detached"):
            asb_sfr_proposal_objective(**arguments)

    arguments = _objective_arguments()
    arguments["candidate_weights"] = torch.tensor([-0.1])
    with pytest.raises(StageBObjectiveError, match="non-negative"):
        asb_sfr_proposal_objective(**arguments)

    arguments = _objective_arguments()
    arguments["candidate_weights"] = torch.tensor([float("nan")])
    with pytest.raises(StageBObjectiveError, match="finite"):
        asb_sfr_proposal_objective(**arguments)

    arguments = _objective_arguments()
    arguments["router_parameters"] = {
        "router_d0.weight": torch.zeros(1, 1, 2, 2, requires_grad=True)
    }
    with pytest.raises(StageBObjectiveError, match="leaf names"):
        asb_sfr_proposal_objective(**arguments)


def test_public_signatures_have_exact_names_and_label_firewall() -> None:
    expected = (
        "student_logits",
        "teacher_probability",
        "teacher_logits",
        "target_weight",
        "background_weight",
        "stable_mask",
        "candidate_cores",
        "candidate_rings",
        "candidate_weights",
        "router_parameters",
        "entropy_margin",
        "foreground_loss_weight",
        "background_loss_weight",
        "candidate_contrast_loss_weight",
        "adapter_l2_weight",
        "spatial_tv_weight",
        "min_active_pixels",
        "contrast_temperature",
        "contrast_ring_weight",
        "contrast_huber_delta",
        "eps",
    )
    assert tuple(inspect.signature(asb_sfr_proposal_objective).parameters) == expected
    forbidden = {
        "gt",
        "ground_truth",
        "outer_mask",
        "label",
        "condition",
        "corruption",
        "severity",
        "dataset_name",
        "split",
    }
    assert set(expected).isdisjoint(forbidden)
    for function in (
        router_coefficient_l2,
        router_coefficient_total_variation,
        router_regularization,
    ):
        assert tuple(inspect.signature(function).parameters) == (
            "router_parameters",
        )
