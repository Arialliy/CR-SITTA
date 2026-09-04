from __future__ import annotations

import inspect
import math

import pytest
import torch
import torch.nn.functional as F

from tta.objectives import (
    SourceFeatureStatistics,
    StageBObjectiveError,
    balanced_binary_entropy,
    bernoulli_entropy_map,
    build_source_anchored_region_weights,
    feature_statistics_alignment_components,
    foreground_mass_guard,
    foreground_mass_guard_components,
    foreground_soft_dice_anchor,
    foreground_soft_iou_anchor,
    multi_layer_feature_statistics_alignment,
    parameter_anchor,
    region_balanced_consistency,
    region_balanced_loss,
    reliable_background_soft_bce,
    source_anchored_multiview_consistency,
)


def _teacher_and_weights() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    teacher = torch.tensor([[[[0.8, 0.2], [0.7, 0.1]]]])
    foreground = torch.tensor([[[[1.0, 0.0], [0.5, 0.0]]]])
    background = torch.tensor([[[[0.0, 0.5], [0.0, 1.0]]]])
    return teacher, foreground, background


def test_o1_normalizes_foreground_and_background_separately() -> None:
    eps = 1.0e-6
    logits = torch.tensor([[[[-2.0, 0.0], [1.0, 3.0]]]], requires_grad=True)
    _, foreground, background = _teacher_and_weights()

    output = balanced_binary_entropy(
        logits, foreground, background, eps=eps
    )
    entropy = bernoulli_entropy_map(logits, eps=eps)
    expected_foreground = (entropy * foreground).sum() / (
        foreground.sum() + eps
    )
    expected_background = (entropy * background).sum() / (
        background.sum() + eps
    )

    assert torch.allclose(output.foreground, expected_foreground)
    assert torch.allclose(output.background, expected_background)
    assert torch.allclose(
        output.total, 0.5 * expected_foreground + 0.5 * expected_background
    )
    assert output.foreground_weight_sum.item() == pytest.approx(1.5)
    assert output.background_weight_sum.item() == pytest.approx(1.5)
    assert output.has_foreground is True


def test_empty_foreground_is_exact_graph_zero_without_nan() -> None:
    logits = torch.tensor([[[[-1.0, 0.5], [2.0, -3.0]]]], requires_grad=True)
    teacher = torch.full_like(logits.detach(), 0.1)
    foreground = torch.zeros_like(logits)
    background = torch.ones_like(logits)

    entropy = balanced_binary_entropy(logits, foreground, background)
    probability = torch.sigmoid(logits)
    dice = foreground_soft_dice_anchor(
        probability, teacher, foreground
    )
    iou = foreground_soft_iou_anchor(probability, teacher, foreground)
    multiview = source_anchored_multiview_consistency(
        torch.stack((logits, logits + 0.1)),
        teacher,
        foreground,
        background,
    )

    for value in (
        entropy.foreground,
        dice,
        iou,
        multiview.foreground,
        *multiview.per_view_foreground,
    ):
        assert value.item() == 0.0
        assert value.requires_grad
        assert torch.isfinite(value)
    entropy.total.backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()


def test_empty_reliable_background_fails_closed_everywhere() -> None:
    logits = torch.zeros(1, 1, 2, 2, requires_grad=True)
    teacher = torch.full_like(logits.detach(), 0.5)
    foreground = torch.ones_like(logits)
    background = torch.zeros_like(logits)

    with pytest.raises(StageBObjectiveError, match="reliable-background"):
        balanced_binary_entropy(logits, foreground, background)
    with pytest.raises(StageBObjectiveError, match="reliable-background"):
        reliable_background_soft_bce(logits, teacher, background)
    with pytest.raises(StageBObjectiveError, match="reliable-background"):
        foreground_mass_guard(
            torch.sigmoid(logits),
            teacher,
            background,
            margin=0.0,
        )
    with pytest.raises(StageBObjectiveError, match="reliable-background"):
        region_balanced_consistency(
            logits, teacher, foreground, background
        )
    with pytest.raises(StageBObjectiveError, match="reliable-background"):
        build_source_anchored_region_weights(
            torch.full_like(teacher, 0.9),
            torch.zeros_like(teacher),
            foreground_threshold=0.7,
            background_threshold=0.2,
        )


def test_region_weights_are_source_anchored_detached_and_guarded() -> None:
    teacher = torch.tensor([[[[0.9, 0.1], [0.5, 0.05]]]])
    uncertainty = torch.tensor([[[[0.0, 0.0], [0.2, 0.0]]]])
    guard = torch.tensor([[[[0.0, 1.0], [0.0, 0.0]]]])
    output = build_source_anchored_region_weights(
        teacher,
        uncertainty,
        foreground_threshold=0.7,
        background_threshold=0.2,
        foreground_gamma=1.0,
        background_gamma=1.0,
        foreground_temperature=0.5,
        background_temperature=0.5,
        foreground_guard=guard,
    )

    assert not output.foreground.requires_grad
    assert not output.background.requires_grad
    assert output.foreground[0, 0, 0, 0].item() == pytest.approx(0.9)
    assert output.foreground[0, 0, 1, 0].item() == 0.0
    assert output.background[0, 0, 0, 1].item() == 0.0
    assert output.background[0, 0, 1, 1].item() == pytest.approx(0.95)
    assert torch.count_nonzero(output.foreground * output.background) == 0


def test_teacher_and_region_weights_must_already_be_detached() -> None:
    logits = torch.zeros(1, 1, 2, 2, requires_grad=True)
    teacher = torch.full_like(logits, 0.25, requires_grad=True)
    foreground = torch.ones_like(logits.detach())
    background = torch.ones_like(logits.detach())

    with pytest.raises(StageBObjectiveError, match="teacher_probability.*detached"):
        reliable_background_soft_bce(logits, teacher, background)
    with pytest.raises(StageBObjectiveError, match="foreground_weight.*detached"):
        balanced_binary_entropy(
            logits,
            foreground.requires_grad_(),
            background,
        )
    with pytest.raises(StageBObjectiveError, match="teacher_probability.*detached"):
        build_source_anchored_region_weights(
            teacher,
            torch.zeros_like(teacher.detach()),
            foreground_threshold=0.7,
            background_threshold=0.2,
        )


def test_soft_dice_and_iou_follow_v5_weighted_formulas() -> None:
    probability = torch.tensor([[[[0.8, 0.4]]]], requires_grad=True)
    teacher = torch.tensor([[[[0.75, 0.5]]]])
    foreground = torch.tensor([[[[1.0, 0.5]]]])
    eps = 1.0e-6
    intersection = (foreground * probability * teacher).sum()
    student_mass = (foreground * probability).sum()
    teacher_mass = (foreground * teacher).sum()
    expected_dice = 1.0 - (2.0 * intersection + eps) / (
        student_mass + teacher_mass + eps
    )
    expected_iou = 1.0 - (intersection + eps) / (
        student_mass + teacher_mass - intersection + eps
    )

    dice = foreground_soft_dice_anchor(
        probability, teacher, foreground, eps
    )
    iou = foreground_soft_iou_anchor(probability, teacher, foreground, eps)
    assert torch.allclose(dice, expected_dice)
    assert torch.allclose(iou, expected_iou)
    (dice + iou).backward()
    assert probability.grad is not None
    assert torch.isfinite(probability.grad).all()


def test_reliable_background_bce_and_mass_guard_are_soft_and_one_sided() -> None:
    logits = torch.tensor([[[[-1.0, 0.5], [1.0, -0.5]]]], requires_grad=True)
    teacher, _, background = _teacher_and_weights()
    eps = 1.0e-6
    bce_map = F.binary_cross_entropy_with_logits(
        logits, teacher, reduction="none"
    )
    expected_bce = (bce_map * background).sum() / (
        background.sum() + eps
    )
    assert torch.allclose(
        reliable_background_soft_bce(
            logits, teacher, background, eps=eps
        ),
        expected_bce,
    )

    probability = torch.tensor([[[[0.2, 0.6]]]], requires_grad=True)
    source = torch.tensor([[[[0.1, 0.2]]]])
    weight = torch.ones_like(source)
    components = foreground_mass_guard_components(
        probability,
        source,
        weight,
        margin=0.1,
        eps=eps,
    )
    expected_current = probability.sum() / (weight.sum() + eps)
    expected_source = source.sum() / (weight.sum() + eps)
    expected_loss = torch.relu(expected_current - expected_source - 0.1)
    assert torch.allclose(components.current_background_mass, expected_current)
    assert torch.allclose(components.source_background_mass, expected_source)
    assert torch.allclose(components.loss, expected_loss)
    components.loss.backward()
    assert probability.grad is not None
    assert torch.isfinite(probability.grad).all()


@pytest.mark.parametrize("divergence", ["bce", "js"])
def test_source_anchored_multiview_is_mean_of_aligned_view_terms(
    divergence: str,
) -> None:
    teacher, foreground, background = _teacher_and_weights()
    first = torch.tensor([[[[1.0, -1.0], [0.5, -0.5]]]], requires_grad=True)
    second = torch.tensor([[[[0.8, -0.8], [0.2, -0.2]]]], requires_grad=True)
    first_output = region_balanced_consistency(
        first,
        teacher,
        foreground,
        background,
        divergence=divergence,
    )
    second_output = region_balanced_consistency(
        second,
        teacher,
        foreground,
        background,
        divergence=divergence,
    )
    output = source_anchored_multiview_consistency(
        (first, second),
        teacher,
        foreground,
        background,
        divergence=divergence,
    )

    assert output.view_count == 2
    assert torch.allclose(
        output.total, 0.5 * (first_output.total + second_output.total)
    )
    assert torch.allclose(
        output.foreground,
        0.5 * (first_output.foreground + second_output.foreground),
    )
    assert torch.allclose(
        output.background,
        0.5 * (first_output.background + second_output.background),
    )
    output.total.backward()
    assert first.grad is not None and torch.isfinite(first.grad).all()
    assert second.grad is not None and torch.isfinite(second.grad).all()


def test_region_balanced_o4_output_is_auditable_and_differentiable() -> None:
    teacher, foreground, background = _teacher_and_weights()
    logits = torch.tensor([[[[1.0, -1.0], [0.5, 0.0]]]], requires_grad=True)
    output = region_balanced_loss(
        logits,
        teacher,
        foreground,
        background,
        mass_margin=0.0,
        foreground_overlap="iou",
    )
    assert torch.allclose(
        output.total,
        output.foreground + output.background + output.mass_guard,
    )
    assert output.has_foreground is True
    assert output.foreground_overlap == "iou"
    assert torch.isfinite(output.total)
    output.total.backward()
    assert logits.grad is not None and torch.isfinite(logits.grad).all()


def test_parameter_anchor_uses_detached_source_and_exact_squared_l2() -> None:
    first = torch.tensor([1.0, 2.0], requires_grad=True)
    second = torch.tensor([3.0], requires_grad=True)
    source_first = torch.tensor([0.0, 1.0])
    source_second = torch.tensor([1.0])
    loss = parameter_anchor(
        {"first": first, "second": second},
        {"first": source_first, "second": source_second},
    )
    assert loss.item() == pytest.approx(6.0)
    loss.backward()
    assert torch.equal(first.grad, torch.tensor([2.0, 2.0]))
    assert torch.equal(second.grad, torch.tensor([4.0]))

    with pytest.raises(StageBObjectiveError, match="must be detached"):
        parameter_anchor((first,), (source_first.requires_grad_(),))


def test_feature_statistics_alignment_matches_moments_and_backpropagates() -> None:
    feature = torch.tensor(
        [[[[1.0, 3.0]], [[2.0, 6.0]]]], requires_grad=True
    )
    source_mean = torch.tensor([1.0, 5.0])
    source_std = torch.tensor([2.0, 1.0])
    output = feature_statistics_alignment_components(
        feature, source_mean, source_std
    )

    assert torch.allclose(output.current_mean, torch.tensor([2.0, 4.0]))
    assert torch.allclose(output.current_std, torch.tensor([1.0, 2.0]))
    assert output.mean_alignment.item() == pytest.approx(2.0)
    assert output.log_std_alignment.item() == pytest.approx(
        2.0 * math.log(2.0)
    )
    output.total.backward()
    assert feature.grad is not None and torch.isfinite(feature.grad).all()

    with pytest.raises(StageBObjectiveError, match="source_mean.*detached"):
        feature_statistics_alignment_components(
            feature.detach(), source_mean.requires_grad_(), source_std
        )
    with pytest.raises(StageBObjectiveError, match="strictly positive"):
        feature_statistics_alignment_components(
            feature.detach(), source_mean.detach(), torch.tensor([2.0, 0.0])
        )


def test_multi_layer_feature_statistics_uses_explicit_layer_weights() -> None:
    feature_a = torch.tensor([[[[0.0, 2.0]]]], requires_grad=True)
    feature_b = torch.tensor([[[[1.0, 3.0]]]], requires_grad=True)
    source = {
        "a": SourceFeatureStatistics(torch.tensor([0.0]), torch.tensor([1.0])),
        "b": SourceFeatureStatistics(torch.tensor([1.0]), torch.tensor([1.0])),
    }
    first = feature_statistics_alignment_components(
        feature_a, source["a"].mean, source["a"].std
    ).total
    second = feature_statistics_alignment_components(
        feature_b, source["b"].mean, source["b"].std
    ).total
    total = multi_layer_feature_statistics_alignment(
        {"a": feature_a, "b": feature_b},
        source,
        layer_weights={"a": 0.25, "b": 2.0},
    )
    assert torch.allclose(total, 0.25 * first + 2.0 * second)


def test_shape_range_finite_and_threshold_validation_is_fail_closed() -> None:
    logits = torch.zeros(1, 1, 2, 2)
    teacher = torch.full_like(logits, 0.5)
    foreground = torch.ones_like(logits)
    background = torch.ones_like(logits)

    with pytest.raises(StageBObjectiveError, match="shape"):
        reliable_background_soft_bce(
            logits, teacher[:, :, :1], background
        )
    with pytest.raises(StageBObjectiveError, match=r"\[0,1\]"):
        balanced_binary_entropy(logits, -foreground, background)
    with pytest.raises(StageBObjectiveError, match="finite"):
        balanced_binary_entropy(
            torch.full_like(logits, float("nan")), foreground, background
        )
    with pytest.raises(StageBObjectiveError, match="strictly below"):
        build_source_anchored_region_weights(
            teacher,
            torch.zeros_like(teacher),
            foreground_threshold=0.2,
            background_threshold=0.2,
        )


def test_public_objective_signatures_expose_no_gt_or_mask_argument() -> None:
    functions = (
        balanced_binary_entropy,
        build_source_anchored_region_weights,
        foreground_soft_dice_anchor,
        foreground_soft_iou_anchor,
        reliable_background_soft_bce,
        source_anchored_multiview_consistency,
        region_balanced_loss,
        foreground_mass_guard,
        parameter_anchor,
        feature_statistics_alignment_components,
    )
    forbidden = {"gt", "ground_truth", "mask", "target"}
    for function in functions:
        names = set(inspect.signature(function).parameters)
        assert names.isdisjoint(forbidden), function.__name__
