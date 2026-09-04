from __future__ import annotations

import json

import pytest
import torch
from torch import Tensor, nn

from analysis.d0_v3_outer_analyzer import FlatParameterLayout
from analysis.foreground_background_gradient_decomposition_v1 import (
    BACKWARD_BASIS,
    ForegroundBackgroundGradientError,
    GradientDecompositionConfig,
    analyze_foreground_background_gradient_decomposition,
)
from tta.binary_tent import binary_entropy_map


NAMES = ("p0", "p1", "p2", "p3", "p4")


def _groups() -> dict[str, tuple[str, ...]]:
    return {
        "P0": NAMES,
        "P1": ("p4",),
        "P2": ("p3", "p4"),
        "P3": ("p2", "p3", "p4"),
        "P4": ("p1", "p2", "p3", "p4"),
    }


def _graph(
    *, base: Tensor | None = None, values: tuple[float, ...] | None = None
) -> tuple[dict[str, nn.Parameter], FlatParameterLayout, Tensor, Tensor]:
    if values is None:
        values = (0.05, -0.04, 0.03, -0.02, 0.01)
    parameters = {
        name: nn.Parameter(torch.tensor([value], dtype=torch.float32))
        for name, value in zip(NAMES, values, strict=True)
    }
    if base is None:
        base = torch.tensor(
            [[[[-2.0, -1.0], [1.0, 2.0]]]], dtype=torch.float32
        )
    features = (
        torch.ones_like(base),
        torch.tensor([[[[1.0, 0.5], [-0.5, -1.0]]]], dtype=torch.float32),
        torch.tensor([[[[0.2, -0.4], [0.6, -0.8]]]], dtype=torch.float32),
        torch.tensor([[[[-0.3, 0.7], [0.9, -0.1]]]], dtype=torch.float32),
        torch.tensor([[[[0.8, -0.6], [0.4, -0.2]]]], dtype=torch.float32),
    )
    logits = base
    for parameter, feature in zip(parameters.values(), features, strict=True):
        logits = logits + parameter.reshape(1, 1, 1, 1) * feature
    layout = FlatParameterLayout.from_named_tensors(tuple(parameters.items()))
    parent_parts = torch.autograd.grad(
        binary_entropy_map(logits, eps=1.0e-6).mean(),
        tuple(parameters.values()),
        retain_graph=True,
    )
    parent = torch.cat([value.detach().reshape(-1) for value in parent_parts])
    return parameters, layout, logits, parent


def _analyze(
    target: Tensor,
    *,
    base: Tensor | None = None,
    values: tuple[float, ...] | None = None,
    task: Tensor | None = None,
):
    parameters, layout, logits, parent = _graph(base=base, values=values)
    if task is None:
        task = torch.tensor([0.5, -0.4, 0.3, -0.2, 0.1], dtype=torch.float32)
    result = analyze_foreground_background_gradient_decomposition(
        source_logits=logits,
        target=target,
        named_parameters=parameters,
        parameter_layout=layout,
        group_parameter_names=_groups(),
        parent_entropy_gradient_flat=parent,
        task_gradient_flat=task,
    )
    return parameters, result


def test_positive_grayscale_target_uses_strict_positive_foreground() -> None:
    target = torch.tensor(
        [[[[0.0, 0.25], [0.75, 0.0]]]], dtype=torch.float32
    )
    _, result = _analyze(target)
    report = result.to_dict()
    stats = report["target_statistics"]
    assert stats["foreground_pixel_count"] == 2
    assert stats["background_pixel_count"] == 2
    assert stats["foreground_subthreshold_pixel_count"] == 1
    assert stats["foreground_suprathreshold_pixel_count"] == 1
    assert stats["foreground_value_min"] == pytest.approx(0.25)
    assert stats["foreground_value_max"] == pytest.approx(0.75)
    assert report["conventions"]["foreground_rule"] == "target>0"
    assert tuple(report["conventions"]["autograd_backward_basis"]) == BACKWARD_BASIS
    assert json.dumps(report, allow_nan=False)


def test_empty_foreground_is_null_and_not_estimable() -> None:
    _, result = _analyze(torch.zeros(1, 1, 2, 2, dtype=torch.float32))
    assert result.vectors.foreground_conditional_mean is None
    assert result.vectors.foreground_subthreshold_conditional_mean is None
    assert result.vectors.foreground_suprathreshold_conditional_mean is None
    assert torch.equal(result.vectors.full_add, result.vectors.background_add)
    report = result.to_dict()
    assert report["target_statistics"]["foreground_estimable"] is False
    for group in report["per_group"].values():
        foreground = group["conditional_entropy_task_alignment"][
            "foreground_entropy_mean"
        ]
        assert foreground["estimable"] is False
        assert foreground["not_estimable_reason"] == "empty_foreground"
        assert foreground["entropy_gradient_norm"] is None
        assert foreground["task_gradient_norm"] == group["task_gradient_norm"]
        assert foreground["entropy_task_dot"] is None
        assert foreground["entropy_task_cosine"] is None
        assert foreground["task_projection"] is None
        assert foreground["unit_descent_task_change"] is None
        assert foreground["direction"] == "undefined"
        assert foreground["cosine_status"] == "not_estimable_region_absent"
        assert group["additive_gradient_norms"]["foreground_add"] is None
        assert group["additive_gradient_norms"][
            "foreground_subthreshold_add"
        ] is None
        assert group["additive_gradient_norms"][
            "foreground_suprathreshold_add"
        ] is None


def test_empty_background_is_null_without_losing_full_gradient() -> None:
    _, result = _analyze(torch.ones(1, 1, 2, 2, dtype=torch.float32))
    assert result.vectors.background_conditional_mean is None
    assert torch.equal(result.vectors.full_add, result.vectors.foreground_add)
    report = result.to_dict()
    assert report["target_statistics"]["background_estimable"] is False
    background = report["per_group"]["P0"][
        "conditional_entropy_task_alignment"
    ]["background_entropy_mean"]
    assert background["estimable"] is False
    assert background["entropy_task_cosine"] is None
    assert background["direction"] == "undefined"


def test_additive_and_weighted_decomposition_reconstruct_parent_gradient() -> None:
    target = torch.tensor(
        [[[[0.0, 1.0], [0.4, 0.0]]]], dtype=torch.float32
    )
    _, result = _analyze(target)
    vectors = result.vectors
    assert torch.equal(
        vectors.foreground_add,
        vectors.foreground_subthreshold_add
        + vectors.foreground_suprathreshold_add,
    )
    assert torch.equal(
        vectors.full_add, vectors.foreground_add + vectors.background_add
    )
    assert torch.allclose(vectors.full_add, vectors.parent_entropy, atol=1e-7, rtol=1e-4)
    report = result.to_dict()
    assert report["decomposition"]["weighted_conditional_reconstruction"][
        "verified"
    ] is True
    assert report["decomposition"]["parent_full_entropy_reconstruction"][
        "verified"
    ] is True
    assert report["decomposition"]["parent_full_entropy_reconstruction"][
        "max_abs_tolerance"
    ] == 1.0e-7
    assert report["decomposition"]["parent_full_entropy_reconstruction"][
        "relative_l2_tolerance"
    ] == 1.0e-4


def test_zero_entropy_gradient_cosine_is_null_not_zero_or_nan() -> None:
    zero = torch.zeros(1, 1, 2, 2, dtype=torch.float32)
    _, result = _analyze(
        torch.tensor([[[[0.0, 1.0], [0.5, 0.0]]]], dtype=torch.float32),
        base=zero,
        values=(0.0, 0.0, 0.0, 0.0, 0.0),
        task=torch.ones(5, dtype=torch.float32),
    )
    alignment = result.to_dict()["per_group"]["P0"][
        "conditional_entropy_task_alignment"
    ]["full_entropy_mean"]
    assert alignment["estimable"] is True
    assert alignment["entropy_gradient_norm"] == 0.0
    assert alignment["entropy_task_dot"] == 0.0
    assert alignment["entropy_task_cosine"] is None
    assert alignment["task_projection"] == 0.0
    assert alignment["unit_descent_task_change"] is None
    assert alignment["direction"] == "undefined"
    assert alignment["cosine_status"] == "not_estimable_entropy_gradient_zero"
    json.dumps(result.to_dict(), allow_nan=False)


@pytest.mark.parametrize(
    ("target", "message"),
    [
        (torch.zeros(1, 1, 2, 2, dtype=torch.uint8), "floating-point"),
        (
            torch.tensor([[[[0.0, -0.1], [0.0, 0.0]]]], dtype=torch.float32),
            r"closed interval \[0,1\]",
        ),
        (
            torch.tensor([[[[0.0, 1.1], [0.0, 0.0]]]], dtype=torch.float32),
            r"closed interval \[0,1\]",
        ),
        (
            torch.tensor([[[[0.0, float("nan")], [0.0, 0.0]]]], dtype=torch.float32),
            "finite",
        ),
        (torch.zeros(1, 1, 3, 2, dtype=torch.float32), "shapes must match"),
    ],
)
def test_invalid_targets_fail_closed(target: Tensor, message: str) -> None:
    parameters, layout, logits, parent = _graph()
    with pytest.raises(ForegroundBackgroundGradientError, match=message):
        analyze_foreground_background_gradient_decomposition(
            source_logits=logits,
            target=target,
            named_parameters=parameters,
            parameter_layout=layout,
            group_parameter_names=_groups(),
            parent_entropy_gradient_flat=parent,
            task_gradient_flat=torch.ones(5),
        )


def test_target_requiring_gradient_is_rejected() -> None:
    parameters, layout, logits, parent = _graph()
    target = torch.zeros(1, 1, 2, 2, dtype=torch.float32, requires_grad=True)
    with pytest.raises(ForegroundBackgroundGradientError, match="must not require"):
        analyze_foreground_background_gradient_decomposition(
            source_logits=logits,
            target=target,
            named_parameters=parameters,
            parameter_layout=layout,
            group_parameter_names=_groups(),
            parent_entropy_gradient_flat=parent,
            task_gradient_flat=torch.ones(5),
        )


def test_autograd_grad_does_not_write_or_replace_parameter_grad_slots() -> None:
    parameters, layout, logits, parent = _graph()
    parameters["p0"].grad = torch.tensor([7.0], dtype=torch.float32)
    grad_objects = {name: parameter.grad for name, parameter in parameters.items()}
    versions = {name: parameter._version for name, parameter in parameters.items()}
    result = analyze_foreground_background_gradient_decomposition(
        source_logits=logits,
        target=torch.tensor([[[[0.0, 1.0], [0.5, 0.0]]]], dtype=torch.float32),
        named_parameters=parameters,
        parameter_layout=layout,
        group_parameter_names=_groups(),
        parent_entropy_gradient_flat=parent,
        task_gradient_flat=torch.ones(5),
        config=GradientDecompositionConfig(),
    )
    assert parameters["p0"].grad is grad_objects["p0"]
    assert torch.equal(parameters["p0"].grad, torch.tensor([7.0]))
    for name in NAMES[1:]:
        assert parameters[name].grad is None
        assert grad_objects[name] is None
    assert {name: parameter._version for name, parameter in parameters.items()} == versions
    assert result.to_dict()["side_effects"] == {
        "uses_filesystem": False,
        "uses_optimizer": False,
        "calls_backward": False,
        "parameter_versions_unchanged": True,
        "parameter_grad_slots_unchanged": True,
    }


def test_parent_entropy_mismatch_and_invalid_group_schema_fail_closed() -> None:
    parameters, layout, logits, parent = _graph()
    target = torch.zeros(1, 1, 2, 2, dtype=torch.float32)
    with pytest.raises(ForegroundBackgroundGradientError, match="parent evidence"):
        analyze_foreground_background_gradient_decomposition(
            source_logits=logits,
            target=target,
            named_parameters=parameters,
            parameter_layout=layout,
            group_parameter_names=_groups(),
            parent_entropy_gradient_flat=parent + 1.0,
            task_gradient_flat=torch.ones(5),
        )

    parameters, layout, logits, parent = _graph()
    groups = _groups()
    groups["P1"] = ("p1",)
    with pytest.raises(ForegroundBackgroundGradientError, match="P1 is not in P2"):
        analyze_foreground_background_gradient_decomposition(
            source_logits=logits,
            target=target,
            named_parameters=parameters,
            parameter_layout=layout,
            group_parameter_names=groups,
            parent_entropy_gradient_flat=parent,
            task_gradient_flat=torch.ones(5),
        )


def test_cross_region_and_task_projection_metrics_are_recorded() -> None:
    _, result = _analyze(
        torch.tensor([[[[0.0, 0.25], [0.75, 0.0]]]], dtype=torch.float32)
    )
    p0 = result.to_dict()["per_group"]["P0"]
    for collection in (
        p0["additive_entropy_task_alignment"],
        p0["conditional_entropy_task_alignment"],
    ):
        for metrics in collection.values():
            assert "task_projection" in metrics
            assert "unit_descent_task_change" in metrics
            assert metrics["direction"] in {
                "beneficial",
                "harmful",
                "neutral",
                "undefined",
            }
    cross = p0["cross_region"]
    assert cross["foreground_background_additive_alignment"][
        "foreground_background_cosine"
    ] is not None
    assert cross["background_to_foreground_additive_norm_ratio"]["value"] >= 0
    assert cross["background_to_foreground_conditional_norm_ratio"]["value"] >= 0
    cancellation = cross["projection_cancellation_ratio"]
    assert cancellation["status"] == "estimable"
    assert 0.0 <= cancellation["value"] <= 1.0
