from __future__ import annotations

import json

import pytest
import torch
from torch import Tensor, nn

from analysis.d0_v3_outer_analyzer import FlatParameterLayout
import analysis.foreground_background_gradient_decomposition_v2 as module
from analysis.foreground_background_gradient_decomposition_v2 import (
    AUDIT_ONLY_RAW_VJPS,
    BACKWARD_BASIS,
    CUMULATIVE_ENDPOINTS,
    DIRECT_VECTOR_FIELDS,
    SCIENTIFIC_BASIS,
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
        task = torch.tensor([0.5, -0.4, 0.3, -0.2, 0.1])
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


def test_five_vjps_have_frozen_order_and_direct_vectors_are_exposed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, bool]] = []
    original = module._v1._autograd_flat

    def recording_autograd_flat(*args, **kwargs):
        calls.append((kwargs["label"], kwargs["retain_graph"]))
        return original(*args, **kwargs)

    monkeypatch.setattr(module._v1, "_autograd_flat", recording_autograd_flat)
    _, result = _analyze(
        torch.tensor([[[[0.0, 0.25], [0.75, 0.0]]]], dtype=torch.float32)
    )
    assert calls == [
        (BACKWARD_BASIS[0], True),
        (BACKWARD_BASIS[1], True),
        (BACKWARD_BASIS[2], True),
        (BACKWARD_BASIS[3], True),
        (BACKWARD_BASIS[4], False),
    ]
    vectors = result.vectors
    assert tuple(DIRECT_VECTOR_FIELDS) == (
        "full_direct",
        "foreground_total_direct",
        "foreground_subthreshold_direct",
        "foreground_suprathreshold_raw_direct",
        "background_raw_direct",
    )
    for field in DIRECT_VECTOR_FIELDS:
        value = getattr(vectors, field)
        assert value.device.type == "cpu"
        assert value.dtype == torch.float64
        assert value.is_contiguous()
        assert not value.requires_grad
    assert torch.equal(vectors.full_add, vectors.full_direct)
    assert torch.equal(vectors.foreground_add, vectors.foreground_total_direct)
    assert torch.equal(
        vectors.foreground_subthreshold_add,
        vectors.foreground_subthreshold_direct,
    )


def test_cumulative_scientific_basis_and_float32_roundtrip_close() -> None:
    _, result = _analyze(
        torch.tensor([[[[0.0, 0.25], [0.75, 0.0]]]], dtype=torch.float32)
    )
    vectors = result.vectors
    assert torch.equal(
        vectors.foreground_suprathreshold_add,
        vectors.foreground_total_direct
        - vectors.foreground_subthreshold_direct,
    )
    assert torch.equal(
        vectors.background_add,
        vectors.full_direct - vectors.foreground_total_direct,
    )
    assert torch.equal(
        vectors.foreground_add,
        vectors.foreground_subthreshold_add
        + vectors.foreground_suprathreshold_add,
    )
    assert torch.equal(
        vectors.full_add, vectors.foreground_add + vectors.background_add
    )
    report = result.to_dict()
    audit = report["numeric_audit"]
    assert audit["evaluation_scheme"] == "cumulative_telescoping_cpu_float64_v2"
    assert tuple(audit["direct_vjp_order"]) == BACKWARD_BASIS
    assert tuple(audit["direct_vector_fields"]) == DIRECT_VECTOR_FIELDS
    assert tuple(audit["scientific_basis_order"]) == SCIENTIFIC_BASIS
    assert audit["direct_vjp_count"] == 5
    assert audit["full_add_is_direct_vjp"] is True
    for section in (
        "cumulative_float64_closure",
        "float32_storage_roundtrip_closure",
        "float32_storage_reconstruction_vs_live_direct",
    ):
        assert tuple(audit[section]) == CUMULATIVE_ENDPOINTS
        for metrics in audit[section].values():
            assert metrics["verified"] is True
            assert metrics["max_abs_tolerance"] == 1.0e-7
            assert metrics["relative_l2_tolerance"] == 1.0e-4
    assert tuple(audit["float32_direct_vjp_storage_drift"]) == (
        DIRECT_VECTOR_FIELDS
    )
    assert all(
        metrics["verified"] is True
        for metrics in audit["float32_direct_vjp_storage_drift"].values()
    )


def test_raw_vjps_are_audit_only_and_cannot_change_scientific_vectors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = module._v1._autograd_flat

    def perturb_only_raw_audits(*args, **kwargs):
        value = original(*args, **kwargs)
        if kwargs["label"] in BACKWARD_BASIS[3:]:
            value = value + 0.25
        return value

    monkeypatch.setattr(module._v1, "_autograd_flat", perturb_only_raw_audits)
    _, result = _analyze(
        torch.tensor([[[[0.0, 0.25], [0.75, 0.0]]]], dtype=torch.float32)
    )
    vectors = result.vectors
    assert torch.equal(
        vectors.foreground_suprathreshold_add,
        vectors.foreground_total_direct
        - vectors.foreground_subthreshold_direct,
    )
    assert torch.equal(
        vectors.background_add,
        vectors.full_direct - vectors.foreground_total_direct,
    )
    raw = result.to_dict()["numeric_audit"]["raw_vjp_residual_audit"]
    assert tuple(raw["global"]) == (
        "foreground_suprathreshold_raw_vs_derived",
        "background_raw_vs_derived",
        "raw_foreground_sum_vs_direct_foreground",
        "raw_independent_three_sum_vs_direct_full",
    )
    assert raw["role"] == "audit_only_backend_reduction_nonclosure"
    assert raw["used_by_scientific_metrics"] is False
    assert raw["used_by_returned_vectors"] is False
    assert raw["failure_does_not_reject_scientific_cumulative_basis"] is True
    assert tuple(AUDIT_ONLY_RAW_VJPS) == (
        "foreground_suprathreshold_add_raw",
        "background_add_raw",
    )
    assert all(
        metrics["within_frozen_tolerance"] is False
        and metrics["audit_only"] is True
        and metrics["used_by_scientific_metrics"] is False
        for metrics in raw["global"].values()
    )


def test_raw_audit_relative_l2_direction_uses_raw_direct_reference(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = module._v1._autograd_flat

    def make_raw_norms_distinct(*args, **kwargs):
        value = original(*args, **kwargs)
        if kwargs["label"] == BACKWARD_BASIS[3]:
            return value + 0.25
        if kwargs["label"] == BACKWARD_BASIS[4]:
            return value - 0.5
        return value

    monkeypatch.setattr(module._v1, "_autograd_flat", make_raw_norms_distinct)
    _, result = _analyze(
        torch.tensor([[[[0.0, 0.25], [0.75, 0.0]]]], dtype=torch.float32)
    )
    vectors = result.vectors
    audit = result.to_dict()["numeric_audit"]["raw_vjp_residual_audit"]

    raw_supra = vectors.foreground_suprathreshold_raw_direct
    derived_supra = vectors.foreground_suprathreshold_add
    supra_metrics = audit["global"][
        "foreground_suprathreshold_raw_vs_derived"
    ]
    assert supra_metrics["reference_l2"] == pytest.approx(
        torch.linalg.vector_norm(raw_supra).item()
    )
    assert supra_metrics["reconstructed_l2"] == pytest.approx(
        torch.linalg.vector_norm(derived_supra).item()
    )

    raw_background = vectors.background_raw_direct
    derived_background = vectors.background_add
    background_metrics = audit["global"]["background_raw_vs_derived"]
    assert background_metrics["reference_l2"] == pytest.approx(
        torch.linalg.vector_norm(raw_background).item()
    )
    assert background_metrics["reconstructed_l2"] == pytest.approx(
        torch.linalg.vector_norm(derived_background).item()
    )

    raw_foreground_sum = vectors.foreground_subthreshold_add + raw_supra
    raw_full_sum = raw_foreground_sum + raw_background
    raw_foreground_metrics = audit["global"][
        "raw_foreground_sum_vs_direct_foreground"
    ]
    raw_full_metrics = audit["global"][
        "raw_independent_three_sum_vs_direct_full"
    ]
    assert raw_foreground_metrics["reference_l2"] == pytest.approx(
        torch.linalg.vector_norm(raw_foreground_sum).item()
    )
    assert raw_foreground_metrics["reconstructed_l2"] == pytest.approx(
        torch.linalg.vector_norm(vectors.foreground_total_direct).item()
    )
    assert raw_full_metrics["reference_l2"] == pytest.approx(
        torch.linalg.vector_norm(raw_full_sum).item()
    )
    assert raw_full_metrics["reconstructed_l2"] == pytest.approx(
        torch.linalg.vector_norm(vectors.full_direct).item()
    )
    # P0 covers every scalar and therefore must use the same direction as the
    # global component audit.
    assert audit["per_group"]["P0"]["background_raw_vs_derived"][
        "reference_l2"
    ] == pytest.approx(background_metrics["reference_l2"])


def test_v1_scientific_report_paths_and_target_semantics_are_preserved() -> None:
    _, result = _analyze(
        torch.tensor([[[[0.0, 0.25], [0.75, 0.0]]]], dtype=torch.float32)
    )
    report = result.to_dict()
    stats = report["target_statistics"]
    assert stats["foreground_pixel_count"] == 2
    assert stats["background_pixel_count"] == 2
    assert stats["foreground_subthreshold_pixel_count"] == 1
    assert stats["foreground_suprathreshold_pixel_count"] == 1
    assert report["conventions"]["foreground_rule"] == "target>0"
    assert tuple(report["per_group"]) == ("P0", "P1", "P2", "P3", "P4")
    p0 = report["per_group"]["P0"]
    assert set(p0) == {
        "parameter_tensor_count",
        "parameter_scalar_count",
        "task_gradient_norm",
        "additive_entropy_task_alignment",
        "conditional_entropy_task_alignment",
        "additive_gradient_norms",
        "cross_region",
    }
    assert set(p0["additive_entropy_task_alignment"]) == {
        "full_entropy_mean",
        "foreground_entropy_add",
        "background_entropy_add",
        "foreground_subthreshold_entropy_add",
        "foreground_suprathreshold_entropy_add",
    }
    assert report["decomposition"]["parent_full_entropy_reconstruction"][
        "verified"
    ] is True
    assert json.dumps(report, allow_nan=False)


def test_empty_regions_remain_null_not_estimable() -> None:
    _, empty_foreground = _analyze(torch.zeros(1, 1, 2, 2))
    assert empty_foreground.vectors.foreground_conditional_mean is None
    assert empty_foreground.vectors.foreground_subthreshold_conditional_mean is None
    assert empty_foreground.vectors.foreground_suprathreshold_conditional_mean is None
    assert torch.equal(
        empty_foreground.vectors.full_add,
        empty_foreground.vectors.background_add,
    )
    group = empty_foreground.to_dict()["per_group"]["P0"]
    fg_alignment = group["conditional_entropy_task_alignment"][
        "foreground_entropy_mean"
    ]
    assert fg_alignment["estimable"] is False
    assert fg_alignment["not_estimable_reason"] == "empty_foreground"

    _, empty_background = _analyze(torch.ones(1, 1, 2, 2))
    assert empty_background.vectors.background_conditional_mean is None
    assert torch.equal(
        empty_background.vectors.full_add,
        empty_background.vectors.foreground_add,
    )
    bg_alignment = empty_background.to_dict()["per_group"]["P0"][
        "conditional_entropy_task_alignment"
    ]["background_entropy_mean"]
    assert bg_alignment["estimable"] is False
    assert bg_alignment["not_estimable_reason"] == "empty_background"


def test_parameter_grad_slots_and_versions_are_unchanged() -> None:
    parameters, layout, logits, parent = _graph()
    parameters["p0"].grad = torch.tensor([7.0])
    grad_objects = {name: value.grad for name, value in parameters.items()}
    versions = {name: value._version for name, value in parameters.items()}
    result = analyze_foreground_background_gradient_decomposition(
        source_logits=logits,
        target=torch.tensor([[[[0.0, 1.0], [0.5, 0.0]]]]),
        named_parameters=parameters,
        parameter_layout=layout,
        group_parameter_names=_groups(),
        parent_entropy_gradient_flat=parent,
        task_gradient_flat=torch.ones(5),
        config=GradientDecompositionConfig(),
    )
    assert parameters["p0"].grad is grad_objects["p0"]
    assert torch.equal(parameters["p0"].grad, torch.tensor([7.0]))
    assert all(parameters[name].grad is None for name in NAMES[1:])
    assert {name: value._version for name, value in parameters.items()} == versions
    assert result.to_dict()["side_effects"] == {
        "uses_filesystem": False,
        "uses_optimizer": False,
        "calls_backward": False,
        "parameter_versions_unchanged": True,
        "parameter_grad_slots_unchanged": True,
    }


def test_parent_interface_still_fails_closed_on_mismatch() -> None:
    parameters, layout, logits, parent = _graph()
    with pytest.raises(ForegroundBackgroundGradientError, match="parent evidence"):
        analyze_foreground_background_gradient_decomposition(
            source_logits=logits,
            target=torch.zeros(1, 1, 2, 2),
            named_parameters=parameters,
            parameter_layout=layout,
            group_parameter_names=_groups(),
            parent_entropy_gradient_flat=parent + 1.0,
            task_gradient_flat=torch.ones(5),
        )


@pytest.mark.parametrize(
    ("target", "message"),
    [
        (torch.zeros(1, 1, 2, 2, dtype=torch.uint8), "floating-point"),
        (
            torch.tensor([[[[0.0, -0.1], [0.0, 0.0]]]], dtype=torch.float32),
            r"closed interval \[0,1\]",
        ),
        (torch.zeros(1, 1, 3, 2), "shapes must match"),
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
