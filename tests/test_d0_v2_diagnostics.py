from __future__ import annotations

from copy import deepcopy

import numpy as np
import pytest

from analysis.d0_v2_diagnostics import (
    D0V2DiagnosticError,
    D0_V2_NOOP_ARTIFACT_TYPE,
    D0_V2_NOOP_REPORT_FIELDS,
    MARGIN_BIN_KEYS,
    THRESHOLD_RULE,
    analyze_noop_episode_v2,
    analyze_noop_episode_v2_from_mapping,
    validate_noop_episode_v2_report,
)
from tta.diagnostics import NoOpThresholds


def _logit(probability: np.ndarray) -> np.ndarray:
    probability = np.asarray(probability, dtype=np.float64)
    return np.log(probability / (1.0 - probability))


def _parameters() -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    before = {"bn.weight": np.array([1.0, 2.0], dtype=np.float32)}
    after = {"bn.weight": np.array([1.0, 2.125], dtype=np.float32)}
    return before, after


def _report(
    pre: np.ndarray,
    post: np.ndarray,
    *,
    thresholds: NoOpThresholds | None = None,
) -> dict[str, object]:
    parameter_pre, parameter_post = _parameters()
    return analyze_noop_episode_v2(
        parameter_pre=parameter_pre,
        parameter_post=parameter_post,
        logits_pre=_logit(pre),
        logits_post=_logit(post),
        probability_pre=np.asarray(pre, dtype=np.float64),
        probability_post=np.asarray(post, dtype=np.float64),
        target=np.zeros_like(pre, dtype=np.float32),
        thresholds=thresholds or NoOpThresholds(),
    )


def test_v2_adds_p95_mass_sum_and_largest_component_geometry() -> None:
    pre = np.array(
        [
            [0.60, 0.60, 0.10],
            [0.60, 0.50, 0.10],
            [0.10, 0.51, 0.51],
        ],
        dtype=np.float64,
    )
    post = np.array(
        [
            [0.60, 0.10, 0.60],
            [0.60, 0.50, 0.60],
            [0.10, 0.60, 0.60],
        ],
        dtype=np.float64,
    )

    report = _report(
        pre,
        post,
        thresholds=NoOpThresholds(connectivity=1, min_component_area=2),
    )

    assert set(report) == D0_V2_NOOP_REPORT_FIELDS
    assert report["artifact_type"] == D0_V2_NOOP_ARTIFACT_TYPE
    assert report["schema_version"] == 2
    assert report["threshold_rule_v2"] == THRESHOLD_RULE
    assert report["p95_abs_delta_logit"] == pytest.approx(
        np.quantile(np.abs(_logit(post) - _logit(pre)), 0.95, method="linear")
    )
    assert report["p95_abs_delta_probability"] == pytest.approx(
        np.quantile(np.abs(post - pre), 0.95, method="linear")
    )
    assert report["foreground_probability_mass_pre"] == pytest.approx(pre.sum())
    assert report["foreground_probability_mass_post"] == pytest.approx(post.sum())
    assert report["foreground_probability_mass_delta"] == pytest.approx(
        post.sum() - pre.sum()
    )
    assert report["foreground_probability_mass_pre"] == report["state"]["pre"][
        "foreground_probability_mass_sum"
    ]
    assert report["foreground_probability_mass_pre"] != report["state"]["pre"][
        "foreground_probability_mass_mean"
    ]
    assert report["largest_component_area_pre"] == 3
    assert report["largest_component_area_post"] == 4
    assert report["largest_component_area_delta"] == 1


def test_margin_bins_are_disjoint_exhaustive_and_publish_outside_count() -> None:
    pre = np.array(
        [[0.5, 0.50005, 0.5005, 0.505, 0.52, 0.55, 0.10]],
        dtype=np.float64,
    )
    report = _report(pre, pre.copy())

    bins = report["near_threshold_pixel_count_by_margin_bin"]
    assert tuple(bins) == MARGIN_BIN_KEYS
    assert bins == {
        "ge_0_lt_1e_minus_4": 2,
        "ge_1e_minus_4_lt_1e_minus_3": 1,
        "ge_1e_minus_3_lt_1e_minus_2": 1,
        "ge_1e_minus_2_lt_5e_minus_2": 1,
    }
    assert report["threshold_margin_outside_ge_5e_minus_2_pixel_count"] == 2
    assert report["threshold_margin_partition_pixel_count"] == pre.size
    assert sum(bins.values()) + report[
        "threshold_margin_outside_ge_5e_minus_2_pixel_count"
    ] == pre.size


def test_prediction_rule_is_strictly_greater_than_point_five() -> None:
    pre = np.array([[0.5, 0.49]], dtype=np.float64)
    post = np.array([[0.5001, 0.49]], dtype=np.float64)

    report = _report(pre, post)

    assert report["state"]["pre"]["foreground_pixel_count"] == 0
    assert report["state"]["post"]["foreground_pixel_count"] == 1
    assert report["BG_to_FG_pixel_count"] == 1
    assert report["binary_pixel_xor_count"] == 1


def test_connectivity_and_min_area_come_from_frozen_thresholds() -> None:
    diagonal = np.array([[0.6, 0.1], [0.1, 0.6]], dtype=np.float64)

    four_neighbour = _report(
        diagonal,
        diagonal,
        thresholds=NoOpThresholds(connectivity=1, min_component_area=1),
    )
    eight_neighbour = _report(
        diagonal,
        diagonal,
        thresholds=NoOpThresholds(connectivity=2, min_component_area=1),
    )
    filtered = _report(
        diagonal,
        diagonal,
        thresholds=NoOpThresholds(connectivity=1, min_component_area=2),
    )

    assert four_neighbour["component_count_pre"] == 2
    assert four_neighbour["largest_component_area_pre"] == 1
    assert eight_neighbour["component_count_pre"] == 1
    assert eight_neighbour["largest_component_area_pre"] == 2
    assert filtered["component_count_pre"] == 0
    assert filtered["largest_component_area_pre"] == 0


def test_thresholds_are_mandatory_frozen_objects_at_exactly_point_five() -> None:
    pre = np.full((2, 2), 0.4, dtype=np.float64)
    parameter_pre, parameter_post = _parameters()
    kwargs = {
        "parameter_pre": parameter_pre,
        "parameter_post": parameter_post,
        "logits_pre": _logit(pre),
        "logits_post": _logit(pre),
        "target": np.zeros_like(pre),
    }

    with pytest.raises(D0V2DiagnosticError, match="exactly 0.5"):
        analyze_noop_episode_v2(
            **kwargs, thresholds=NoOpThresholds(prediction_threshold=0.51)
        )
    with pytest.raises(D0V2DiagnosticError, match="frozen NoOpThresholds"):
        analyze_noop_episode_v2(**kwargs, thresholds={})  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("field", "bad_value"),
    (("logits_pre", np.nan), ("logits_post", np.inf)),
)
def test_nonfinite_logit_inputs_fail_closed(field: str, bad_value: float) -> None:
    pre = np.full((2, 2), 0.4, dtype=np.float64)
    parameter_pre, parameter_post = _parameters()
    values: dict[str, object] = {
        "parameter_pre": parameter_pre,
        "parameter_post": parameter_post,
        "logits_pre": _logit(pre),
        "logits_post": _logit(pre),
        "target": np.zeros_like(pre),
        "thresholds": NoOpThresholds(),
    }
    values[field] = np.full((2, 2), bad_value)

    with pytest.raises(D0V2DiagnosticError, match="NaN or Inf"):
        analyze_noop_episode_v2(**values)  # type: ignore[arg-type]


def test_nonfinite_parameter_and_inconsistent_probability_fail_closed() -> None:
    pre = np.full((2, 2), 0.4, dtype=np.float64)
    before, after = _parameters()
    before["bn.weight"][0] = np.nan
    with pytest.raises(D0V2DiagnosticError, match="NaN or Inf"):
        analyze_noop_episode_v2(
            parameter_pre=before,
            parameter_post=after,
            logits_pre=_logit(pre),
            logits_post=_logit(pre),
            target=np.zeros_like(pre),
            thresholds=NoOpThresholds(),
        )

    before, after = _parameters()
    with pytest.raises(D0V2DiagnosticError, match="do not match sigmoid"):
        analyze_noop_episode_v2(
            parameter_pre=before,
            parameter_post=after,
            logits_pre=_logit(pre),
            logits_post=_logit(pre),
            probability_pre=np.full((2, 2), 0.9),
            probability_post=np.full((2, 2), 0.9),
            target=np.zeros_like(pre),
            thresholds=NoOpThresholds(),
        )


def test_exact_input_mapping_accepts_only_complete_probability_pair() -> None:
    pre = np.full((2, 2), 0.4, dtype=np.float64)
    before, after = _parameters()
    value: dict[str, object] = {
        "parameter_pre": before,
        "parameter_post": after,
        "logits_pre": _logit(pre),
        "logits_post": _logit(pre),
        "target": np.zeros_like(pre),
        "thresholds": NoOpThresholds().to_dict(),
    }

    assert analyze_noop_episode_v2_from_mapping(value)["schema_version"] == 2
    with pytest.raises(D0V2DiagnosticError, match="exact schema"):
        analyze_noop_episode_v2_from_mapping({**value, "unknown": 1})
    with pytest.raises(D0V2DiagnosticError, match="exact schema"):
        analyze_noop_episode_v2_from_mapping(
            {**value, "probability_pre": pre.copy()}
        )


@pytest.mark.parametrize(
    "mutation",
    (
        "unknown_top_level",
        "unknown_nested",
        "nonfinite",
        "partition_mismatch",
        "mass_mean_substitution",
        "area_delta_mismatch",
    ),
)
def test_output_verifier_fails_closed(mutation: str) -> None:
    pre = np.array([[0.4, 0.5], [0.6, 0.7]], dtype=np.float64)
    post = np.array([[0.45, 0.51], [0.55, 0.7]], dtype=np.float64)
    value = deepcopy(_report(pre, post))
    if mutation == "unknown_top_level":
        value["unknown"] = 1
    elif mutation == "unknown_nested":
        value["state"]["pre"]["unknown"] = 1
    elif mutation == "nonfinite":
        value["p95_abs_delta_logit"] = float("nan")
    elif mutation == "partition_mismatch":
        value["threshold_margin_partition_pixel_count"] += 1
    elif mutation == "mass_mean_substitution":
        value["foreground_probability_mass_pre"] = value["state"]["pre"][
            "foreground_probability_mass_mean"
        ]
    elif mutation == "area_delta_mismatch":
        value["largest_component_area_delta"] += 1
    else:  # pragma: no cover - parametrization is frozen above.
        raise AssertionError(mutation)

    with pytest.raises(D0V2DiagnosticError):
        validate_noop_episode_v2_report(value)


def test_analysis_is_deterministic_and_does_not_mutate_inputs() -> None:
    pre = np.array([[0.49, 0.51], [0.10, 0.90]], dtype=np.float64)
    post = np.array([[0.48, 0.52], [0.20, 0.80]], dtype=np.float64)
    before, after = _parameters()
    snapshots = (
        pre.copy(),
        post.copy(),
        before["bn.weight"].copy(),
        after["bn.weight"].copy(),
    )
    kwargs = {
        "parameter_pre": before,
        "parameter_post": after,
        "logits_pre": _logit(pre),
        "logits_post": _logit(post),
        "probability_pre": pre,
        "probability_post": post,
        "target": np.zeros_like(pre),
        "thresholds": NoOpThresholds(),
    }

    first = analyze_noop_episode_v2(**kwargs)
    second = analyze_noop_episode_v2(**kwargs)

    assert first == second
    np.testing.assert_array_equal(pre, snapshots[0])
    np.testing.assert_array_equal(post, snapshots[1])
    np.testing.assert_array_equal(before["bn.weight"], snapshots[2])
    np.testing.assert_array_equal(after["bn.weight"], snapshots[3])
