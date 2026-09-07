"""Synthetic count-only checks; no project experiment artifacts are read."""

import copy

import pytest

from analysis import o3_multiscale_guard_metrics_v2 as metrics


def _endpoint(tp=840, fp=180, detected=82, fa=30, target=1000):
    total = 8 * 256 * 256
    fn = target - tp
    union = tp + fp + fn
    return {
        "iou": tp / union, "normalized_iou": tp / union,
        "pd": detected / 100, "fa_per_million": fa / total * 1e6,
        "foreground_fraction": (tp + fp) / total,
        "intersection_pixels": tp, "false_positive_pixels": fp,
        "false_negative_pixels": fn, "true_negative_pixels": total - tp - fp - fn,
        "predicted_positive_pixels": tp + fp, "target_positive_pixels": target,
        "detected_targets": detected, "total_targets": 100,
        "false_alarm_pixels": fa, "total_image_pixels": total, "image_count": 8,
    }


def _cells():
    return [{"condition": metrics.b4._condition_key(corruption, severity),
             "corruption": corruption, "severity": severity,
             "source": _endpoint(tp=790, fp=200, detected=75, fa=60),
             "o3": _endpoint(tp=800, fp=200, detected=80, fa=50),
             "previous": _endpoint(tp=820, fp=200, detected=80, fa=40),
             "guarded": _endpoint()}
            for corruption, severity in metrics.CONDITIONS]


def _losses(segmentation, guard=0.01):
    return {"segmentation_full_fit_loss": segmentation,
            "background_guard_full_fit_loss": guard,
            "augmented_full_fit_loss": segmentation + guard}


def _summarize(cells=None, initial=None, final=None):
    return metrics.summarize(_cells() if cells is None else cells,
                             _losses(0.5) if initial is None else initial,
                             _losses(0.4) if final is None else final)


def test_expected_goals_deltas_and_scope():
    cells = _cells()
    result = _summarize(cells)
    assert result["condition_count"] == 13
    assert result["nonclean_condition_count"] == 12
    assert result["family_condition_counts"] == {family: 3 for family in metrics.FAMILIES}
    assert result["goal_count"] == result["passed_goal_count"] == 13
    assert result["all_goals_met"]
    assert result["failed_goals"] == []
    assert result["learning_check_passed"]
    assert result["fit_performance_signal_passed"]
    guarded, previous = cells[1]["guarded"], cells[1]["previous"]
    delta = result["deltas"]["nonclean"]["vs_previous"]
    assert delta["iou_pp"] == pytest.approx((guarded["iou"] - previous["iou"]) * 100)
    assert delta["normalized_iou_pp"] == pytest.approx(delta["iou_pp"])
    assert delta["pd_pp"] == pytest.approx(2.0)
    assert delta["fa_per_million"] == pytest.approx(guarded["fa_per_million"] - previous["fa_per_million"])
    for key in ("generalization_claim", "statistical_significance_claim", "paper_result", "formal_test",
                "automatic_full_training_allowed", "automatic_formal_test_allowed",
                "automatic_retry_or_hyperparameter_search_allowed"):
        assert result[key] is False
    assert result["fit_and_measure_on_same_images"] is True
    assert result["no_validation_split"] is True


def test_clean_excluded_from_nonclean_macro_and_reported_with_counts():
    cells = _cells()
    cells[0]["guarded"] = _endpoint(tp=950, fp=10, detected=99, fa=1)
    result = _summarize(cells)
    assert result["nonclean_macro"]["guarded"]["iou"] == pytest.approx(cells[1]["guarded"]["iou"])
    assert result["all_conditions_macro_descriptive_only"]["guarded"]["iou"] > result["nonclean_macro"]["guarded"]["iou"]
    assert result["clean"]["guarded"]["false_alarm_pixels"] == 1
    assert result["clean"]["condition"] == "clean_S0"


def test_arbitrary_input_order_is_canonicalized_without_mutation():
    cells = list(reversed(_cells()))
    before = copy.deepcopy(cells)
    result = _summarize(cells)
    assert cells == before
    assert result == _summarize()


def test_nonclean_iou_must_strictly_exceed_previous():
    cells = _cells()
    for row in cells[1:]:
        row["guarded"] = copy.deepcopy(row["previous"])
    result = _summarize(cells)
    assert result["failed_goals"] == ["nonclean_iou_above_previous"]
    assert not result["all_goals_met"]


@pytest.mark.parametrize("metric,endpoint,goal", [
    ("pd", _endpoint(detected=79), "nonclean_pd_at_least_previous"),
    ("fa", _endpoint(fa=41), "nonclean_fa_at_most_previous"),
])
def test_nonclean_pd_fa_constraints(metric, endpoint, goal):
    cells = _cells()
    for row in cells[1:]:
        row["guarded"] = dict(endpoint)
    result = _summarize(cells)
    assert not result["goals"][goal]
    assert result["goals"]["nonclean_iou_above_previous"]


@pytest.mark.parametrize("family", ["low_contrast", "stripe_noise"])
def test_existing_family_iou_must_be_preserved(family):
    cells = _cells()
    for row in cells:
        if row["corruption"] == family:
            row["guarded"] = _endpoint(tp=810, fp=200)
    result = _summarize(cells)
    assert not result["goals"][f"{family}_iou_at_least_previous"]


@pytest.mark.parametrize("family", ["gaussian_blur", "gaussian_noise"])
def test_gaussian_family_equality_to_o3_passes_but_degradation_fails(family):
    cells = _cells()
    for row in cells:
        if row["corruption"] == family:
            row["guarded"] = dict(row["o3"])
    result = _summarize(cells)
    assert result["goals"][f"{family}_iou_at_least_o3"]
    assert result["goals"][f"{family}_fa_at_most_o3"]
    for row in cells:
        if row["corruption"] == family:
            row["guarded"] = _endpoint(tp=790, fp=200, fa=51)
    result = _summarize(cells)
    assert not result["goals"][f"{family}_iou_at_least_o3"]
    assert not result["goals"][f"{family}_fa_at_most_o3"]


@pytest.mark.parametrize("endpoint,goal", [
    (_endpoint(tp=790, fp=200), "clean_iou_at_least_o3"),
    (_endpoint(detected=79), "clean_pd_at_least_o3"),
    (_endpoint(fa=51), "clean_fa_at_most_o3"),
])
def test_clean_metrics_are_independent_constraints(endpoint, goal):
    cells = _cells()
    cells[0]["guarded"] = endpoint
    result = _summarize(cells)
    assert result["failed_goals"] == [goal]
    assert result["passed_goal_count"] == 12


def test_only_augmented_loss_controls_learning_check():
    initial = _losses(0.5, 0.0)
    final = _losses(0.4, 0.11)
    result = _summarize(initial=initial, final=final)
    assert result["fit_performance_signal_passed"]
    assert not result["learning_check_passed"]
    assert result["failed_goals"] == ["augmented_full_fit_loss_decreased"]
    assert not result["all_goals_met"]


def test_learning_change_below_tolerance_is_not_an_improvement():
    initial, final = _losses(0.5, 0.0), _losses(0.5 - 0.5e-12, 0.0)
    assert not _summarize(initial=initial, final=final)["learning_check_passed"]
    final = _losses(0.5 - 2e-12, 0.0)
    assert _summarize(initial=initial, final=final)["learning_check_passed"]


@pytest.mark.parametrize("case", ["missing", "duplicate", "unknown", "severity", "bool_severity", "family", "dataset", "not_mapping"])
def test_invalid_condition_sets_rejected(case):
    cells = _cells()
    if case == "missing":
        cells.pop()
    elif case == "duplicate":
        cells[-1] = copy.deepcopy(cells[0])
    elif case == "unknown":
        cells[1]["condition"] = "gaussian_noise_S2"
    elif case == "severity":
        cells[1]["severity"] = 3
    elif case == "bool_severity":
        cells[1]["severity"] = True
    elif case == "family":
        cells[1]["corruption"] = "stripe_noise"
    elif case == "dataset":
        cells[0]["dataset"] = "IRSTD-1K"
    else:
        cells[0] = None
    with pytest.raises(metrics.GuardMetricsError):
        _summarize(cells)


@pytest.mark.parametrize("case", ["missing_method", "count", "float_count", "bool_count", "derived", "nonfinite", "niou", "image_count", "extra_field"])
def test_invalid_endpoint_rejected(case):
    cells = _cells()
    value = cells[1]["guarded"]
    if case == "missing_method":
        del cells[1]["guarded"]
    elif case == "count":
        value["intersection_pixels"] += 1
    elif case == "float_count":
        value["image_count"] = 8.0
    elif case == "bool_count":
        value["false_alarm_pixels"] = True
    elif case == "derived":
        value["iou"] += 0.01
    elif case == "nonfinite":
        value["iou"] = float("nan")
    elif case == "niou":
        value["normalized_iou"] = 1.1
    elif case == "image_count":
        value["image_count"] = 64
    else:
        value["not_an_endpoint_field"] = 0
    with pytest.raises(metrics.GuardMetricsError):
        _summarize(cells)


def test_target_accounting_must_match_across_methods_and_conditions():
    cells = _cells()
    cells[-1]["guarded"] = _endpoint(target=1001)
    with pytest.raises(metrics.GuardMetricsError, match="unchanged train8 GT"):
        _summarize(cells)


@pytest.mark.parametrize("case", ["missing", "unknown", "nan", "negative", "bool", "augmented_below_segmentation"])
def test_invalid_losses_rejected(case):
    initial = _losses(0.5)
    if case == "missing":
        del initial["augmented_full_fit_loss"]
    elif case == "unknown":
        initial["wrong_key"] = 0.0
    elif case == "nan":
        initial["segmentation_full_fit_loss"] = float("nan")
    elif case == "negative":
        initial["background_guard_full_fit_loss"] = -0.01
    elif case == "bool":
        initial["background_guard_full_fit_loss"] = False
    else:
        initial["augmented_full_fit_loss"] = 0.4
    with pytest.raises(metrics.GuardMetricsError):
        _summarize(initial=initial)


def test_completely_empty_foreground_uses_frozen_count_convention():
    cells = _cells()
    for row in cells:
        for method in metrics.METHODS:
            value = row[method]
            value.update({"intersection_pixels": 0, "false_positive_pixels": 0,
                "false_negative_pixels": 0, "true_negative_pixels": 8 * 256 * 256,
                "predicted_positive_pixels": 0, "target_positive_pixels": 0,
                "detected_targets": 0, "total_targets": 0, "false_alarm_pixels": 0,
                "iou": 1.0, "normalized_iou": 1.0, "pd": 0.0,
                "fa_per_million": 0.0, "foreground_fraction": 0.0})
    result = _summarize(cells)
    assert result["nonclean_macro"]["guarded"]["iou"] == 1.0
    assert not result["goals"]["nonclean_iou_above_previous"]
