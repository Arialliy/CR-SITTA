"""Synthetic masks and official v3 transitions only; no experiment reads."""

import copy
from functools import lru_cache

import numpy as np
import pytest

from analysis import o3_anchored_metrics_v4 as metrics
from analysis.o3_reachability_objects_v3 import object_transitions


def _target():
    gt = np.zeros((256, 256), dtype=bool)
    gt[20:30, 20:30] = True
    return gt


def _mask(rows=10, fp=1):
    value = np.zeros((256, 256), dtype=bool)
    value[20:20+rows, 20:30] = True
    value[60, 60:60+fp] = True
    return value


def _endpoint(mask, target=None):
    gt = _target() if target is None else target
    one = object_transitions(gt, mask, mask)["previous"]
    tp = int((gt & mask).sum()) * 8
    fp = int((~gt & mask).sum()) * 8
    fn = int((gt & ~mask).sum()) * 8
    total = 8 * 256 * 256
    union = tp + fp + fn
    iou = tp / union if union else 1.0
    return {"iou": iou, "normalized_iou": iou, "pd": one["pd"],
        "fa_per_million": one["false_alarm_pixels"] * 8 / total * 1e6,
        "foreground_fraction": (tp + fp) / total,
        "intersection_pixels": tp, "false_positive_pixels": fp, "false_negative_pixels": fn,
        "true_negative_pixels": total - tp - fp - fn, "predicted_positive_pixels": tp + fp,
        "target_positive_pixels": int(gt.sum()) * 8,
        "detected_targets": one["detected_targets"] * 8,
        "total_targets": (one["detected_targets"] + one["false_negative_targets"]) * 8,
        "false_alarm_pixels": one["false_alarm_pixels"] * 8,
        "total_image_pixels": total, "image_count": 8}


@lru_cache(maxsize=1)
def _base():
    masks = {"source": _mask(7, 4), "o3": _mask(7, 3), "v1": _mask(8, 3),
             "control": _mask(9, 2), "candidate": _mask(10, 1)}
    endpoints = {name: _endpoint(mask) for name, mask in masks.items()}
    arms = {arm: object_transitions(_target(), masks["v1"], masks[arm]) for arm in metrics.ARMS}
    cells, transitions = [], []
    for family, severity in metrics.CONDITIONS:
        condition = metrics.b4._condition_key(family, severity)
        cells.append({"dataset": metrics.DATASET, "condition": condition,
                      "corruption": family, "severity": severity, **copy.deepcopy(endpoints)})
        for identifier in metrics.ORDERED_IMAGE_IDS:
            transitions.append({"condition": condition, "image_id": identifier, **copy.deepcopy(arms)})
    return cells, transitions


def _bundle():
    return copy.deepcopy(_base())


def _summarize(bundle=None, initial=None, final=None):
    cells, transitions = _bundle() if bundle is None else bundle
    return metrics.summarize(cells, transitions,
        {"control": 0.4, "candidate": 0.4} if initial is None else initial,
        {"control": 0.35, "candidate": 0.3} if final is None else final)


def _replace_arm(bundle, arm, indexes, mask):
    cells, transitions = bundle
    for ci in indexes:
        cells[ci][arm] = _endpoint(mask)
        replacement = object_transitions(_target(), _mask(8, 3), mask)
        for ii in range(8):
            transitions[ci * 8 + ii][arm] = copy.deepcopy(replacement)


def test_all_goals_and_control_comparison_are_separate():
    result = _summarize()
    assert result["goal_count"] == result["passed_goal_count"] == 20
    assert result["all_goals_met"] and result["learning_checks_passed"] and result["fit_performance_signal_passed"]
    assert result["failed_goals"] == []
    means = result["nonclean_macro"]
    changes = result["deltas"]["nonclean"]
    assert changes["candidate_minus_v1"]["iou_pp"] == pytest.approx((means["candidate"]["iou"] - means["v1"]["iou"]) * 100)
    assert changes["candidate_minus_control"]["iou_pp"] == pytest.approx((means["candidate"]["iou"] - means["control"]["iou"]) * 100)
    assert changes["control_minus_v1"]["iou_pp"] > 0
    assert changes["candidate_minus_v1"]["iou_pp"] == pytest.approx(changes["candidate_minus_control"]["iou_pp"] + changes["control_minus_v1"]["iou_pp"])
    for arm in metrics.ARMS:
        assert result["target_transitions"][arm]["all"]["total_targets"] == 104
        assert result["target_transitions"][arm]["nonclean"]["total_targets"] == 96
        assert result["target_transitions"][arm]["all"]["transition_counts"]["TP→TP"] == 104
    for key in ("generalization_claim", "statistical_significance_claim", "paper_result", "formal_test",
                "automatic_full_training_allowed", "automatic_formal_test_allowed",
                "automatic_retry_or_hyperparameter_search_allowed", "engineering_tests_are_performance_evidence"):
        assert result[key] is False
    assert result["fit_and_measure_on_same_images"] is True


def test_inputs_are_not_mutated():
    bundle = _bundle()
    before = copy.deepcopy(bundle)
    _summarize(bundle)
    assert bundle == before


def test_clean_not_included_in_nonclean_mean():
    bundle = _bundle()
    _replace_arm(bundle, "candidate", [0], _mask(10, 0))
    result = _summarize(bundle)
    assert result["nonclean_macro"] == _summarize()["nonclean_macro"]
    assert result["clean"]["candidate"]["iou"] == 1.0
    assert result["all_conditions_macro_descriptive_only"]["candidate"]["iou"] > result["nonclean_macro"]["candidate"]["iou"]


def test_candidate_must_beat_extra_training_control_not_only_v1():
    bundle = _bundle()
    _replace_arm(bundle, "candidate", list(range(13)), _mask(9, 2))
    result = _summarize(bundle)
    assert result["goals"]["candidate_nonclean_iou_above_v1"]
    assert not result["goals"]["candidate_nonclean_iou_above_control"]
    assert result["failed_goals"] == ["candidate_nonclean_iou_above_control"]


def test_candidate_equal_v1_fails_strict_iou_improvement():
    bundle = _bundle()
    _replace_arm(bundle, "candidate", list(range(13)), _mask(8, 3))
    result = _summarize(bundle)
    assert not result["goals"]["candidate_nonclean_iou_above_v1"]
    assert result["goals"]["candidate_clean_iou_at_least_v1"]
    assert result["goals"]["candidate_no_v1_tp_to_fn_all_conditions"]


@pytest.mark.parametrize("arm", metrics.ARMS)
def test_learning_checks_do_not_substitute_for_performance(arm):
    final = {"control": 0.35, "candidate": 0.3}
    final[arm] = 0.4
    result = _summarize(final=final)
    assert result["fit_performance_signal_passed"]
    assert not result["learning_checks_passed"]
    assert not result["all_goals_met"]
    assert result["failed_goals"] == [f"{arm}_full_fit_loss_decreased"]


def test_learning_tolerance_requires_real_decrease():
    final = {"control": 0.4 - 0.5e-12, "candidate": 0.3}
    assert not _summarize(final=final)["learning_checks"]["control_full_fit_loss_decreased"]
    final["control"] = 0.4 - 2e-12
    assert _summarize(final=final)["learning_checks"]["control_full_fit_loss_decreased"]


@pytest.mark.parametrize("family", metrics.FAMILIES)
def test_every_family_iou_constraint_is_enforced(family):
    bundle = _bundle()
    indexes = [i for i, row in enumerate(bundle[0]) if row["corruption"] == family]
    _replace_arm(bundle, "candidate", indexes, _mask(7, 3))
    result = _summarize(bundle)
    assert not result["goals"][f"candidate_{family}_iou_at_least_v1"]


def test_family_fa_reported_without_an_unregistered_family_gate():
    bundle = _bundle()
    _replace_arm(bundle, "candidate", [1, 2, 3], _mask(10, 4))
    _replace_arm(bundle, "candidate", list(range(4, 13)), _mask(10, 0))
    result = _summarize(bundle)
    assert result["deltas"]["families"]["gaussian_noise"]["candidate_minus_v1"]["fa_per_million"] > 0
    assert result["all_goals_met"]
    assert not any("gaussian_noise_fa" in name for name in result["goals"])


def test_clean_fa_cannot_be_excused_by_nonclean_gains():
    bundle = _bundle()
    _replace_arm(bundle, "candidate", [0], _mask(10, 4))
    result = _summarize(bundle)
    assert result["failed_goals"] == ["candidate_clean_fa_at_most_v1"]


def test_lost_clean_target_counts_toward_global_target_erasure_gate():
    bundle = _bundle()
    _replace_arm(bundle, "candidate", [0], np.zeros((256, 256), dtype=bool))
    result = _summarize(bundle)
    assert result["target_transitions"]["candidate"]["all"]["transition_counts"]["TP→FN"] == 8
    assert result["target_transitions"]["candidate"]["nonclean"]["transition_counts"]["TP→FN"] == 0
    assert not result["goals"]["candidate_no_v1_tp_to_fn_all_conditions"]
    assert not result["goals"]["candidate_clean_pd_at_least_v1"]


def test_target_exchange_is_not_hidden_by_unchanged_pd():
    target = np.zeros((256, 256), dtype=bool)
    target[20, 20] = target[40, 40] = True
    previous = np.zeros_like(target);previous[20, 20] = True
    candidate = np.zeros_like(target);candidate[40, 40] = True
    cells, transitions = _bundle()
    old_endpoint, new_endpoint = _endpoint(previous, target), _endpoint(candidate, target)
    old_objects = object_transitions(target, previous, previous)
    new_objects = object_transitions(target, previous, candidate)
    for cell in cells:
        for name in metrics.METHODS:
            cell[name] = copy.deepcopy(new_endpoint if name == "candidate" else old_endpoint)
    for row in transitions:
        row["control"], row["candidate"] = copy.deepcopy(old_objects), copy.deepcopy(new_objects)
    result = _summarize((cells, transitions))
    assert result["goals"]["candidate_nonclean_pd_at_least_v1"]
    assert result["target_transitions"]["candidate"]["all"]["transition_counts"]["TP→FN"] == 104
    assert result["target_transitions"]["candidate"]["all"]["transition_counts"]["FN→TP"] == 104
    assert not result["goals"]["candidate_no_v1_tp_to_fn_all_conditions"]


@pytest.mark.parametrize("case", ["missing", "duplicate", "wrong_order", "dataset", "severity_bool", "bad_method", "count_drift", "missing_v1"])
def test_cell_identity_and_endpoint_validation(case):
    cells, transitions = _bundle()
    if case == "missing":
        cells.pop()
    elif case == "duplicate":
        cells[-1] = copy.deepcopy(cells[0])
    elif case == "wrong_order":
        cells[1], cells[2] = cells[2], cells[1]
    elif case == "dataset":
        cells[0]["dataset"] = "IRSTD-1K"
    elif case == "severity_bool":
        cells[1]["severity"] = True
    elif case == "bad_method":
        cells[0]["candidate"]["iou"] = float("nan")
    elif case == "count_drift":
        cells[0]["candidate"]["intersection_pixels"] += 1
    else:
        del cells[0]["v1"]
    with pytest.raises(metrics.AnchoredMetricsError):
        _summarize((cells, transitions))


@pytest.mark.parametrize("case", ["missing", "duplicate", "reorder", "wrong_id", "wrong_condition", "missing_arm", "bad_protocol", "bad_shape", "wrong_counts", "hit_flags", "distance_three", "bad_previous", "net_change_bool", "false_gt_geometry"])
def test_transition_validation_fails_closed(case):
    cells, transitions = _bundle()
    obj = transitions[0]["candidate"]
    if case == "missing":
        transitions.pop()
    elif case == "duplicate":
        transitions[-1] = copy.deepcopy(transitions[0])
    elif case == "reorder":
        transitions[0], transitions[1] = transitions[1], transitions[0]
    elif case == "wrong_id":
        transitions[0]["image_id"] = "not_the_frozen_train8"
    elif case == "wrong_condition":
        transitions[0]["condition"] = "gaussian_noise_S1"
    elif case == "missing_arm":
        del transitions[0]["control"]
    elif case == "bad_protocol":
        obj["protocol"]["connectivity"] = 4
    elif case == "bad_shape":
        obj["image_shape"] = [224, 224]
    elif case == "wrong_counts":
        obj["transition_counts"]["TP→TP"] += 1
    elif case == "hit_flags":
        obj["targets"][0]["prev_hit"] = 1
    elif case == "distance_three":
        obj["targets"][0]["new_match"]["centroid_distance"] = 3.0
    elif case == "bad_previous":
        obj["previous"]["false_alarm_pixels"] += 1
    elif case == "net_change_bool":
        obj["net_detected_target_change"] = False
    else:
        obj["targets"][0]["centroid"][0] += 0.1
    with pytest.raises(metrics.AnchoredMetricsError):
        _summarize((cells, transitions))


def test_object_cell_fa_mismatch_is_detected():
    bundle = _bundle()
    for row in bundle[1][:8]:
        row["candidate"]["diagnostic"]["false_alarm_pixels"] += 1
    with pytest.raises(metrics.AnchoredMetricsError, match="candidate.false_alarm_pixels"):
        _summarize(bundle)


def test_object_cell_target_area_mismatch_is_detected():
    cells, transitions = _bundle()
    for row in transitions:
        for arm in metrics.ARMS:
            row[arm]["targets"][0]["area"] -= 1
    with pytest.raises(metrics.AnchoredMetricsError, match="object GT totals"):
        _summarize((cells, transitions))


@pytest.mark.parametrize("value", [{"control": .4}, {"control": .4, "candidate": .4, "extra": 1},
    {"control": float("nan"), "candidate": .4}, {"control": -.1, "candidate": .4},
    {"control": True, "candidate": .4}])
def test_loss_mapping_is_exact_finite_and_nonnegative(value):
    with pytest.raises(metrics.AnchoredMetricsError):
        _summarize(initial=value)
