"""NumPy-only synthetic reachability checks; no data loading or training."""

from __future__ import annotations

import itertools
import json

import numpy as np
import pytest

from analysis.o3_residual_reachability_v3 import analyze_reachability


MASK_KEYS = {"oracle_mask", "can_positive", "can_negative", "v1_false_negative",
             "v1_false_positive", "repairable_fn", "repairable_fp",
             "unrepairable_fn", "unrepairable_fp"}


def assert_consistent(result, z0, z1, target):
    counts, arrays = result["counts"], result["arrays"]
    z0, z1, target = np.asarray(z0, dtype=np.float64), np.asarray(z1, dtype=np.float64), np.asarray(target, dtype=bool)
    gain = arrays["oracle_gain"]
    generated = np.where(gain == 0, z0, np.where(gain == 2, 2 * z1 - z0, z1)) > 0
    assert np.array_equal(generated, arrays["oracle_mask"])
    assert set(arrays) == MASK_KEYS | {"lower_logit", "upper_logit", "oracle_gain"}
    assert all(arrays[key].dtype == np.bool_ for key in MASK_KEYS)
    assert all(arrays[key].dtype == np.float64 for key in ("lower_logit", "upper_logit", "oracle_gain"))
    assert all(value.shape == z0.shape for value in arrays.values())
    assert np.isin(gain, [0, 1, 2]).all()
    repaired = arrays["repairable_fn"] | arrays["repairable_fp"]
    assert (gain[~repaired] == 1).all()
    assert (gain[repaired] != 1).all()
    assert counts["v1_fn"] == counts["repairable_fn"] + counts["unrepairable_fn"]
    assert counts["v1_fp"] == counts["repairable_fp"] + counts["unrepairable_fp"]
    assert counts["oracle_fn"] == counts["unrepairable_fn"]
    assert counts["oracle_fp"] == counts["unrepairable_fp"]
    assert counts["oracle_tp"] == counts["v1_tp"] + counts["repairable_fn"]
    assert counts["gt_positive_pixels"] == counts["v1_tp"] + counts["v1_fn"]
    assert counts["pixels"] == counts["gt_positive_pixels"] + counts["gt_negative_pixels"]
    assert counts["oracle_iou"] >= counts["v1_iou"]
    assert sum(counts["oracle_gain_counts"].values()) == counts["pixels"]
    json.dumps(counts, allow_nan=False)


def test_zero_correction_cannot_repair_errors_and_keeps_unit_gain():
    z = np.array([-2.0, 0.0, 3.0, 0.0])
    target = np.array([1, 1, 0, 0])
    result = analyze_reachability(z, z, target)
    counts, arrays = result["counts"], result["arrays"]
    assert counts["repairable_fn"] == counts["repairable_fp"] == 0
    assert counts["unrepairable_fn"] == 2 and counts["unrepairable_fp"] == 1
    assert np.array_equal(arrays["oracle_mask"], z > 0)
    assert (arrays["oracle_gain"] == 1).all()
    assert counts["zero_logit_counts"] == dict(z_o3=2, z_v1=2, z2=2, lower=2, upper=2)
    assert_consistent(result, z, z, target)


def test_increasing_and_decreasing_residuals_choose_both_endpoints():
    z0 = np.array([-3.0, 2.0, 3.0, -2.0])
    z1 = np.array([-1.0, -1.0, 1.0, 1.0])
    target = np.array([1, 1, 0, 0])
    result = analyze_reachability(z0, z1, target)
    assert result["arrays"]["oracle_gain"].tolist() == [2, 0, 2, 0]
    assert result["arrays"]["oracle_mask"].tolist() == [True, True, False, False]
    assert result["counts"]["repairable_fn"] == result["counts"]["repairable_fp"] == 2
    assert result["counts"]["oracle_iou"] == 1.0
    assert_consistent(result, z0, z1, target)


def test_only_closed_zero_endpoint_can_remove_false_positive():
    z0 = np.array([0.0, 2.0, -2.0])
    z1 = np.array([1.0, 1.0, -1.0])
    target = np.array([0, 0, 1])
    result = analyze_reachability(z0, z1, target)
    assert result["arrays"]["oracle_gain"].tolist() == [0, 2, 1]
    assert result["counts"]["repairable_fp_only_at_zero_endpoint"] == 2
    assert result["counts"]["repairable_fp"] == 2
    assert result["counts"]["unrepairable_fn"] == 1  # upper == 0 is not positive.
    for alpha in (1e-9, 0.1, 1.0, 1.9, 2 - 1e-9):
        assert ((z0 + alpha * (z1 - z0))[:2] > 0).all()
    assert_consistent(result, z0, z1, target)


def test_zero_v1_is_negative_and_can_be_repaired_for_foreground():
    z0 = np.array([-1.0, 1.0, -1.0, 1.0])
    z1 = np.zeros(4)
    target = np.array([1, 1, 0, 0])
    result = analyze_reachability(z0, z1, target)
    assert result["arrays"]["oracle_gain"].tolist() == [2, 0, 1, 1]
    assert result["counts"]["zero_logit_counts"]["z_v1"] == 4
    assert_consistent(result, z0, z1, target)


@pytest.mark.parametrize("dtype", [np.float16, np.float32, np.float64])
@pytest.mark.parametrize("shape", [(), (1,), (2, 3), (2, 1, 3, 5)])
def test_arbitrary_nonempty_shape_and_float64_outputs(dtype, shape):
    z0 = np.full(shape, -2.0, dtype=dtype)
    z1 = np.full(shape, -0.5, dtype=dtype)
    target = np.ones(shape, dtype=bool)
    result = analyze_reachability(z0, z1, target)
    assert result["counts"]["oracle_iou"] == 1.0
    assert_consistent(result, z0, z1, target)


def test_all_background_empty_union_iou_convention():
    z = np.array([-1.0, 0.0])
    target = np.zeros(2, dtype=np.uint8)
    result = analyze_reachability(z, z, target)
    assert result["counts"]["gt_positive_pixels"] == 0
    assert result["counts"]["v1_iou"] == result["counts"]["oracle_iou"] == 1.0


def test_subnormal_signs_are_strict_without_epsilon_tolerance():
    tiny = np.nextafter(np.float64(0), np.float64(1))
    z0 = np.array([-tiny, tiny])
    z1 = np.array([0.0, tiny])
    target = np.array([1, 0])
    result = analyze_reachability(z0, z1, target)
    assert result["counts"]["repairable_fn"] == 1
    assert result["counts"]["unrepairable_fp"] == 1
    assert_consistent(result, z0, z1, target)


def test_random_grid_matches_reachability_masks_and_count_conservation():
    generator = np.random.default_rng(3003)
    alpha = np.linspace(0.0, 2.0, 1001)[:, None, None]
    for _ in range(50):
        z0 = generator.integers(-8, 9, (3, 5)).astype(np.float64) / 2
        z1 = generator.integers(-8, 9, (3, 5)).astype(np.float64) / 2
        target = generator.integers(0, 2, (3, 5))
        result = analyze_reachability(z0, z1, target)
        grid = z0 + alpha * (z1 - z0)
        assert np.array_equal(result["arrays"]["can_positive"], (grid > 0).any(axis=0))
        assert np.array_equal(result["arrays"]["can_negative"], (grid <= 0).any(axis=0))
        brute_mask = np.where(target, (grid > 0).any(axis=0), ~(grid <= 0).any(axis=0))
        assert np.array_equal(result["arrays"]["oracle_mask"], brute_mask)
        assert_consistent(result, z0, z1, target)


def test_exhaustive_independent_endpoint_choices_cannot_exceed_oracle_iou():
    generator = np.random.default_rng(3004)
    for _ in range(30):
        z0 = generator.integers(-4, 5, 4).astype(np.float64)
        z1 = generator.integers(-4, 5, 4).astype(np.float64)
        truth = generator.integers(0, 2, 4).astype(bool)
        result = analyze_reachability(z0, z1, truth)
        best = 0.0
        for gains in itertools.product((0, 1, 2), repeat=4):
            gains = np.asarray(gains)
            prediction = z0 + gains * (z1 - z0) > 0
            tp = np.count_nonzero(prediction & truth)
            union = np.count_nonzero(prediction | truth)
            iou = tp / union if union else 1.0
            best = max(best, iou)
        assert result["counts"]["oracle_iou"] == best


def test_no_input_mutation_or_alias_and_noncontiguous_inputs():
    z0 = np.array([[-3.0, 2.0], [3.0, -2.0]]).T
    z1 = np.array([[-1.0, -1.0], [1.0, 1.0]]).T
    target = np.array([[1, 1], [0, 0]]).T
    before = [value.copy() for value in (z0, z1, target)]
    result = analyze_reachability(z0, z1, target)
    for array in result["arrays"].values():
        assert all(not np.shares_memory(array, value) for value in (z0, z1, target))
    assert all(np.array_equal(value, snapshot) for value, snapshot in zip((z0, z1, target), before))


@pytest.mark.parametrize("which", [0, 1, 2])
@pytest.mark.parametrize("fault", ["empty", "shape", "nan", "inf"])
def test_invalid_inputs_are_rejected(which, fault):
    values = [np.zeros((2, 3)), np.zeros((2, 3)), np.zeros((2, 3))]
    if fault == "empty":
        values[which] = np.empty((0, 3))
    elif fault == "shape":
        values[which] = values[which][:, :2]
    else:
        values[which][0, 0] = float(fault)
    with pytest.raises(ValueError):
        analyze_reachability(*values)


@pytest.mark.parametrize("value", [-1, 0.5, 2])
def test_nonbinary_target_rejected(value):
    with pytest.raises(ValueError, match="binary"):
        analyze_reachability(np.zeros(1), np.zeros(1), np.array([value]))


@pytest.mark.parametrize("which", [0, 1])
@pytest.mark.parametrize("dtype", [np.int64, np.complex128, object, str])
def test_nonfloating_logits_rejected(which, dtype):
    values = [np.zeros(1), np.zeros(1), np.zeros(1)]
    values[which] = np.zeros(1).astype(dtype)
    with pytest.raises(TypeError, match="floating"):
        analyze_reachability(*values)


@pytest.mark.parametrize("dtype", [np.complex128, object, str])
def test_nonreal_or_nonnumeric_target_rejected(dtype):
    with pytest.raises(TypeError, match="binary"):
        analyze_reachability(np.zeros(1), np.zeros(1), np.zeros(1).astype(dtype))


def test_finite_inputs_with_overflowing_extrapolation_fail_closed():
    extreme = np.finfo(np.float64).max
    with pytest.raises(ValueError, match="endpoint"):
        analyze_reachability(np.array([-extreme]), np.array([extreme]), np.zeros(1))


def test_extreme_finite_identity_is_not_rejected_due_to_temporary_doubling():
    z = np.array([1e308, -1e308])
    result = analyze_reachability(z, z, np.array([1, 0]))
    assert np.array_equal(result["arrays"]["lower_logit"], z)
    assert np.array_equal(result["arrays"]["upper_logit"], z)
    assert (result["arrays"]["oracle_gain"] == 1).all()
    assert result["counts"]["oracle_iou"] == result["counts"]["v1_iou"] == 1.0
