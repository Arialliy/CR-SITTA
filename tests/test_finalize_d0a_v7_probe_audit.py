from copy import deepcopy
import pytest
from analysis.finalize_d0a_v7_probe_audit import validate_records


def fixture_rows():
    return [{"image_id": "a", "dataset": "fixture", "view": view, "probe_id": probe,
             "target_count": 0, "input_metadata": {"target_tensor_sha256": "same"},
             "gt_tensor_unchanged": True, "deterministic_repeat_exact": True,
             "original_runner_parity_exact": True, "preclip_clamp_parity_exact": True}
            for view in ("full_256", "train_crop_224") for probe in ("clean", "lf_mask", "hf_noise")]


def test_all_empty_target_records_can_be_complete():
    validate_records(fixture_rows(), [], "fixture", ["a"])


def test_duplicate_or_missing_view_cannot_be_finalized():
    rows = fixture_rows()
    with pytest.raises(ValueError):
        validate_records(rows[:-1], [], "fixture", ["a"])
    with pytest.raises(ValueError):
        validate_records(rows + [deepcopy(rows[0])], [], "fixture", ["a"])


@pytest.mark.parametrize("field", ["gt_tensor_unchanged", "original_runner_parity_exact", "deterministic_repeat_exact"])
def test_failed_numerical_checks_cannot_be_finalized(field):
    rows = fixture_rows()
    rows[0][field] = False
    with pytest.raises(ValueError):
        validate_records(rows, [], "fixture", ["a"])


def test_missing_target_records_and_changed_crop_fail():
    rows = fixture_rows()
    rows[0]["target_count"] = 1
    with pytest.raises(ValueError):
        validate_records(rows, [], "fixture", ["a"])
    rows = fixture_rows()
    rows[0]["input_metadata"] = {"target_tensor_sha256": "changed"}
    with pytest.raises(ValueError):
        validate_records(rows, [], "fixture", ["a"])
