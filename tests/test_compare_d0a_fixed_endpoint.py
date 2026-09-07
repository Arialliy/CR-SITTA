import json
import math
import pytest
from analysis.compare_d0a_fixed_endpoint import compare, extract_metrics, read_epoch


def test_mixed_endpoint_does_not_trigger_e0():
    actual = compare({"miou": .7912439594201043, "pd": .9724867724867725, "fa_per_pixel_x1e6": 25.461955242846386},
                     {"miou": .7923859142552256, "pd": .9703703703703703, "fa_per_pixel_x1e6": 17.786600503576807})
    assert actual["E0_all_three_harm_condition"] is False
    assert actual["tradeoff"] == "mixed_or_tied"
    assert actual["delta_percentage_points"]["pd"] == pytest.approx(.21164021164022007)
    assert actual["delta_percentage_points"]["miou"] == pytest.approx(-.1141954835121366)


def test_all_harm_and_zero_reference_fa():
    result = compare({"miou": .7, "pd": .8, "fa_per_pixel_x1e6": 2},
                     {"miou": .8, "pd": .9, "fa_per_pixel_x1e6": 0})
    assert result["E0_all_three_harm_condition"] is True
    assert result["relative_fa_change_percent"] is None


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -1, 80])
def test_bad_metric_units_rejected(value):
    with pytest.raises(ValueError):
        extract_metrics({"miou": value, "pd": .9, "fa_per_pixel_x1e6": 20})


def test_read_unique_complete_endpoint(tmp_path):
    path = tmp_path / "log.jsonl"
    rows = [{"epoch": i} for i in range(500, 1001)]
    path.write_text("\n".join(map(json.dumps, rows)))
    assert read_epoch(path, 1000)["epoch"] == 1000
    path.write_text("\n".join(map(json.dumps, rows + [{"epoch": 1000}])))
    with pytest.raises(ValueError):
        read_epoch(path, 1000)


def test_missing_log_epoch_rejected(tmp_path):
    path = tmp_path / "log.jsonl"
    path.write_text("\n".join(json.dumps({"epoch": i}) for i in range(501,1001)))
    with pytest.raises(ValueError):
        read_epoch(path, 1000)
