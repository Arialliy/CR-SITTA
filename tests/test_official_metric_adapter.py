from __future__ import annotations

import numpy as np
import pytest

from metrics.official_metric_adapter import OfficialMetricAdapter


def test_official_adapter_smoke_and_reset() -> None:
    logits = np.full((1, 1, 4, 4), -1.0, dtype=np.float32)
    targets = np.zeros((1, 1, 4, 4), dtype=np.float32)
    logits[0, 0, 1, 1] = 1.0
    targets[0, 0, 1, 1] = 1.0
    evaluator = OfficialMetricAdapter(bins=2, image_size=4)

    evaluator.update(logits, targets)
    result = evaluator.compute()

    assert result.image_count == 1
    assert result.pixel_accuracy == pytest.approx(1.0)
    assert result.mean_iou == pytest.approx(1.0)
    assert result.detection_probability[0] == pytest.approx(1.0)
    assert np.isfinite(result.false_alarm_pixel_rate).all()
    assert result.to_dict()["image_count"] == 1

    evaluator.reset()
    reset_result = evaluator.compute()
    assert reset_result.image_count == 0
    assert np.isfinite(reset_result.detection_probability).all()
    assert np.isfinite(reset_result.false_alarm_pixel_rate).all()


def test_official_adapter_rejects_non_official_batch_or_size() -> None:
    evaluator = OfficialMetricAdapter(image_size=4)

    with pytest.raises(ValueError, match="batch_size=1"):
        evaluator.update(np.zeros((2, 1, 4, 4)), np.zeros((2, 1, 4, 4)))
    with pytest.raises(ValueError, match="square image size"):
        evaluator.update(np.zeros((1, 1, 3, 4)), np.zeros((1, 1, 3, 4)))
