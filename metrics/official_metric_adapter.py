"""A thin, explicit adapter around NS-FPN's legacy official metrics.

This class exists only for source-reproduction comparisons.  New experiments
should use :class:`metrics.irstd_metrics.UnifiedResearchEvaluator`, whose
thresholding and matching rules are internally consistent.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import numpy as np
from numpy.typing import NDArray
import torch

from utils.metric import PD_FA, ROCMetric, mIoU


@dataclass(frozen=True)
class OfficialMetricResult:
    """Outputs produced by the legacy NS-FPN metric implementations."""

    pixel_accuracy: float
    mean_iou: float
    probability_thresholds: NDArray[np.float64]
    true_positive_rate: NDArray[np.float64]
    false_positive_rate: NDArray[np.float64]
    recall: NDArray[np.float64]
    precision: NDArray[np.float64]
    legacy_pd_fa_raw_thresholds: NDArray[np.float64]
    detection_probability: NDArray[np.float64]
    false_alarm_pixel_rate: NDArray[np.float64]
    image_count: int

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serialisable representation."""

        return {
            "pixel_accuracy": self.pixel_accuracy,
            "mean_iou": self.mean_iou,
            "probability_thresholds": self.probability_thresholds.tolist(),
            "true_positive_rate": self.true_positive_rate.tolist(),
            "false_positive_rate": self.false_positive_rate.tolist(),
            "recall": self.recall.tolist(),
            "precision": self.precision.tolist(),
            "legacy_pd_fa_raw_thresholds": (
                self.legacy_pd_fa_raw_thresholds.tolist()
            ),
            "detection_probability": self.detection_probability.tolist(),
            "false_alarm_pixel_rate": self.false_alarm_pixel_rate.tolist(),
            "image_count": self.image_count,
        }


def _as_official_tensor(value: Any, *, name: str) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        tensor = value.detach().cpu()
    else:
        tensor = torch.as_tensor(value)
    if tensor.ndim == 2:
        tensor = tensor[None, None, ...]
    elif tensor.ndim == 3:
        tensor = tensor[:, None, ...]
    elif tensor.ndim == 4 and tensor.shape[1] == 1:
        pass
    else:
        raise ValueError(
            f"{name} must have shape [H,W], [B,H,W], or [B,1,H,W]; "
            f"got {tuple(tensor.shape)}."
        )
    if tensor.numel() == 0:
        raise ValueError(f"{name} cannot be empty.")
    if not torch.isfinite(tensor).all():
        raise ValueError(f"{name} contains NaN or Inf values.")
    return tensor.float()


class OfficialMetricAdapter:
    """Delegate to the repository's metrics without changing their semantics.

    The official ``PD_FA`` implementation assumes a square, batch-size-one
    image and thresholds its input in the raw range ``[0, 255]``.  These quirks
    are deliberately retained and exposed in the result field names.
    """

    def __init__(self, *, nclass: int = 1, bins: int = 10, image_size: int = 256):
        if nclass != 1:
            raise ValueError("NS-FPN's official IRSTD metrics require nclass=1.")
        if isinstance(bins, bool) or not isinstance(bins, int) or bins < 1:
            raise ValueError("bins must be a positive integer.")
        if (
            isinstance(image_size, bool)
            or not isinstance(image_size, int)
            or image_size < 1
        ):
            raise ValueError("image_size must be a positive integer.")
        self.nclass = nclass
        self.bins = bins
        self.image_size = image_size
        self.reset()

    def reset(self) -> None:
        # Re-instantiation also avoids the legacy reset methods' hard-coded
        # eleven-bin array size.
        self._roc = ROCMetric(self.nclass, self.bins)
        self._pd_fa = PD_FA(self.nclass, self.bins, self.image_size)
        self._miou = mIoU(self.nclass)
        self._image_count = 0

    def update(self, logits: Any, targets: Any) -> None:
        prediction_tensor = _as_official_tensor(logits, name="logits")
        target_tensor = _as_official_tensor(targets, name="targets")
        if prediction_tensor.shape != target_tensor.shape:
            raise ValueError(
                f"logits and targets must have the same shape, got "
                f"{tuple(prediction_tensor.shape)} and {tuple(target_tensor.shape)}."
            )
        if prediction_tensor.shape[0] != 1:
            raise ValueError("Official PD_FA requires batch_size=1.")
        if tuple(prediction_tensor.shape[-2:]) != (
            self.image_size,
            self.image_size,
        ):
            raise ValueError(
                "Official PD_FA requires the configured square image size "
                f"{self.image_size}, got {tuple(prediction_tensor.shape[-2:])}."
            )

        self._roc.update(prediction_tensor, target_tensor)
        self._miou.update(prediction_tensor, target_tensor)
        self._pd_fa.update(prediction_tensor, target_tensor)
        self._image_count += 1

    def compute(self) -> OfficialMetricResult:
        pixel_accuracy, mean_iou = self._miou.get()
        true_positive_rate, false_positive_rate, recall, precision = self._roc.get()

        denominator = self.image_size * self.image_size * self._image_count
        if denominator:
            false_alarm_pixel_rate = self._pd_fa.FA / denominator
        else:
            false_alarm_pixel_rate = np.zeros(self.bins + 1, dtype=np.float64)
        detection_probability = np.divide(
            self._pd_fa.PD,
            self._pd_fa.target,
            out=np.zeros(self.bins + 1, dtype=np.float64),
            where=self._pd_fa.target != 0,
        )

        return OfficialMetricResult(
            pixel_accuracy=float(pixel_accuracy),
            mean_iou=float(mean_iou),
            probability_thresholds=np.linspace(0.0, 1.0, self.bins + 1),
            true_positive_rate=np.asarray(true_positive_rate, dtype=np.float64),
            false_positive_rate=np.asarray(false_positive_rate, dtype=np.float64),
            recall=np.asarray(recall, dtype=np.float64),
            precision=np.asarray(precision, dtype=np.float64),
            legacy_pd_fa_raw_thresholds=np.linspace(
                0.0, 255.0, self.bins + 1
            ),
            detection_probability=np.asarray(
                detection_probability, dtype=np.float64
            ),
            false_alarm_pixel_rate=np.asarray(
                false_alarm_pixel_rate, dtype=np.float64
            ),
            image_count=self._image_count,
        )


__all__ = ["OfficialMetricAdapter", "OfficialMetricResult"]
