"""Geometry-preserving, label-free deteriorations for Stage-C ASB-SFR."""

from .fourier_low_mask import (
    FourierMaskOutput,
    identity_deterioration,
    mask_low_frequency_amplitude,
)
from .high_frequency_noise import (
    HighFrequencyNoiseOutput,
    inject_high_frequency_noise,
    inject_high_frequency_noise_components,
)
from .image_space import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    imagenet_denormalize,
    imagenet_normalize,
)

__all__ = [
    "FourierMaskOutput",
    "HighFrequencyNoiseOutput",
    "IMAGENET_MEAN",
    "IMAGENET_STD",
    "identity_deterioration",
    "imagenet_denormalize",
    "imagenet_normalize",
    "inject_high_frequency_noise",
    "inject_high_frequency_noise_components",
    "mask_low_frequency_amplitude",
]
