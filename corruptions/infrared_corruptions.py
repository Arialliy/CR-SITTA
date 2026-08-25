"""Infrared image corruptions in the unnormalised physical intensity domain."""

from __future__ import annotations

from typing import Mapping

import numpy as np
from scipy.ndimage import gaussian_filter, gaussian_filter1d

from .corruption_protocol import (
    get_default_severity_table,
    validate_corruption_request,
)


_CHANNEL_COUNTS = frozenset((1, 3, 4))


def _infer_channel_axis(shape: tuple[int, ...]) -> int | None:
    if len(shape) == 2:
        return None
    first_is_channel = shape[0] in _CHANNEL_COUNTS
    last_is_channel = shape[-1] in _CHANNEL_COUNTS
    if first_is_channel and not last_is_channel:
        return 0
    if last_is_channel and not first_is_channel:
        return len(shape) - 1
    if first_is_channel and last_is_channel:
        # Prefer the smaller plausible channel dimension.  Exact ties are
        # inherently ambiguous without metadata, so follow NumPy's common HWC
        # convention.  Shape preservation remains guaranteed either way.
        return 0 if shape[0] < shape[-1] else len(shape) - 1
    raise ValueError(
        "a 3D image must be CHW or HWC with 1, 3, or 4 channels"
    )


def _prepare_image(
    image_01: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.dtype, int | None, tuple[int, int]]:
    if not isinstance(image_01, np.ndarray):
        raise TypeError("image_01 must be a numpy.ndarray")
    if image_01.ndim not in {2, 3}:
        raise ValueError("image_01 must be grayscale HW, CHW, or HWC")
    if any(dimension <= 0 for dimension in image_01.shape):
        raise ValueError("image_01 dimensions must be non-empty")
    if image_01.dtype.kind not in {"u", "i", "f"}:
        raise TypeError("image_01 must have a real numeric dtype")

    work = image_01.astype(np.float64, copy=True)
    if not np.isfinite(work).all():
        raise ValueError("image_01 must contain only finite values")
    if work.min() < 0.0 or work.max() > 1.0:
        raise ValueError("image_01 must be in the physical intensity range [0, 1]")

    output_dtype = image_01.dtype if image_01.dtype.kind == "f" else np.dtype(np.float32)
    channel_axis = _infer_channel_axis(tuple(image_01.shape))
    spatial_axes = tuple(axis for axis in range(image_01.ndim) if axis != channel_axis)
    if len(spatial_axes) != 2:
        raise AssertionError("exactly two spatial axes are required")
    return image_01, work, output_dtype, channel_axis, spatial_axes  # type: ignore[return-value]


def _gaussian_noise(
    image: np.ndarray,
    parameters: Mapping[str, float],
    rng: np.random.Generator,
    spatial_axes: tuple[int, int],
) -> np.ndarray:
    sigma = float(parameters["sigma"])
    noise_shape = list(image.shape)
    if image.ndim == 3:
        channel_axis = next(axis for axis in range(3) if axis not in spatial_axes)
        noise_shape[channel_axis] = 1
    noise = rng.normal(loc=0.0, scale=sigma, size=tuple(noise_shape))
    return image + noise


def _gaussian_blur(
    image: np.ndarray,
    parameters: Mapping[str, float],
    rng: np.random.Generator,
    spatial_axes: tuple[int, int],
) -> np.ndarray:
    del rng
    sigma = float(parameters["sigma"])
    axis_sigmas = [0.0] * image.ndim
    for axis in spatial_axes:
        axis_sigmas[axis] = sigma
    return gaussian_filter(image, sigma=tuple(axis_sigmas), mode="reflect")


def _low_contrast(
    image: np.ndarray,
    parameters: Mapping[str, float],
    rng: np.random.Generator,
    spatial_axes: tuple[int, int],
) -> np.ndarray:
    del rng
    factor = float(parameters["contrast_factor"])
    centre = image.mean(axis=spatial_axes, keepdims=True)
    return centre + factor * (image - centre)


def _stripe_noise(
    image: np.ndarray,
    parameters: Mapping[str, float],
    rng: np.random.Generator,
    spatial_axes: tuple[int, int],
) -> np.ndarray:
    amplitude = float(parameters["amplitude"])
    smooth_sigma = float(parameters["smooth_sigma_pixels"])
    width_axis = spatial_axes[1]
    width = image.shape[width_axis]
    if width == 1:
        return image.copy()

    profile = rng.normal(size=width)
    if smooth_sigma > 0.0:
        profile = gaussian_filter1d(profile, sigma=smooth_sigma, mode="reflect")
    profile -= profile.mean()
    root_mean_square = float(np.sqrt(np.mean(np.square(profile))))
    if root_mean_square <= np.finfo(np.float64).eps:
        # This is practically unreachable for random width>1 data but keeps the
        # result well-defined for custom bit generators and tiny images.
        profile = np.linspace(-1.0, 1.0, num=width, dtype=np.float64)
        root_mean_square = float(np.sqrt(np.mean(np.square(profile))))
    profile *= amplitude / root_mean_square

    broadcast_shape = [1] * image.ndim
    broadcast_shape[width_axis] = width
    return image + profile.reshape(broadcast_shape)


_IMPLEMENTATIONS = {
    "gaussian_noise": _gaussian_noise,
    "gaussian_blur": _gaussian_blur,
    "low_contrast": _low_contrast,
    "stripe_noise": _stripe_noise,
}


def apply_corruption(
    image_01: np.ndarray,
    corruption: str,
    severity: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Apply one corruption without changing shape or mutating the input.

    Parameters
    ----------
    image_01:
        A NumPy grayscale ``HW``, channel-first ``CHW``, or channel-last
        ``HWC`` image in the physical intensity range ``[0, 1]``. Corruption
        must occur before ImageNet or other model normalisation.
    corruption, severity:
        ``clean`` is valid only at severity 0. All other supported corruptions
        are valid only at severities 1 through 5.
    rng:
        Caller-owned ``numpy.random.Generator``. Dataset runners should create
        it with ``make_sample_rng(image_id, corruption, severity, base_seed)``.

    Returns
    -------
    numpy.ndarray
        A new array, clipped to ``[0, 1]``, with the input shape preserved.
        Floating dtypes are preserved. Non-clean integer inputs produce
        ``float32``; clean is an exact dtype-preserving copy.

    Notes
    -----
    No mask argument exists by design: ground-truth masks must never enter or
    be modified by the corruption path.
    """

    corruption, severity = validate_corruption_request(corruption, severity)
    if not isinstance(rng, np.random.Generator):
        raise TypeError("rng must be a numpy.random.Generator")

    original, work, output_dtype, _channel_axis, spatial_axes = _prepare_image(image_01)
    if corruption == "clean":
        return original.copy()

    parameters = get_default_severity_table().parameters(corruption, severity)
    implementation = _IMPLEMENTATIONS[corruption]
    corrupted = implementation(work, parameters, rng, spatial_axes)
    if corrupted.shape != work.shape:
        raise RuntimeError("corruption implementation changed image shape")
    if not np.isfinite(corrupted).all():
        raise RuntimeError("corruption implementation produced NaN or infinity")
    corrupted = np.clip(corrupted, 0.0, 1.0)
    return corrupted.astype(output_dtype, copy=False)
