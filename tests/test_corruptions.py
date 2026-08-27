from __future__ import annotations

import inspect

import numpy as np
import pytest

from corruptions import (
    NON_CLEAN_CORRUPTIONS,
    SUPPORTED_CORRUPTIONS,
    apply_corruption,
    apply_sample_corruption,
    derive_sample_seed,
    get_default_severity_table,
    make_sample_rng,
    validate_corruption_request,
)


def _base_hwc() -> np.ndarray:
    values = np.linspace(0.2, 0.8, num=8 * 11, dtype=np.float32).reshape(8, 11)
    return np.stack((values, values, values), axis=-1)


def _layout_cases() -> tuple[np.ndarray, ...]:
    hwc = _base_hwc()
    return (
        hwc[..., 0],
        hwc,
        np.moveaxis(hwc, -1, 0),
        hwc[..., :1],
        np.moveaxis(hwc[..., :1], -1, 0),
    )


def test_shipped_table_is_fixed_split_pilot_frozen_and_complete() -> None:
    table = get_default_severity_table()

    assert table.status == "fixed_split_v1_frozen_after_round2_train_pilot"
    assert table.frozen is True
    assert table.calibration_required is True
    assert table.calibration_completed is True
    assert table.calibration_scope == (
        "fixed_train_sha256_ranked_64_per_dataset_no_test_images_or_labels"
    )
    assert set(table.levels) == set(SUPPORTED_CORRUPTIONS)
    assert set(table.levels["clean"]) == {0}
    for corruption in NON_CLEAN_CORRUPTIONS:
        assert set(table.levels[corruption]) == {1, 2, 3, 4, 5}


def test_shipped_strength_parameters_are_strictly_monotonic() -> None:
    table = get_default_severity_table()

    for corruption in NON_CLEAN_CORRUPTIONS:
        name = table.strength_parameters[corruption]
        assert name is not None
        values = [table.parameters(corruption, severity)[name] for severity in range(1, 6)]
        if table.monotonic_directions[corruption] == "increasing":
            assert all(left < right for left, right in zip(values, values[1:]))
        else:
            assert all(left > right for left, right in zip(values, values[1:]))


def test_shipped_table_is_deeply_read_only() -> None:
    table = get_default_severity_table()
    with pytest.raises(TypeError):
        table.levels["gaussian_noise"][1]["sigma"] = 99.0  # type: ignore[index]


@pytest.mark.parametrize("severity", [1, 2, 3, 4, 5])
@pytest.mark.parametrize("corruption", NON_CLEAN_CORRUPTIONS)
def test_all_layouts_preserve_shape_dtype_range_and_input(
    corruption: str, severity: int
) -> None:
    for image in _layout_cases():
        before = image.copy()
        result = apply_corruption(image, corruption, severity, np.random.default_rng(123))

        assert result.shape == image.shape
        assert result.dtype == image.dtype
        assert np.isfinite(result).all()
        assert float(result.min()) >= 0.0
        assert float(result.max()) <= 1.0
        assert np.array_equal(image, before)
        assert not np.shares_memory(result, image)


def test_clean_is_exact_copy_and_does_not_consume_rng() -> None:
    image = _base_hwc()
    rng = np.random.default_rng(9)
    control = np.random.default_rng(9)

    result = apply_corruption(image, "clean", 0, rng)

    assert np.array_equal(result, image)
    assert result is not image
    assert not np.shares_memory(result, image)
    assert rng.integers(0, 2**32) == control.integers(0, 2**32)


@pytest.mark.parametrize("corruption", ("gaussian_noise", "stripe_noise"))
def test_stochastic_corruptions_reproduce_with_equivalent_caller_rng(
    corruption: str,
) -> None:
    image = _base_hwc()
    first = apply_corruption(image, corruption, 4, np.random.default_rng(77))
    second = apply_corruption(image, corruption, 4, np.random.default_rng(77))
    different = apply_corruption(image, corruption, 4, np.random.default_rng(78))

    assert np.array_equal(first, second)
    assert not np.array_equal(first, different)


def test_sample_wrapper_matches_explicit_stable_generator() -> None:
    image = _base_hwc()
    kwargs = {
        "image_id": "XDU189",
        "corruption": "gaussian_noise",
        "severity": 3,
        "base_seed": 42,
    }
    seed = derive_sample_seed(**kwargs)

    wrapped = apply_sample_corruption(image, **kwargs)
    explicit = apply_corruption(
        image,
        kwargs["corruption"],
        kwargs["severity"],
        np.random.default_rng(seed),
    )

    assert np.array_equal(wrapped, explicit)


def test_seed_derivation_has_golden_value_and_distinguishes_key_fields() -> None:
    # The golden value ensures this protocol cannot silently regress to the
    # process-randomised built-in hash() or change tuple serialisation.
    expected = 7_260_107_400_448_367_691
    assert derive_sample_seed("XDU189", "gaussian_noise", 3, 42) == expected

    seeds = {
        derive_sample_seed("XDU189", "gaussian_noise", 3, 42),
        derive_sample_seed("XDU190", "gaussian_noise", 3, 42),
        derive_sample_seed("XDU189", "gaussian_noise", 4, 42),
        derive_sample_seed("XDU189", "stripe_noise", 3, 42),
        derive_sample_seed("XDU189", "gaussian_noise", 3, 43),
    }
    assert len(seeds) == 5


def test_make_sample_rng_reproduces_across_independent_instances() -> None:
    first = make_sample_rng("图像-01", "stripe_noise", 5, 123).normal(size=32)
    second = make_sample_rng("图像-01", "stripe_noise", 5, 123).normal(size=32)

    assert np.array_equal(first, second)


def test_noise_and_stripe_empirical_strength_increase_with_severity() -> None:
    image = np.full((128, 160), 0.5, dtype=np.float64)
    for corruption in ("gaussian_noise", "stripe_noise"):
        distortions = []
        for severity in range(1, 6):
            result = apply_corruption(
                image, corruption, severity, np.random.default_rng(1234)
            )
            distortions.append(float(np.sqrt(np.mean(np.square(result - image)))))
        assert all(left < right for left, right in zip(distortions, distortions[1:]))


def test_blur_and_low_contrast_empirical_strength_increase_with_severity() -> None:
    checkerboard = (np.indices((64, 64)).sum(axis=0) % 2).astype(np.float64)
    blur_contrasts = [
        float(
            apply_corruption(
                checkerboard, "gaussian_blur", severity, np.random.default_rng(0)
            ).std()
        )
        for severity in range(1, 6)
    ]
    low_contrasts = [
        float(
            apply_corruption(
                checkerboard, "low_contrast", severity, np.random.default_rng(0)
            ).std()
        )
        for severity in range(1, 6)
    ]

    assert all(left > right for left, right in zip(blur_contrasts, blur_contrasts[1:]))
    assert all(left > right for left, right in zip(low_contrasts, low_contrasts[1:]))


@pytest.mark.parametrize("channel_first", [False, True])
def test_blur_never_mixes_channels(channel_first: bool) -> None:
    image = np.zeros((17, 19, 3), dtype=np.float64)
    image[8, 9, 0] = 1.0
    if channel_first:
        image = np.moveaxis(image, -1, 0)

    result = apply_corruption(
        image, "gaussian_blur", 3, np.random.default_rng(0)
    )
    channel_axis = 0 if channel_first else -1
    untouched_1 = np.take(result, 1, axis=channel_axis)
    untouched_2 = np.take(result, 2, axis=channel_axis)

    assert not untouched_1.any()
    assert not untouched_2.any()


@pytest.mark.parametrize("channel_first", [False, True])
def test_gaussian_sensor_noise_is_shared_across_rgb_channels(
    channel_first: bool,
) -> None:
    image = np.full((17, 19, 3), 0.5, dtype=np.float64)
    if channel_first:
        image = np.moveaxis(image, -1, 0)

    result = apply_corruption(
        image, "gaussian_noise", 3, np.random.default_rng(4)
    )
    channel_axis = 0 if channel_first else -1

    assert np.array_equal(
        np.take(result, 0, axis=channel_axis),
        np.take(result, 1, axis=channel_axis),
    )
    assert np.array_equal(
        np.take(result, 1, axis=channel_axis),
        np.take(result, 2, axis=channel_axis),
    )


@pytest.mark.parametrize("channel_first", [False, True])
def test_stripe_profile_is_shared_across_rows_and_channels(channel_first: bool) -> None:
    image = np.full((13, 17, 3), 0.5, dtype=np.float64)
    if channel_first:
        image = np.moveaxis(image, -1, 0)
    result = apply_corruption(image, "stripe_noise", 3, np.random.default_rng(5))
    delta = result - image

    if channel_first:
        assert np.allclose(delta, delta[0:1, 0:1, :])
    else:
        assert np.allclose(delta, delta[0:1, :, 0:1])


def test_clipping_handles_extreme_pixels() -> None:
    image = np.zeros((64, 64), dtype=np.float32)
    image[:, 32:] = 1.0
    result = apply_corruption(
        image, "gaussian_noise", 5, np.random.default_rng(3)
    )

    assert result.min() == 0.0
    assert result.max() == 1.0
    assert np.isfinite(result).all()


@pytest.mark.parametrize(
    ("corruption", "severity", "error"),
    [
        ("clean", 1, ValueError),
        ("gaussian_noise", 0, ValueError),
        ("gaussian_noise", 6, ValueError),
        ("unknown", 1, ValueError),
        ("gaussian_noise", 1.0, TypeError),
        ("gaussian_noise", True, TypeError),
    ],
)
def test_invalid_corruption_requests_are_rejected(
    corruption: str, severity: object, error: type[Exception]
) -> None:
    with pytest.raises(error):
        validate_corruption_request(corruption, severity)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "image",
    [
        np.zeros((4,), dtype=np.float32),
        np.zeros((1, 1, 4, 4), dtype=np.float32),
        np.zeros((5, 6, 7), dtype=np.float32),
        np.full((4, 4), -0.01, dtype=np.float32),
        np.full((4, 4), 1.01, dtype=np.float32),
        np.full((4, 4), np.nan, dtype=np.float32),
        np.zeros((4, 4), dtype=np.complex64),
        np.zeros((4, 4), dtype=np.bool_),
    ],
)
def test_invalid_images_are_rejected(image: np.ndarray) -> None:
    with pytest.raises((TypeError, ValueError)):
        apply_corruption(image, "clean", 0, np.random.default_rng(0))


def test_requires_numpy_array_and_generator() -> None:
    image = np.zeros((4, 4), dtype=np.float32)
    with pytest.raises(TypeError, match="numpy.ndarray"):
        apply_corruption([[0.0]], "clean", 0, np.random.default_rng(0))  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="Generator"):
        apply_corruption(image, "clean", 0, 1)  # type: ignore[arg-type]


def test_public_kernel_has_no_mask_path() -> None:
    parameters = inspect.signature(apply_corruption).parameters
    assert tuple(parameters) == ("image_01", "corruption", "severity", "rng")

    mask = np.ones((4, 4), dtype=np.uint8)
    before = mask.copy()
    with pytest.raises(TypeError, match="mask"):
        apply_corruption(  # type: ignore[call-arg]
            np.zeros((4, 4), dtype=np.float32),
            "clean",
            0,
            np.random.default_rng(0),
            mask=mask,
        )
    assert np.array_equal(mask, before)
