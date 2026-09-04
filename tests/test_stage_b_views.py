import inspect

import pytest
import torch

from tta.views import (
    GEOMETRIC_VIEW_NAMES,
    IMAGENET_MEAN,
    IMAGENET_STD,
    INVERTIBLE_VIEW_REGISTRY,
    MILD_CONTRAST_FACTOR,
    NORMALIZED_RGB_INPUT_CONTRACT,
    STUDENT_PERTURBATION_REGISTRY,
    apply_fixed_mild_contrast,
    build_detached_region_weights,
    validated_student_perturbations,
    validated_weak_views,
)


def _normalize(physical: torch.Tensor) -> torch.Tensor:
    mean = physical.new_tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
    std = physical.new_tensor(IMAGENET_STD).view(1, 3, 1, 1)
    return (physical - mean) / std


def _denormalize(image: torch.Tensor) -> torch.Tensor:
    mean = image.new_tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
    std = image.new_tensor(IMAGENET_STD).view(1, 3, 1, 1)
    return image * std + mean


@pytest.mark.parametrize("name", GEOMETRIC_VIEW_NAMES)
def test_every_registered_geometric_view_has_bit_exact_inverse(name: str) -> None:
    tensor = torch.arange(2 * 3 * 5 * 9).reshape(2, 3, 5, 9)
    view = INVERTIBLE_VIEW_REGISTRY[name]
    transformed = view.forward(tensor)
    recovered = view.inverse_prediction(transformed)

    assert transformed.shape == tensor.shape
    assert torch.equal(recovered, tensor)


def test_geometric_registry_is_exact_frozen_whitelist() -> None:
    assert GEOMETRIC_VIEW_NAMES == ("identity", "hflip", "vflip", "hvflip")
    assert tuple(INVERTIBLE_VIEW_REGISTRY) == GEOMETRIC_VIEW_NAMES
    with pytest.raises(TypeError):
        INVERTIBLE_VIEW_REGISTRY["rot90"] = INVERTIBLE_VIEW_REGISTRY["identity"]
    with pytest.raises(ValueError, match="unknown weak view"):
        validated_weak_views(("identity", "rot90"))


def test_requested_view_order_is_preserved() -> None:
    views = validated_weak_views(("vflip", "identity", "hflip"))
    assert tuple(view.name for view in views) == ("vflip", "identity", "hflip")


def test_student_perturbations_are_fixed_deterministic_and_domain_explicit() -> None:
    physical = torch.linspace(0.1, 0.9, 35).reshape(1, 1, 5, 7).repeat(1, 3, 1, 1)
    normalized = _normalize(physical)
    identity, contrast = validated_student_perturbations(
        ("identity", "mild_contrast_0p95")
    )

    rng_before = torch.random.get_rng_state().clone()
    identity_result = identity.forward(normalized)
    first = contrast.forward(normalized)
    second = contrast.forward(normalized.clone())
    rng_after = torch.random.get_rng_state()

    expected_physical = (physical - 0.5) * MILD_CONTRAST_FACTOR + 0.5
    assert torch.equal(identity_result, normalized)
    assert torch.equal(first, second)
    assert torch.equal(rng_before, rng_after)
    assert torch.allclose(_denormalize(first), expected_physical, atol=1e-6, rtol=0)
    assert torch.allclose(
        _denormalize(first)[:, 0], _denormalize(first)[:, 1], atol=1e-7, rtol=0
    )
    assert contrast.input_contract == NORMALIZED_RGB_INPUT_CONTRACT
    assert tuple(STUDENT_PERTURBATION_REGISTRY) == (
        "identity",
        "mild_contrast_0p95",
    )


def test_photometric_perturbation_rejects_wrong_domain_or_channels() -> None:
    with pytest.raises(ValueError, match="three RGB channels"):
        apply_fixed_mild_contrast(torch.zeros(1, 1, 5, 7))
    with pytest.raises(ValueError, match="normalized-input contract"):
        apply_fixed_mild_contrast(torch.full((1, 3, 5, 7), 100.0))
    with pytest.raises(ValueError, match="unknown student perturbation"):
        validated_student_perturbations(("random_gamma",))


def test_region_weights_follow_formula_are_detached_and_protect_neighbors() -> None:
    teacher_mean = torch.full((1, 1, 5, 7), 0.1, requires_grad=True)
    teacher_mean = teacher_mean.clone()
    teacher_mean[0, 0, 2, 3] = 0.9
    teacher_mean[0, 0, 0, 0] = 0.5
    teacher_variance = torch.zeros_like(teacher_mean, requires_grad=True)
    teacher_variance = teacher_variance.clone()
    teacher_variance[0, 0, 4, 6] = 0.05

    output = build_detached_region_weights(
        teacher_mean,
        teacher_variance,
        tau_background=0.2,
        tau_foreground=0.8,
        gamma_foreground=2.0,
        gamma_background=1.0,
        temperature_foreground=0.1,
        temperature_background=0.1,
        protection_radius=1,
    )

    tensors = (
        output.foreground_weight,
        output.background_weight,
        output.protection_band,
        output.foreground_indicator,
        output.background_indicator,
    )
    assert all(not tensor.requires_grad and tensor.grad_fn is None for tensor in tensors)
    assert all(tensor.shape == teacher_mean.shape for tensor in tensors)

    assert output.foreground_weight[0, 0, 2, 3].item() == pytest.approx(0.9**2)
    assert output.foreground_weight.count_nonzero().item() == 1
    assert output.protection_band[0, 0, 1:4, 2:5].sum().item() == 9
    assert output.background_weight[0, 0, 1:4, 2:5].count_nonzero().item() == 0
    assert output.background_weight[0, 0, 0, 0].item() == 0.0
    assert output.background_weight[0, 0, 0, 1].item() == pytest.approx(0.9)
    assert output.background_weight[0, 0, 4, 6].item() == pytest.approx(
        0.9 * torch.exp(torch.tensor(-0.5)).item()
    )


@pytest.mark.parametrize(
    ("tau_background", "tau_foreground"),
    [(0.5, 0.5), (0.7, 0.5), (-0.1, 0.5), (0.2, 1.1)],
)
def test_region_weight_thresholds_fail_closed(
    tau_background: float, tau_foreground: float
) -> None:
    mean = torch.full((1, 1, 3, 3), 0.5)
    variance = torch.zeros_like(mean)
    with pytest.raises(ValueError):
        build_detached_region_weights(
            mean,
            variance,
            tau_background=tau_background,
            tau_foreground=tau_foreground,
        )


def test_public_method_inputs_have_no_label_or_mask_channel() -> None:
    public_methods = (
        validated_weak_views,
        validated_student_perturbations,
        apply_fixed_mild_contrast,
        build_detached_region_weights,
    )
    forbidden = ("gt", "ground_truth", "target", "label", "mask")
    for method in public_methods:
        parameter_names = tuple(inspect.signature(method).parameters)
        assert not any(
            token in parameter_name.lower()
            for parameter_name in parameter_names
            for token in forbidden
        )
