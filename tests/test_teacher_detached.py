import pytest
import torch
from torch import nn

from tta.model_adapter import IRSTDModelAdapter
from tta.proposals.source_multiview_teacher import (
    aggregate_aligned_probabilities,
    build_aligned_multiview_teacher,
    build_aligned_view_probabilities,
    frozen_probability_forward,
)


class GuardedNSFPNLike(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(1, 2, 1)
        self.bn = nn.BatchNorm2d(2)
        self.dropout = nn.Dropout2d(p=0.75)
        self.head = nn.Conv2d(2, 1, 1)
        self.observed_inference_modes: list[bool] = []
        self.observed_grad_modes: list[bool] = []
        self.observed_training_modes: list[bool] = []

    def forward(self, image: torch.Tensor, warm_flag: bool):
        del warm_flag
        self.observed_inference_modes.append(torch.is_inference_mode_enabled())
        self.observed_grad_modes.append(torch.is_grad_enabled())
        self.observed_training_modes.append(self.training)
        features = self.dropout(self.bn(self.conv(image)))
        return [], self.head(features)


def _state_clone(model: nn.Module) -> dict[str, torch.Tensor]:
    return {name: value.detach().clone() for name, value in model.state_dict().items()}


def test_teacher_is_detached_finite_bounded_and_model_is_not_mutated() -> None:
    torch.manual_seed(7)
    model = GuardedNSFPNLike()
    model.train()
    adapter = IRSTDModelAdapter(model)
    image = torch.randn(1, 1, 256, 256, requires_grad=True)

    state_before = _state_clone(model)
    modes_before = [module.training for module in model.modules()]
    requires_grad_before = [parameter.requires_grad for parameter in model.parameters()]

    teacher, uncertainty = build_aligned_multiview_teacher(
        adapter,
        image,
        include_context_tile=True,
        aggregation="source_anchor",
        beta=0.6,
    )

    assert teacher.shape == uncertainty.shape == (1, 1, 256, 256)
    for output in (teacher, uncertainty):
        assert output.grad_fn is None
        assert not output.requires_grad
        assert torch.isfinite(output).all()
        assert (output >= 0).all() and (output <= 1).all()

    assert model.observed_inference_modes
    assert all(model.observed_inference_modes)
    assert not any(model.observed_grad_modes)
    assert not any(model.observed_training_modes)
    assert [module.training for module in model.modules()] == modes_before
    assert [parameter.requires_grad for parameter in model.parameters()] == (
        requires_grad_before
    )
    assert all(
        torch.equal(value, state_before[name])
        for name, value in model.state_dict().items()
    )


def test_every_public_probability_result_is_detached() -> None:
    image = torch.randn(1, 1, 8, 9, requires_grad=True)
    model = nn.Conv2d(1, 1, 1)

    direct = frozen_probability_forward(model, image)
    aligned, _ = build_aligned_view_probabilities(model, image)
    aggregate = aggregate_aligned_probabilities(aligned, "mean")

    assert not direct.requires_grad and direct.grad_fn is None
    assert not aligned.requires_grad and aligned.grad_fn is None
    assert not aggregate.requires_grad and aggregate.grad_fn is None


def test_source_anchor_beta_zero_is_exact_identity_probability() -> None:
    torch.manual_seed(11)
    image = torch.randn(1, 1, 9, 13)
    model = nn.Conv2d(1, 1, 1)
    aligned, _ = build_aligned_view_probabilities(model, image)
    teacher = aggregate_aligned_probabilities(
        aligned, "source_anchor", beta=0.0
    )
    assert torch.equal(teacher, aligned[0])


@pytest.mark.parametrize(
    "bad_image",
    [
        torch.zeros(1, 8, 8),
        torch.zeros(1, 1, 8, 8, dtype=torch.int64),
        torch.full((1, 1, 8, 8), float("nan")),
    ],
)
def test_strict_image_validation(bad_image: torch.Tensor) -> None:
    with pytest.raises((TypeError, ValueError)):
        frozen_probability_forward(nn.Conv2d(1, 1, 1), bad_image)


def test_strict_output_shape_and_finiteness_validation() -> None:
    class WrongChannels(nn.Module):
        def forward(self, image: torch.Tensor) -> torch.Tensor:
            return image.repeat(1, 2, 1, 1)

    class NonFinite(nn.Module):
        def forward(self, image: torch.Tensor) -> torch.Tensor:
            return torch.full_like(image[:, :1], float("inf"))

    image = torch.zeros(1, 1, 8, 8)
    with pytest.raises(ValueError, match="logits must have shape"):
        frozen_probability_forward(WrongChannels(), image)
    with pytest.raises(ValueError, match="finite"):
        frozen_probability_forward(NonFinite(), image)


def test_context_tile_rejects_non_256_input() -> None:
    with pytest.raises(ValueError, match="256 x 256"):
        build_aligned_view_probabilities(
            nn.Conv2d(1, 1, 1),
            torch.zeros(1, 1, 255, 256),
            include_context_tile=True,
        )
