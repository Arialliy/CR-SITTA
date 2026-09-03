import pytest
import torch
from torch import nn

from tta.proposals.source_multiview_teacher import (
    GEOMETRIC_VIEWS,
    aggregate_aligned_probabilities,
    apply_geometric_view,
    build_aligned_view_probabilities,
    infer_context_tile_probability,
    inverse_geometric_view,
    stitch_context_tile_probabilities,
)


class PointwiseLogitModel(nn.Module):
    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return image[:, :1]


class BatchRecordingPointwiseModel(PointwiseLogitModel):
    def __init__(self) -> None:
        super().__init__()
        self.forward_batch_sizes: list[int] = []

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        self.forward_batch_sizes.append(int(image.shape[0]))
        return super().forward(image)


@pytest.mark.parametrize("view", GEOMETRIC_VIEWS)
def test_geometric_apply_then_inverse_is_bit_exact(view: str) -> None:
    image = torch.arange(2 * 3 * 7 * 11).reshape(2, 3, 7, 11)
    recovered = inverse_geometric_view(apply_geometric_view(image, view), view)
    assert torch.equal(recovered, image)


def test_only_preregistered_geometric_views_are_accepted() -> None:
    with pytest.raises(ValueError, match="unknown geometric view"):
        apply_geometric_view(torch.zeros(1, 1, 4, 4), "rot90")


def test_aligned_base_views_preserve_identity_and_order() -> None:
    image = torch.linspace(-2, 2, 63).reshape(1, 1, 7, 9)
    probabilities, names = build_aligned_view_probabilities(
        PointwiseLogitModel(), image
    )
    expected = torch.sigmoid(image)
    assert names == ("identity", "hflip", "vflip", "hvflip")
    assert probabilities.shape == (4, 1, 1, 7, 9)
    assert torch.equal(probabilities[0], expected)
    for probability in probabilities:
        assert torch.equal(probability, expected)


def test_context_tile_constant_reconstruction() -> None:
    value = 0.37
    tiles = torch.full((4, 2, 1, 224, 224), value)
    reconstructed = stitch_context_tile_probabilities(tiles)
    assert reconstructed.shape == (2, 1, 256, 256)
    assert torch.equal(reconstructed, torch.full_like(reconstructed, value))

    logits = torch.full((1, 1, 256, 256), -0.75)
    inferred = infer_context_tile_probability(PointwiseLogitModel(), logits)
    expected = torch.full_like(inferred, torch.sigmoid(torch.tensor(-0.75)))
    assert torch.allclose(inferred, expected, rtol=0, atol=1e-7)


def test_optional_tile_is_appended_after_all_base_views() -> None:
    image = torch.zeros(1, 1, 256, 256)
    probabilities, names = build_aligned_view_probabilities(
        PointwiseLogitModel(), image, include_context_tile=True
    )
    assert names == (
        "identity",
        "hflip",
        "vflip",
        "hvflip",
        "context_tile",
    )
    assert probabilities.shape == (5, 1, 1, 256, 256)
    assert torch.equal(probabilities, torch.full_like(probabilities, 0.5))


def test_context_tiles_are_forwarded_separately_without_view_batching() -> None:
    model = BatchRecordingPointwiseModel()
    probabilities, _ = build_aligned_view_probabilities(
        model,
        torch.zeros(1, 1, 256, 256),
        include_context_tile=True,
    )

    assert probabilities.shape == (5, 1, 1, 256, 256)
    # Four full-image flips followed by four independent context-tile calls.
    assert model.forward_batch_sizes == [1] * 8


def test_all_aggregation_methods_and_anchor_boundaries() -> None:
    probabilities = torch.tensor([0.1, 0.2, 0.8, 0.9]).reshape(4, 1, 1, 1, 1)

    mean = aggregate_aligned_probabilities(probabilities, "mean")
    trimmed = aggregate_aligned_probabilities(
        probabilities, "trimmed_mean", trim_each_side=1
    )
    weighted = aggregate_aligned_probabilities(
        probabilities, "disagreement_weighted_mean", tau=0.2
    )
    anchor_zero = aggregate_aligned_probabilities(
        probabilities, "source_anchor", beta=0
    )
    anchor_one = aggregate_aligned_probabilities(
        probabilities, "source_anchor", beta=1
    )

    assert mean.item() == pytest.approx(0.5)
    assert trimmed.item() == pytest.approx(0.5)
    assert weighted.item() == pytest.approx(0.5)
    assert anchor_zero.item() == pytest.approx(0.1)
    assert anchor_one.item() == pytest.approx(mean.item())


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"method": "trimmed_mean", "trim_each_side": 2}, "removes every"),
        ({"method": "disagreement_weighted_mean", "tau": 0}, "greater than zero"),
        ({"method": "source_anchor", "beta": -0.1}, r"lie in \[0, 1\]"),
        ({"method": "source_anchor", "beta": 1.1}, r"lie in \[0, 1\]"),
    ],
)
def test_aggregation_rejects_invalid_boundaries(kwargs, message: str) -> None:
    probabilities = torch.full((4, 1, 1, 2, 2), 0.5)
    with pytest.raises(ValueError, match=message):
        aggregate_aligned_probabilities(probabilities, **kwargs)
