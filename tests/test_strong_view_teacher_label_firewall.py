from __future__ import annotations

import inspect

import pytest
import torch
from torch import nn

from tta.proposals.source_multiview_teacher import (
    build_aligned_view_probabilities,
)
from tta.teachers import (
    build_strong_teacher,
    build_strong_teacher_from_aligned_probabilities,
)


class ToyPredictor(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.gain = nn.Parameter(torch.tensor(2.0))

    def forward(self, image: torch.Tensor):
        # A fixed canvas bias makes aligned flip predictions differ, which
        # exercises robust aggregation rather than an equivariant shortcut.
        width = image.shape[-1]
        ramp = torch.linspace(-0.4, 0.4, width, device=image.device)
        logits = self.gain * image[:, :1] + ramp.reshape(1, 1, 1, width)
        return [], logits


def _image() -> torch.Tensor:
    image = torch.zeros(1, 3, 16, 18)
    image[..., 6:9, 7:10] = 1.0
    return image


def test_strong_teacher_matches_explicit_median_and_is_fully_detached() -> None:
    model = ToyPredictor()
    model.train()
    image = _image().requires_grad_()
    output = build_strong_teacher(
        model,
        image,
        ("identity", "hflip", "vflip", "hvflip"),
        aggregation="median",
        candidate_threshold=0.75,
    )
    aligned, _ = build_aligned_view_probabilities(model, image.detach())
    expected = aligned.sort(dim=0, stable=True).values[(aligned.shape[0] - 1) // 2]
    assert torch.equal(output.probability, expected)
    assert model.training is True
    assert model.gain.requires_grad is True
    assert image.grad is None
    assert output.view_names == ("identity", "hflip", "vflip", "hvflip")
    assert output.aggregation == "median"

    tensors = (
        output.probability,
        output.logits,
        output.view_variance,
        output.stability_mask,
        output.local_contrast_map,
        output.target_weight,
        output.background_weight,
        output.guard_union,
        *output.candidate_core,
        *output.candidate_ring,
    )
    assert all(not tensor.requires_grad and tensor.grad_fn is None for tensor in tensors)


def test_candidate_core_ring_and_guard_are_prediction_derived_and_aligned() -> None:
    output = build_strong_teacher(
        ToyPredictor(),
        _image(),
        ("identity", "hflip", "vflip", "hvflip"),
        target_threshold=0.75,
        candidate_threshold=0.75,
        candidate_min_area=2,
        ring_inner_radius=1,
        ring_outer_radius=3,
    )
    assert len(output.candidate_core) >= 1
    assert len(output.candidate_core) == len(output.candidate_ring)
    for core, ring in zip(output.candidate_core, output.candidate_ring):
        assert core.dtype == torch.bool and ring.dtype == torch.bool
        assert core.shape == output.probability.shape
        assert ring.shape == output.probability.shape
        assert not torch.logical_and(core, ring).any()
        assert core.any()
    assert output.guard_union.shape == output.probability.shape
    assert output.target_weight.shape == output.probability.shape
    assert output.background_weight.shape == output.probability.shape
    assert not torch.logical_and(
        output.target_weight > 0, output.background_weight > 0
    ).any()


def test_trimmed_mean_is_explicit_and_plain_mean_is_rejected() -> None:
    output = build_strong_teacher(
        ToyPredictor(),
        _image(),
        ("identity", "hflip", "vflip", "hvflip"),
        aggregation="trimmed_mean",
        trim_each_side=1,
    )
    assert output.aggregation == "trimmed_mean"
    with pytest.raises(ValueError, match="median.*trimmed_mean"):
        build_strong_teacher(
            ToyPredictor(),
            _image(),
            ("identity", "hflip"),
            aggregation="mean",  # type: ignore[arg-type]
        )


def test_sealed_aligned_stack_path_matches_live_teacher() -> None:
    model = ToyPredictor()
    image = _image()
    names = ("identity", "hflip", "vflip", "hvflip")
    aligned, aligned_names = build_aligned_view_probabilities(model, image)
    cached = build_strong_teacher_from_aligned_probabilities(
        aligned,
        aligned_names,
        expected_view_names=names,
        aggregation="median",
        candidate_threshold=0.75,
    )
    live = build_strong_teacher(
        model,
        image,
        names,
        aggregation="median",
        candidate_threshold=0.75,
    )
    assert torch.equal(cached.probability, live.probability)
    assert torch.equal(cached.view_variance, live.view_variance)
    assert torch.equal(cached.target_weight, live.target_weight)
    assert torch.equal(cached.background_weight, live.background_weight)
    assert all(
        torch.equal(left, right)
        for left, right in zip(
            cached.candidate_core, live.candidate_core, strict=True
        )
    )


def test_sealed_teacher_enforces_exact_expected_view_sequence_and_identity() -> None:
    model = ToyPredictor()
    aligned, names = build_aligned_view_probabilities(model, _image())

    with pytest.raises(ValueError, match="exactly match.*order"):
        build_strong_teacher_from_aligned_probabilities(
            aligned,
            names,
            expected_view_names=tuple(reversed(names)),
        )

    nonidentity_names = names[1:]
    with pytest.raises(ValueError, match="include identity"):
        build_strong_teacher_from_aligned_probabilities(
            aligned[1:],
            nonidentity_names,
            expected_view_names=nonidentity_names,
        )


def test_teacher_rejects_dynamic_views_and_multisample_batch() -> None:
    with pytest.raises(ValueError, match="unknown"):
        build_strong_teacher(ToyPredictor(), _image(), ("identity", "rotate17"))
    with pytest.raises(ValueError, match=r"\[1,C,H,W\]"):
        build_strong_teacher(
            ToyPredictor(), _image().repeat(2, 1, 1, 1), ("identity",)
        )


def test_strong_teacher_signature_has_strict_label_and_condition_firewall() -> None:
    forbidden = {
            "gt",
            "ground_truth",
            "outer_mask",
            "mask",
            "target",
            "label",
            "condition",
            "corruption",
            "severity",
            "dataset_name",
            "split",
    }
    for function in (
        build_strong_teacher,
        build_strong_teacher_from_aligned_probabilities,
    ):
        names = set(inspect.signature(function).parameters)
        assert names.isdisjoint(forbidden)
