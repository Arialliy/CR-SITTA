from __future__ import annotations

from pathlib import Path
import sys

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tta.parameter_vector import ParameterVectorError, vectorize_named_tensors


def test_layout_aligns_names_and_round_trips_without_order_dependence() -> None:
    layout, vector = vectorize_named_tensors(
        [
            ("decoder.weight", torch.tensor([[1.0, 2.0]], dtype=torch.float32)),
            ("decoder.bias", torch.tensor([3.0], dtype=torch.float32)),
        ],
        label="parameters",
    )
    assert layout.names == ("decoder.weight", "decoder.bias")
    assert vector.dtype == torch.float64
    assert vector.tolist() == [1.0, 2.0, 3.0]
    reordered = layout.flatten(
        {
            "decoder.bias": torch.tensor([30.0]),
            "decoder.weight": torch.tensor([[10.0, 20.0]]),
        },
        label="reordered",
    )
    assert reordered.tolist() == [10.0, 20.0, 30.0]
    restored = layout.unflatten(reordered)
    assert tuple(restored["decoder.weight"].shape) == (1, 2)
    assert restored["decoder.bias"].tolist() == [30.0]
    assert len(layout.parameter_names_sha256) == 64
    assert len(layout.topology_sha256) == 64


def test_none_gradient_can_only_be_explicitly_zero_filled() -> None:
    layout, _ = vectorize_named_tensors(
        {"a": torch.tensor([1.0]), "b": torch.tensor([2.0, 3.0])},
        label="parameters",
    )
    with pytest.raises(ParameterVectorError, match="is None"):
        layout.flatten(
            {"a": torch.tensor([4.0]), "b": None}, label="gradients"
        )
    zero_filled = layout.flatten(
        {"a": torch.tensor([4.0]), "b": None},
        label="gradients",
        none_as_zero=True,
    )
    assert zero_filled.tolist() == [4.0, 0.0, 0.0]


def test_topology_and_group_mapping_fail_closed() -> None:
    layout, _ = vectorize_named_tensors(
        {"a": torch.tensor([1.0]), "b": torch.tensor([2.0])},
        label="parameters",
    )
    with pytest.raises(ParameterVectorError, match="topology mismatch"):
        layout.flatten({"a": torch.tensor([1.0])}, label="missing")
    with pytest.raises(ParameterVectorError, match="shape mismatch"):
        layout.flatten(
            {"a": torch.tensor([[1.0]]), "b": torch.tensor([2.0])},
            label="shape",
        )
    with pytest.raises(ParameterVectorError, match="cover the layout exactly"):
        layout.validated_group_assignment({"a": "encoder"})
    groups = layout.group_indices({"a": "encoder", "b": "decoder"})
    assert tuple(groups) == ("encoder", "decoder")
    assert groups["encoder"].tolist() == [0]
    assert groups["decoder"].tolist() == [1]


def test_nonfinite_and_duplicate_names_are_rejected() -> None:
    with pytest.raises(ParameterVectorError, match="NaN or Inf"):
        vectorize_named_tensors(
            {"a": torch.tensor([float("nan")])}, label="bad"
        )
    with pytest.raises(ParameterVectorError, match="duplicate"):
        vectorize_named_tensors(
            [("a", torch.tensor([1.0])), ("a", torch.tensor([2.0]))],
            label="duplicate",
        )
