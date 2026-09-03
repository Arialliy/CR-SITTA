from __future__ import annotations

import copy

import pytest
import torch

from analysis.d0_v3_outer_analyzer import (
    D0V3OuterAnalyzerError,
    FlatParameterLayout,
    analyze_formal_stage_a_episode,
)
from analysis.source_train_provenance import SourceTrainAnalysisProvenance
from tta.diagnostics import NoOpThresholds


SHA_A = "a" * 64
SHA_B = "b" * 64


def _layout() -> tuple[FlatParameterLayout, dict[str, torch.Tensor]]:
    named = {
        "encoder.bn.weight": torch.tensor([1.0, 2.0], dtype=torch.float32),
        "decoder.bn.bias": torch.tensor([0.0], dtype=torch.float32),
    }
    layout = FlatParameterLayout.from_named_tensors(tuple(named.items()))
    return layout, named


def _provenance() -> SourceTrainAnalysisProvenance:
    return SourceTrainAnalysisProvenance(
        dataset="NUAA-SIRST",
        split_name="train",
        split_sha256=SHA_A,
        checkpoint_sha256=SHA_B,
        seed=42,
        oracle_analysis=True,
        outer_evaluator_label_accesses=1,
        supervised_gradient_role="outer_oracle_train_labels_only",
    )


def test_layout_round_trip_and_tamper_rejection() -> None:
    layout, named = _layout()
    restored = FlatParameterLayout.from_mapping(layout.to_dict())
    flat = restored.pack(named)
    unpacked = restored.unpack(flat, label="flat")
    assert tuple(unpacked) == tuple(named)
    assert all(torch.equal(unpacked[key], named[key]) for key in named)

    tampered = copy.deepcopy(layout.to_dict())
    tampered["scalar_count"] += 1
    with pytest.raises(D0V3OuterAnalyzerError):
        FlatParameterLayout.from_mapping(tampered)


def test_outer_episode_reports_four_level_metrics_and_twenty_ready_alignment() -> None:
    layout, named = _layout()
    source = layout.pack(named)
    after = source.clone()
    after[0] += 0.01
    entropy_gradient = torch.tensor([0.2, -0.1, 0.3], dtype=torch.float32)
    supervised_gradient = torch.tensor([0.1, -0.2, 0.2], dtype=torch.float32)
    logits_pre = torch.tensor(
        [[[[0.0, 1.0], [-1.0, -2.0]]]], dtype=torch.float32
    )
    logits_post = logits_pre + torch.tensor(
        [[[[0.1, -0.1], [0.2, 0.0]]]], dtype=torch.float32
    )
    target = torch.tensor(
        [[[[1.0, 1.0], [0.0, 0.0]]]], dtype=torch.float32
    )
    report = analyze_formal_stage_a_episode(
        layout=layout,
        source_parameters_flat=source,
        parameters_after_flat=after,
        entropy_gradient_flat=entropy_gradient,
        supervised_gradient_flat=supervised_gradient,
        logits_pre=logits_pre,
        logits_post=logits_post,
        target=target,
        thresholds=NoOpThresholds(),
        provenance=_provenance(),
        fine_group_assignment={
            "encoder.bn.weight": "encoder_0",
            "decoder.bn.bias": "decoder_0",
        },
        first_order_zero_tolerance=0.0,
    )
    assert report["stage2_authorized"] is False
    assert report["noop"]["threshold_rule_v2"] == (
        "strict_probability_greater_than_0_5"
    )
    assert set(report["entropy_task_alignment"]["per_group"]) == {
        "encoder_0",
        "decoder_0",
    }
    bins = report["threshold_margin_bin_response"]
    assert sum(value["pixel_count"] for value in bins.values()) == 4
