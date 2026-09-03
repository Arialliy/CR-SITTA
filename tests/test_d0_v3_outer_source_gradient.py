from __future__ import annotations

import torch
from torch import nn

from analysis.d0_v2_task_loss import D0V2TaskLossConfig
from analysis.d0_v3_outer_analyzer import FlatParameterLayout
from analysis.source_train_provenance import SourceTrainAnalysisProvenance
from tta.d0_v3_outer_source_gradient import (
    D0V3OuterSourceModel,
    compute_d0_v3_outer_source_gradient,
)
from tta.model_adapter import IRSTDModelAdapter
from tta.state_manager import EpisodicStateManager


class _TinySource(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.bn = nn.BatchNorm2d(3)
        self.head = nn.Conv2d(3, 1, kernel_size=1, bias=False)

    def forward(self, value, warm_flag=False):
        logits = self.head(self.bn(value))
        return (), logits


def _worker() -> D0V3OuterSourceModel:
    model = _TinySource()
    adapter = IRSTDModelAdapter(model, warm_flag=False)
    adapter.set_source_eval_mode()
    manager = EpisodicStateManager(model)
    adapter.set_tent_mode(use_batch_stats=False)
    parameters, names = adapter.collect_adaptable_params()
    named = tuple(zip(names, parameters, strict=True))
    layout = FlatParameterLayout.from_named_tensors(named)
    source = layout.pack(
        {name: parameter.detach().cpu() for name, parameter in named}
    )
    manager.reset_to_source()
    return D0V3OuterSourceModel(
        model=model,
        adapter=adapter,
        state_manager=manager,
        parameters=tuple(parameters),
        parameter_names=tuple(names),
        layout=layout,
        source_parameters_flat=source,
        fine_group_assignment={name: "tiny_bn" for name in names},
        checkpoint_wrapper="state_dict",
        checkpoint_sha256="a" * 64,
    )


def test_outer_source_gradient_has_no_optimizer_and_resets_exactly() -> None:
    worker = _worker()
    source = worker.state_manager.source_fingerprint
    result = compute_d0_v3_outer_source_gradient(
        worker,
        image=torch.rand(1, 3, 4, 4, dtype=torch.float32),
        target=torch.zeros(1, 1, 4, 4, dtype=torch.float32),
        task_loss_config=D0V2TaskLossConfig(
            lambda_bce=1.0,
            lambda_soft_iou=1.0,
            eps=1.0e-6,
        ),
        provenance=SourceTrainAnalysisProvenance(
            dataset="synthetic-train-only",
            split_name="train",
            split_sha256="b" * 64,
            checkpoint_sha256="a" * 64,
            seed=42,
            oracle_analysis=True,
            outer_evaluator_label_accesses=1,
            supervised_gradient_role="outer_oracle_train_labels_only",
        ),
    )
    assert result.source_logits.shape == (1, 1, 4, 4)
    assert result.supervised_gradient_flat.numel() == 6
    assert torch.isfinite(result.supervised_gradient_flat).all()
    assert result.source_state_sha256 == source.full_sha256
    assert result.reset_source_state_sha256 == source.full_sha256
    assert worker.state_manager.assert_source_state() == source
    assert not hasattr(worker, "optimizer")
