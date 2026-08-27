from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn
import yaml

from run_adabn_source_pilot import (
    DEFAULT_CONFIG,
    SourcePilotSample,
    execute_episode_order_gate,
    finalize_and_publish,
    load_config,
    materialize_label_free_inputs,
    validate_protocol_contract,
)
import test_source as source_runner
from tta.episodic_runner import EpisodicRunner
from tta.model_adapter import IRSTDModelAdapter
from tta.state_manager import EpisodicStateManager


class TinyPilotModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(1, 1, 1, bias=False)
        self.bn = nn.BatchNorm2d(1)
        self.dropout = nn.Dropout2d(p=0.99)
        self.head = nn.Conv2d(1, 1, 1, bias=False)
        with torch.no_grad():
            self.conv.weight.fill_(1.0)
            self.bn.weight.fill_(1.0)
            self.bn.bias.zero_()
            self.bn.running_mean.fill_(5.0)
            self.bn.running_var.fill_(4.0)
            self.head.weight.fill_(1.0)

    def forward(self, image: torch.Tensor, warm_flag: bool):
        features = self.dropout(self.bn(self.conv(image)))
        return ([features] if warm_flag else []), self.head(features)


def _sample(image_id: str, image: torch.Tensor) -> SourcePilotSample:
    return SourcePilotSample(
        image_id=image_id,
        image=image.clone(),
        metadata={
            "image_id": image_id,
            "original_size": [4, 4],
            "dataset": "IRSTD-1K",
            "corruption": "gaussian_noise",
            "severity": 3,
            "seed": 42,
        },
        input_raw_sha256=f"input-{image_id}",
    )


def _tiny_runner() -> tuple[IRSTDModelAdapter, EpisodicRunner]:
    model = TinyPilotModel()
    adapter = IRSTDModelAdapter(model)
    adapter.set_source_eval_mode()
    state = EpisodicStateManager(model, optimizer=None)
    return adapter, EpisodicRunner(adapter, state)


def test_fixed_contract_is_exact_train_only_parent_prefix() -> None:
    config, _paths, selected_ids, _files, validation = validate_protocol_contract()

    assert len(selected_ids) == 32
    assert selected_ids[:3] == ("XDU250", "XDU827", "XDU647")
    assert selected_ids[-1] == "XDU540"
    assert config["scope"]["paper_result"] is False
    assert config["scope"]["performance_metrics_computed"] is False
    assert validation["selected_ids_absent_from_fixed_test"] is True
    assert validation["test_dataset_constructed"] is False
    assert validation["test_images_opened"] == 0
    assert validation["test_masks_opened"] == 0
    assert validation["parent_pilot"]["selected_prefix_exact"] is True


@pytest.mark.parametrize(
    ("section", "key", "replacement", "label"),
    (
        ("dataset", "name", "NUDT-SIRST", "dataset.name"),
        ("dataset", "selected_count", 31, "dataset.selected_count"),
        ("condition", "severity", 5, "condition.severity"),
        ("condition", "seed", 7, "condition.seed"),
        ("execution", "batch_size", 2, "execution.batch_size"),
        (
            "execution",
            "order_check",
            "canonical_only",
            "execution.order_check",
        ),
        (
            "execution",
            "fixed_probability_threshold",
            0.4,
            "execution.fixed_probability_threshold",
        ),
        (
            "outputs",
            "condition_directory",
            "gaussian_noise_S5",
            "outputs.condition_directory",
        ),
    ),
)
def test_protocol_literals_fail_closed(
    tmp_path: Path,
    section: str,
    key: str,
    replacement: object,
    label: str,
) -> None:
    config = yaml.safe_load(DEFAULT_CONFIG.read_text(encoding="utf-8"))
    config[section][key] = replacement
    path = tmp_path / "drifted.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    with pytest.raises(ValueError, match=label.replace(".", r"\.")):
        load_config(path)


def test_fixed_inputs_materialize_exactly_without_retaining_a_label() -> None:
    config, paths, selected_ids, selected_files, _validation = (
        validate_protocol_contract()
    )
    samples, audit = materialize_label_free_inputs(
        config, paths, selected_ids, selected_files
    )

    assert len(samples) == 32
    assert tuple(sample.image_id for sample in samples) == selected_ids
    assert all(not hasattr(sample, "mask") for sample in samples)
    assert all(set(sample.metadata) == {
        "image_id",
        "original_size",
        "dataset",
        "corruption",
        "severity",
        "seed",
    } for sample in samples)
    assert audit["input_tensor_sequence_sha256"] == (
        "d09021a7c46d95bad657bef79fbbe3dbf35e5b5ed0a8fc565578a60784c60885"
    )
    assert audit["source_train_mask_retained_after_hashing"] is False
    assert audit["method_received_mask"] is False
    assert audit["io_guard"]["image_open_count"] == 32
    assert audit["io_guard"]["mask_open_count"] == 32
    assert audit["io_guard"]["forbidden_open_count"] == 0


def test_episode_gate_saves_all_outputs_and_proves_reverse_order(tmp_path: Path) -> None:
    adapter, runner = _tiny_runner()
    samples = (
        _sample("a", torch.linspace(-1.0, 1.0, 16).reshape(1, 1, 4, 4)),
        _sample("b", torch.linspace(2.0, 6.0, 16).reshape(1, 1, 4, 4)),
    )
    condition_dir = tmp_path / "IRSTD-1K" / "gaussian_noise_S3"

    result = execute_episode_order_gate(
        samples=samples,
        adapter=adapter,
        runner=runner,
        condition_dir=condition_dir,
        expected_bn_count=1,
        expected_spatial_size=4,
    )

    metrics = result["metrics"]
    assert metrics["paper_result"] is False
    assert metrics["performance_metrics_computed"] is False
    assert metrics["behavioral_diagnostics"][
        "source_adabn_probability_changed_images"
    ] >= 1
    assert metrics["checks"]["canonical_reverse_logits_bit_exact"] is True
    assert metrics["checks"]["all_parameters_and_buffers_unchanged_before_reset"]
    assert metrics["behavioral_diagnostics"]["saved_output_counts"] == {
        "source_probability_maps": 2,
        "adabn_probability_maps": 2,
        "source_prediction_masks": 2,
        "adabn_prediction_masks": 2,
    }
    for directory, suffix in (
        ("source_probability_maps_256", ".npy"),
        ("probability_maps_256", ".npy"),
        ("source_prediction_masks_256", ".png"),
        ("prediction_masks_256", ".png"),
    ):
        assert len(list((condition_dir / directory).glob(f"*{suffix}"))) == 2

    records = [
        json.loads(line)
        for line in (condition_dir / "per_image.jsonl").read_text().splitlines()
    ]
    assert all(record["canonical_reverse_source_logits_bit_exact"] for record in records)
    assert all(record["canonical_reverse_adabn_logits_bit_exact"] for record in records)
    assert all(record["canonical_reverse_state_bit_exact"] for record in records)
    assert all("mask" not in record["metadata"] for record in records)
    for record in records:
        source_map = np.load(condition_dir / record["source_probability_map"])
        adabn_map = np.load(condition_dir / record["adabn_probability_map"])
        assert source_map.shape == adabn_map.shape == (4, 4)


def _minimal_staging(staging: Path) -> None:
    condition = staging / "IRSTD-1K" / "gaussian_noise_S3"
    condition.mkdir(parents=True)
    (staging / "run_config.yaml").write_text("schema_version: 1\n")
    source_runner.write_json_atomic(staging / "provenance.json", {"ok": True})
    source_runner.write_json_atomic(staging / "aggregate_metrics.json", {"ok": True})
    source_runner.write_json_atomic(condition / "metrics.json", {"ok": True})
    source_runner.write_jsonl_atomic(condition / "per_image.jsonl", [{"ok": True}])
    source_runner.write_jsonl_atomic(
        condition / "adaptation_diagnostics.jsonl", [{"ok": True}]
    )


def test_complete_is_fail_closed_and_publish_is_atomic(tmp_path: Path) -> None:
    staging = tmp_path / ".smoke.build"
    final = tmp_path / "smoke"
    _minimal_staging(staging)

    with pytest.raises(RuntimeError, match="gates failed"):
        finalize_and_publish(
            staging=staging,
            final_root=final,
            protocol_id="test",
            protocol_sha256="abc",
            aggregate_checks={"state_reset": False},
        )
    assert not (staging / "COMPLETE.json").exists()
    assert not final.exists()

    complete = finalize_and_publish(
        staging=staging,
        final_root=final,
        protocol_id="test",
        protocol_sha256="abc",
        aggregate_checks={"state_reset": True},
    )
    assert complete["complete"] is True
    assert not staging.exists()
    assert (final / "artifact_manifest.json").is_file()
    assert json.loads((final / "COMPLETE.json").read_text())["complete"] is True
