from __future__ import annotations

from copy import deepcopy
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image
import pytest
import torch
from torch import nn
from torch.utils.data import Dataset

from dataio.corruption_cache import TensorSequenceHasher, ordered_ids_sha256, sha256_file
from metrics.irstd_metrics import (
    IRSTDEvaluationProtocol,
    UnifiedResearchEvaluator,
    probabilities_from_logits,
)
from metrics.official_metric_adapter import OfficialMetricAdapter
from run_adabn_corruption_benchmark import (
    CONDITIONS,
    DATASETS,
    DEFAULT_PROTOCOL,
    DatasetContext,
    SourceConditionReference,
    build_parser,
    execute_condition,
    load_protocol,
    probability_float32_from_logits,
    resolve_selection,
)
import run_adabn_corruption_benchmark as benchmark
import run_source_corruption_benchmark as source_benchmark
import test_fixed_split_source as fixed_source
import test_source as source_runner
from tta.episodic_runner import EpisodicRunner
from tta.model_adapter import IRSTDModelAdapter
from tta.state_manager import EpisodicStateManager


class TinyAdaBNModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(1, 1, 1, bias=False)
        self.bn = nn.BatchNorm2d(1)
        self.head = nn.Conv2d(1, 1, 1, bias=False)
        with torch.no_grad():
            self.conv.weight.fill_(1.0)
            self.bn.weight.fill_(1.0)
            self.bn.bias.zero_()
            self.bn.running_mean.fill_(5.0)
            self.bn.running_var.fill_(4.0)
            self.head.weight.fill_(1.0)

    def forward(self, image: torch.Tensor, warm_flag: bool):
        features = self.bn(self.conv(image))
        return ([features] if warm_flag else []), self.head(features)


class TinyConditionDataset(Dataset[dict[str, object]]):
    def __init__(self) -> None:
        first = torch.linspace(-1.0, 1.0, 256 * 256, dtype=torch.float32).reshape(
            1, 256, 256
        )
        second = torch.linspace(2.0, 6.0, 256 * 256, dtype=torch.float32).reshape(
            1, 256, 256
        )
        first_target = torch.zeros((1, 256, 256), dtype=torch.float32)
        second_target = torch.zeros((1, 256, 256), dtype=torch.float32)
        first_target[:, 120:124, 120:124] = 1.0
        second_target[:, 130:134, 130:134] = 1.0
        self.samples = (
            ("image-a", first, first_target),
            ("nested/image-b", second, second_target),
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, object]:
        image_id, image, target = self.samples[index]
        return {
            "image": image.clone(),
            "mask": target.clone(),
            "image_id": image_id,
            "original_size": (256, 256),
            "dataset": "IRSTD-1K",
            "corruption": "clean",
            "severity": 0,
            "seed": 42,
        }


def _selection(*arguments: str):
    _path, protocol = load_protocol(DEFAULT_PROTOCOL)
    args = build_parser().parse_args(list(arguments))
    return resolve_selection(args, protocol), protocol


def test_protocol_freezes_probability_shards_and_completion_layers() -> None:
    _path, protocol = load_protocol(DEFAULT_PROTOCOL)

    assert tuple(protocol["datasets_order"]) == DATASETS
    assert tuple(tuple(value) for value in protocol["ordered_conditions"]) == CONDITIONS
    assert protocol["scope"]["formal_conditions_per_dataset"] == 13
    assert protocol["scope"]["formal_global_condition_count"] == 39
    assert protocol["scope"]["full_condition_shards_are_formal"] is True
    assert protocol["outputs"]["probability_storage"] == {
        "format": "one_numpy_npy_shard_per_condition",
        "filename": "probabilities_256.npy",
        "dtype": "little_endian_float32",
        "shape": ["N", 256, 256],
        "per_image_lookup": ["probability_shard", "probability_shard_index"],
    }
    assert protocol["outputs"]["formal_condition_sentinel"] == (
        "CONDITION_COMPLETE.json"
    )
    assert protocol["outputs"]["formal_dataset_sentinel"] == "DATASET_COMPLETE.json"
    assert protocol["outputs"]["formal_global_sentinel"] == "COMPLETE.json"


def test_full_condition_and_dataset_shards_are_formal() -> None:
    condition, protocol = _selection(
        "--dataset", "IRSTD-1K", "--condition", "gaussian_noise_S3"
    )
    assert condition.formal_artifact is True
    assert condition.datasets == ("IRSTD-1K",)
    assert condition.conditions == (("gaussian_noise", 3),)
    assert condition.max_images is None
    assert condition.output_root == benchmark._project_path(
        protocol["outputs"]["formal_root"]
    )
    assert condition.mode != "smoke"

    all_datasets_for_one_condition, _ = _selection(
        "--condition", "gaussian_noise_S3"
    )
    assert all_datasets_for_one_condition.formal_artifact is True
    assert all_datasets_for_one_condition.datasets == DATASETS
    assert all_datasets_for_one_condition.conditions == (("gaussian_noise", 3),)
    assert all_datasets_for_one_condition.mode != "smoke"

    dataset, _ = _selection("--dataset", "IRSTD-1K")
    assert dataset.formal_artifact is True
    assert dataset.datasets == ("IRSTD-1K",)
    assert dataset.conditions == CONDITIONS
    assert dataset.max_images is None
    assert dataset.mode == "formal_dataset_shard"

    complete, _ = _selection()
    assert complete.formal_artifact is True
    assert complete.datasets == DATASETS
    assert complete.conditions == CONDITIONS
    assert complete.mode == "formal_all"


@pytest.mark.parametrize(
    "arguments",
    (
        ("--dataset", "IRSTD-1K", "--condition", "clean_S0", "--max-images", "1"),
        ("--dataset", "IRSTD-1K", "--smoke"),
    ),
)
def test_max_images_and_explicit_smoke_are_never_formal(
    arguments: tuple[str, ...],
) -> None:
    selection, _protocol = _selection(*arguments)

    assert selection.mode == "smoke"
    assert selection.formal_artifact is False


def test_custom_output_is_never_formal(tmp_path: Path) -> None:
    output = tmp_path / "custom"
    selection, _protocol = _selection(
        "--dataset", "IRSTD-1K", "--condition", "clean_S0", "--output-dir", str(output)
    )

    assert selection.mode == "smoke"
    assert selection.formal_artifact is False
    assert selection.output_root == output.resolve()


def test_aggregate_only_rejects_every_run_filter() -> None:
    _path, protocol = load_protocol(DEFAULT_PROTOCOL)
    parser = build_parser()
    for conflicting in (
        ("--dataset", "IRSTD-1K"),
        ("--condition", "clean_S0"),
        ("--max-images", "1"),
        ("--smoke",),
        ("--output-dir", "/tmp/adabn-test-output"),
    ):
        args = parser.parse_args(["--aggregate-only", *conflicting])
        with pytest.raises(ValueError, match="cannot be combined"):
            resolve_selection(args, protocol)


def test_dataset_manifest_condition_keys_survive_sorted_json_order(
    tmp_path: Path,
) -> None:
    path = tmp_path / "manifest.json"
    expected_order = tuple(
        source_benchmark.condition_key(*item) for item in CONDITIONS
    )
    source_runner.write_json_atomic(
        path,
        {"conditions": {key: {"complete": True} for key in expected_order}},
    )
    links = json.loads(path.read_text(encoding="utf-8"))["conditions"]

    assert tuple(links) != expected_order
    benchmark._require_exact_condition_link_keys(links)

    missing = dict(links)
    missing.pop(next(iter(missing)))
    with pytest.raises(ValueError, match="key set drifted"):
        benchmark._require_exact_condition_link_keys(missing)


def test_probability_conversion_matches_frozen_numpy_path_bit_exactly() -> None:
    logits = torch.tensor(
        [[[[-19.999500274658203, 0.0]]]],
        dtype=torch.float32,
    )
    expected = probabilities_from_logits(logits)[0, 0].astype(np.float32)

    actual = probability_float32_from_logits(logits)
    torch_probability = torch.sigmoid(logits)[0, 0].numpy().astype(np.float32)

    assert np.array_equal(actual, expected)
    assert actual.dtype.str == "<f4"
    # This value differs by one float32 ULP on the frozen CPU implementation.
    # The assertion prevents an apparently harmless torch.sigmoid substitution.
    assert not np.array_equal(actual, torch_probability)


def _tensor_hashes(dataset: TinyConditionDataset) -> tuple[str, str]:
    input_hasher = TensorSequenceHasher()
    target_hasher = TensorSequenceHasher()
    for sample in dataset:
        image_id = str(sample["image_id"])
        input_hasher.update(image_id, sample["image"])
        target_hasher.update(image_id, sample["mask"])
    return input_hasher.hexdigest(), target_hasher.hexdigest()


def _tiny_adapter_and_runner() -> tuple[IRSTDModelAdapter, EpisodicRunner]:
    model = TinyAdaBNModel()
    adapter = IRSTDModelAdapter(model)
    adapter.set_source_eval_mode()
    state = EpisodicStateManager(model, optimizer=None)
    return adapter, EpisodicRunner(adapter, state)


def _source_reference(
    root: Path,
    dataset: TinyConditionDataset,
    adapter: IRSTDModelAdapter,
    evaluation_protocol: IRSTDEvaluationProtocol,
) -> SourceConditionReference:
    condition_root = root / "conditions" / "clean_S0"
    probability_path = condition_root / "probabilities_256.npy"
    partial, probability_map = source_benchmark._atomic_probability_memmap(
        probability_path, len(dataset)
    )
    official = OfficialMetricAdapter(image_size=256)
    unified = UnifiedResearchEvaluator(evaluation_protocol)
    records: list[dict[str, object]] = []
    for index, sample in enumerate(dataset):
        image = sample["image"]
        target = sample["mask"]
        assert isinstance(image, torch.Tensor)
        assert isinstance(target, torch.Tensor)
        with torch.no_grad():
            logits = adapter.forward_logits(image.unsqueeze(0)).detach().cpu()
        target_batch = target.unsqueeze(0)
        probability = probability_float32_from_logits(logits)
        probability_map[index] = probability
        source_runner._update_official_evaluator(official, logits, target_batch)
        unified.update_logits(logits, target_batch)
        mask = np.where(probability > 0.5, 255, 0).astype(np.uint8)
        relative_mask = Path("prediction_masks_256") / fixed_source._relative_prediction_path(
            str(sample["image_id"]), ".png"
        )
        fixed_source._write_png_atomic(condition_root / relative_mask, mask)
        records.append(
            {
                "index": index,
                "image_id": str(sample["image_id"]),
                "probability_tensor_raw_sha256": benchmark._raw_array_sha256(
                    probability
                ),
                "prediction_mask": str(relative_mask),
                "prediction_mask_sha256": sha256_file(condition_root / relative_mask),
            }
        )
    probability_map.flush()
    del probability_map
    os.replace(partial, probability_path)
    probability_hash = sha256_file(probability_path)
    for record in records:
        record["probability_shard_sha256"] = probability_hash
    official_result = official.compute()
    unified_result = unified.compute()
    metrics = {
        "condition_key": "clean_S0",
        "evaluated_images": len(dataset),
        "summary": source_benchmark._condition_summary(
            official_result, unified_result
        ),
        "official": official_result.to_dict(),
        "unified": unified_result.to_dict(),
    }
    return SourceConditionReference(
        condition_root=condition_root,
        probabilities=np.load(probability_path, mmap_mode="r", allow_pickle=False),
        records=tuple(records),
        metrics=metrics,
        probability_shard_sha256=probability_hash,
    )


def _tiny_context(
    tmp_path: Path,
    dataset: TinyConditionDataset,
) -> DatasetContext:
    image_ids = tuple(str(sample["image_id"]) for sample in dataset)
    input_hash, target_hash = _tensor_hashes(dataset)
    return DatasetContext(
        dataset_name="IRSTD-1K",
        config_contract={"test_images": len(dataset)},
        source_protocol={},
        source_dataset_contract={},
        cache_dir=tmp_path / "cache",
        cache_manifest={
            "cache_content_sha256": "cache-content",
            "ordered_ids_sha256": ordered_ids_sha256(image_ids),
            "conditions": [
                {
                    "key": "clean_S0",
                    "tensor_sequence_sha256": input_hash,
                }
            ],
            "targets": {"tensor_sequence_sha256": target_hash},
        },
        cache_audit={"manifest_sha256": "cache-manifest"},
        image_ids=image_ids,
        checkpoint=tmp_path / "checkpoint.pth.tar",
        checkpoint_payload={},
        checkpoint_summary={},
        source_root=tmp_path / "source",
        source_benchmark={},
        source_artifact_manifest={},
    )


def _run_tiny_condition(
    tmp_path: Path,
    *,
    tamper_source_official: bool = False,
) -> tuple[Path, dict[str, object]]:
    dataset = TinyConditionDataset()
    context = _tiny_context(tmp_path, dataset)
    adapter, runner = _tiny_adapter_and_runner()
    evaluation_protocol = IRSTDEvaluationProtocol(
        froc_probability_thresholds=(0.0, 0.5, 1.0)
    )
    reference = _source_reference(
        tmp_path / "source",
        dataset,
        adapter,
        evaluation_protocol,
    )
    if tamper_source_official:
        metrics = deepcopy(dict(reference.metrics))
        metrics["official"] = deepcopy(dict(metrics["official"]))
        metrics["official"]["mean_iou"] = float(
            metrics["official"]["mean_iou"]
        ) + 1e-12
        reference = SourceConditionReference(
            condition_root=reference.condition_root,
            probabilities=reference.probabilities,
            records=reference.records,
            metrics=metrics,
            probability_shard_sha256=reference.probability_shard_sha256,
        )
    destination = tmp_path / "adabn-condition"
    metrics = execute_condition(
        context=context,
        corruption="clean",
        severity=0,
        dataset=dataset,
        source_reference=reference,
        adapter=adapter,
        runner=runner,
        evaluation_protocol=evaluation_protocol,
        destination=destination,
        expected_batchnorm_count=1,
        max_images=None,
        formal_artifact=True,
    )
    return destination, metrics


def test_condition_saves_one_float32_shard_and_passes_all_source_parity(
    tmp_path: Path,
) -> None:
    destination, metrics = _run_tiny_condition(tmp_path)

    probability_path = destination / "probabilities_256.npy"
    probability = np.load(probability_path, mmap_mode="r", allow_pickle=False)
    assert probability.shape == (2, 256, 256)
    assert probability.dtype.str == "<f4"
    assert not (destination / "probability_maps_256").exists()
    assert len(list((destination / "prediction_masks_256").rglob("*.png"))) == 2

    records = [
        json.loads(line)
        for line in (destination / "per_image.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    shard_hash = sha256_file(probability_path)
    assert [record["probability_shard_index"] for record in records] == [0, 1]
    assert all(record["probability_shard"] == "probabilities_256.npy" for record in records)
    assert all(record["probability_shard_sha256"] == shard_hash for record in records)
    assert all(record["source_pre_probability_reference_bit_exact"] for record in records)
    assert all(record["source_pre_binary_mask_reference_bit_exact"] for record in records)
    assert all(record["source_pre_binary_mask_file_hash_exact"] for record in records)

    parity = metrics["source_pre_parity"]
    assert parity["stored_source_logits_available"] is False
    assert parity["logits_proxy_gate"] == (
        "exact_float32_probability_plus_binary_mask"
    )
    assert parity["official_metrics_exact"] is True
    assert parity["unified_metrics_exact"] is True
    assert parity["summary_exact"] is True
    assert parity["numeric_tolerance_used"] is False
    assert metrics["delta_role"] == "report_only_never_used_for_selection_or_tuning"
    assert set(metrics["deltas_from_source"]) == set(metrics["summary"])
    assert "niou" not in json.dumps(metrics).lower()


def test_source_aggregate_parity_is_exact_and_fails_on_one_float_drift(
    tmp_path: Path,
) -> None:
    with pytest.raises(RuntimeError, match="Source-pre parity gate failed"):
        _run_tiny_condition(tmp_path, tamper_source_official=True)

    assert not (tmp_path / "adabn-condition" / "metrics.json").exists()


def test_condition_internal_semantics_reject_index_tampering(
    tmp_path: Path,
) -> None:
    destination, generated_metrics = _run_tiny_condition(tmp_path)
    metrics = deepcopy(generated_metrics)
    # The tiny helper uses the intentionally slower generic runner.  Its two
    # episodes nevertheless perform full SHA on both expected indexes.
    metrics["state_audit_schedule"]["exact"] = True
    per_image = benchmark._load_jsonl(destination / "per_image.jsonl")
    diagnostics = benchmark._load_jsonl(
        destination / "adaptation_diagnostics.jsonl"
    )
    manifest_files = {
        str(path.relative_to(destination)): {
            "sha256": sha256_file(path),
            "bytes": path.stat().st_size,
        }
        for path in destination.rglob("*")
        if path.is_file()
    }
    probabilities = np.load(
        destination / "probabilities_256.npy", mmap_mode="r", allow_pickle=False
    )

    benchmark._verify_condition_internal_semantics(
        condition_root=destination,
        metrics=metrics,
        probabilities=probabilities,
        per_image=per_image,
        diagnostics=diagnostics,
        manifest_files=manifest_files,
    )
    tampered = deepcopy(per_image)
    tampered[0]["probability_shard_index"] = 1
    with pytest.raises(ValueError, match="shard index"):
        benchmark._verify_condition_internal_semantics(
            condition_root=destination,
            metrics=metrics,
            probabilities=probabilities,
            per_image=tampered,
            diagnostics=diagnostics,
            manifest_files=manifest_files,
        )


def _materialize_39_completion_stubs(
    output_root: Path,
    protocol: dict[str, object],
) -> None:
    sentinel = protocol["outputs"]["formal_condition_sentinel"]
    for dataset_name in DATASETS:
        for corruption, severity in CONDITIONS:
            key = benchmark.condition_key(corruption, severity)
            condition_root = output_root / dataset_name / "conditions" / key
            condition_root.mkdir(parents=True)
            source_runner.write_json_atomic(
                condition_root / sentinel,
                {
                    "complete": True,
                    "dataset": dataset_name,
                    "condition_key": key,
                    "repository_code_bundle_sha256": "test-code-bundle",
                },
            )


def _patch_lightweight_dataset_aggregation(
    monkeypatch: pytest.MonkeyPatch,
    protocol: dict[str, object],
) -> list[str]:
    finalized: list[str] = []
    condition_sentinel = protocol["outputs"]["formal_condition_sentinel"]
    dataset_sentinel = protocol["outputs"]["formal_dataset_sentinel"]

    monkeypatch.setattr(
        benchmark,
        "_repository_contract",
        lambda: {
            "file_sha256": {"run_adabn_corruption_benchmark.py": "test-code"},
            "code_bundle_sha256": "test-code-bundle",
        },
    )

    def fake_finalize_dataset_from_conditions(**kwargs):
        dataset_name = kwargs["dataset_name"]
        output_root = kwargs["output_root"]
        assert tuple(kwargs["conditions"]) == CONDITIONS
        assert kwargs["formal_artifact"] is True
        for corruption, severity in CONDITIONS:
            key = benchmark.condition_key(corruption, severity)
            completion_path = (
                output_root
                / dataset_name
                / "conditions"
                / key
                / condition_sentinel
            )
            if not completion_path.is_file():
                raise FileNotFoundError(f"missing formal condition shard: {completion_path}")
            completion = json.loads(completion_path.read_text(encoding="utf-8"))
            if completion.get("repository_code_bundle_sha256") != (
                "test-code-bundle"
            ):
                raise ValueError(f"repository code bundle mismatch: {completion_path}")
        finalized.append(dataset_name)
        source_runner.write_json_atomic(
            output_root / dataset_name / dataset_sentinel,
            {"complete": True, "dataset": dataset_name},
        )
        return {"completion": {"complete": True}}

    def fake_verify_dataset_artifact(**kwargs):
        dataset_name = kwargs["dataset_name"]
        output_root = kwargs["output_root"]
        assert (output_root / dataset_name / dataset_sentinel).is_file()
        conditions = [
            {
                "condition_index": index,
                "condition_key": benchmark.condition_key(corruption, severity),
                "corruption": corruption,
                "severity": severity,
            }
            for index, (corruption, severity) in enumerate(CONDITIONS)
        ]
        return {
            "completion": {"dataset_summary_sha256": f"summary-{dataset_name}"},
            "completion_sha256": f"complete-{dataset_name}",
            "manifest_sha256": f"manifest-{dataset_name}",
            "summary": {"conditions": conditions},
        }

    monkeypatch.setattr(
        benchmark,
        "finalize_dataset_from_conditions",
        fake_finalize_dataset_from_conditions,
    )
    monkeypatch.setattr(
        benchmark,
        "verify_dataset_artifact",
        fake_verify_dataset_artifact,
    )
    return finalized


def test_aggregate_only_publishes_global_complete_from_exactly_39_conditions(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    protocol_path, loaded = load_protocol(DEFAULT_PROTOCOL)
    protocol = deepcopy(loaded)
    output_root = tmp_path / "formal"
    _materialize_39_completion_stubs(output_root, protocol)
    finalized = _patch_lightweight_dataset_aggregation(monkeypatch, protocol)

    result = benchmark.aggregate_formal_results(
        protocol_path, protocol, output_root
    )

    assert finalized == list(DATASETS)
    assert result["completion"]["dataset_count"] == 3
    assert result["completion"]["condition_count_per_dataset"] == 13
    assert result["completion"]["global_dataset_condition_count"] == 39
    assert len(result["aggregate"]["datasets"]) == 3
    assert sum(
        len(record["conditions"]) for record in result["aggregate"]["datasets"]
    ) == 39
    assert (output_root / "global_index" / "aggregate_metrics.json").is_file()
    assert (output_root / "global_index" / "artifact_manifest.json").is_file()
    complete = json.loads(
        (output_root / "COMPLETE.json").read_text(encoding="utf-8")
    )
    assert complete["complete"] is True
    assert complete["all_required_gates_passed"] is True


@pytest.mark.parametrize("failure", ("missing_shard", "code_lineage_drift"))
def test_aggregate_only_fails_closed_before_global_complete(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure: str,
) -> None:
    protocol_path, loaded = load_protocol(DEFAULT_PROTOCOL)
    protocol = deepcopy(loaded)
    output_root = tmp_path / "formal"
    _materialize_39_completion_stubs(output_root, protocol)
    first = (
        output_root
        / DATASETS[0]
        / "conditions"
        / benchmark.condition_key(*CONDITIONS[0])
        / protocol["outputs"]["formal_condition_sentinel"]
    )
    if failure == "missing_shard":
        first.unlink()
        expected = "missing formal condition shard"
    else:
        completion = json.loads(first.read_text(encoding="utf-8"))
        completion["repository_code_bundle_sha256"] = "drifted-code"
        source_runner.write_json_atomic(first, completion)
        expected = "repository code bundle mismatch"
    _patch_lightweight_dataset_aggregation(monkeypatch, protocol)

    with pytest.raises((FileNotFoundError, ValueError), match=expected):
        benchmark.aggregate_formal_results(protocol_path, protocol, output_root)

    assert not (output_root / "COMPLETE.json").exists()
    assert not (output_root / "global_index").exists()
