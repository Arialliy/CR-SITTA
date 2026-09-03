from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest

from benchmark import checkpoint_axis
from benchmark import adabn_axis_runner_v2 as axis_runner
from dataio.corruption_cache import condition_key, ordered_ids_sha256, sha256_file


def _json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n",
        encoding="utf-8",
    )


def _jsonl(path: Path, values: list[object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(
            json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            + "\n"
            for value in values
        ),
        encoding="utf-8",
    )


def _axis(
    *,
    role: str = "best_pd",
    dataset: str = "IRSTD-1K",
    artifact_kind: str = "source",
    output_dir: Path = Path("/artifact"),
) -> SimpleNamespace:
    return SimpleNamespace(
        role=role,
        dataset=dataset,
        checkpoint_sha256="a" * 64,
        checkpoint_path=Path("/checkpoint.pth.tar"),
        config_sha256="b" * 64,
        artifact_kind=artifact_kind,
        protocol_id="cr-sitta-development-best-pd-axis-v1",
        expected_epoch=1,
        split_sha256="d" * 64,
        threshold_transform="sigmoid",
        threshold_rule="strict_greater_than",
        threshold_value=0.5,
        expected_images=1,
        output_dir=output_dir,
        parity_receipt_path=(Path("/PARITY_RECEIPT.json") if role == "best_pd" else None),
        parity_receipt_sha256=("e" * 64 if role == "best_pd" else None),
    )


def _development_fields() -> dict[str, object]:
    return {
        "development_only": True,
        "main_paper_table": False,
        "extra_best_pd_tuning_episodes": 0,
    }


def _source_fixture(
    root: Path,
    *,
    role: str = "best_pd",
    dataset: str = "IRSTD-1K",
    axis_sha: str = "b" * 64,
) -> Path:
    image_id = "sample"
    condition_summaries: list[dict[str, object]] = []
    condition_files: dict[str, dict[str, dict[str, str]]] = {}
    for corruption, severity in axis_runner.CONDITIONS:
        key = condition_key(corruption, severity)
        condition_root = root / "conditions" / key
        condition_root.mkdir(parents=True)
        probability = np.zeros((1, 256, 256), dtype="<f4")
        probability_path = condition_root / "probabilities_256.npy"
        np.save(probability_path, probability, allow_pickle=False)
        mask_path = condition_root / "prediction_masks_256" / "sample.png"
        mask_path.parent.mkdir(parents=True)
        Image.fromarray(np.zeros((256, 256), dtype=np.uint8), mode="L").save(mask_path)
        shard_sha = sha256_file(probability_path)
        _jsonl(
            condition_root / "per_image.jsonl",
            [
                {
                    "index": 0,
                    "image_id": image_id,
                    "probability_shard": "probabilities_256.npy",
                    "probability_shard_index": 0,
                    "probability_shard_sha256": shard_sha,
                    "prediction_mask": "prediction_masks_256/sample.png",
                    "prediction_mask_sha256": sha256_file(mask_path),
                }
            ],
        )
        _json(
            condition_root / "metrics.json",
            {
                "schema_version": 2,
                "artifact_kind": "source",
                "method": "Source",
                "checkpoint_role": role,
                "dataset": dataset,
                "condition_key": key,
                "evaluated_images": 1,
                **_development_fields(),
            },
        )
        condition_summaries.append(
            {
                "condition_key": key,
                "corruption": corruption,
                "severity": severity,
            }
        )
        condition_files[key] = {
            "metrics": {
                "path": f"conditions/{key}/metrics.json",
                "sha256": sha256_file(condition_root / "metrics.json"),
            },
            "per_image": {
                "path": f"conditions/{key}/per_image.jsonl",
                "sha256": sha256_file(condition_root / "per_image.jsonl"),
            },
            "probability_shard": {
                "path": f"conditions/{key}/probabilities_256.npy",
                "sha256": sha256_file(condition_root / "probabilities_256.npy"),
            },
        }
    common = {
        "schema_version": 2,
        "artifact_kind": "source",
        "method": "Source",
        "checkpoint_role": role,
        "dataset": dataset,
        "axis_config_sha256": axis_sha,
        "checkpoint_sha256": "a" * 64,
        **_development_fields(),
    }
    parity_gate = (
        {
            "passed": True,
            "mode": "consumer",
            "receipt": "/PARITY_RECEIPT.json",
            "receipt_sha256": "e" * 64,
        }
        if role == "best_pd"
        else {"passed": True, "mode": "producer"}
    )
    benchmark = {
        **common,
        "condition_count": 13,
        "evaluated_images_per_condition": 1,
        "ordered_ids_sha256": ordered_ids_sha256((image_id,)),
        "conditions": condition_summaries,
        "best_miou_parity_gate": parity_gate,
    }
    _json(root / "benchmark.json", benchmark)
    _json(root / "run_config.json", {"checkpoint_role": role})
    source_axis = _axis(role=role, dataset=dataset, output_dir=root)
    manifest = checkpoint_axis.build_artifact_manifest(
        root,
        axis=source_axis,
        required_payloads=("benchmark.json", "run_config.json"),
        extra={
            "method": "Source",
            **_development_fields(),
            "condition_count": 13,
            "condition_files": condition_files,
            "files": {},
            "best_miou_parity_gate": parity_gate,
        },
    )
    manifest["files"] = {
        record["path"]: {
            "sha256": record["sha256"],
            "bytes": record["size_bytes"],
        }
        for record in manifest["payload_tree"]["files"]
    }
    _json(root / "artifact_manifest.json", manifest)
    manifest_sha = sha256_file(root / "artifact_manifest.json")
    _json(
        root / "COMPLETE.json",
        {
            "schema_version": 1,
            "complete": True,
            "artifact_contract": checkpoint_axis.ARTIFACT_CONTRACT,
            "artifact_kind": "source",
            "protocol_id": source_axis.protocol_id,
            "axis_config_sha256": axis_sha,
            "dataset": dataset,
            "checkpoint_role": role,
            "checkpoint_sha256": "a" * 64,
            "checkpoint_epoch": 1,
            "split_sha256": "d" * 64,
            "manifest_sha256": manifest_sha,
            "payload_tree_sha256": manifest["payload_tree"]["sha256"],
            "payload_file_count": manifest["payload_tree"]["file_count"],
            "method": "Source",
            **_development_fields(),
            "condition_count": 13,
            "benchmark_sha256": sha256_file(root / "benchmark.json"),
            "parity_receipt_sha256": ("e" * 64 if role == "best_pd" else None),
            "best_miou_parity_passed": True,
        },
    )
    return root


def _args(**overrides: object) -> argparse.Namespace:
    values: dict[str, object] = {
        "checkpoint_role": "best_pd",
        "parity_only": False,
        "output_dir": None,
        "source_artifact_root": None,
        "aggregate_only": False,
        "dataset": None,
        "condition": None,
        "max_images": None,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def test_source_v2_is_recursively_verified(tmp_path: Path) -> None:
    root = _source_fixture(tmp_path / "source")
    result = axis_runner.verify_source_dataset_artifact(
        root,
        expected_axis=_axis(output_dir=root),
        axis_config_sha256="b" * 64,
        dataset="IRSTD-1K",
    )

    assert result.image_ids == ("sample",)
    assert result.completion["condition_count"] == 13
    assert len(result.manifest["files"]) == 13 * 4 + 2


def test_source_v2_probability_tamper_is_rejected(tmp_path: Path) -> None:
    root = _source_fixture(tmp_path / "source")
    path = root / "conditions" / "clean_S0" / "probabilities_256.npy"
    with path.open("ab") as handle:
        handle.write(b"tamper")

    with pytest.raises(ValueError, match="recursive payload tree drift"):
        axis_runner.verify_source_dataset_artifact(
            root,
            expected_axis=_axis(output_dir=root),
            axis_config_sha256="b" * 64,
            dataset="IRSTD-1K",
        )


def test_source_v2_unlisted_file_is_rejected(tmp_path: Path) -> None:
    root = _source_fixture(tmp_path / "source")
    (root / "unlisted.txt").write_text("not sealed", encoding="utf-8")

    with pytest.raises(ValueError, match="recursive payload tree drift"):
        axis_runner.verify_source_dataset_artifact(
            root,
            expected_axis=_axis(output_dir=root),
            axis_config_sha256="b" * 64,
            dataset="IRSTD-1K",
        )


def test_source_v2_symlink_root_is_rejected(tmp_path: Path) -> None:
    real = _source_fixture(tmp_path / "real")
    link = tmp_path / "link"
    link.symlink_to(real, target_is_directory=True)

    with pytest.raises((FileNotFoundError, ValueError), match="real directory|symlink"):
        axis_runner.verify_source_dataset_artifact(
            link,
            expected_axis=_axis(output_dir=link),
            axis_config_sha256="b" * 64,
            dataset="IRSTD-1K",
        )


def test_source_v2_wrong_checkpoint_role_is_rejected(tmp_path: Path) -> None:
    root = _source_fixture(tmp_path / "source", role="best_miou")

    with pytest.raises(ValueError, match="checkpoint_role"):
        axis_runner.verify_source_dataset_artifact(
            root,
            expected_axis=_axis(role="best_pd", output_dir=root),
            axis_config_sha256="b" * 64,
            dataset="IRSTD-1K",
        )


def test_recursive_manifest_rejects_parent_escape(tmp_path: Path) -> None:
    root = tmp_path / "artifact"
    root.mkdir(parents=True)

    with pytest.raises(ValueError, match="unsafe"):
        axis_runner.verify_recursive_files(
            root,
            {"../outside": {"sha256": "a" * 64, "bytes": 1}},
            allowed_unlisted=(),
        )


def test_best_miou_requires_explicit_parity_and_source_roots() -> None:
    config = checkpoint_axis.load_axis_config()
    with pytest.raises(ValueError, match="parity-only"):
        axis_runner.resolve_selection(
            _args(checkpoint_role="best_miou"), config
        )
    with pytest.raises(ValueError, match="frozen AdaBN parity candidate root"):
        axis_runner.resolve_selection(
            _args(
                checkpoint_role="best_miou",
                parity_only=True,
                output_dir=Path("results/not-the-frozen-adabn-candidate"),
            ),
            config,
        )


def test_best_miou_parity_output_is_noncanonical() -> None:
    config = checkpoint_axis.load_axis_config()
    selection = axis_runner.resolve_selection(
        _args(
            checkpoint_role="best_miou",
            parity_only=True,
            output_dir=Path(
                "results/checkpoint_axis_v2_parity_candidates/best_miou/adabn"
            ),
            source_artifact_root=Path(
                "results/checkpoint_axis_v2_parity_candidates/best_miou/source"
            ),
        ),
        config,
    )

    assert selection.formal_development_artifact is False
    assert selection.parity_only is True
    assert selection.output_root.name == "adabn"


def test_best_miou_parity_rejects_partial_image_execution() -> None:
    config = checkpoint_axis.load_axis_config()
    with pytest.raises(ValueError, match="--max-images is forbidden"):
        axis_runner.resolve_selection(
            _args(
                checkpoint_role="best_miou",
                parity_only=True,
                max_images=1,
                output_dir=Path(
                    "results/checkpoint_axis_v2_parity_candidates/best_miou/adabn"
                ),
            ),
            config,
        )


def test_best_pd_formal_selection_is_role_first() -> None:
    config = checkpoint_axis.load_axis_config()
    selection = axis_runner.resolve_selection(
        _args(checkpoint_role="best_pd"), config
    )

    assert selection.formal_development_artifact is True
    assert selection.output_root == (
        axis_runner.PROJECT_ROOT
        / "results"
        / "adabn"
        / "adabn_checkpoint_axis_v2"
        / "best_pd"
    )


@pytest.mark.parametrize(
    "overrides, message",
    [
        ({"max_images": 1}, "--max-images is forbidden"),
        (
            {"output_dir": Path("results/best-pd-noncanonical")},
            "--output-dir is forbidden",
        ),
    ],
)
def test_best_pd_rejects_partial_or_noncanonical_execution(
    overrides: dict[str, object], message: str
) -> None:
    config = checkpoint_axis.load_axis_config()
    with pytest.raises(ValueError, match=message):
        axis_runner.resolve_selection(
            _args(checkpoint_role="best_pd", **overrides), config
        )


def test_formal_best_pd_rejects_source_override() -> None:
    config = checkpoint_axis.load_axis_config()
    with pytest.raises(ValueError, match="canonical role-matched Source"):
        axis_runner.resolve_selection(
            _args(
                checkpoint_role="best_pd",
                source_artifact_root=Path("results/synthetic-source"),
            ),
            config,
        )


def test_prepare_context_invokes_fail_closed_global_gate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    marker = RuntimeError("global parity gate reached")

    def blocked(*args: object, **kwargs: object) -> object:
        raise marker

    monkeypatch.setattr(checkpoint_axis, "resolve_axis", blocked)
    with pytest.raises(RuntimeError, match="global parity gate reached"):
        axis_runner.prepare_dataset_context(
            axis_config_path=Path("axis.yaml"),
            axis_config={},
            role="best_pd",
            dataset="IRSTD-1K",
            verify_all_cache_file_hashes=False,
        )


def test_directory_publication_is_no_replace(tmp_path: Path) -> None:
    first = tmp_path / ".result.build-first"
    first.mkdir()
    (first / "payload").write_text("first", encoding="utf-8")
    final = tmp_path / "result"
    axis_runner._publish_directory_noreplace(first, final)
    second = tmp_path / ".result.build-second"
    second.mkdir()
    (second / "payload").write_text("second", encoding="utf-8")

    with pytest.raises(FileExistsError):
        axis_runner._publish_directory_noreplace(second, final)
    assert (final / "payload").read_text(encoding="utf-8") == "first"


def test_condition_staging_path_is_reserved_but_not_precreated(
    tmp_path: Path,
) -> None:
    final = tmp_path / "conditions" / "clean_S0"
    staging = axis_runner._staging_sibling(final)

    assert final.parent.is_dir()
    assert staging.parent == final.parent
    assert staging.name.startswith(".clean_S0.build-")
    assert not staging.exists()


def test_dataset_metadata_postverify_failure_rolls_back_only_metadata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset_root = tmp_path / "axis" / "IRSTD-1K"
    conditions_root = dataset_root / "conditions"
    condition_payloads: list[Path] = []
    for corruption, severity in axis_runner.CONDITIONS:
        condition_root = conditions_root / condition_key(corruption, severity)
        condition_root.mkdir(parents=True)
        for name in (
            "metrics.json",
            "per_image.jsonl",
            "adaptation_diagnostics.jsonl",
            "probabilities_256.npy",
        ):
            path = condition_root / name
            path.write_bytes(f"{condition_root.name}:{name}".encode())
            condition_payloads.append(path)

    axis_config_path = tmp_path / "axis.yaml"
    axis_config_path.write_text("schema_version: 1\n", encoding="utf-8")
    axis_sha = sha256_file(axis_config_path)
    axis = SimpleNamespace(
        artifact_kind="adabn",
        protocol_id="cr-sitta-development-best-pd-axis-v1",
        config_sha256=axis_sha,
        dataset="IRSTD-1K",
        role="best_miou",
        checkpoint_sha256="a" * 64,
        expected_epoch=1,
        split_sha256="d" * 64,
        threshold_transform="sigmoid",
        threshold_rule="strict_greater_than",
        threshold_value=0.5,
        checkpoint_path=tmp_path / "checkpoint.pth.tar",
        output_dir=dataset_root,
        parity_receipt_path=None,
        parity_receipt_sha256=None,
    )
    source = axis_runner.SourceArtifact(
        root=tmp_path / "source",
        benchmark={},
        manifest={},
        completion={},
        manifest_sha256="1" * 64,
        completion_sha256="2" * 64,
        benchmark_sha256="3" * 64,
        image_ids=("sample",),
    )
    context = SimpleNamespace(
        axis=axis,
        source_axis=SimpleNamespace(),
        axis_config_path=axis_config_path,
        axis_config={"schema_version": 1},
        axis_config_sha256=axis_sha,
        source_artifact=source,
        legacy_context=SimpleNamespace(dataset_name="IRSTD-1K"),
    )

    monkeypatch.setattr(checkpoint_axis, "verify_checkpoint_file", lambda axis: None)
    monkeypatch.setattr(
        axis_runner,
        "verify_source_dataset_artifact",
        lambda *args, **kwargs: source,
    )
    monkeypatch.setattr(
        axis_runner,
        "verify_condition_artifact",
        lambda root, **kwargs: {
            "metrics": {
                "summary": {},
                "source_summary": {},
                "deltas_from_source": {},
            },
            "manifest_sha256": "4" * 64,
            "completion_sha256": "5" * 64,
        },
    )
    monkeypatch.setattr(
        axis_runner,
        "verify_dataset_artifact",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("postverify")),
    )

    with pytest.raises(RuntimeError, match="postverify"):
        axis_runner.finalize_dataset(
            context=context,
            dataset_root=dataset_root,
            role="best_miou",
            formal_development_artifact=False,
            parity_only=True,
        )

    for name in (
        "benchmark.json",
        "run_config.yaml",
        "provenance.json",
        "artifact_manifest.json",
        axis_runner.DATASET_SENTINEL,
    ):
        assert not (dataset_root / name).exists()
    assert all(path.is_file() for path in condition_payloads)
    assert not tuple(dataset_root.parent.glob(".IRSTD-1K.metadata-build-*"))


def test_condition_verifier_uses_condition_sentinel_not_dataset_complete(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "dataset" / "conditions" / "clean_S0"
    root.mkdir(parents=True)
    np.save(
        root / "probabilities_256.npy",
        np.zeros((1, 256, 256), dtype=np.float32),
    )
    _jsonl(root / "per_image.jsonl", [{"image_id": "sample"}])
    _jsonl(root / "adaptation_diagnostics.jsonl", [{}])
    order_sentinel = {"passed": True}
    _json(root / "order_isolation_sentinel.json", order_sentinel)
    (root / "condition_run_config.yaml").write_text("runtime: {}\n", encoding="utf-8")
    _json(root / "condition_provenance.json", {"method": "AdaBN"})
    mask = root / "prediction_masks_256" / "sample.png"
    mask.parent.mkdir()
    mask.write_bytes(b"mask")
    contract = {
        "schema_version": axis_runner.SCHEMA_VERSION,
        "protocol_id": axis_runner.PROTOCOL_ID,
        "axis_config_sha256": "b" * 64,
        "checkpoint_role": "best_pd",
        "checkpoint_sha256": "a" * 64,
        "dataset": "IRSTD-1K",
        "condition_key": "clean_S0",
        "method": "AdaBN",
        "corruption": "clean",
        "severity": 0,
        "parity_receipt_path": "/PARITY_RECEIPT.json",
        "parity_receipt_sha256": "e" * 64,
        "formal_development_artifact": True,
        "paper_result": False,
        "parity_only": False,
        **_development_fields(),
    }
    metrics = {
        **contract,
        "evaluated_images": 1,
        "full_fixed_test_split": True,
        "probability_shard_sha256": sha256_file(root / "probabilities_256.npy"),
        "order_isolation_sentinel": order_sentinel,
        "checks": {"synthetic_contract_check": True},
    }
    _json(root / "metrics.json", metrics)
    files = axis_runner._artifact_files(
        root,
        excluded=("artifact_manifest.json", axis_runner.CONDITION_SENTINEL),
    )
    manifest = {**contract, "files": files}
    _json(root / "artifact_manifest.json", manifest)
    completion = {
        **contract,
        "complete": True,
        "scope": "condition",
        "formal_development_artifact": True,
        "evaluated_images": 1,
        "full_fixed_test_split": True,
        "all_required_gates_passed": True,
        "metrics_sha256": files["metrics.json"]["sha256"],
        "artifact_manifest_sha256": sha256_file(root / "artifact_manifest.json"),
    }
    _json(root / axis_runner.CONDITION_SENTINEL, completion)
    monkeypatch.setattr(
        axis_runner.legacy,
        "_verify_condition_internal_semantics",
        lambda **kwargs: None,
    )

    verified = axis_runner.verify_condition_artifact(
        root,
        expected_axis=_axis(
            role="best_pd",
            artifact_kind="adabn",
            output_dir=root.parent.parent,
        ),
        axis_config_sha256="b" * 64,
        expected_source=None,
        expected_corruption="clean",
        expected_severity=0,
        formal_development_artifact=True,
        parity_only=False,
    )

    assert verified["completion"]["complete"] is True
    assert not (root / "COMPLETE.json").exists()


def test_best_miou_scientific_payload_parity_is_exact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project = tmp_path / "project"
    reference = (
        project
        / "results"
        / "adabn"
        / "adabn_batch_stats_v1"
        / "D"
        / "conditions"
        / "clean_S0"
    )
    candidate = tmp_path / "candidate"
    for root in (reference, candidate):
        root.mkdir(parents=True)
        np.save(root / "probabilities_256.npy", np.array([[[0.25, 0.75]]], dtype=np.float32))
        mask = root / "prediction_masks_256" / "a.png"
        mask.parent.mkdir()
        mask.write_bytes(b"same png bytes")
        _jsonl(root / "per_image.jsonl", [{"image_id": "a"}])
        _json(
            root / "metrics.json",
            {
                "summary": {"unified_global_iou": 0.5},
                "official": {"miou": 0.5},
                "unified": {"global_iou": 0.5, "curve": [0.1, 0.2]},
                "source_summary": {"unified_global_iou": 0.6},
                "deltas_from_source": {"unified_global_iou": -0.1},
                "probability_shard_sha256": sha256_file(root / "probabilities_256.npy"),
                "adabn_probability_tensor_sequence_sha256": "c" * 64,
            },
        )
    monkeypatch.setattr(axis_runner, "PROJECT_ROOT", project)
    metrics = json.loads((candidate / "metrics.json").read_text(encoding="utf-8"))
    metrics["unified"]["curve"] = (0.1, 0.2)

    receipt = axis_runner._best_miou_parity(
        staging=candidate, dataset="D", key="clean_S0", metrics=metrics
    )

    assert receipt["passed"] is True
    assert receipt["numeric_tolerance_used"] is False


def test_best_miou_one_probability_bit_drift_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project = tmp_path / "project"
    reference = (
        project
        / "results"
        / "adabn"
        / "adabn_batch_stats_v1"
        / "D"
        / "conditions"
        / "clean_S0"
    )
    candidate = tmp_path / "candidate"
    for root, probability in (
        (reference, np.array([0.5], dtype=np.float32)),
        (candidate, np.nextafter(np.array([0.5], dtype=np.float32), np.array([1.0], dtype=np.float32))),
    ):
        root.mkdir(parents=True)
        np.save(root / "probabilities_256.npy", probability)
        mask = root / "prediction_masks_256" / "a.png"
        mask.parent.mkdir()
        mask.write_bytes(b"same")
        _jsonl(root / "per_image.jsonl", [{"image_id": "a"}])
        _json(root / "metrics.json", {})
    monkeypatch.setattr(axis_runner, "PROJECT_ROOT", project)

    with pytest.raises(ValueError, match="probability shard parity"):
        axis_runner._best_miou_parity(
            staging=candidate, dataset="D", key="clean_S0", metrics={}
        )


def test_frozen_v1_compute_files_are_unchanged() -> None:
    assert sha256_file(axis_runner.FROZEN_ADABN_V1_CONFIG) == (
        axis_runner.FROZEN_ADABN_V1_CONFIG_SHA256
    )
    assert sha256_file(axis_runner.FROZEN_ADABN_V1_RUNNER) == (
        axis_runner.FROZEN_ADABN_V1_RUNNER_SHA256
    )
