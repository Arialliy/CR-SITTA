from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from dataio.corruption_cache import sha256_file
from benchmark import checkpoint_axis as axis_api
from benchmark import source_corruption_axis_runner_v2 as source_v2
import run_source_corruption_benchmark as source_v1


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )


def _axis(*, role: str = "best_pd", dataset: str = "IRSTD-1K") -> SimpleNamespace:
    return SimpleNamespace(
        role=role,
        dataset=dataset,
        checkpoint_path=PROJECT_ROOT
        / "results/retraining_fixed_split"
        / dataset
        / f"{role}.pth.tar",
        checkpoint_sha256="a" * 64,
        config_sha256="b" * 64,
        development_only=True,
        artifact_kind="source",
        protocol_id="cr-sitta-development-best-pd-axis-v1",
        expected_epoch=1,
        split_sha256="c" * 64,
        threshold_transform="sigmoid",
        threshold_rule="strict_greater_than",
        threshold_value=0.5,
    )


def _sealed_source_artifact(root: Path, *, role: str = "best_pd") -> None:
    axis = _axis(role=role)
    benchmark = {
        "schema_version": 2,
        "artifact_kind": "source",
        "method": "Source",
        "checkpoint_role": role,
        "dataset": "IRSTD-1K",
        "condition_count": 13,
        "evaluated_images_per_condition": 1,
        "ordered_ids_sha256": source_v2.ordered_ids_sha256(("a",)),
        "conditions": [
            {"condition_key": f"{corruption}_S{severity}"}
            for corruption, severity in source_v1._conditions(
                source_v1._load_protocol(
                    source_v2.DEFAULT_SOURCE_PROTOCOL, "IRSTD-1K"
                )[0]
            )
        ],
    }
    _write_json(root / "benchmark.json", benchmark)
    _write_json(root / "run_config.json", {"checkpoint_role": role})
    condition_files: dict[str, object] = {}
    for corruption, severity in source_v1._conditions(
        source_v1._load_protocol(source_v2.DEFAULT_SOURCE_PROTOCOL, "IRSTD-1K")[0]
    ):
        key = f"{corruption}_S{severity}"
        directory = root / "conditions" / key
        _write_json(
            directory / "metrics.json",
            {
                "condition_key": key,
                "checkpoint_role": role,
                "evaluated_images": 1,
            },
        )
        shard = directory / "probabilities_256.npy"
        shard.parent.mkdir(parents=True, exist_ok=True)
        np.save(shard, np.zeros((1, 256, 256), dtype=np.float32), allow_pickle=False)
        mask = directory / "prediction_masks_256" / "a.png"
        mask.parent.mkdir()
        mask.write_bytes(b"fixture-png")
        record = {
            "index": 0,
            "image_id": "a",
            "prediction_mask": "prediction_masks_256/a.png",
            "prediction_mask_sha256": sha256_file(mask),
            "probability_shard_sha256": sha256_file(shard),
        }
        (directory / "per_image.jsonl").write_text(
            json.dumps(record, sort_keys=True) + "\n", encoding="utf-8"
        )
    ledger = axis_api.artifact_tree_ledger(
        root, exclude=("artifact_manifest.json", "COMPLETE.json")
    )
    files = {
        record["path"]: {
            "sha256": record["sha256"],
            "bytes": record["size_bytes"],
        }
        for record in ledger["files"]
    }
    for corruption, severity in source_v1._conditions(
        source_v1._load_protocol(source_v2.DEFAULT_SOURCE_PROTOCOL, "IRSTD-1K")[0]
    ):
        key = f"{corruption}_S{severity}"
        condition_files[key] = {
            name: {"path": path, "sha256": files[path]["sha256"]}
            for name, path in {
                "metrics": f"conditions/{key}/metrics.json",
                "per_image": f"conditions/{key}/per_image.jsonl",
                "probability_shard": f"conditions/{key}/probabilities_256.npy",
            }.items()
        }
    manifest = axis_api.build_artifact_manifest(
        root,
        axis=axis,
        required_payloads=("benchmark.json", "run_config.json"),
        extra={
            "method": "Source",
            "development_only": True,
            "main_paper_table": False,
            "extra_best_pd_tuning_episodes": 0,
            "condition_count": 13,
            "condition_files": condition_files,
            "files": files,
            "best_miou_parity_gate": (
                {
                    "mode": "producer",
                    "passed": True,
                    "authorization_receipt": False,
                }
                if role == "best_miou"
                else {
                    "mode": "consumer",
                    "passed": True,
                    "receipt_sha256": "d" * 64,
                }
            ),
        },
    )
    _write_json(root / "artifact_manifest.json", manifest)
    _write_json(
        root / "COMPLETE.json",
        {
            "complete": True,
            "schema_version": 1,
            "artifact_contract": axis_api.ARTIFACT_CONTRACT,
            "artifact_kind": "source",
            "protocol_id": axis.protocol_id,
            "dataset": "IRSTD-1K",
            "checkpoint_role": role,
            "axis_config_sha256": "b" * 64,
            "checkpoint_sha256": "a" * 64,
            "checkpoint_epoch": axis.expected_epoch,
            "split_sha256": axis.split_sha256,
            "payload_tree_sha256": ledger["sha256"],
            "payload_file_count": ledger["file_count"],
            "condition_count": 13,
            "manifest_sha256": sha256_file(root / "artifact_manifest.json"),
        },
    )


def test_frozen_v1_runner_and_protocol_remain_byte_exact() -> None:
    assert sha256_file(PROJECT_ROOT / "run_source_corruption_benchmark.py") == (
        "d6e5815a5ddda2aa68e9429f19fbe65ca3de29e2828d35595e6e00de129c0317"
    )
    assert sha256_file(
        PROJECT_ROOT / "configs/source_corruption_benchmark_fixed_splits.yaml"
    ) == source_v2.SOURCE_PROTOCOL_SHA256


def test_v2_reuses_the_exact_v1_thirteen_condition_contract() -> None:
    protocol, _ = source_v1._load_protocol(
        source_v2.DEFAULT_SOURCE_PROTOCOL, "IRSTD-1K"
    )

    assert source_v1._conditions(protocol) == (
        ("clean", 0),
        ("gaussian_noise", 1),
        ("gaussian_noise", 3),
        ("gaussian_noise", 5),
        ("gaussian_blur", 1),
        ("gaussian_blur", 3),
        ("gaussian_blur", 5),
        ("low_contrast", 1),
        ("low_contrast", 3),
        ("low_contrast", 5),
        ("stripe_noise", 1),
        ("stripe_noise", 3),
        ("stripe_noise", 5),
    )


def test_source_artifact_recursive_verifier_accepts_sealed_tree(tmp_path: Path) -> None:
    root = tmp_path / "best_pd" / "IRSTD-1K"
    _sealed_source_artifact(root)

    audit = source_v2.verify_source_artifact(root, expected_axis=_axis())

    assert audit["payload_tree"]["file_count"] == 54
    assert audit["benchmark"]["condition_count"] == 13


def test_source_artifact_recursive_verifier_rejects_payload_tamper(tmp_path: Path) -> None:
    root = tmp_path / "best_pd" / "IRSTD-1K"
    _sealed_source_artifact(root)
    (root / "benchmark.json").write_text("{}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="recursive payload tree"):
        source_v2.verify_source_artifact(root)


def test_source_artifact_recursive_verifier_rejects_unbound_extra_file(
    tmp_path: Path,
) -> None:
    root = tmp_path / "best_pd" / "IRSTD-1K"
    _sealed_source_artifact(root)
    (root / "unbound.txt").write_text("not sealed\n", encoding="utf-8")

    with pytest.raises(ValueError, match="recursive payload tree"):
        source_v2.verify_source_artifact(root)


def test_source_artifact_recursive_verifier_rejects_symlink_root(
    tmp_path: Path,
) -> None:
    real = tmp_path / "real"
    _sealed_source_artifact(real)
    alias = tmp_path / "alias"
    alias.symlink_to(real, target_is_directory=True)

    with pytest.raises(ValueError, match="symlink"):
        source_v2.verify_source_artifact(alias)


def test_checkpoint_axis_enforces_role_first_and_frozen_parity_destinations() -> None:
    config = axis_api.load_axis_config()
    best_pd = axis_api.resolve_axis(
        config,
        dataset="IRSTD-1K",
        role="best_pd",
        artifact_kind="source",
        verify_files=False,
        verify_parity_gate=False,
    )
    assert best_pd.output_dir == (
        PROJECT_ROOT
        / "results/source_corruption_checkpoint_axis_v2/best_pd/IRSTD-1K"
    )
    parity_output = (
        PROJECT_ROOT
        / "results/checkpoint_axis_v2_parity_candidates/best_miou/source/IRSTD-1K"
    )
    best_miou = axis_api.resolve_axis(
        config,
        dataset="IRSTD-1K",
        role="best_miou",
        artifact_kind="source",
        output_override=parity_output,
        verify_files=False,
    )
    assert best_miou.output_dir == parity_output

    with pytest.raises(ValueError, match="parity destination"):
        axis_api.resolve_axis(
            config,
            dataset="IRSTD-1K",
            role="best_miou",
            artifact_kind="source",
            output_override=PROJECT_ROOT / "results/arbitrary/IRSTD-1K",
            verify_files=False,
        )


def test_best_pd_receipt_rejects_arbitrary_path_before_read(tmp_path: Path) -> None:
    arbitrary = tmp_path / "PARITY_RECEIPT.json"
    _write_json(arbitrary, {"passed": True})

    with pytest.raises(ValueError, match="fixed global path"):
        source_v2.verify_best_miou_parity_receipt(arbitrary, axis=_axis())


def test_checkpoint_role_validation_is_fail_closed() -> None:
    with pytest.raises(KeyError):
        source_v2._validate_checkpoint_against_source_protocol(
            {"dataset": "IRSTD-1K", "selection_metric": "pd"},
            axis=_axis(role="not_a_role"),
            dataset_contract={},
        )
