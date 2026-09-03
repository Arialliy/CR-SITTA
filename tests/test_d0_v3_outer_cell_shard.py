from __future__ import annotations

from dataclasses import replace
import hashlib
import io
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from analysis import d0_v3_outer_cell_shard as outer
from analysis.d0_v3_formal_contract import (
    FINE_ALIGNMENT_GROUP_IDS,
    FROZEN_CANDIDATES,
    PROTOCOL_ID,
)
from analysis.d0_v3_label_free_shard import VerifiedLabelFreeShard
from analysis.d0_v3_phase_receipt import (
    OUTER_ACCESS_ARTIFACT_TYPE,
    PROTOCOL_ID as PHASE_PROTOCOL_ID,
)


DATASET = "NUAA-SIRST"
CONDITION = "clean_S0"
CONFIG_SHA = "1" * 64
SPLIT_SHA = "2" * 64
CHECKPOINT_SHA = "3" * 64
CACHE_PROTOCOL_SHA = "4" * 64
METHOD_SHA = "5" * 64
PHASE_SHA = "6" * 64
LABEL_MANIFEST_SHA = "7" * 64
LABEL_COMPLETE_SHA = "8" * 64
LAYOUT_FILE_SHA = "9" * 64
LAYOUT_SHA = "a" * 64
SOURCE_STATE_SHA = "b" * 64


def _file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _npy_bytes(value: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    np.save(buffer, np.ascontiguousarray(value, dtype="<f4"), allow_pickle=False)
    return buffer.getvalue()


def _write(path: Path, value: bytes) -> None:
    path.write_bytes(value)


def _scope() -> dict[str, object]:
    return {
        "dataset": DATASET,
        "split_name": "train",
        "split_sha256": SPLIT_SHA,
        "checkpoint_sha256": CHECKPOINT_SHA,
        "seed": 42,
        "source_train_derived": True,
        "paper_test_result": False,
        "use_test_images": False,
        "use_test_labels": False,
        "oracle_analysis": True,
        "method_label_accesses": 0,
        "outer_evaluator_label_accesses": 1,
        "adaptation_gradient_uses_labels": False,
        "supervised_gradient_role": "outer_oracle_train_labels_only",
    }


def _analysis() -> dict[str, object]:
    return {
        "schema_version": 3,
        "artifact_type": "cr_sitta_p3_stage_a_outer_episode",
        "scope": _scope(),
        "label_isolation": {
            "label_free_payload_complete_before_target_open": True,
            "method_label_accesses": 0,
            "outer_evaluator_label_accesses": 1,
            "supervised_gradient_used_by_adaptation": False,
            "adaptation_optimizer_executed_by_outer_evaluator": False,
            "test_payload_accesses": 0,
        },
        "layout_sha256": LAYOUT_SHA,
        "noop": {},
        "threshold_margin_bin_response": {},
        "entropy_task_alignment": {
            "global": {},
            "per_group": {key: {} for key in FINE_ALIGNMENT_GROUP_IDS},
        },
        "scientific_selection_performed": False,
        "stage2_authorized": False,
    }


def _task_loss() -> dict[str, object]:
    return {
        "schema_version": 1,
        "task_loss_type": "d0_v2_outer_oracle_bce_soft_iou",
        "scope": _scope(),
        "label_isolation": {
            "role": "outer_oracle_train_labels_only",
            "source_train_only": True,
            "outer_oracle_analysis": True,
            "used_by_adaptation": False,
            "adaptation_gradient_uses_labels": False,
            "method_label_accesses": 0,
            "outer_evaluator_label_accesses": 1,
        },
        "config": {},
        "input": {},
        "components": {"total_loss": 1.0},
    }


def _build_fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    repo = tmp_path / "repo"
    label = repo / "label"
    shard = repo / "outer"
    label.mkdir(parents=True)
    shard.mkdir()
    image_ids = [f"pilot_{index:02d}" for index in range(64)]
    ids_sha = hashlib.sha256(
        json.dumps(image_ids, ensure_ascii=False, separators=(",", ":")).encode()
    ).hexdigest()
    label_manifest = {
        "ordered_image_ids": image_ids,
        "ordered_image_ids_sha256": ids_sha,
        "dataset_binding": {
            "train_split_sha256": SPLIT_SHA,
            "checkpoint_role": "best_miou",
            "checkpoint_path": "results/checkpoint.pth.tar",
            "checkpoint_sha256": CHECKPOINT_SHA,
        },
        "input_seal": {
            "files": [
                {
                    "role": "cache_method_manifest",
                    "path": "cache/method_input_manifest.json",
                    "sha256": METHOD_SHA,
                },
                {
                    "role": "cache_protocol",
                    "path": "configs/cache.yaml",
                    "sha256": CACHE_PROTOCOL_SHA,
                },
            ]
        },
    }
    label_manifest_bytes = outer.canonical_json_bytes(label_manifest, newline=True)
    label_source = np.zeros((64, 1, 256, 256), dtype="<f4")
    label_source_bytes = _npy_bytes(label_source)
    label_episodes = [
        {
            "independent_candidate_receipt": {
                "numeric_evidence": {"changed_parameter_tensor_count": 1}
            }
        }
        for _ in range(640)
    ]
    label_episode_bytes = outer.canonical_jsonl_bytes(label_episodes, ordered=False)
    label_layout_bytes = outer.canonical_json_bytes(
        {"synthetic_layout": True}, newline=True
    )
    label_complete_bytes = outer.canonical_json_bytes(
        {"complete": True}, newline=True
    )
    stable_payloads = {
        "manifest.json": (label_manifest_bytes, LABEL_MANIFEST_SHA),
        "parameter_layout.json": (label_layout_bytes, LAYOUT_FILE_SHA),
        "source_logits_pre.npy": (
            label_source_bytes,
            hashlib.sha256(label_source_bytes).hexdigest(),
        ),
        "episode_receipts.jsonl": (
            label_episode_bytes,
            hashlib.sha256(label_episode_bytes).hexdigest(),
        ),
        "COMPLETE.json": (label_complete_bytes, LABEL_COMPLETE_SHA),
    }
    verified = VerifiedLabelFreeShard(
        path=label,
        dataset=DATASET,
        condition=CONDITION,
        replicate="R0",
        formal=True,
        dry_run=False,
        image_count=64,
        candidate_count=10,
        episode_count=640,
        manifest_sha256=LABEL_MANIFEST_SHA,
        complete_sha256=LABEL_COMPLETE_SHA,
        phase_receipt_sha256=PHASE_SHA,
    )
    fake_layout = SimpleNamespace(
        names=tuple(f"bn_{index}" for index in range(106)),
        scalar_count=8736,
        layout_sha256=LAYOUT_SHA,
    )
    monkeypatch.setattr(outer, "verify_label_free_shard", lambda *_a, **_k: verified)
    monkeypatch.setattr(outer, "validate_parameter_layout", lambda _v: fake_layout)
    monkeypatch.setattr(
        outer,
        "_recompute_changed_parameter_tensor_counts",
        lambda *_a, **_k: (1,) * 640,
    )
    fake_outer_code_seal = {
        "files": [{"path": "outer.py", "sha256": "c" * 64}],
        "bundle_sha256": "d" * 64,
    }
    monkeypatch.setattr(
        outer, "build_outer_code_seal", lambda _root: fake_outer_code_seal
    )
    original_read = outer.read_stable_regular_file

    def fake_read(path: Path):
        path = Path(path)
        if path.parent == label and path.name in stable_payloads:
            data, digest = stable_payloads[path.name]
            return SimpleNamespace(data=data, sha256=digest)
        return original_read(path)

    monkeypatch.setattr(outer, "read_stable_regular_file", fake_read)

    outer_logits = label_source.copy()
    gradients = np.ones((64, 8736), dtype="<f4")
    _write(shard / outer.OUTER_SOURCE_LOGITS_FILENAME, _npy_bytes(outer_logits))
    _write(shard / outer.SUPERVISED_GRADIENTS_FILENAME, _npy_bytes(gradients))
    records = []
    for image_index, image_id in enumerate(image_ids):
        for candidate_index, candidate in enumerate(FROZEN_CANDIDATES):
            records.append(
                {
                    "schema_version": 3,
                    "artifact_type": outer.RECORD_ARTIFACT_TYPE,
                    "dataset": DATASET,
                    "condition": CONDITION,
                    "replicate_id": "R0",
                    "image_index": image_index,
                    "image_id": image_id,
                    "candidate": {
                        "candidate_id": candidate.candidate_id,
                        "optimizer": candidate.optimizer,
                        "learning_rate": candidate.learning_rate,
                    },
                    "finite_gradient": True,
                    "changed_parameter_tensor_count": 1,
                    "analysis": _analysis(),
                }
            )
    _write(
        shard / outer.OUTER_RECORDS_FILENAME,
        outer.canonical_jsonl_bytes(records, ordered=True),
    )
    audits = []
    for index, image_id in enumerate(image_ids):
        source_hash = outer.array_slice_sha256(outer_logits[index])
        audits.append(
            {
                "schema_version": 3,
                "artifact_type": outer.TASK_AUDIT_ARTIFACT_TYPE,
                "dataset": DATASET,
                "condition": CONDITION,
                "replicate_id": "R0",
                "image_index": index,
                "image_id": image_id,
                "source_logits_bit_exact": True,
                "label_free_source_logits_slice_sha256": source_hash,
                "outer_source_logits_slice_sha256": source_hash,
                "supervised_gradient_slice_sha256": outer.array_slice_sha256(
                    gradients[index]
                ),
                "supervised_gradient_finite": True,
                "source_state_sha256": SOURCE_STATE_SHA,
                "reset_source_state_sha256": SOURCE_STATE_SHA,
                "task_loss": _task_loss(),
                "forward_runtime": {
                    "seed": 42,
                    "device": "cuda:0",
                    "deterministic_algorithms": True,
                    "deterministic_warn_only": False,
                    "cudnn_benchmark": False,
                    "cudnn_deterministic": True,
                    "cublas_workspace_config": ":4096:8",
                    "visible_cuda_device_count": 1,
                },
                "used_by_adaptation": False,
                "stage2_authorized": False,
            }
        )
    _write(
        shard / outer.TASK_LOSS_AUDITS_FILENAME,
        outer.canonical_jsonl_bytes(audits, ordered=False),
    )
    access = {
        "schema_version": 3,
        "artifact_type": OUTER_ACCESS_ARTIFACT_TYPE,
        "protocol_id": PHASE_PROTOCOL_ID,
        "cell_binding": {"dataset": DATASET, "condition": CONDITION, "replicate": 0},
        "phase_evidence": {
            "label_free_cell_receipt_sha256": PHASE_SHA,
            "complete_candidate_episode_count": 640,
            "candidate_phase_target_access_count": 0,
        },
        "outer_access": {
            "loader_call_count": 1,
            "used_by_adaptation": False,
            "adaptation_target_indexing_count": 0,
        },
        "authorization": {"stage2_authorized": False},
    }
    _write(
        shard / outer.OUTER_ACCESS_RECEIPT_FILENAME,
        outer.canonical_json_bytes(access),
    )
    arrays = {
        outer.OUTER_SOURCE_LOGITS_FILENAME: outer_logits,
        outer.SUPERVISED_GRADIENTS_FILENAME: gradients,
    }
    manifest = {
        "schema_version": 3,
        "artifact_type": outer.ARTIFACT_TYPE,
        "protocol_id": PROTOCOL_ID,
        "config_sha256": CONFIG_SHA,
        "cell": {"dataset": DATASET, "condition": CONDITION, "replicate": "R0"},
        "mode": {
            "formal": True,
            "dry_run": False,
            "image_count": 64,
            "candidate_count": 10,
            "record_count": 640,
            "record_order": outer.EPISODE_ORDER,
        },
        "label_free_shard": {
            "path": "label",
            "manifest_sha256": LABEL_MANIFEST_SHA,
            "complete_sha256": LABEL_COMPLETE_SHA,
            "phase_receipt_sha256": PHASE_SHA,
            "cpu_verified_before_target_load": True,
        },
        "dataset_binding": {
            "train_split_sha256": SPLIT_SHA,
            "checkpoint_path": "results/checkpoint.pth.tar",
            "checkpoint_sha256": CHECKPOINT_SHA,
            "cache_protocol_sha256": CACHE_PROTOCOL_SHA,
            "cache_method_manifest_sha256": METHOD_SHA,
        },
        "ordered_image_ids": image_ids,
        "ordered_image_ids_sha256": ids_sha,
        "parameter_layout": {
            "source_path": "label/parameter_layout.json",
            "source_file_sha256": LAYOUT_FILE_SHA,
            "layout_sha256": LAYOUT_SHA,
            "parameter_tensor_count": 106,
            "scalar_parameter_count": 8736,
        },
        "outer_code_seal": fake_outer_code_seal,
        "arrays": {
            filename: {
                "path": filename,
                "sha256": _file_sha(shard / filename),
                "shape": list(value.shape),
                "dtype": "<f4",
                "c_order": True,
                "finite": True,
                "lossless": True,
            }
            for filename, value in arrays.items()
        },
        "outer_records": {
            "path": outer.OUTER_RECORDS_FILENAME,
            "sha256": _file_sha(shard / outer.OUTER_RECORDS_FILENAME),
            "count": 640,
            "order": outer.EPISODE_ORDER,
        },
        "task_loss_audits": {
            "path": outer.TASK_LOSS_AUDITS_FILENAME,
            "sha256": _file_sha(shard / outer.TASK_LOSS_AUDITS_FILENAME),
            "count": 64,
            "one_supervised_gradient_per_image": True,
            "used_by_adaptation": False,
        },
        "outer_access_receipt": {
            "path": outer.OUTER_ACCESS_RECEIPT_FILENAME,
            "sha256": _file_sha(shard / outer.OUTER_ACCESS_RECEIPT_FILENAME),
            "used_by_adaptation": False,
            "outer_target_loader_call_count": 1,
        },
        "execution": dict(outer._EXECUTION),
        "data_boundary": dict(outer._DATA_BOUNDARY),
        "authorization": dict(outer._AUTHORIZATION),
    }
    _write(
        shard / outer.MANIFEST_FILENAME,
        outer.canonical_json_bytes(manifest, newline=True),
    )
    payload_files = [
        {"path": name, "sha256": _file_sha(shard / name)}
        for name in sorted(outer.MEMBERS - {outer.COMPLETE_FILENAME})
    ]
    complete = {
        "schema_version": 3,
        "artifact_type": outer.COMPLETE_ARTIFACT_TYPE,
        "complete": True,
        "candidate_phase_complete": True,
        "outer_phase_complete": True,
        "formal": True,
        "dry_run": False,
        "config_sha256": CONFIG_SHA,
        "dataset": DATASET,
        "condition": CONDITION,
        "replicate": "R0",
        "image_count": 64,
        "candidate_count": 10,
        "record_count": 640,
        "manifest": {
            "path": outer.MANIFEST_FILENAME,
            "sha256": _file_sha(shard / outer.MANIFEST_FILENAME),
        },
        "payload_files": payload_files,
        "atomic_no_replace": True,
        "paper_result": False,
        "paper_test_result": False,
        "formal_protocol_complete": False,
        "scientific_gate_status": "not_evaluated",
        "stage2_authorized": False,
    }
    _write(
        shard / outer.COMPLETE_FILENAME,
        outer.canonical_json_bytes(complete, newline=True),
    )
    return repo, label, shard, verified


def test_public_cpu_verifier_accepts_complete_bound_outer_cell(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo, label, shard, _ = _build_fixture(tmp_path, monkeypatch)
    verified = outer.verify_outer_cell_shard(
        shard,
        repository_root=repo,
        label_free_shard_path=label,
        expected_config_sha256=CONFIG_SHA,
    )
    assert verified.record_count == 640
    assert verified.image_count == 64
    assert verified.label_free_phase_receipt_sha256 == PHASE_SHA


def test_outer_source_logit_tamper_fails_bit_exact_gate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo, label, shard, _ = _build_fixture(tmp_path, monkeypatch)
    tampered = np.zeros((64, 1, 256, 256), dtype="<f4")
    tampered[0, 0, 0, 0] = 1.0
    (shard / outer.OUTER_SOURCE_LOGITS_FILENAME).write_bytes(_npy_bytes(tampered))
    with pytest.raises(outer.D0V3OuterCellShardError, match="bit-exact"):
        outer.verify_outer_cell_shard(
            shard,
            repository_root=repo,
            label_free_shard_path=label,
            expected_config_sha256=CONFIG_SHA,
        )


def test_record_loss_or_stage2_tamper_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo, label, shard, _ = _build_fixture(tmp_path, monkeypatch)
    lines = (shard / outer.OUTER_RECORDS_FILENAME).read_bytes().splitlines()
    value = json.loads(lines[0])
    value["analysis"]["stage2_authorized"] = True
    lines[0] = outer.canonical_ordered_json_bytes(value)
    (shard / outer.OUTER_RECORDS_FILENAME).write_bytes(b"\n".join(lines) + b"\n")
    with pytest.raises(outer.D0V3OuterCellShardError, match="identity"):
        outer.verify_outer_cell_shard(
            shard,
            repository_root=repo,
            label_free_shard_path=label,
            expected_config_sha256=CONFIG_SHA,
        )


def test_symlink_member_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo, label, shard, _ = _build_fixture(tmp_path, monkeypatch)
    access = shard / outer.OUTER_ACCESS_RECEIPT_FILENAME
    backup = shard.parent / "access-backup.json"
    access.rename(backup)
    access.symlink_to(backup)
    with pytest.raises(outer.D0V3OuterCellShardError, match="snapshot"):
        outer.verify_outer_cell_shard(
            shard,
            repository_root=repo,
            label_free_shard_path=label,
            expected_config_sha256=CONFIG_SHA,
        )
