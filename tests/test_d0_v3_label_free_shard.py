from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from analysis.d0_v2_independent_candidate_contract import (
    FROZEN_CANDIDATES,
    IndependentCandidateReceipt,
)
from analysis.d0_v3_label_free_shard import (
    ARRAY_FILENAMES,
    ARTIFACT_TYPE,
    COMPLETE_ARTIFACT_TYPE,
    COMPLETE_FILENAME,
    D0V3LabelFreeShardError,
    ENTROPY_GRADIENTS_FILENAME,
    EPISODES_FILENAME,
    EPISODE_ARTIFACT_TYPE,
    FLOAT_DTYPE,
    LAYOUT_FILENAME,
    MANIFEST_FILENAME,
    PARAMETERS_AFTER_FILENAME,
    PARAMETER_SCALAR_COUNT,
    PARAMETER_TENSOR_COUNT,
    POST_LOGITS_FILENAME,
    REQUIRED_CRITICAL_CODE_PATHS,
    SOURCE_LOGITS_FILENAME,
    SOURCE_PARAMETERS_FILENAME,
    canonical_json_bytes,
    canonical_sha256,
    named_bundle_sha256,
    raw_float32_sha256,
    verify_label_free_shard,
)
from analysis.d0_v3_outer_analyzer import FlatParameterLayout
from analysis.source_train_provenance import SourceTrainAnalysisProvenance
from scripts.run_d0_v3_formal_stage_a_label_free import (
    EpisodeEvidence,
    _build_episode_receipt,
)
from tta.d0_v2_native_step import NativeFirstStepObservation, frozen_first_step_spec
from tta.d0_v2_parameter_groups import FROZEN_D0_V2_FINE_INVENTORY_SHA256


SHA_A = "a" * 64
SHA_B = "b" * 64
CONFIG_SHA = "c" * 64


def _write(path: Path, data: bytes) -> None:
    path.write_bytes(data)


def _layout() -> FlatParameterLayout:
    values = [("bn.0", torch.zeros(PARAMETER_SCALAR_COUNT - 105))]
    values.extend((f"bn.{index}", torch.zeros(1)) for index in range(1, 106))
    layout = FlatParameterLayout.from_named_tensors(values)
    assert len(layout.names) == PARAMETER_TENSOR_COUNT
    assert layout.scalar_count == PARAMETER_SCALAR_COUNT
    return layout


def _input_seal(checkpoint_path: str) -> dict:
    base = {
        "formal_config": ("configs/formal.yaml", CONFIG_SHA),
        "parent_engineering_config": ("configs/parent.yaml", SHA_A),
        "engineering_smoke_aggregate": ("results/smoke/aggregate.json", SHA_A),
        "cache_execution_protocol": ("configs/cache_execution.yaml", SHA_A),
        "cache_protocol": ("configs/cache.yaml", SHA_A),
        "source_train_split": ("datasets/NUAA-SIRST/train.txt", SHA_A),
        "frozen_pilot_ids": ("datasets/NUAA-SIRST/pilot64.txt", SHA_A),
        "source_checkpoint": (checkpoint_path, SHA_B),
        "cache_manifest": ("results/cache/manifest.json", SHA_A),
        "cache_method_manifest": ("results/cache/method_input_manifest.json", SHA_A),
        "cache_complete": ("results/cache/COMPLETE.json", SHA_A),
        "method_condition": ("results/cache/conditions/clean_S0.npy", SHA_A),
    }
    records = [
        {"role": role, "path": path, "sha256": digest}
        for role, (path, digest) in base.items()
    ]
    records.extend(
        {
            "role": f"critical_code:{path}",
            "path": path,
            "sha256": SHA_A,
        }
        for path in REQUIRED_CRITICAL_CODE_PATHS
    )
    records.sort(key=lambda value: value["role"])
    return {
        "files": records,
        "seal_sha256": canonical_sha256(records),
        "target_payload_bytes_opened": 0,
        "target_payload_deserialized": False,
        "validation_payload_opens": 0,
        "test_split_files_opened": 0,
        "test_images_opened": 0,
        "test_masks_opened": 0,
        "test_labels_opened": 0,
    }


def _native(
    *,
    candidate_index: int,
    layout: FlatParameterLayout,
    before_hash: str,
    gradient_hash: str,
    after_hash: str,
    delta_hash: str,
) -> dict:
    candidate = FROZEN_CANDIDATES[candidate_index]
    state_tensors = 3 * PARAMETER_TENSOR_COUNT if candidate.optimizer == "Adam" else PARAMETER_TENSOR_COUNT
    return NativeFirstStepObservation(
        optimizer_name=candidate.optimizer,
        learning_rate=candidate.learning_rate,
        parameter_names=layout.names,
        device="cuda:0",
        runtime_optimizer={},
        parameter_before_bundle_sha256=before_hash,
        gradient_bundle_sha256=gradient_hash,
        reference_parameter_after_bundle_sha256=after_hash,
        actual_parameter_after_bundle_sha256=after_hash,
        parameter_delta_bundle_sha256=delta_hash,
        reference_optimizer_state_bundle_sha256=SHA_A,
        actual_optimizer_state_bundle_sha256=SHA_A,
        parameter_tensor_count=PARAMETER_TENSOR_COUNT,
        gradient_tensor_count=PARAMETER_TENSOR_COUNT,
        scalar_parameter_count=PARAMETER_SCALAR_COUNT,
        changed_parameter_tensor_count=0,
        native_reference_parameter_tensor_count=PARAMETER_TENSOR_COUNT,
        native_reference_optimizer_state_tensor_count=state_tensors,
        optimizer_state_parameter_count=PARAMETER_TENSOR_COUNT,
        optimizer_state_tensor_count=state_tensors,
        bit_exact_parameter_tensor_count=PARAMETER_TENSOR_COUNT,
        bit_exact_optimizer_state_tensor_count=state_tensors,
        step_norm_l2=0.0,
        gradient_callback_invoked=True,
    ).to_dict()


def _geometry(candidate_index: int) -> dict:
    candidate = FROZEN_CANDIDATES[candidate_index]
    provenance = SourceTrainAnalysisProvenance(
        dataset="NUAA-SIRST",
        split_name="train",
        split_sha256=SHA_A,
        checkpoint_sha256=SHA_B,
        seed=42,
        oracle_analysis=False,
        outer_evaluator_label_accesses=0,
        supervised_gradient_role="none",
    )
    return {
        "schema_version": 3,
        "analysis_type": "tent_optimizer_first_step_geometry",
        "scope": provenance.to_dict(),
        "optimizer": frozen_first_step_spec(
            candidate.optimizer, candidate.learning_rate
        ).to_dict(),
        "gate_policy": {
            "runtime_same_device_hard_gate_is_sole_acceptance_gate": True
        },
    }


def _make_dry_shard(root: Path) -> Path:
    root.mkdir()
    layout = _layout()
    _write(root / LAYOUT_FILENAME, canonical_json_bytes(layout.to_dict(), newline=True))
    dtype = np.dtype(FLOAT_DTYPE)
    arrays = {
        SOURCE_LOGITS_FILENAME: np.zeros((1, 1, 256, 256), dtype=dtype),
        POST_LOGITS_FILENAME: np.zeros((1, 10, 1, 256, 256), dtype=dtype),
        SOURCE_PARAMETERS_FILENAME: np.zeros((1, 10, PARAMETER_SCALAR_COUNT), dtype=dtype),
        PARAMETERS_AFTER_FILENAME: np.zeros((1, 10, PARAMETER_SCALAR_COUNT), dtype=dtype),
        ENTROPY_GRADIENTS_FILENAME: np.zeros((1, 10, PARAMETER_SCALAR_COUNT), dtype=dtype),
    }
    for filename, value in arrays.items():
        with (root / filename).open("xb") as handle:
            np.save(handle, value, allow_pickle=False)
    file_hashes = {
        filename: hashlib.sha256((root / filename).read_bytes()).hexdigest()
        for filename in ARRAY_FILENAMES
    }
    flat_zero = arrays[SOURCE_PARAMETERS_FILENAME][0, 0]
    bundle_zero = named_bundle_sha256(flat_zero, layout)
    process = {
        "logical_process_id": "R0-synthetic",
        "os_process_id": 1,
        "process_start_time_ticks": 1,
        "parent_run_nonce": SHA_A,
        "child_launch_nonce": SHA_B,
        "command_sha256": CONFIG_SHA,
    }
    evidences: list[EpisodeEvidence] = []
    source_logit_hash = raw_float32_sha256(arrays[SOURCE_LOGITS_FILENAME][0])
    post_logit_hash = raw_float32_sha256(arrays[POST_LOGITS_FILENAME][0, 0])
    for index, candidate in enumerate(FROZEN_CANDIDATES):
        identity = {
            "model_instance_id": f"synthetic:{index}:model",
            "method_instance_id": f"synthetic:{index}:method",
            "optimizer_instance_id": f"synthetic:{index}:optimizer",
            "autograd_graph_id": f"synthetic:{index}:graph",
            "backward_execution_id": f"synthetic:{index}:backward",
            "gradient_buffer_owner_id": f"synthetic:{index}:gradient",
        }
        native = _native(
            candidate_index=index,
            layout=layout,
            before_hash=bundle_zero,
            gradient_hash=bundle_zero,
            after_hash=bundle_zero,
            delta_hash=bundle_zero,
        )
        receipt = IndependentCandidateReceipt(
            candidate_index=index,
            candidate=candidate,
            config_sha256=CONFIG_SHA,
            dataset="NUAA-SIRST",
            condition="clean_S0",
            sample_index=0,
            sample_id="synthetic_0",
            split_sha256=SHA_A,
            checkpoint_sha256=SHA_B,
            source_state_sha256=SHA_A,
            runtime_sha256=SHA_A,
            determinism_sha256=SHA_A,
            input_sha256=SHA_A,
            selected_parameter_names_sha256=SHA_A,
            pre_logits_sha256=source_logit_hash,
            post_logits_sha256=post_logit_hash,
            entropy_gradient_bundle_sha256=bundle_zero,
            parameter_delta_bundle_sha256=bundle_zero,
            **identity,
            gradient_tensor_count=PARAMETER_TENSOR_COUNT,
            changed_parameter_tensor_count=0,
            optimizer_state_entry_count_after_step=PARAMETER_TENSOR_COUNT,
            native_reference_parameter_tensor_count=PARAMETER_TENSOR_COUNT,
            native_reference_optimizer_state_tensor_count=native["counts"][
                "native_reference_optimizer_state_tensor_count"
            ],
            step_norm_l2=0.0,
        ).to_dict()
        evidences.append(
            EpisodeEvidence(
                image_index=0,
                image_id="synthetic_0",
                candidate_index=index,
                independent=receipt,
                native=native,
                optimizer_geometry=_geometry(index),
            )
        )
    episode_receipts = [
        _build_episode_receipt(
            evidence,
            config_sha256=CONFIG_SHA,
            dataset="NUAA-SIRST",
            condition="clean_S0",
            replicate="R0",
            process_identity=process,
            arrays=arrays,
            array_sha256s=file_hashes,
            layout=layout,
        )
        for evidence in evidences
    ]
    episode_bytes = b"".join(
        canonical_json_bytes(value, newline=True) for value in episode_receipts
    )
    _write(root / EPISODES_FILENAME, episode_bytes)
    episode_hashes = [
        hashlib.sha256(canonical_json_bytes(value)).hexdigest()
        for value in episode_receipts
    ]
    checkpoint_path = "results/baseline/NUAA-SIRST/best_miou.pth.tar"
    dataset_binding = {
        "train_split_sha256": SHA_A,
        "checkpoint_role": "best_miou",
        "checkpoint_path": checkpoint_path,
        "checkpoint_sha256": SHA_B,
    }
    array_records = {
        filename: {
            "path": filename,
            "sha256": file_hashes[filename],
            "shape": list(value.shape),
            "dtype": FLOAT_DTYPE,
            "c_order": True,
            "finite": True,
            "lossless": True,
        }
        for filename, value in arrays.items()
    }
    image_ids = ["synthetic_0"]
    manifest = {
        "schema_version": 3,
        "artifact_type": ARTIFACT_TYPE,
        "protocol_id": "cr-sitta-d0-v3-formal-stage-a",
        "config_sha256": CONFIG_SHA,
        "cell": {"dataset": "NUAA-SIRST", "condition": "clean_S0", "replicate": "R0"},
        "mode": {
            "formal": False,
            "dry_run": True,
            "image_count": 1,
            "candidate_count": 10,
            "episode_count": 10,
            "episode_order": "image_major_candidate_minor",
        },
        "process_identity": process,
        "dataset_binding": dataset_binding,
        "ordered_image_ids": image_ids,
        "ordered_image_ids_sha256": hashlib.sha256(
            json.dumps(image_ids, ensure_ascii=False, separators=(",", ":")).encode()
        ).hexdigest(),
        "candidate_slugs": [value.slug for value in FROZEN_CANDIDATES],
        "parameter_layout": {
            "path": LAYOUT_FILENAME,
            "sha256": hashlib.sha256((root / LAYOUT_FILENAME).read_bytes()).hexdigest(),
            "layout_sha256": layout.layout_sha256,
            "parameter_tensor_count": PARAMETER_TENSOR_COUNT,
            "scalar_parameter_count": PARAMETER_SCALAR_COUNT,
        },
        "fine_inventory_sha256": FROZEN_D0_V2_FINE_INVENTORY_SHA256,
        "input_seal": _input_seal(checkpoint_path),
        "arrays": array_records,
        "episode_receipts": {
            "path": EPISODES_FILENAME,
            "sha256": hashlib.sha256(episode_bytes).hexdigest(),
            "count": 10,
            "order": "image_major_candidate_minor",
            "episode_receipt_hashes_sha256": canonical_sha256(episode_hashes),
        },
        "phase_receipt": None,
        "data_boundary": {
            "source_train_derived": True,
            "split_name": "train",
            "split_role": "frozen_pilot64",
            "no_validation_split": True,
            "method_label_accesses": 0,
            "outer_evaluator_label_accesses": 0,
            "train_target_payload_bytes_opened": 0,
            "train_target_payload_deserialization_count": 0,
            "validation_payload_opens": 0,
            "test_split_files_opened": 0,
            "test_images_opened": 0,
            "test_masks_opened": 0,
            "test_labels_opened": 0,
        },
        "authorization": {
            "paper_result": False,
            "paper_test_result": False,
            "scientific_gate_status": "not_evaluated",
            "scientific_selection_performed": False,
            "formal_protocol_complete": False,
            "stage2_authorized": False,
        },
    }
    _write(root / MANIFEST_FILENAME, canonical_json_bytes(manifest, newline=True))
    payload_names = sorted(
        {
            *ARRAY_FILENAMES,
            LAYOUT_FILENAME,
            EPISODES_FILENAME,
            MANIFEST_FILENAME,
        }
    )
    complete = {
        "schema_version": 3,
        "artifact_type": COMPLETE_ARTIFACT_TYPE,
        "complete": True,
        "candidate_phase_complete": True,
        "formal": False,
        "dry_run": True,
        "config_sha256": CONFIG_SHA,
        "dataset": "NUAA-SIRST",
        "condition": "clean_S0",
        "replicate": "R0",
        "image_count": 1,
        "candidate_count": 10,
        "episode_count": 10,
        "manifest": {
            "path": MANIFEST_FILENAME,
            "sha256": hashlib.sha256((root / MANIFEST_FILENAME).read_bytes()).hexdigest(),
        },
        "payload_files": [
            {"path": name, "sha256": hashlib.sha256((root / name).read_bytes()).hexdigest()}
            for name in payload_names
        ],
        "atomic_no_replace": True,
        "paper_result": False,
        "formal_protocol_complete": False,
        "scientific_gate_status": "not_evaluated",
        "stage2_authorized": False,
    }
    _write(root / COMPLETE_FILENAME, canonical_json_bytes(complete, newline=True))
    return root


def test_cpu_verifier_accepts_complete_lossless_dry_shard(tmp_path: Path) -> None:
    root = _make_dry_shard(tmp_path / "dry")
    verified = verify_label_free_shard(root, expected_config_sha256=CONFIG_SHA)
    assert verified.formal is False
    assert verified.dry_run is True
    assert verified.image_count == 1
    assert verified.candidate_count == 10
    assert verified.episode_count == 10
    assert verified.phase_receipt_sha256 is None


def test_cpu_verifier_rejects_lossless_array_tamper(tmp_path: Path) -> None:
    root = _make_dry_shard(tmp_path / "dry")
    with (root / ENTROPY_GRADIENTS_FILENAME).open("r+b") as handle:
        handle.seek(-4, 2)
        handle.write(b"\x00\x00\x80?")
    with pytest.raises(D0V3LabelFreeShardError):
        verify_label_free_shard(root, expected_config_sha256=CONFIG_SHA)
