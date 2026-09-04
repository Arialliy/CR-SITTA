from __future__ import annotations

import copy
import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from analysis import p3_stage_b1_outer_cell_shard as v1
from analysis import p3_stage_b1_outer_cell_shard_v2 as cell
from analysis.d0_v3_label_free_shard import array_slice_sha256, canonical_json_bytes
from tta.d0_v3_atomic_shard import publish_flat_directory_noreplace


PROTOCOL = "cr-sitta-p3-stage-b1-gradient-decomposition-v2"
CONFIG_SHA = "a" * 64
DATASET = "NUAA-SIRST"
CONDITION = "clean_S0"
LAYOUT_SHA = "b" * 64


def _source_layout() -> tuple[dict[str, Any], dict[str, tuple[str, ...]]]:
    sizes = [16] * 6 + [32] * 10 + [92] * 17 + [100] + [64] * 8 + [96] * 64
    assert len(sizes) == 106 and sum(sizes) == 8736
    names = [f"parameter_{index:03d}" for index in range(106)]
    offsets, cursor = [], 0
    for size in sizes:
        offsets.append(cursor)
        cursor += size
    return (
        {
            "protocol": "test-layout",
            "names": names,
            "shapes": [[size] for size in sizes],
            "offsets": offsets,
            "scalar_count": cursor,
            "dtype": "torch.float32",
            "layout_sha256": LAYOUT_SHA,
        },
        {
            "P0": tuple(names),
            "P1": tuple(names[:6]),
            "P2": tuple(names[:16]),
            "P3": tuple(names[:34]),
            "P4": tuple(names[:42]),
        },
    )


def _config(image_ids: list[str]) -> dict[str, Any]:
    return {
        "protocol_id": PROTOCOL,
        "datasets": {
            DATASET: {
                "checkpoint_role": "best_miou",
                "checkpoint_path": "results/baseline/NUAA-SIRST/best_miou.pth.tar",
                "checkpoint_sha256": "c" * 64,
                "train_split_sha256": "d" * 64,
                "ordered_pilot64_image_ids_sha256": hashlib.sha256(
                    canonical_json_bytes(image_ids)
                ).hexdigest(),
            }
        },
        "conditions": [CONDITION],
        "parent_cell_artifacts": {
            "candidate_shard_template": "parent/candidate/{dataset}/{condition}",
            "outer_shard_template": "parent/outer/{dataset}/{condition}",
            "parameter_layout": {
                "filename": "parameter_layout.json",
                "layout_sha256": LAYOUT_SHA,
                "parameter_tensor_count": 106,
                "scalar_parameter_count": 8736,
            },
        },
        "entropy_gradient_consistency": {
            "max_abs_tolerance": 1.0e-7,
            "relative_l2_tolerance": 1.0e-4,
        },
    }


def _alignment(vector: np.ndarray | None, task: np.ndarray, reason: str | None):
    value = v1._alignment_metrics(vector, task)
    value["not_estimable_reason"] = reason
    return value


def _groups_for(
    science: np.ndarray,
    task: np.ndarray,
    target: Mapping[str, Any],
    group_layout: Mapping[str, Any],
) -> dict[str, Any]:
    _, indices = cell.validate_coarse_group_layout(
        group_layout, protocol_id=PROTOCOL, expected_layout_sha256=LAYOUT_SHA
    )
    total = int(target["total_pixel_count"])
    counts = {
        "fg": int(target["foreground_pixel_count"]),
        "bg": int(target["background_pixel_count"]),
        "sub": int(target["foreground_subthreshold_pixel_count"]),
        "supra": int(target["foreground_suprathreshold_pixel_count"]),
    }
    result: dict[str, Any] = {}
    for group_id in cell.GROUP_IDS:
        index = indices[group_id]
        sub, supra, bg = science[:, index]
        fg, full = sub + supra, sub + supra + bg
        selected_task = task[index]
        additive_spec = {
            "full_entropy_mean": (full, total, None),
            "foreground_entropy_add": (fg, counts["fg"], "empty_foreground"),
            "background_entropy_add": (bg, counts["bg"], "empty_background"),
            "foreground_subthreshold_entropy_add": (
                sub, counts["sub"], "empty_foreground_subthreshold"
            ),
            "foreground_suprathreshold_entropy_add": (
                supra, counts["supra"], "empty_foreground_suprathreshold"
            ),
        }
        conditional_spec = {
            "full_entropy_mean": (full, total, None),
            "foreground_entropy_mean": (fg, counts["fg"], "empty_foreground"),
            "background_entropy_mean": (bg, counts["bg"], "empty_background"),
            "foreground_subthreshold_entropy_mean": (
                sub, counts["sub"], "empty_foreground_subthreshold"
            ),
            "foreground_suprathreshold_entropy_mean": (
                supra, counts["supra"], "empty_foreground_suprathreshold"
            ),
        }
        additive = {
            name: _alignment(None if count == 0 else vector, selected_task, reason if count == 0 else None)
            for name, (vector, count, reason) in additive_spec.items()
        }
        conditional = {
            name: _alignment(
                None if count == 0 else vector * total / count,
                selected_task,
                reason if count == 0 else None,
            )
            for name, (vector, count, reason) in conditional_spec.items()
        }
        fg_value = None if counts["fg"] == 0 else fg
        bg_value = None if counts["bg"] == 0 else bg
        result[group_id] = {
            "parameter_tensor_count": group_layout["groups"][group_id]["parameter_tensor_count"],
            "parameter_scalar_count": len(index),
            "task_gradient_norm": float(np.linalg.norm(selected_task)),
            "additive_entropy_task_alignment": additive,
            "conditional_entropy_task_alignment": conditional,
            "additive_gradient_norms": {
                "foreground_subthreshold_add": float(np.linalg.norm(sub)) if counts["sub"] else None,
                "foreground_suprathreshold_add": float(np.linalg.norm(supra)) if counts["supra"] else None,
                "background_add": float(np.linalg.norm(bg)) if counts["bg"] else None,
                "foreground_add": float(np.linalg.norm(fg)) if counts["fg"] else None,
                "full_add": float(np.linalg.norm(full)),
            },
            "cross_region": v1._cross_region_metrics(
                fg_value,
                bg_value,
                None if fg_value is None else fg_value * total / counts["fg"],
                None if bg_value is None else bg_value * total / counts["bg"],
                selected_task,
            ),
        }
    return result


def _fixture_payloads(*, raw_shift: float = 0.0, tamper_group: bool = False):
    image_ids = [f"pilot_{index:02d}" for index in range(64)]
    config = _config(image_ids)
    source_layout, group_names = _source_layout()
    group_layout = cell.build_coarse_group_layout(
        protocol_id=PROTOCOL,
        source_parameter_layout=source_layout,
        group_parameter_names=group_names,
    )
    rng = np.random.default_rng(7)
    sub = rng.normal(0, 1e-4, size=(64, 8736)).astype("<f4")
    supra = rng.normal(0, 1e-4, size=(64, 8736)).astype("<f4")
    bg = rng.normal(0, 1e-4, size=(64, 8736)).astype("<f4")
    sub[0] = 0
    supra[0] = 0
    foreground = np.ascontiguousarray(sub + supra, dtype="<f4")
    full = np.ascontiguousarray(foreground + bg, dtype="<f4")
    raw_supra = np.ascontiguousarray(supra + np.float32(raw_shift), dtype="<f4")
    raw_bg = np.ascontiguousarray(bg - np.float32(raw_shift), dtype="<f4")
    direct = np.ascontiguousarray(
        np.stack((full, foreground, sub, raw_supra, raw_bg), axis=1), dtype="<f4"
    )
    parent = np.repeat(direct[:, None, 0, :], 10, axis=1)
    task = rng.normal(0, 1e-3, size=(64, 8736)).astype("<f4")
    records = []
    for index, image_id in enumerate(image_ids):
        empty = index == 0
        target = {
            "target_present": not empty,
            "total_pixel_count": 4,
            "foreground_pixel_count": 0 if empty else 2,
            "background_pixel_count": 4 if empty else 2,
            "foreground_subthreshold_pixel_count": 0 if empty else 1,
            "foreground_suprathreshold_pixel_count": 0 if empty else 1,
            "target_value_sum": 0.0 if empty else 2.0,
            "target_slice_sha256": hashlib.sha256(f"target:{index}".encode()).hexdigest(),
            "partition_disjoint": True,
            "partition_exhaustive": True,
        }
        science = cell.derive_science_basis(direct[index])
        groups = _groups_for(science, task[index].astype(np.float64), target, group_layout)
        if tamper_group and index == 0:
            groups["P0"]["additive_gradient_norms"]["full_add"] += 1.0
        records.append(
            {
                "schema_version": 2,
                "artifact_type": cell.RECORD_ARTIFACT_TYPE,
                "protocol_id": PROTOCOL,
                "config_sha256": CONFIG_SHA,
                "dataset": DATASET,
                "condition": CONDITION,
                "corruption_family": "clean",
                "severity": "S0",
                "replicate": "R0",
                "image_index": index,
                "image_id": image_id,
                "target": target,
                "source_integrity": {
                    "source_logits_bit_exact": True,
                    "parent_source_logits_slice_sha256": "e" * 64,
                    "recomputed_source_logits_slice_sha256": "e" * 64,
                    "source_state_before_sha256": "f" * 64,
                    "source_state_after_sha256": "f" * 64,
                    "state_restored": True,
                    "rng_before_sha256": "1" * 64,
                    "rng_after_sha256": "1" * 64,
                    "rng_restored": True,
                },
                "gradient_integrity": cell.build_gradient_integrity(
                    direct[index],
                    parent_entropy_gradients=parent[index],
                    max_abs_tolerance=1e-7,
                    relative_l2_tolerance=1e-4,
                ),
                "groups": groups,
                "data_boundary": dict(cell.CELL_DATA_BOUNDARY),
                "authorization": dict(cell.CELL_AUTHORIZATION),
            }
        )
    receipt = canonical_json_bytes({"parent_outer_access": True})
    lineage = {
        "candidate_shard_path": f"parent/candidate/{DATASET}/{CONDITION}",
        "outer_shard_path": f"parent/outer/{DATASET}/{CONDITION}",
        "candidate_manifest_sha256": "2" * 64,
        "candidate_complete_sha256": "3" * 64,
        "candidate_phase_receipt_sha256": "4" * 64,
        "outer_manifest_sha256": "5" * 64,
        "outer_complete_sha256": "6" * 64,
        "outer_access_receipt_sha256": hashlib.sha256(receipt).hexdigest(),
        "parent_entropy_gradients_sha256": "7" * 64,
        "parent_task_gradients_sha256": "8" * 64,
    }
    code_files = [{"path": "analysis/test.py", "sha256": "9" * 64}]
    code_seal = {
        "files": code_files,
        "bundle_sha256": hashlib.sha256(canonical_json_bytes(code_files)).hexdigest(),
    }
    parameter_layout = {
        "parent_source_path": f"parent/candidate/{DATASET}/{CONDITION}/parameter_layout.json",
        "parent_source_file_sha256": "0" * 64,
        "source_layout_sha256": LAYOUT_SHA,
        "source_parameter_tensor_count": 106,
        "source_scalar_parameter_count": 8736,
        "coarse_group_layout_path": cell.COARSE_GROUP_LAYOUT_FILENAME,
        "coarse_group_layout_sha256": hashlib.sha256(
            canonical_json_bytes(group_layout, newline=True)
        ).hexdigest(),
        "group_order": list(cell.GROUP_IDS),
        "group_scalar_counts": dict(cell.GROUP_SCALAR_COUNTS),
    }
    execution = {"runtime": dict(cell.FORMAL_RUNTIME), **cell.FORMAL_EXECUTION_COUNTS}
    payloads = cell.build_stage_b1_cell_payloads(
        protocol_id=PROTOCOL,
        config_sha256=CONFIG_SHA,
        cell={
            "dataset": DATASET,
            "condition": CONDITION,
            "corruption_family": "clean",
            "severity": "S0",
            "replicate": "R0",
        },
        ordered_image_ids=image_ids,
        dataset_binding={
            "split_name": "train",
            "train_split_sha256": "d" * 64,
            "checkpoint_role": "best_miou",
            "checkpoint_path": "results/baseline/NUAA-SIRST/best_miou.pth.tar",
            "checkpoint_sha256": "c" * 64,
        },
        parent_lineage=lineage,
        parameter_layout=parameter_layout,
        region_gradient_basis=direct,
        episode_records=records,
        coarse_group_layout=group_layout,
        outer_access_receipt_bytes=receipt,
        code_seal=code_seal,
        execution=execution,
    )
    return payloads, config, code_seal, direct, records


def _write(tmp_path: Path, payloads: Mapping[str, bytes], name: str = "cell") -> Path:
    path = tmp_path / name
    path.mkdir()
    for filename, value in payloads.items():
        (path / filename).write_bytes(value)
    return path


def test_telescoping_science_basis_ignores_raw_audit_slots() -> None:
    direct = np.zeros((5, 8736), dtype="<f4")
    direct[0] = 7
    direct[1] = 5
    direct[2] = 2
    direct[3] = -999
    direct[4] = 999
    science = cell.derive_science_basis(direct)
    assert np.array_equal(science[0], np.full(8736, 2.0))
    assert np.array_equal(science[1], np.full(8736, 3.0))
    assert np.array_equal(science[2], np.full(8736, 2.0))


def test_raw_audit_failure_is_record_only() -> None:
    direct = np.zeros((5, 8736), dtype="<f4")
    direct[0] = 1
    direct[1] = 0.5
    direct[2] = 0.2
    direct[3] = 9
    direct[4] = -9
    parent = np.repeat(direct[None, 0], 10, axis=0)
    report = cell.build_gradient_integrity(
        direct,
        parent_entropy_gradients=parent,
        max_abs_tolerance=1e-7,
        relative_l2_tolerance=1e-4,
    )
    assert report["parent_direct_full_consistency_passed"] is True
    assert report["raw_consistency"]["components"]["background"]["within_dual_tolerance"] is False
    assert report["raw_consistency"]["failure_action"] == "record_only_never_protocol_or_science_gate"


def test_parent_direct_full_dual_tolerance_is_a_hard_gate() -> None:
    direct = np.zeros((5, 8736), dtype="<f4")
    direct[0] = 1.0
    parent = np.zeros((10, 8736), dtype="<f4")
    report = cell.build_gradient_integrity(
        direct,
        parent_entropy_gradients=parent,
        max_abs_tolerance=1e-7,
        relative_l2_tolerance=1e-4,
    )
    assert report["parent_direct_full_consistency_passed"] is False
    with pytest.raises(cell.P3StageB1OuterCellShardV2Error, match="dual tolerance"):
        cell._validate_gradient_integrity(
            report,
            direct=direct,
            parent_entropy=parent,
            max_abs_tolerance=1e-7,
            relative_l2_tolerance=1e-4,
        )


def test_builder_and_offline_verifier_recompute_groups_and_closure(tmp_path: Path) -> None:
    payloads, config, seal, _, records = _fixture_payloads(raw_shift=1e-3)
    artifact = _write(tmp_path, payloads)
    verified = cell.verify_stage_b1_cell_shard(
        artifact,
        repository_root=tmp_path,
        config=config,
        expected_config_sha256=CONFIG_SHA,
        verify_live_parents=False,
        expected_code_seal=seal,
    )
    assert verified.record_count == 64
    assert verified.cumulative_vjps_sha256 == verified.basis_sha256
    assert records[1]["gradient_integrity"]["raw_consistency"]["components"]["background"]["within_dual_tolerance"] is False


def test_live_parent_verification_requires_explicit_code_seal(tmp_path: Path) -> None:
    payloads, config, _seal, _, _ = _fixture_payloads()
    artifact = _write(tmp_path, payloads)
    with pytest.raises(
        cell.P3StageB1OuterCellShardV2Error,
        match="requires the expected code seal",
    ):
        cell.verify_stage_b1_cell_shard(
            artifact,
            repository_root=tmp_path,
            config=config,
            expected_config_sha256=CONFIG_SHA,
            verify_live_parents=True,
            expected_code_seal=None,
        )


def test_verifier_rejects_rebuilt_but_false_group_metric(tmp_path: Path) -> None:
    payloads, config, seal, _, _ = _fixture_payloads(tamper_group=True)
    artifact = _write(tmp_path, payloads)
    with pytest.raises(cell.P3StageB1OuterCellShardV2Error, match="full_add.norm differs"):
        cell.verify_stage_b1_cell_shard(
            artifact,
            repository_root=tmp_path,
            config=config,
            expected_config_sha256=CONFIG_SHA,
            verify_live_parents=False,
            expected_code_seal=seal,
        )


def test_builder_rejects_wrong_shape_and_execution_type() -> None:
    payloads, _, _, direct, _ = _fixture_payloads()
    assert set(payloads) == cell.MEMBERS
    with pytest.raises(cell.P3StageB1OuterCellShardV2Error, match="array schema"):
        cell.build_stage_b1_cell_payloads(
            protocol_id=PROTOCOL,
            config_sha256=CONFIG_SHA,
            cell={"dataset": DATASET, "condition": CONDITION, "corruption_family": "clean", "severity": "S0", "replicate": "R0"},
            ordered_image_ids=[f"x{i}" for i in range(64)],
            dataset_binding={}, parent_lineage={}, parameter_layout={},
            region_gradient_basis=direct[:, :4], episode_records=[{}] * 64,
            coarse_group_layout={}, outer_access_receipt_bytes=b"{}",
            code_seal={}, execution={},
        )


def test_byte_tamper_and_atomic_no_replace_are_rejected(tmp_path: Path) -> None:
    payloads, config, seal, _, _ = _fixture_payloads()
    artifact = _write(tmp_path, payloads, "tampered")
    path = artifact / cell.EPISODE_RECORDS_FILENAME
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(cell.P3StageB1OuterCellShardV2Error):
        cell.verify_stage_b1_cell_shard(
            artifact,
            repository_root=tmp_path,
            config=config,
            expected_config_sha256=CONFIG_SHA,
            verify_live_parents=False,
            expected_code_seal=seal,
        )
    staging = _write(tmp_path, payloads, "staging")
    destination = tmp_path / "destination"
    destination.mkdir()
    with pytest.raises(FileExistsError):
        publish_flat_directory_noreplace(
            staging,
            destination,
            expected_members=cell.MEMBERS,
            semantic_verifier=lambda _: None,
        )
