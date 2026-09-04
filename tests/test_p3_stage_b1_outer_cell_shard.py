from __future__ import annotations

import copy
import hashlib
from pathlib import Path
from collections.abc import Callable
from typing import Any

import numpy as np
import pytest

from analysis import p3_stage_b1_outer_cell_shard as cell
from analysis.d0_v3_label_free_shard import array_slice_sha256, canonical_json_bytes


PROTOCOL = "cr-sitta-p3-stage-b1-gradient-decomposition-v1"
CONFIG_SHA = "a" * 64
DATASET = "NUAA-SIRST"
CONDITION = "clean_S0"
LAYOUT_SHA = "b" * 64


def _source_layout() -> tuple[dict[str, Any], dict[str, tuple[str, ...]]]:
    sizes = [16] * 6 + [32] * 10 + [92] * 17 + [100] + [64] * 8 + [96] * 64
    assert len(sizes) == 106 and sum(sizes) == 8736
    names = [f"parameter_{index:03d}" for index in range(106)]
    offsets: list[int] = []
    cursor = 0
    for size in sizes:
        offsets.append(cursor)
        cursor += size
    layout = {
        "protocol": "test-layout",
        "names": names,
        "shapes": [[size] for size in sizes],
        "offsets": offsets,
        "scalar_count": cursor,
        "dtype": "torch.float32",
        "layout_sha256": LAYOUT_SHA,
    }
    groups = {
        "P0": tuple(names),
        "P1": tuple(names[:6]),
        "P2": tuple(names[:16]),
        "P3": tuple(names[:34]),
        "P4": tuple(names[:42]),
    }
    return layout, groups


def _config(image_ids: list[str]) -> dict[str, Any]:
    ids_sha = hashlib.sha256(canonical_json_bytes(image_ids)).hexdigest()
    return {
        "protocol_id": PROTOCOL,
        "datasets": {
            DATASET: {
                "checkpoint_role": "best_miou",
                "checkpoint_path": "results/baseline/NUAA-SIRST/best_miou.pth.tar",
                "checkpoint_sha256": "c" * 64,
                "train_split_sha256": "d" * 64,
                "ordered_pilot64_image_ids_sha256": ids_sha,
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


def _alignment(
    vector: np.ndarray | None,
    task: np.ndarray,
    *,
    reason: str | None,
) -> dict[str, Any]:
    value = cell._alignment_metrics(vector, task)
    value["not_estimable_reason"] = reason
    return value


def _groups_for(
    basis: np.ndarray,
    task: np.ndarray,
    target: dict[str, Any],
    group_layout: dict[str, Any],
) -> dict[str, Any]:
    _, indices = cell.validate_coarse_group_layout(
        group_layout, protocol_id=PROTOCOL, expected_layout_sha256=LAYOUT_SHA
    )
    total = target["total_pixel_count"]
    fg_count = target["foreground_pixel_count"]
    bg_count = target["background_pixel_count"]
    sub_count = target["foreground_subthreshold_pixel_count"]
    supra_count = target["foreground_suprathreshold_pixel_count"]
    result: dict[str, Any] = {}
    for group_id in cell.GROUP_IDS:
        selected = indices[group_id]
        sub, supra, bg = basis[:, selected].astype(np.float64)
        fg = sub + supra
        full = fg + bg
        task_part = task[selected]
        additive_vectors = {
            "full_entropy_mean": (full, total, None),
            "foreground_entropy_add": (fg, fg_count, "empty_foreground"),
            "background_entropy_add": (bg, bg_count, "empty_background"),
            "foreground_subthreshold_entropy_add": (
                sub,
                sub_count,
                "empty_foreground_subthreshold",
            ),
            "foreground_suprathreshold_entropy_add": (
                supra,
                supra_count,
                "empty_foreground_suprathreshold",
            ),
        }
        additive_alignment = {
            name: _alignment(
                None if count == 0 else vector,
                task_part,
                reason=reason if count == 0 else None,
            )
            for name, (vector, count, reason) in additive_vectors.items()
        }
        conditional_vectors = {
            "full_entropy_mean": (full, total, None),
            "foreground_entropy_mean": (fg, fg_count, "empty_foreground"),
            "background_entropy_mean": (bg, bg_count, "empty_background"),
            "foreground_subthreshold_entropy_mean": (
                sub,
                sub_count,
                "empty_foreground_subthreshold",
            ),
            "foreground_suprathreshold_entropy_mean": (
                supra,
                supra_count,
                "empty_foreground_suprathreshold",
            ),
        }
        conditional_alignment = {
            name: _alignment(
                None if count == 0 else vector * (total / count),
                task_part,
                reason=reason if count == 0 else None,
            )
            for name, (vector, count, reason) in conditional_vectors.items()
        }
        fg_value = None if fg_count == 0 else fg
        bg_value = None if bg_count == 0 else bg
        result[group_id] = {
            "parameter_tensor_count": group_layout["groups"][group_id][
                "parameter_tensor_count"
            ],
            "parameter_scalar_count": len(selected),
            "task_gradient_norm": float(np.linalg.norm(task_part)),
            "additive_entropy_task_alignment": additive_alignment,
            "conditional_entropy_task_alignment": conditional_alignment,
            "additive_gradient_norms": {
                "foreground_subthreshold_add": (
                    float(np.linalg.norm(sub)) if sub_count else None
                ),
                "foreground_suprathreshold_add": (
                    float(np.linalg.norm(supra)) if supra_count else None
                ),
                "background_add": float(np.linalg.norm(bg)) if bg_count else None,
                "foreground_add": float(np.linalg.norm(fg)) if fg_count else None,
                "full_add": float(np.linalg.norm(full)),
            },
            "cross_region": cell._cross_region_metrics(
                fg_value,
                bg_value,
                None if fg_value is None else fg_value * total / fg_count,
                None if bg_value is None else bg_value * total / bg_count,
                task_part,
            ),
        }
    return result


def _fixture_payloads(
    *,
    authorization_tamper: bool = False,
    execution_mutator: Callable[[dict[str, Any]], None] | None = None,
):
    image_ids = [f"pilot_{index:02d}" for index in range(64)]
    config = _config(image_ids)
    source_layout, group_names = _source_layout()
    group_layout = cell.build_coarse_group_layout(
        protocol_id=PROTOCOL,
        source_parameter_layout=source_layout,
        group_parameter_names=group_names,
    )
    rng = np.random.default_rng(7)
    basis = rng.normal(0.0, 1.0e-4, size=(64, 3, 8736)).astype("<f4")
    basis[0, :2] = 0.0
    task = rng.normal(0.0, 1.0e-3, size=8736).astype(np.float64)
    records: list[dict[str, Any]] = []
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
            "target_slice_sha256": hashlib.sha256(
                f"target:{index}".encode()
            ).hexdigest(),
            "partition_disjoint": True,
            "partition_exhaustive": True,
        }
        authorization = dict(cell.CELL_AUTHORIZATION)
        if authorization_tamper and index == 0:
            authorization["stage_b3_authorized"] = True
        records.append(
            {
                "schema_version": 1,
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
                "gradient_integrity": {
                    "basis_order": list(cell.BASIS_ORDER),
                    "basis_slice_sha256": array_slice_sha256(basis[index]),
                    "finite": True,
                    "parent_candidate_slice_count": 10,
                    "parent_max_abs_error": 0.0,
                    "parent_max_relative_l2_error": 0.0,
                    "max_abs_tolerance": 1.0e-7,
                    "relative_l2_tolerance": 1.0e-4,
                    "parent_consistency_passed": True,
                },
                "groups": _groups_for(basis[index], task, target, group_layout),
                "data_boundary": dict(cell.CELL_DATA_BOUNDARY),
                "authorization": authorization,
            }
        )
    receipt = canonical_json_bytes({"parent_outer_access": True})
    parent_lineage = {
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
        "parent_source_path": (
            f"parent/candidate/{DATASET}/{CONDITION}/parameter_layout.json"
        ),
        "parent_source_file_sha256": "0" * 64,
        "source_layout_sha256": LAYOUT_SHA,
        "source_parameter_tensor_count": 106,
        "source_scalar_parameter_count": 8736,
        "coarse_group_layout_path": cell.COARSE_GROUP_LAYOUT_FILENAME,
        # The builder fills the file but deliberately accepts this value from
        # the caller, so compute its canonical newline digest here.
        "coarse_group_layout_sha256": hashlib.sha256(
            canonical_json_bytes(group_layout, newline=True)
        ).hexdigest(),
        "group_order": list(cell.GROUP_IDS),
        "group_scalar_counts": dict(cell.GROUP_SCALAR_COUNTS),
    }
    execution = {"runtime": dict(cell.FORMAL_RUNTIME), **cell.FORMAL_EXECUTION_COUNTS}
    if execution_mutator is not None:
        execution_mutator(execution)
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
        parent_lineage=parent_lineage,
        parameter_layout=parameter_layout,
        region_gradient_basis=basis,
        episode_records=records,
        coarse_group_layout=group_layout,
        outer_access_receipt_bytes=receipt,
        code_seal=code_seal,
        execution=execution,
        data_boundary=cell.MANIFEST_DATA_BOUNDARY,
        authorization=cell.CELL_AUTHORIZATION,
    )
    return payloads, config, code_seal


def _write_artifact(tmp_path: Path, payloads: dict[str, bytes]) -> Path:
    path = tmp_path / "cell"
    path.mkdir()
    for name, value in payloads.items():
        (path / name).write_bytes(value)
    return path


def test_cell_builder_and_cpu_verifier_preserve_empty_region_nulls(tmp_path: Path) -> None:
    payloads, config, code_seal = _fixture_payloads()
    artifact = _write_artifact(tmp_path, payloads)
    verified = cell.verify_stage_b1_cell_shard(
        artifact,
        repository_root=tmp_path,
        config=config,
        expected_config_sha256=CONFIG_SHA,
        verify_live_parents=False,
        expected_code_seal=code_seal,
    )
    assert verified.record_count == 64
    empty = verified.records[0]["groups"]["P0"]
    assert empty["additive_gradient_norms"]["foreground_add"] is None
    assert (
        empty["additive_entropy_task_alignment"]["foreground_entropy_add"]
        ["cosine_status"]
        == "not_estimable_region_absent"
    )
    assert verified.records[1]["target"]["target_present"] is True


def test_cell_verifier_rejects_record_authorization_even_if_hashes_rebuilt(
    tmp_path: Path,
) -> None:
    payloads, config, code_seal = _fixture_payloads(authorization_tamper=True)
    artifact = _write_artifact(tmp_path, payloads)
    with pytest.raises(cell.P3StageB1OuterCellShardError, match="record identity"):
        cell.verify_stage_b1_cell_shard(
            artifact,
            repository_root=tmp_path,
            config=config,
            expected_config_sha256=CONFIG_SHA,
            verify_live_parents=False,
            expected_code_seal=code_seal,
        )


def test_cell_verifier_rejects_byte_tampering(tmp_path: Path) -> None:
    payloads, config, code_seal = _fixture_payloads()
    artifact = _write_artifact(tmp_path, payloads)
    records = artifact / cell.EPISODE_RECORDS_FILENAME
    records.write_bytes(records.read_bytes() + b" ")
    with pytest.raises(cell.P3StageB1OuterCellShardError):
        cell.verify_stage_b1_cell_shard(
            artifact,
            repository_root=tmp_path,
            config=config,
            expected_config_sha256=CONFIG_SHA,
            verify_live_parents=False,
            expected_code_seal=code_seal,
        )


def test_coarse_group_layout_rejects_non_nested_group() -> None:
    source, names = _source_layout()
    broken = copy.deepcopy(names)
    broken["P2"] = tuple(source["names"][6:16])
    with pytest.raises(cell.P3StageB1OuterCellShardError):
        cell.build_coarse_group_layout(
            protocol_id=PROTOCOL,
            source_parameter_layout=source,
            group_parameter_names=broken,
        )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda execution: execution.pop("source_forward_count"),
            "fields differ",
        ),
        (
            lambda execution: execution.__setitem__("invented_count", 0),
            "fields differ",
        ),
        (
            lambda execution: execution["runtime"].__setitem__("device", "cpu"),
            "runtime differs",
        ),
        (
            lambda execution: execution.__setitem__("checkpoint_write_count", 1),
            "counts differ",
        ),
        (
            lambda execution: execution["runtime"].__setitem__(
                "deterministic_algorithms", False
            ),
            "runtime differs",
        ),
        (
            lambda execution: execution.__setitem__(
                "source_model_build_count", True
            ),
            "counts differ",
        ),
        (
            lambda execution: execution.__setitem__("optimizer_build_count", False),
            "counts differ",
        ),
        (
            lambda execution: execution["runtime"].__setitem__(
                "visible_cuda_device_count", True
            ),
            "runtime differs",
        ),
    ],
)
def test_cell_verifier_rejects_execution_schema_and_runtime_tampering(
    tmp_path: Path,
    mutation: Callable[[dict[str, Any]], None],
    message: str,
) -> None:
    payloads, config, code_seal = _fixture_payloads(execution_mutator=mutation)
    artifact = _write_artifact(tmp_path, payloads)
    with pytest.raises(cell.P3StageB1OuterCellShardError, match=message):
        cell.verify_stage_b1_cell_shard(
            artifact,
            repository_root=tmp_path,
            config=config,
            expected_config_sha256=CONFIG_SHA,
            verify_live_parents=False,
            expected_code_seal=code_seal,
        )
