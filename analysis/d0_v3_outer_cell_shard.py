"""CPU-only verifier for one formal P3 Stage-A outer-analysis cell shard.

The verifier opens no cache target and imports no model runner.  It first
revalidates the complete label-free shard, then validates the immutable outer
payload and proves that the separately stored outer Source logits are bit
exact to the label-free Source logits.  Target-derived diagnostics remain
train-only, non-adaptive evidence and never authorize Stage 2.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
from typing import Any, Final

import numpy as np

from analysis.d0_v3_formal_contract import (
    CONDITIONS,
    DATASETS,
    FINE_ALIGNMENT_GROUP_IDS,
    FROZEN_CANDIDATES,
    PROTOCOL_ID,
)
from analysis.d0_v3_label_free_shard import (
    CANDIDATE_COUNT,
    COMPLETE_FILENAME as LABEL_FREE_COMPLETE_FILENAME,
    EPISODES_FILENAME as LABEL_FREE_EPISODES_FILENAME,
    FLOAT_DTYPE,
    FORMAL_IMAGE_COUNT,
    LAYOUT_FILENAME as LABEL_FREE_LAYOUT_FILENAME,
    MANIFEST_FILENAME as LABEL_FREE_MANIFEST_FILENAME,
    PARAMETERS_AFTER_FILENAME as LABEL_FREE_PARAMETERS_AFTER_FILENAME,
    SOURCE_LOGITS_FILENAME as LABEL_FREE_SOURCE_LOGITS_FILENAME,
    SOURCE_PARAMETERS_FILENAME as LABEL_FREE_SOURCE_PARAMETERS_FILENAME,
    VerifiedLabelFreeShard,
    array_slice_sha256,
    canonical_json_bytes,
    parse_canonical_json,
    validate_parameter_layout,
    verify_label_free_shard,
)
from analysis.d0_v3_phase_receipt import (
    OUTER_ACCESS_ARTIFACT_TYPE,
    PROTOCOL_ID as PHASE_PROTOCOL_ID,
)
from tta.d0_secure_io import read_stable_regular_file, snapshot_regular_directory


SCHEMA_VERSION: Final = 3
ARTIFACT_TYPE: Final = "cr_sitta_d0_v3_formal_stage_a_outer_cell_shard"
COMPLETE_ARTIFACT_TYPE: Final = (
    "cr_sitta_d0_v3_formal_stage_a_outer_cell_complete"
)
RECORD_ARTIFACT_TYPE: Final = "cr_sitta_d0_v3_stage_a_outer_episode_record"
TASK_AUDIT_ARTIFACT_TYPE: Final = "cr_sitta_d0_v3_outer_task_loss_audit"
EPISODE_ORDER: Final = "image_major_candidate_minor"

OUTER_SOURCE_LOGITS_FILENAME: Final = "outer_source_logits.npy"
SUPERVISED_GRADIENTS_FILENAME: Final = "supervised_gradients.npy"
OUTER_RECORDS_FILENAME: Final = "outer_records.jsonl"
TASK_LOSS_AUDITS_FILENAME: Final = "task_loss_audits.jsonl"
OUTER_ACCESS_RECEIPT_FILENAME: Final = "outer_access_receipt.json"
MANIFEST_FILENAME: Final = "manifest.json"
COMPLETE_FILENAME: Final = "COMPLETE.json"
MEMBERS: Final = frozenset(
    {
        OUTER_SOURCE_LOGITS_FILENAME,
        SUPERVISED_GRADIENTS_FILENAME,
        OUTER_RECORDS_FILENAME,
        TASK_LOSS_AUDITS_FILENAME,
        OUTER_ACCESS_RECEIPT_FILENAME,
        MANIFEST_FILENAME,
        COMPLETE_FILENAME,
    }
)
OUTER_CRITICAL_CODE_PATHS: Final = (
    "scripts/run_d0_v3_formal_stage_a_outer.py",
    "analysis/d0_v3_outer_cell_shard.py",
    "tta/d0_v3_outer_source_gradient.py",
    "analysis/d0_v3_outer_analyzer.py",
    "analysis/d0_v3_phase_receipt.py",
    "analysis/d0_v3_label_free_shard.py",
    "analysis/d0_v2_task_loss.py",
    "tta/d0_v3_atomic_shard.py",
)

_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z", re.ASCII)
_RECORD_FIELDS = {
    "schema_version",
    "artifact_type",
    "dataset",
    "condition",
    "replicate_id",
    "image_index",
    "image_id",
    "candidate",
    "finite_gradient",
    "changed_parameter_tensor_count",
    "analysis",
}
_ANALYSIS_FIELDS = {
    "schema_version",
    "artifact_type",
    "scope",
    "label_isolation",
    "layout_sha256",
    "noop",
    "threshold_margin_bin_response",
    "entropy_task_alignment",
    "scientific_selection_performed",
    "stage2_authorized",
}
_TASK_AUDIT_FIELDS = {
    "schema_version",
    "artifact_type",
    "dataset",
    "condition",
    "replicate_id",
    "image_index",
    "image_id",
    "source_logits_bit_exact",
    "label_free_source_logits_slice_sha256",
    "outer_source_logits_slice_sha256",
    "supervised_gradient_slice_sha256",
    "supervised_gradient_finite",
    "source_state_sha256",
    "reset_source_state_sha256",
    "task_loss",
    "forward_runtime",
    "used_by_adaptation",
    "stage2_authorized",
}
_FORWARD_RUNTIME_FIELDS = {
    "seed",
    "device",
    "deterministic_algorithms",
    "deterministic_warn_only",
    "cudnn_benchmark",
    "cudnn_deterministic",
    "cublas_workspace_config",
    "visible_cuda_device_count",
}
_AUTHORIZATION = {
    "source_train_derived": True,
    "paper_result": False,
    "paper_test_result": False,
    "development_test_selected_result": False,
    "scientific_selection_performed": False,
    "formal_protocol_complete": False,
    "stage2_authorized": False,
}
_DATA_BOUNDARY = {
    "candidate_phase_cpu_verified_before_outer_target_load": True,
    "canonical_phase_receipt_verified_before_outer_target_load": True,
    "candidate_phase_target_access_count": 0,
    "outer_target_loader_call_count": 1,
    "outer_target_indexing_count": FORMAL_IMAGE_COUNT,
    "method_label_accesses": 0,
    "supervised_gradient_used_by_adaptation": False,
    "adaptation_optimizer_constructed_by_outer": False,
    "validation_payload_opens": 0,
    "test_split_files_opened": 0,
    "test_images_opened": 0,
    "test_masks_opened": 0,
    "test_labels_opened": 0,
}
_EXECUTION = {
    "outer_source_model_build_count": 1,
    "outer_optimizer_build_count": 0,
    "source_forward_count": FORMAL_IMAGE_COUNT,
    "supervised_backward_count": FORMAL_IMAGE_COUNT,
    "supervised_gradient_count": FORMAL_IMAGE_COUNT,
    "source_logits_bit_exact_count": FORMAL_IMAGE_COUNT,
    "outer_episode_analysis_count": FORMAL_IMAGE_COUNT * CANDIDATE_COUNT,
    "candidate_optimizer_step_count_in_outer": 0,
}


class D0V3OuterCellShardError(ValueError):
    """An outer cell shard is incomplete, unbound, or unsafe."""


@dataclass(frozen=True, slots=True)
class VerifiedOuterCellShard:
    path: Path
    dataset: str
    condition: str
    replicate: str
    image_count: int
    candidate_count: int
    record_count: int
    manifest_sha256: str
    complete_sha256: str
    outer_access_receipt_sha256: str
    label_free_manifest_sha256: str
    label_free_phase_receipt_sha256: str


def _mapping(value: Any, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise D0V3OuterCellShardError(f"{label} must be a mapping")
    return value


def _exact(value: Any, fields: set[str], *, label: str) -> Mapping[str, Any]:
    result = _mapping(value, label=label)
    if set(result) != fields:
        missing = sorted(fields - set(result))
        unknown = sorted(set(result) - fields)
        raise D0V3OuterCellShardError(
            f"{label} fields differ; missing={missing}, unknown={unknown}"
        )
    return result


def _sha256(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise D0V3OuterCellShardError(f"{label} must be lowercase SHA-256")
    return value


def _integer(value: Any, *, label: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise D0V3OuterCellShardError(f"{label} must be integer >= {minimum}")
    return value


def _finite(value: Any, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise D0V3OuterCellShardError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise D0V3OuterCellShardError(f"{label} must be finite")
    return result


def _unique_object(pairs: Sequence[tuple[str, Any]], label: str) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise D0V3OuterCellShardError(
                f"{label} contains duplicate JSON key {key!r}"
            )
        result[key] = value
    return result


def canonical_ordered_json_bytes(value: Mapping[str, Any]) -> bytes:
    """Canonical JSON preserving the validated fine-group insertion order."""

    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=False,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise D0V3OuterCellShardError("ordered record is not finite JSON") from exc


def _parse_json_line(line: bytes, *, label: str, ordered: bool) -> Mapping[str, Any]:
    try:
        value = json.loads(
            line.decode("utf-8"),
            object_pairs_hook=lambda pairs: _unique_object(pairs, label),
            parse_constant=lambda token: (_ for _ in ()).throw(
                D0V3OuterCellShardError(f"{label} contains {token}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise D0V3OuterCellShardError(f"{label} is not strict JSON") from exc
    result = _mapping(value, label=label)
    expected = (
        canonical_ordered_json_bytes(result)
        if ordered
        else canonical_json_bytes(result)
    )
    if expected != line:
        raise D0V3OuterCellShardError(f"{label} is not canonical")
    return result


def _parse_jsonl(data: bytes, *, label: str, ordered: bool) -> list[Mapping[str, Any]]:
    if not data or not data.endswith(b"\n"):
        raise D0V3OuterCellShardError(f"{label} must be non-empty newline JSONL")
    lines = data[:-1].split(b"\n")
    if any(not line for line in lines):
        raise D0V3OuterCellShardError(f"{label} contains an empty line")
    return [
        _parse_json_line(line, label=f"{label}[{index}]", ordered=ordered)
        for index, line in enumerate(lines)
    ]


def canonical_jsonl_bytes(
    records: Sequence[Mapping[str, Any]], *, ordered: bool
) -> bytes:
    if isinstance(records, (str, bytes)) or not isinstance(records, Sequence):
        raise D0V3OuterCellShardError("records must be a sequence")
    serializer = canonical_ordered_json_bytes if ordered else canonical_json_bytes
    return b"".join(serializer(record) + b"\n" for record in records)


def _npy(data: bytes, *, label: str) -> np.ndarray:
    try:
        value = np.load(io.BytesIO(data), allow_pickle=False)
    except (OSError, ValueError, EOFError) as exc:
        raise D0V3OuterCellShardError(f"{label} is not safe NPY") from exc
    if not isinstance(value, np.ndarray):
        raise D0V3OuterCellShardError(f"{label} is not ndarray")
    return value


def _relative(root: Path, value: Path, *, label: str) -> str:
    try:
        relative = value.relative_to(root)
    except ValueError as exc:
        raise D0V3OuterCellShardError(f"{label} is outside repository root") from exc
    return relative.as_posix()


def _input_seal_by_role(manifest: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    seal = _mapping(manifest.get("input_seal"), label="label-free input_seal")
    files = seal.get("files")
    if isinstance(files, (str, bytes)) or not isinstance(files, Sequence):
        raise D0V3OuterCellShardError("label-free input seal files are missing")
    by_role: dict[str, Mapping[str, Any]] = {}
    for index, raw in enumerate(files):
        record = _mapping(raw, label=f"label-free input_seal.files[{index}]")
        if set(record) != {"role", "path", "sha256"}:
            raise D0V3OuterCellShardError("label-free input seal record differs")
        role = record["role"]
        if not isinstance(role, str) or not role or role in by_role:
            raise D0V3OuterCellShardError("label-free input seal role differs")
        _sha256(record["sha256"], label=f"label-free input role {role}")
        by_role[role] = record
    for role in ("cache_protocol", "cache_method_manifest"):
        if role not in by_role:
            raise D0V3OuterCellShardError(f"label-free input seal lacks {role}")
    return by_role


def build_outer_code_seal(repository_root: str | os.PathLike[str]) -> dict[str, Any]:
    """Hash the exact additive outer implementation without target access."""

    root = Path(os.path.abspath(os.fspath(repository_root)))
    files = [
        {
            "path": relative,
            "sha256": read_stable_regular_file(root / relative).sha256,
        }
        for relative in OUTER_CRITICAL_CODE_PATHS
    ]
    return {
        "files": files,
        "bundle_sha256": hashlib.sha256(canonical_json_bytes(files)).hexdigest(),
    }


def _label_free_context(
    path: Path,
    *,
    repository_root: Path,
    expected_config_sha256: str | None,
) -> tuple[
    VerifiedLabelFreeShard,
    Mapping[str, Any],
    Any,
    np.ndarray,
    list[Mapping[str, Any]],
]:
    verified = verify_label_free_shard(
        path,
        expected_config_sha256=expected_config_sha256,
        verify_live_inputs=False,
    )
    if (
        not verified.formal
        or verified.dry_run
        or verified.replicate != "R0"
        or verified.image_count != FORMAL_IMAGE_COUNT
        or verified.episode_count != FORMAL_IMAGE_COUNT * CANDIDATE_COUNT
    ):
        raise D0V3OuterCellShardError(
            "outer analysis requires one complete formal R0/64 label-free shard"
        )
    manifest_snapshot = read_stable_regular_file(path / LABEL_FREE_MANIFEST_FILENAME)
    if manifest_snapshot.sha256 != verified.manifest_sha256:
        raise D0V3OuterCellShardError("label-free manifest changed after verification")
    manifest = parse_canonical_json(
        manifest_snapshot.data, label="label-free manifest", newline=True
    )
    layout = validate_parameter_layout(
        parse_canonical_json(
            read_stable_regular_file(path / LABEL_FREE_LAYOUT_FILENAME).data,
            label="label-free parameter layout",
            newline=True,
        )
    )
    source_snapshot = read_stable_regular_file(path / LABEL_FREE_SOURCE_LOGITS_FILENAME)
    source_logits = _npy(source_snapshot.data, label="label-free Source logits")
    if (
        source_logits.dtype.str != FLOAT_DTYPE
        or source_logits.shape != (FORMAL_IMAGE_COUNT, 1, 256, 256)
        or not source_logits.flags.c_contiguous
        or not np.isfinite(source_logits).all()
    ):
        raise D0V3OuterCellShardError("label-free Source logits schema differs")
    episode_snapshot = read_stable_regular_file(path / LABEL_FREE_EPISODES_FILENAME)
    episodes = _parse_jsonl(
        episode_snapshot.data, label="label-free episodes", ordered=False
    )
    if len(episodes) != FORMAL_IMAGE_COUNT * CANDIDATE_COUNT:
        raise D0V3OuterCellShardError("label-free episode count differs")
    _relative(repository_root, path, label="label-free shard")
    return verified, manifest, layout, source_logits, episodes


def _recompute_changed_parameter_tensor_counts(
    label_free_path: Path, layout: Any
) -> tuple[int, ...]:
    """Recompute all 640 changed-tensor counts from lossless label-free arrays."""

    try:
        source = np.load(
            label_free_path / LABEL_FREE_SOURCE_PARAMETERS_FILENAME,
            mmap_mode="r",
            allow_pickle=False,
        )
        after = np.load(
            label_free_path / LABEL_FREE_PARAMETERS_AFTER_FILENAME,
            mmap_mode="r",
            allow_pickle=False,
        )
    except (OSError, ValueError) as exc:
        raise D0V3OuterCellShardError(
            "cannot reopen verified parameter arrays for changed-count audit"
        ) from exc
    expected_shape = (FORMAL_IMAGE_COUNT, CANDIDATE_COUNT, layout.scalar_count)
    if (
        not isinstance(source, np.ndarray)
        or not isinstance(after, np.ndarray)
        or source.shape != expected_shape
        or after.shape != expected_shape
        or source.dtype.str != FLOAT_DTYPE
        or after.dtype.str != FLOAT_DTYPE
        or not np.isfinite(source).all()
        or not np.isfinite(after).all()
    ):
        raise D0V3OuterCellShardError("label-free parameter array schema differs")
    counts: list[int] = []
    for image_index in range(FORMAL_IMAGE_COUNT):
        for candidate_index in range(CANDIDATE_COUNT):
            changed = 0
            for tensor_index, start in enumerate(layout.offsets):
                end = (
                    layout.offsets[tensor_index + 1]
                    if tensor_index + 1 < len(layout.offsets)
                    else layout.scalar_count
                )
                if not np.array_equal(
                    source[image_index, candidate_index, start:end],
                    after[image_index, candidate_index, start:end],
                ):
                    changed += 1
            counts.append(changed)
    if len(counts) != FORMAL_IMAGE_COUNT * CANDIDATE_COUNT:
        raise D0V3OuterCellShardError("changed-count grid is incomplete")
    return tuple(counts)


def _validate_task_loss(
    value: Any,
    *,
    dataset: str,
    split_sha256: str,
    checkpoint_sha256: str,
) -> None:
    audit = _mapping(value, label="task_loss")
    expected = {
        "schema_version",
        "task_loss_type",
        "scope",
        "label_isolation",
        "config",
        "input",
        "components",
    }
    if set(audit) != expected or audit["schema_version"] != 1:
        raise D0V3OuterCellShardError("task-loss audit schema differs")
    scope = _mapping(audit["scope"], label="task_loss.scope")
    if (
        scope.get("dataset") != dataset
        or scope.get("split_name") != "train"
        or scope.get("split_sha256") != split_sha256
        or scope.get("checkpoint_sha256") != checkpoint_sha256
        or scope.get("paper_test_result") is not False
        or scope.get("use_test_images") is not False
        or scope.get("use_test_labels") is not False
        or scope.get("method_label_accesses") != 0
        or scope.get("outer_evaluator_label_accesses") != 1
        or scope.get("adaptation_gradient_uses_labels") is not False
    ):
        raise D0V3OuterCellShardError("task-loss train-only scope differs")
    isolation = _mapping(audit["label_isolation"], label="task_loss.label_isolation")
    if (
        isolation.get("used_by_adaptation") is not False
        or isolation.get("adaptation_gradient_uses_labels") is not False
        or isolation.get("method_label_accesses") != 0
        or isolation.get("outer_evaluator_label_accesses") != 1
    ):
        raise D0V3OuterCellShardError("task-loss label isolation differs")
    components = _mapping(audit["components"], label="task_loss.components")
    if not components or any(
        not math.isfinite(_finite(value, label=f"task loss {key}"))
        for key, value in components.items()
    ):
        raise D0V3OuterCellShardError("task-loss components are not finite")


def _validate_analysis(
    value: Any,
    *,
    dataset: str,
    split_sha256: str,
    checkpoint_sha256: str,
    layout_sha256: str,
) -> None:
    analysis = _exact(value, _ANALYSIS_FIELDS, label="record.analysis")
    if (
        analysis["schema_version"] != 3
        or analysis["artifact_type"] != "cr_sitta_p3_stage_a_outer_episode"
        or analysis["layout_sha256"] != layout_sha256
        or analysis["scientific_selection_performed"] is not False
        or analysis["stage2_authorized"] is not False
    ):
        raise D0V3OuterCellShardError("outer analysis identity differs")
    scope = _mapping(analysis["scope"], label="analysis.scope")
    if (
        scope.get("dataset") != dataset
        or scope.get("split_name") != "train"
        or scope.get("split_sha256") != split_sha256
        or scope.get("checkpoint_sha256") != checkpoint_sha256
        or scope.get("method_label_accesses") != 0
        or scope.get("outer_evaluator_label_accesses") != 1
        or scope.get("adaptation_gradient_uses_labels") is not False
        or scope.get("paper_test_result") is not False
    ):
        raise D0V3OuterCellShardError("outer analysis scope differs")
    isolation = _mapping(analysis["label_isolation"], label="analysis.label_isolation")
    if (
        isolation.get("label_free_payload_complete_before_target_open") is not True
        or isolation.get("method_label_accesses") != 0
        or isolation.get("outer_evaluator_label_accesses") != 1
        or isolation.get("supervised_gradient_used_by_adaptation") is not False
        or isolation.get("adaptation_optimizer_executed_by_outer_evaluator") is not False
        or isolation.get("test_payload_accesses") != 0
    ):
        raise D0V3OuterCellShardError("outer analysis label isolation differs")
    alignment = _mapping(
        analysis["entropy_task_alignment"], label="analysis.entropy_task_alignment"
    )
    per_group = _mapping(alignment.get("per_group"), label="alignment.per_group")
    if tuple(per_group) != FINE_ALIGNMENT_GROUP_IDS:
        raise D0V3OuterCellShardError("fine-group alignment order differs")


def verify_outer_cell_shard(
    path: str | os.PathLike[str],
    *,
    repository_root: str | os.PathLike[str],
    label_free_shard_path: str | os.PathLike[str],
    expected_config_sha256: str | None = None,
) -> VerifiedOuterCellShard:
    """Fully verify one immutable formal outer cell without opening a target."""

    root = Path(os.path.abspath(os.fspath(path)))
    repo = Path(os.path.abspath(os.fspath(repository_root)))
    label_path = Path(os.path.abspath(os.fspath(label_free_shard_path)))
    (
        label_verified,
        label_manifest,
        layout,
        label_source_logits,
        label_episodes,
    ) = _label_free_context(
        label_path,
        repository_root=repo,
        expected_config_sha256=expected_config_sha256,
    )
    recomputed_changed_counts = _recompute_changed_parameter_tensor_counts(
        label_path, layout
    )
    try:
        snapshot = snapshot_regular_directory(root)
    except (OSError, RuntimeError, ValueError) as exc:
        raise D0V3OuterCellShardError(
            f"cannot securely snapshot outer shard: {root}"
        ) from exc
    by_name = {member.path.name: member for member in snapshot.members}
    if set(by_name) != MEMBERS:
        raise D0V3OuterCellShardError("outer shard member set differs")

    manifest = parse_canonical_json(
        by_name[MANIFEST_FILENAME].data, label=MANIFEST_FILENAME, newline=True
    )
    manifest_fields = {
        "schema_version",
        "artifact_type",
        "protocol_id",
        "config_sha256",
        "cell",
        "mode",
        "label_free_shard",
        "dataset_binding",
        "ordered_image_ids",
        "ordered_image_ids_sha256",
        "parameter_layout",
        "outer_code_seal",
        "arrays",
        "outer_records",
        "task_loss_audits",
        "outer_access_receipt",
        "execution",
        "data_boundary",
        "authorization",
    }
    if set(manifest) != manifest_fields:
        raise D0V3OuterCellShardError("outer manifest fields differ")
    if (
        manifest["schema_version"] != SCHEMA_VERSION
        or manifest["artifact_type"] != ARTIFACT_TYPE
        or manifest["protocol_id"] != PROTOCOL_ID
    ):
        raise D0V3OuterCellShardError("outer manifest protocol differs")
    config_sha256 = _sha256(manifest["config_sha256"], label="config_sha256")
    if expected_config_sha256 is not None and config_sha256 != _sha256(
        expected_config_sha256, label="expected_config_sha256"
    ):
        raise D0V3OuterCellShardError("outer config SHA differs")
    cell = _mapping(manifest["cell"], label="manifest.cell")
    if set(cell) != {"dataset", "condition", "replicate"}:
        raise D0V3OuterCellShardError("outer cell fields differ")
    dataset, condition, replicate = (
        cell["dataset"],
        cell["condition"],
        cell["replicate"],
    )
    if (
        dataset not in DATASETS
        or condition not in CONDITIONS
        or replicate != "R0"
        or dataset != label_verified.dataset
        or condition != label_verified.condition
    ):
        raise D0V3OuterCellShardError("outer/label-free cell binding differs")
    if manifest["mode"] != {
        "formal": True,
        "dry_run": False,
        "image_count": FORMAL_IMAGE_COUNT,
        "candidate_count": CANDIDATE_COUNT,
        "record_count": FORMAL_IMAGE_COUNT * CANDIDATE_COUNT,
        "record_order": EPISODE_ORDER,
    }:
        raise D0V3OuterCellShardError("outer formal R0/64 mode differs")

    label_binding = _mapping(manifest["label_free_shard"], label="label_free_shard")
    expected_label_binding = {
        "path": _relative(repo, label_path, label="label-free shard"),
        "manifest_sha256": label_verified.manifest_sha256,
        "complete_sha256": label_verified.complete_sha256,
        "phase_receipt_sha256": label_verified.phase_receipt_sha256,
        "cpu_verified_before_target_load": True,
    }
    if label_binding != expected_label_binding:
        raise D0V3OuterCellShardError("label-free lineage binding differs")

    dataset_binding = _mapping(manifest["dataset_binding"], label="dataset_binding")
    label_dataset_binding = _mapping(
        label_manifest["dataset_binding"], label="label-free dataset_binding"
    )
    input_by_role = _input_seal_by_role(label_manifest)
    if set(label_dataset_binding) != {
        "train_split_sha256",
        "checkpoint_role",
        "checkpoint_path",
        "checkpoint_sha256",
    } or label_dataset_binding["checkpoint_role"] != "best_miou":
        raise D0V3OuterCellShardError("label-free dataset binding differs")
    expected_dataset_binding = {
        "train_split_sha256": label_dataset_binding["train_split_sha256"],
        "checkpoint_path": label_dataset_binding["checkpoint_path"],
        "checkpoint_sha256": label_dataset_binding["checkpoint_sha256"],
        "cache_protocol_sha256": input_by_role["cache_protocol"]["sha256"],
        "cache_method_manifest_sha256": input_by_role[
            "cache_method_manifest"
        ]["sha256"],
    }
    if dataset_binding != expected_dataset_binding:
        raise D0V3OuterCellShardError("outer dataset/cache lineage differs")

    image_ids = manifest["ordered_image_ids"]
    if (
        image_ids != label_manifest["ordered_image_ids"]
        or not isinstance(image_ids, list)
        or len(image_ids) != FORMAL_IMAGE_COUNT
        or len(set(image_ids)) != FORMAL_IMAGE_COUNT
    ):
        raise D0V3OuterCellShardError("outer ordered Pilot IDs differ")
    if manifest["ordered_image_ids_sha256"] != label_manifest[
        "ordered_image_ids_sha256"
    ]:
        raise D0V3OuterCellShardError("outer Pilot ID seal differs")
    expected_layout = {
        "source_path": (
            _relative(repo, label_path, label="label-free shard")
            + f"/{LABEL_FREE_LAYOUT_FILENAME}"
        ),
        "source_file_sha256": read_stable_regular_file(
            label_path / LABEL_FREE_LAYOUT_FILENAME
        ).sha256,
        "layout_sha256": layout.layout_sha256,
        "parameter_tensor_count": len(layout.names),
        "scalar_parameter_count": layout.scalar_count,
    }
    if manifest["parameter_layout"] != expected_layout:
        raise D0V3OuterCellShardError("outer parameter layout binding differs")
    if manifest["outer_code_seal"] != build_outer_code_seal(repo):
        raise D0V3OuterCellShardError("outer implementation code seal differs")

    outer_logits = _npy(
        by_name[OUTER_SOURCE_LOGITS_FILENAME].data,
        label=OUTER_SOURCE_LOGITS_FILENAME,
    )
    supervised = _npy(
        by_name[SUPERVISED_GRADIENTS_FILENAME].data,
        label=SUPERVISED_GRADIENTS_FILENAME,
    )
    for filename, array, shape in (
        (
            OUTER_SOURCE_LOGITS_FILENAME,
            outer_logits,
            (FORMAL_IMAGE_COUNT, 1, 256, 256),
        ),
        (
            SUPERVISED_GRADIENTS_FILENAME,
            supervised,
            (FORMAL_IMAGE_COUNT, layout.scalar_count),
        ),
    ):
        if (
            array.dtype.str != FLOAT_DTYPE
            or array.shape != shape
            or not array.flags.c_contiguous
            or not np.isfinite(array).all()
        ):
            raise D0V3OuterCellShardError(f"outer array schema differs: {filename}")
    if not np.array_equal(outer_logits, label_source_logits):
        raise D0V3OuterCellShardError(
            "outer Source logits are not bit-exact to label-free Source logits"
        )
    expected_arrays = {
        filename: {
            "path": filename,
            "sha256": by_name[filename].sha256,
            "shape": list(array.shape),
            "dtype": FLOAT_DTYPE,
            "c_order": True,
            "finite": True,
            "lossless": True,
        }
        for filename, array in (
            (OUTER_SOURCE_LOGITS_FILENAME, outer_logits),
            (SUPERVISED_GRADIENTS_FILENAME, supervised),
        )
    }
    if manifest["arrays"] != expected_arrays:
        raise D0V3OuterCellShardError("outer array manifest differs")

    records = _parse_jsonl(
        by_name[OUTER_RECORDS_FILENAME].data,
        label=OUTER_RECORDS_FILENAME,
        ordered=True,
    )
    if len(records) != FORMAL_IMAGE_COUNT * CANDIDATE_COUNT:
        raise D0V3OuterCellShardError("outer record count differs")
    split_sha256 = dataset_binding["train_split_sha256"]
    checkpoint_sha256 = dataset_binding["checkpoint_sha256"]
    for position, raw in enumerate(records):
        record = _exact(raw, _RECORD_FIELDS, label=f"records[{position}]")
        image_index, candidate_index = divmod(position, CANDIDATE_COUNT)
        candidate = FROZEN_CANDIDATES[candidate_index]
        if (
            record["schema_version"] != SCHEMA_VERSION
            or record["artifact_type"] != RECORD_ARTIFACT_TYPE
            or record["dataset"] != dataset
            or record["condition"] != condition
            or record["replicate_id"] != "R0"
            or record["image_index"] != image_index
            or record["image_id"] != image_ids[image_index]
            or record["candidate"]
            != {
                "candidate_id": candidate.candidate_id,
                "optimizer": candidate.optimizer,
                "learning_rate": candidate.learning_rate,
            }
            or record["finite_gradient"] is not True
        ):
            raise D0V3OuterCellShardError("outer record grid/identity differs")
        changed = _integer(
            record["changed_parameter_tensor_count"],
            label="changed_parameter_tensor_count",
        )
        nested = _mapping(
            label_episodes[position]["independent_candidate_receipt"],
            label="label-free independent receipt",
        )
        nested_numeric = _mapping(
            nested["numeric_evidence"], label="label-free numeric evidence"
        )
        if (
            changed != nested_numeric["changed_parameter_tensor_count"]
            or changed != recomputed_changed_counts[position]
        ):
            raise D0V3OuterCellShardError(
                "outer changed-parameter count differs from lossless label-free payload"
            )
        _validate_analysis(
            record["analysis"],
            dataset=dataset,
            split_sha256=split_sha256,
            checkpoint_sha256=checkpoint_sha256,
            layout_sha256=layout.layout_sha256,
        )
    expected_records = {
        "path": OUTER_RECORDS_FILENAME,
        "sha256": by_name[OUTER_RECORDS_FILENAME].sha256,
        "count": FORMAL_IMAGE_COUNT * CANDIDATE_COUNT,
        "order": EPISODE_ORDER,
    }
    if manifest["outer_records"] != expected_records:
        raise D0V3OuterCellShardError("outer record manifest differs")

    audits = _parse_jsonl(
        by_name[TASK_LOSS_AUDITS_FILENAME].data,
        label=TASK_LOSS_AUDITS_FILENAME,
        ordered=False,
    )
    if len(audits) != FORMAL_IMAGE_COUNT:
        raise D0V3OuterCellShardError("task-loss audit count differs")
    source_state_sha256: str | None = None
    for index, raw in enumerate(audits):
        audit = _exact(raw, _TASK_AUDIT_FIELDS, label=f"task_audits[{index}]")
        runtime = _exact(
            audit["forward_runtime"],
            _FORWARD_RUNTIME_FIELDS,
            label=f"task_audits[{index}].forward_runtime",
        )
        if (
            audit["schema_version"] != SCHEMA_VERSION
            or audit["artifact_type"] != TASK_AUDIT_ARTIFACT_TYPE
            or audit["dataset"] != dataset
            or audit["condition"] != condition
            or audit["replicate_id"] != "R0"
            or audit["image_index"] != index
            or audit["image_id"] != image_ids[index]
            or audit["source_logits_bit_exact"] is not True
            or audit["supervised_gradient_finite"] is not True
            or audit["used_by_adaptation"] is not False
            or audit["stage2_authorized"] is not False
            or runtime["seed"] != 42
            or runtime["device"] != "cuda:0"
            or runtime["deterministic_algorithms"] is not True
            or runtime["deterministic_warn_only"] is not False
            or runtime["cudnn_benchmark"] is not False
            or runtime["cudnn_deterministic"] is not True
            or runtime["cublas_workspace_config"] != ":4096:8"
            or runtime["visible_cuda_device_count"] != 1
        ):
            raise D0V3OuterCellShardError("task-loss audit identity differs")
        label_hash = array_slice_sha256(label_source_logits[index])
        outer_hash = array_slice_sha256(outer_logits[index])
        gradient_hash = array_slice_sha256(supervised[index])
        if (
            audit["label_free_source_logits_slice_sha256"] != label_hash
            or audit["outer_source_logits_slice_sha256"] != outer_hash
            or label_hash != outer_hash
            or audit["supervised_gradient_slice_sha256"] != gradient_hash
        ):
            raise D0V3OuterCellShardError("task-loss tensor slice seal differs")
        source_sha = _sha256(audit["source_state_sha256"], label="source_state_sha256")
        if audit["reset_source_state_sha256"] != source_sha:
            raise D0V3OuterCellShardError("outer Source reset state differs")
        if source_state_sha256 is None:
            source_state_sha256 = source_sha
        elif source_state_sha256 != source_sha:
            raise D0V3OuterCellShardError("outer Source state changed across images")
        _validate_task_loss(
            audit["task_loss"],
            dataset=dataset,
            split_sha256=split_sha256,
            checkpoint_sha256=checkpoint_sha256,
        )
    expected_audits = {
        "path": TASK_LOSS_AUDITS_FILENAME,
        "sha256": by_name[TASK_LOSS_AUDITS_FILENAME].sha256,
        "count": FORMAL_IMAGE_COUNT,
        "one_supervised_gradient_per_image": True,
        "used_by_adaptation": False,
    }
    if manifest["task_loss_audits"] != expected_audits:
        raise D0V3OuterCellShardError("task-loss audit manifest differs")

    access = parse_canonical_json(
        by_name[OUTER_ACCESS_RECEIPT_FILENAME].data,
        label=OUTER_ACCESS_RECEIPT_FILENAME,
        newline=False,
    )
    if (
        access.get("schema_version") != SCHEMA_VERSION
        or access.get("artifact_type") != OUTER_ACCESS_ARTIFACT_TYPE
        or access.get("protocol_id") != PHASE_PROTOCOL_ID
        or access.get("cell_binding")
        != {"dataset": dataset, "condition": condition, "replicate": 0}
        or access.get("phase_evidence", {}).get(
            "label_free_cell_receipt_sha256"
        )
        != label_verified.phase_receipt_sha256
        or access.get("phase_evidence", {}).get(
            "complete_candidate_episode_count"
        )
        != FORMAL_IMAGE_COUNT * CANDIDATE_COUNT
        or access.get("phase_evidence", {}).get(
            "candidate_phase_target_access_count"
        )
        != 0
        or access.get("outer_access", {}).get("loader_call_count") != 1
        or access.get("outer_access", {}).get("used_by_adaptation") is not False
        or access.get("outer_access", {}).get("adaptation_target_indexing_count")
        != 0
        or access.get("authorization", {}).get("stage2_authorized") is not False
    ):
        raise D0V3OuterCellShardError("outer target access receipt differs")
    expected_access = {
        "path": OUTER_ACCESS_RECEIPT_FILENAME,
        "sha256": by_name[OUTER_ACCESS_RECEIPT_FILENAME].sha256,
        "used_by_adaptation": False,
        "outer_target_loader_call_count": 1,
    }
    if manifest["outer_access_receipt"] != expected_access:
        raise D0V3OuterCellShardError("outer access manifest differs")
    if manifest["execution"] != _EXECUTION:
        raise D0V3OuterCellShardError("outer execution counts differ")
    if manifest["data_boundary"] != _DATA_BOUNDARY:
        raise D0V3OuterCellShardError("outer data boundary differs")
    if manifest["authorization"] != _AUTHORIZATION:
        raise D0V3OuterCellShardError("outer authorization differs")

    complete = parse_canonical_json(
        by_name[COMPLETE_FILENAME].data, label=COMPLETE_FILENAME, newline=True
    )
    payload_files = [
        {"path": name, "sha256": by_name[name].sha256}
        for name in sorted(set(by_name) - {COMPLETE_FILENAME})
    ]
    expected_complete = {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": COMPLETE_ARTIFACT_TYPE,
        "complete": True,
        "candidate_phase_complete": True,
        "outer_phase_complete": True,
        "formal": True,
        "dry_run": False,
        "config_sha256": config_sha256,
        "dataset": dataset,
        "condition": condition,
        "replicate": "R0",
        "image_count": FORMAL_IMAGE_COUNT,
        "candidate_count": CANDIDATE_COUNT,
        "record_count": FORMAL_IMAGE_COUNT * CANDIDATE_COUNT,
        "manifest": {
            "path": MANIFEST_FILENAME,
            "sha256": by_name[MANIFEST_FILENAME].sha256,
        },
        "payload_files": payload_files,
        "atomic_no_replace": True,
        "paper_result": False,
        "paper_test_result": False,
        "formal_protocol_complete": False,
        "scientific_gate_status": "not_evaluated",
        "stage2_authorized": False,
    }
    if complete != expected_complete:
        raise D0V3OuterCellShardError("outer COMPLETE does not rebuild from payload")
    if read_stable_regular_file(label_path / LABEL_FREE_COMPLETE_FILENAME).sha256 != (
        label_verified.complete_sha256
    ):
        raise D0V3OuterCellShardError("label-free COMPLETE changed after outer verify")
    return VerifiedOuterCellShard(
        path=root,
        dataset=dataset,
        condition=condition,
        replicate="R0",
        image_count=FORMAL_IMAGE_COUNT,
        candidate_count=CANDIDATE_COUNT,
        record_count=FORMAL_IMAGE_COUNT * CANDIDATE_COUNT,
        manifest_sha256=by_name[MANIFEST_FILENAME].sha256,
        complete_sha256=by_name[COMPLETE_FILENAME].sha256,
        outer_access_receipt_sha256=by_name[OUTER_ACCESS_RECEIPT_FILENAME].sha256,
        label_free_manifest_sha256=label_verified.manifest_sha256,
        label_free_phase_receipt_sha256=str(label_verified.phase_receipt_sha256),
    )


__all__ = [
    "ARTIFACT_TYPE",
    "COMPLETE_ARTIFACT_TYPE",
    "COMPLETE_FILENAME",
    "D0V3OuterCellShardError",
    "EPISODE_ORDER",
    "MANIFEST_FILENAME",
    "MEMBERS",
    "OUTER_ACCESS_RECEIPT_FILENAME",
    "OUTER_CRITICAL_CODE_PATHS",
    "OUTER_RECORDS_FILENAME",
    "OUTER_SOURCE_LOGITS_FILENAME",
    "RECORD_ARTIFACT_TYPE",
    "SCHEMA_VERSION",
    "SUPERVISED_GRADIENTS_FILENAME",
    "TASK_AUDIT_ARTIFACT_TYPE",
    "TASK_LOSS_AUDITS_FILENAME",
    "VerifiedOuterCellShard",
    "build_outer_code_seal",
    "canonical_jsonl_bytes",
    "canonical_ordered_json_bytes",
    "verify_outer_cell_shard",
]
