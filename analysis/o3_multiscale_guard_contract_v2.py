"""Read-only byte/metadata verification of the sealed train8 v1 parent.

No image, NPY or checkpoint deserializer is imported. The new experiment may
reuse parent payloads only after this independent verifier has checked them.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

REPOSITORY = Path(__file__).resolve().parents[1]
PARENT_RELATIVE = "results/cr_sitta/o3_multiscale_train8_v1/R0"
PARENT_ROOT = REPOSITORY / PARENT_RELATIVE
PARENT_MANIFEST_SHA256 = "ba2fadbd15b01c5f9e331f78678420c6993917ffff246ea31f7e52ba681801d4"
PARENT_LEDGER_SHA256 = "06a9c5e2f78c9194d5c6cebae7ab518d838bd86390cc9953bebc31dbb69fc329"
PARENT_PROTOCOL = "cr-sitta-o3-multiscale-train8-v1"
DATASET = "NUDT-SIRST"
IMAGE_IDS = ("000891", "001005", "000641", "001179", "000328", "001048", "000460", "001228")
CONDITIONS = (("clean", 0), *((family, severity)
    for family in ("gaussian_noise", "gaussian_blur", "low_contrast", "stripe_noise")
    for severity in (1, 3, 5)))
TRAIN_SPLIT = "datasets/NUDT-SIRST/img_idx/train_NUDT-SIRST.txt"
CACHE_MANIFEST = "results/binary_tent/ss_calibration_cache_v2/NUDT-SIRST/manifest.json"
TEACHER_MANIFEST = "results/cr_sitta/nonadaptive_teacher_screen_v1/candidate_phase/NUDT-SIRST/manifest.json"
SCOPE = {"role": "source_supervised_fit_smoke", "train_only": True,
         "no_validation_split": True, "formal_test": False, "paper_result": False,
         "full_training": False, "fit_and_measure_on_same_images": True}
ROOT_PAYLOADS = frozenset(("cells.jsonl", "episodes.jsonl", "initial.pth.tar", "manifest.json",
    "o3_features.npy", "o3_probabilities.npy", "runtime.json", "source_probabilities.npy",
    "step_0128.pth.tar", "summary.json", "train_target_receipt.json", "train_targets.npy",
    "trained_probabilities.npy", "training.jsonl", "training_order.json"))


class ParentRunVerificationError(RuntimeError):
    """A sealed v1 input, file roster or source-training boundary changed."""


def _same(actual: Any, expected: Any, label: str) -> None:
    if actual != expected or type(actual) is not type(expected):
        raise ParentRunVerificationError(f"{label} differs")


def _safe(path: Path, repository: Path) -> Path:
    if not path.is_absolute() or not path.is_relative_to(repository) or ".." in path.parts:
        raise ParentRunVerificationError(f"path escapes repository: {path}")
    for part in (path, *path.parents):
        if part == repository.parent:
            break
        if part.is_symlink():
            raise ParentRunVerificationError(f"symlink is forbidden: {part}")
    return path


def _relative(root: Path, value: Any, repository: Path) -> Path:
    if (not isinstance(value, str) or not value or Path(value).is_absolute()
            or ".." in Path(value).parts or Path(value).as_posix() != value):
        raise ParentRunVerificationError(f"unsafe relative path: {value!r}")
    return _safe(root / value, repository)


def _digest(path: Path, repository: Path) -> dict[str, Any]:
    _safe(path, repository)
    if not path.is_file():
        raise ParentRunVerificationError(f"regular file is missing: {path}")
    before = path.stat()
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    after = path.stat()
    identity = lambda st: (st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns, st.st_ctime_ns)
    if identity(before) != identity(after):
        raise ParentRunVerificationError(f"file changed while hashing: {path}")
    return {"sha256": digest.hexdigest(), "bytes": after.st_size}


def _reject_constant(value: str) -> None:
    raise ParentRunVerificationError(f"nonfinite JSON value: {value}")


def _pairs(values: list[tuple[str, Any]]) -> dict[str, Any]:
    result = {}
    for key, value in values:
        if key in result:
            raise ParentRunVerificationError(f"duplicate JSON field: {key}")
        result[key] = value
    return result


def _decode(payload: str) -> Any:
    try:
        return json.loads(payload, object_pairs_hook=_pairs, parse_constant=_reject_constant)
    except (json.JSONDecodeError, UnicodeError) as exc:
        raise ParentRunVerificationError("invalid parent JSON metadata") from exc


def _json(path: Path, repository: Path, expected: str | None = None) -> dict[str, Any]:
    _safe(path, repository)
    payload = path.read_bytes()
    if expected is not None:
        _same(hashlib.sha256(payload).hexdigest(), expected, f"{path.name} SHA256")
    result = _decode(payload.decode("utf-8"))
    if not isinstance(result, dict):
        raise ParentRunVerificationError(f"JSON mapping required: {path}")
    return result


def _jsonl(path: Path, repository: Path) -> list[dict[str, Any]]:
    _safe(path, repository)
    records = [_decode(line) for line in path.read_text(encoding="utf-8").splitlines()]
    if any(not isinstance(row, dict) for row in records):
        raise ParentRunVerificationError(f"JSONL mappings required: {path}")
    return records


def _condition_key(family: str, severity: int) -> str:
    return f"{family}_S{severity}"


def _expected_files() -> set[str]:
    files = set(ROOT_PAYLOADS)
    for condition in CONDITIONS:
        for method in ("source", "o3", "trained"):
            prefix = f"conditions/{_condition_key(*condition)}/{method}"
            files.add(f"{prefix}/metrics.json")
            files.update(f"{prefix}/{image_id}.png" for image_id in IMAGE_IDS)
    return files


def _file_set(root: Path, repository: Path) -> set[str]:
    found = set()
    for path in root.rglob("*"):
        _safe(path, repository)
        if path.is_dir():
            continue
        if not path.is_file():
            raise ParentRunVerificationError(f"unsupported artifact node: {path}")
        found.add(path.relative_to(root).as_posix())
    return found


def _verify_scope(manifest: dict[str, Any], summary: dict[str, Any], receipt: dict[str, Any]) -> None:
    for key, value in {"protocol_id": PARENT_PROTOCOL, "phase": SCOPE["role"],
                       "image_ids": list(IMAGE_IDS), "fit_gt_access_authorized": True,
                       "formal_test": False, "no_validation_split": True,
                       "paper_result": False, "original_o3_method_label_accesses": 0}.items():
        _same(manifest.get(key), value, f"manifest.{key}")
    config = manifest.get("configuration", {})
    for key, value in {"protocol_id": PARENT_PROTOCOL, "dataset": DATASET, "image_count": 8,
                       "result_root": PARENT_RELATIVE, "seed": 42, "scope": SCOPE,
                       "condition_source": "unchanged_parent_13_conditions"}.items():
        _same(config.get(key), value, f"configuration.{key}")
    for key, value in {"steps": 128, "batch_size": 4,
                       "checkpoint_rule": "fixed_last_step_no_metric_selection"}.items():
        _same(config.get("training", {}).get(key), value, f"configuration.training.{key}")
    for key, value in {"dataset": DATASET, "image_ids": list(IMAGE_IDS), "scope": SCOPE,
                       "samples": 104, "optimizer_steps": 128, "test_payload_opens": 0,
                       "original_o3_method_label_accesses": 0, "host_restored": True,
                       "generalization_claim": False, "automatic_full_training_allowed": False}.items():
        _same(summary.get(key), value, f"summary.{key}")
    for key, value in {"dataset": DATASET, "image_ids": list(IMAGE_IDS), "split_name": "train",
                       "role": "source_supervised_residual_training_train8",
                       "sampling": "first8_in_original_frozen_B4_Pilot64_order",
                       "target_shape": [8, 1, 256, 256], "target_dtype": "float32",
                       "formal_test": False, "paper_result": False, "no_validation_split": True,
                       "supervised_labels_enter_original_o3_update": False,
                       "train_mask_png_decodes": 8, "unique_train_mask_files": 8,
                       "outer_target_loader_calls": 0, "sealed_outer_target_payload_opens": 0,
                       "test_image_decodes": 0, "test_mask_decodes": 0, "test_split_reads": 0,
                       "validation_image_decodes": 0, "validation_mask_decodes": 0,
                       "validation_split_reads": 0, "other_pilot_mask_decodes": 0}.items():
        _same(receipt.get(key), value, f"train_target_receipt.{key}")


def verify_parent_run(
    root: Path = PARENT_ROOT, *, expected_manifest_sha256: str = PARENT_MANIFEST_SHA256,
    expected_ledger_sha256: str = PARENT_LEDGER_SHA256, fixture_repository: Path | None = None,
) -> dict[str, Any]:
    """Verify the canonical v1 run; explicit alternative hashes are fixture-only.

    Production callers supply at most ``root``. A noncanonical synthetic root
    requires an explicit separate ``fixture_repository`` and both noncanonical
    expected hashes; those overrides can never weaken the real parent check.
    """
    root = Path(root)
    if not root.is_absolute():
        root = REPOSITORY / root
    if root == PARENT_ROOT:
        if (fixture_repository is not None or expected_manifest_sha256 != PARENT_MANIFEST_SHA256
                or expected_ledger_sha256 != PARENT_LEDGER_SHA256):
            raise ParentRunVerificationError("canonical parent hash/root overrides are forbidden")
        repository = REPOSITORY
    else:
        if (fixture_repository is None or Path(fixture_repository) == REPOSITORY
                or expected_manifest_sha256 == PARENT_MANIFEST_SHA256
                or expected_ledger_sha256 == PARENT_LEDGER_SHA256):
            raise ParentRunVerificationError("noncanonical parent requires explicit synthetic fixture bindings")
        repository = Path(fixture_repository)
        if not repository.is_absolute() or root != repository / PARENT_RELATIVE:
            raise ParentRunVerificationError("synthetic parent must retain canonical relative layout")
    _safe(root, repository)
    manifest = _json(root / "manifest.json", repository, expected_manifest_sha256)
    ledger = _json(root / "artifact_ledger.json", repository, expected_ledger_sha256)
    complete = _json(root / "COMPLETE.json", repository)
    _same(complete, {"automatic_full_training_allowed": False, "complete": True,
                    "ledger_sha256": expected_ledger_sha256, "manifest_sha256": expected_manifest_sha256,
                    "paper_result": False, "samples": 104, "steps": 128}, "COMPLETE")
    _same(set(ledger), _expected_files(), "parent ledger file roster")
    _same(len(ledger), 366, "parent ledger count")
    expected_files = set(ledger) | {"artifact_ledger.json", "COMPLETE.json"}
    _same(_file_set(root, repository), expected_files, "complete parent file set")
    for relative, descriptor in ledger.items():
        _same(_digest(_relative(root, relative, repository), repository), descriptor,
              f"parent payload {relative}")
    bindings = manifest.get("bindings")
    if not isinstance(bindings, list) or len(bindings) != 87:
        raise ParentRunVerificationError("parent must contain exactly 87 bindings")
    bound = {}
    for item in bindings:
        if not isinstance(item, dict) or set(item) != {"path", "sha256", "bytes"}:
            raise ParentRunVerificationError("parent binding descriptor differs")
        path = _relative(repository, item["path"], repository)
        if item["path"] in bound:
            raise ParentRunVerificationError("duplicate parent binding")
        _same(_digest(path, repository), {"sha256": item["sha256"], "bytes": item["bytes"]},
              f"external binding {item['path']}")
        bound[item["path"]] = item
    summary = _json(root / "summary.json", repository)
    target_receipt = _json(root / "train_target_receipt.json", repository)
    _verify_scope(manifest, summary, target_receipt)
    for name in (CACHE_MANIFEST, TEACHER_MANIFEST):
        if name not in bound:
            raise ParentRunVerificationError(f"required input metadata binding is absent: {name}")
    cache = _json(_relative(repository, CACHE_MANIFEST, repository), repository, bound[CACHE_MANIFEST]["sha256"])
    teacher = _json(_relative(repository, TEACHER_MANIFEST, repository), repository, bound[TEACHER_MANIFEST]["sha256"])
    pilot_ids = teacher.get("image_ids")
    if (not isinstance(pilot_ids, list) or len(pilot_ids) != 64 or len(set(pilot_ids)) != 64
            or pilot_ids[:8] != list(IMAGE_IDS)):
        raise ParentRunVerificationError("fixed train8 is not the original Pilot64 prefix")
    _same(cache.get("dataset"), DATASET, "cache dataset")
    _same(cache.get("image_ids"), pilot_ids, "cache/teacher ID order")
    _same(cache.get("train_split"), TRAIN_SPLIT, "cache train split")
    train_path = _relative(repository, TRAIN_SPLIT, repository)
    train_binding = _digest(train_path, repository)
    _same(train_binding["sha256"], cache.get("train_split_sha256"), "actual train split hash")
    _same(target_receipt.get("train_split_sha256"), train_binding["sha256"], "supervised train split hash")
    train_ids = train_path.read_text(encoding="utf-8").splitlines()
    if len(train_ids) != 663 or len(set(train_ids)) != 663 or not set(pilot_ids).issubset(train_ids):
        raise ParentRunVerificationError("Pilot64 must remain a subset of the 663 official train IDs")
    cells = _jsonl(root / "cells.jsonl", repository)
    _same([(row.get("corruption"), row.get("severity")) for row in cells], list(CONDITIONS), "13 cell order")
    for row in cells:
        _same(row.get("dataset"), DATASET, "cell dataset")
        _same(row.get("condition"), _condition_key(row["corruption"], row["severity"]), "cell condition")
        for method in ("source", "o3", "trained"):
            _same(row.get(method, {}).get("image_count"), 8, f"{method} cell image count")
    episodes = _jsonl(root / "episodes.jsonl", repository)
    _same([(row.get("condition"), row.get("image_id")) for row in episodes],
          [(_condition_key(*condition), image_id) for condition in CONDITIONS for image_id in IMAGE_IDS],
          "104 episode condition/ID order")
    order = _json(root / "training_order.json", repository)
    _same(order.get("sample_order"), "condition_major_then_image_id", "training sample order")
    indices = order.get("indices")
    if (not isinstance(indices, list) or len(indices) != 128
            or any(not isinstance(batch, list) or len(batch) != 4
                   or any(type(i) is not int or not 0 <= i < 104 for i in batch) for batch in indices)):
        raise ParentRunVerificationError("training order must contain the original 128x4 indices")
    training = _jsonl(root / "training.jsonl", repository)
    _same([row.get("step") for row in training], list(range(1, 129)), "128 training steps")
    _same([row.get("sample_indices") for row in training], indices, "executed training order")
    if any(not isinstance(row.get("loss"), (int, float)) or not math.isfinite(row["loss"]) for row in training):
        raise ParentRunVerificationError("nonfinite training loss")
    _same(_file_set(root, repository), expected_files, "parent final file set")
    receipt = {"status": "verified_read_only_parent", "parent_root": PARENT_RELATIVE,
               "parent_manifest_sha256": expected_manifest_sha256,
               "parent_ledger_sha256": expected_ledger_sha256,
               "parent_complete_sha256": _digest(root / "COMPLETE.json", repository)["sha256"],
               "artifact_files_verified": 366, "external_bindings_verified": 87,
               "dataset": DATASET, "image_ids": list(IMAGE_IDS), "condition_count": 13,
               "train_split_sha256": train_binding["sha256"], "samples": 104, "steps": 128,
               "arrays_deserialized": 0, "checkpoints_deserialized": 0, "images_decoded": 0,
               "test_payload_opens": 0, "old_outer_target_payload_opens": 0,
               "no_validation_split": True, "paper_result": False}
    return {"parent_root": root, "parent_manifest": manifest, "parent_summary": summary,
            "parent_cells": cells, "parent_ledger": ledger,
            "parent_ledger_sha256": expected_ledger_sha256, "parent_ledger_sha": expected_ledger_sha256,
            "bindings": bindings, "receipt": receipt}


__all__ = ["ParentRunVerificationError", "verify_parent_run"]
