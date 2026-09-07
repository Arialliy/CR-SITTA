"""Hash-only, append-only contract for the train-side O3 spatial candidate.

This module does not deserialize checkpoints, images, probabilities or targets.
The original B4 helpers are used without monkey-patching their frozen protocol.
In particular, the all-dataset completion barrier is checked before an outer
caller is allowed to deserialize any train targets.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Mapping

import yaml

from scripts import run_p3_stage_b4_full_pilot64_v1 as b4


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs/cr_sitta_o3_spatial_residual_v1.yaml"
FROZEN_CONFIG_SHA256 = "083d0ebc7ed798339a2595bf1cb0a4c22e83f787de11796e22fe7c64e308abdd"
PROTOCOL_ID = "cr-sitta-o3-spatial-residual-v1"
RESULT_ROOT = "results/cr_sitta/o3_spatial_residual_v1"
PARENT_CONFIG = "configs/p3_stage_b4_full_pilot64_proposal_gate_v1.yaml"
PARENT_AGGREGATE = "results/cr_sitta/p3_stage_b4_full_pilot64_proposal_gate_v1/aggregate_phase/R0"
PARENT_MANIFEST_SHA256 = "fd5f3d2df4abed1ee399f3298946a48a85ce00151d4d55b7f32a0d350c5e718c"
PARENT_COMPLETE_SHA256 = "c52ea760401a9bdffee68f6de42b83906208c1ec615bb6fddd9d40bf0b01b30b"
HISTORICAL_EXPORT_ONLY_DRIFT = {
    "tta/adapters/__init__.py": {
        "historical_sha256": "0289b3f17b61da5dc48fc6d92731d2fe6a1dd07b8dcce7181545be56c41f603e",
        "current_sha256": "078ac5abfcf5b77d6254ff8184115beed1fd7438609143ec7c60c68a07d82415",
    },
    "tta/objectives/__init__.py": {
        "historical_sha256": "4f55167b3302528e72a53a757a949bf34dcdd04a70201db8440dde63dc7c1f84",
        "current_sha256": "8f1335dc08765c5863f38a8cd68c432058b3e4deb946f632105ccacbaf360d57",
    },
}
DATASETS = b4.DATASETS
CONDITIONS = b4.CONDITIONS
PILOT_COUNT = 64
EPISODE_COUNT = PILOT_COUNT * len(CONDITIONS)
REQUIRED_NEW_CODE = frozenset((
    "analysis/spatial_residual_contract_v1.py",
    "analysis/evaluate_o3_spatial_residual_v1.py",
    "scripts/run_o3_spatial_residual_v1.py",
    "tta/adapters/decoder_spatial_residual_v1.py",
    "tta/spatial_residual_episode_v1.py",
    "tests/test_spatial_residual_contract_v1.py",
    "tests/test_evaluate_o3_spatial_residual_v1.py",
    "tests/test_o3_spatial_residual_runner_v1.py",
    "tests/test_decoder_spatial_residual_v1.py",
    "tests/test_spatial_residual_episode_v1.py",
))


class SpatialResidualProtocolError(RuntimeError):
    """The candidate's independent preregistration or saved artifact drifted."""


def _fail(message: str) -> None:
    raise SpatialResidualProtocolError(message)


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True,
                       separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")


def _path(relative: str) -> Path:
    if not isinstance(relative, str) or not relative:
        _fail("path must be a nonempty repository-relative string")
    candidate = Path(relative)
    if candidate.is_absolute() or ".." in candidate.parts:
        _fail(f"unsafe repository-relative path: {relative}")
    result = ROOT / candidate
    _reject_symlinks(result)
    return result


def _reject_symlinks(path: Path) -> None:
    if not path.is_absolute() or not path.is_relative_to(ROOT):
        _fail(f"path escapes repository: {path}")
    for part in (path, *path.parents):
        if part == ROOT.parent:
            break
        if part.is_symlink():
            _fail(f"symlink is not allowed: {part}")


def sha256_file(path: Path) -> str:
    _reject_symlinks(path)
    if not path.is_file():
        _fail(f"expected regular file: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(4 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _json(path: Path) -> dict[str, Any]:
    sha256_file(path)  # reject unsafe paths before opening as text
    try:
        result = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise SpatialResidualProtocolError(f"invalid JSON: {path}") from exc
    if not isinstance(result, dict):
        _fail(f"JSON must be a mapping: {path}")
    return result


def _binding(path: Path, expected: str | None = None) -> dict[str, Any]:
    actual = sha256_file(path)
    if expected is not None and actual != expected:
        _fail(f"bound bytes changed: {path}")
    return {"path": path.relative_to(ROOT).as_posix(), "sha256": actual,
            "bytes": path.stat().st_size}


def _write_json(path: Path, value: Any) -> None:
    _reject_symlinks(path)
    with path.open("xb") as stream:
        stream.write(_json_bytes(value))
        stream.flush()
        os.fsync(stream.fileno())


def read_config(config_path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    path = Path(config_path)
    if not path.is_absolute():
        path = ROOT / path
    if path != DEFAULT_CONFIG:
        _fail("only the canonical independent config path is accepted")
    if sha256_file(path) != FROZEN_CONFIG_SHA256:
        _fail("independent config byte hash differs from preregistration")
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        raise SpatialResidualProtocolError("invalid independent YAML") from exc
    if not isinstance(raw, dict):
        _fail("independent YAML must be a mapping")
    _validate_config(raw)
    return raw


def _validate_config(raw: Mapping[str, Any]) -> None:
    if (raw.get("protocol_id") != PROTOCOL_ID or raw.get("result_root") != RESULT_ROOT
            or raw.get("parent_config") != PARENT_CONFIG
            or raw.get("parent_config_sha256") != b4.FROZEN_CONFIG_SHA256
            or tuple(raw.get("datasets", ())) != DATASETS):
        _fail("independent protocol identity, output, parent or datasets drifted")
    aggregate = raw.get("parent_aggregate", {})
    if aggregate != {"path": PARENT_AGGREGATE,
                     "manifest_sha256": PARENT_MANIFEST_SHA256,
                     "complete_sha256": PARENT_COMPLETE_SHA256}:
        _fail("historical aggregate binding drifted")
    if raw.get("historical_export_only_drift") != HISTORICAL_EXPORT_ONLY_DRIFT:
        _fail("historical export-only exception roster or exact hashes drifted")
    scope = raw.get("scope", {})
    expected_scope = {"split_name": "train", "no_validation_split": True,
                      "formal_test_allowed": False, "full_source_training_allowed": False,
                      "new_validation_split": False, "paper_result": False}
    if any(scope.get(key) != value or type(scope.get(key)) is not type(value)
           for key, value in expected_scope.items()):
        _fail("scope must remain train-only, without a validation split or full training")
    conditions = tuple((str(row[0]), int(row[1])) for row in raw.get("ordered_conditions", ()))
    if conditions != CONDITIONS:
        _fail("condition order differs from frozen B4 Pilot64")
    paths = raw.get("implementation_files", ())
    if (not isinstance(paths, list) or len(paths) != len(set(paths))
            or not REQUIRED_NEW_CODE.issubset(paths)):
        _fail("new critical code roster is incomplete or duplicated")
    for path in paths:
        _path(path)


def _validate_historical_code(
    historical: Any, current: Mapping[str, str], raw: Mapping[str, Any]
) -> None:
    """Accept only the two preregistered export-only historical transitions."""
    exceptions = raw.get("historical_export_only_drift")
    if exceptions != HISTORICAL_EXPORT_ONLY_DRIFT:
        _fail("historical critical code exception roster or exact hashes drifted")
    if (not isinstance(historical, dict) or set(historical) != set(current)
            or not set(exceptions).issubset(historical)):
        _fail("historical critical code path roster differs from frozen B4 result")
    for path, actual in current.items():
        if path in exceptions:
            pair = exceptions[path]
            if historical[path] != pair["historical_sha256"] or actual != pair["current_sha256"]:
                _fail(f"historical critical code export transition differs: {path}")
        elif historical[path] != actual:
            _fail(f"historical critical code bytes differ from frozen B4 result: {path}")


def _historical_bindings(parent: Any, raw: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Verify saved B4 metadata/metric bytes; never decode historical arrays."""
    root = _path(raw["parent_aggregate"]["path"])
    bindings = [_binding(parent.config_path, b4.FROZEN_CONFIG_SHA256),
                _binding(root / "manifest.json", PARENT_MANIFEST_SHA256),
                _binding(root / "COMPLETE.json", PARENT_COMPLETE_SHA256)]
    manifest, complete = _json(root / "manifest.json"), _json(root / "COMPLETE.json")
    if (manifest.get("protocol_id") != b4.PROTOCOL_ID or manifest.get("phase") != "aggregate"
            or complete.get("complete") is not True
            or complete.get("manifest_sha256") != PARENT_MANIFEST_SHA256):
        _fail("historical aggregate identity/completion is invalid")
    current_code = b4._capture_code_hashes(parent)
    historical_code = manifest.get("code_sha256")
    _validate_historical_code(historical_code, current_code, raw)
    for name, expected in current_code.items():
        bindings.append(_binding(_path(name), expected))
    ledger = manifest.get("files")
    if not isinstance(ledger, dict) or not ledger:
        _fail("historical aggregate file ledger is missing")
    for relative, item in ledger.items():
        path = _path((root.relative_to(ROOT) / relative).as_posix())
        binding = _binding(path, item["sha256"])
        if binding["bytes"] != item["bytes"]:
            _fail("historical aggregate file size drifted")
        bindings.append(binding)
    parents = manifest.get("parent_artifacts", {})
    if set(parents) != set(DATASETS):
        _fail("historical parent dataset roster differs")
    for dataset in DATASETS:
        record = parents[dataset]
        for phase in ("candidate", "outer"):
            artifact = _path(record[f"{phase}_path"])
            manifest_hash = record[f"{phase}_manifest_sha256"]
            bindings.extend((_binding(artifact / "manifest.json", manifest_hash),
                             _binding(artifact / "COMPLETE.json", record[f"{phase}_complete_sha256"])))
            old = _json(artifact / "manifest.json")
            done = _json(artifact / "COMPLETE.json")
            if (old.get("dataset") != dataset or old.get("phase") != phase
                    or old.get("code_sha256") != historical_code
                    or done.get("complete") is not True
                    or done.get("manifest_sha256") != manifest_hash):
                _fail(f"historical {dataset}/{phase} lineage is invalid")
    return bindings


def _input_bindings(parent: Any, dataset: str) -> tuple[list[str], list[dict[str, Any]]]:
    # This helper hashes the 13 image caches and 26 teacher arrays, but explicitly
    # does not hash or deserialize the outer-target payload.
    b4._verify_consumed_payloads(parent, dataset, include_outer_target=False)
    teacher = b4._teacher_manifest(parent, dataset)
    record = parent.raw["datasets"][dataset]
    cache_root, teacher_root = _path(record["cache_root"]), _path(record["teacher_artifact_root"])
    bindings = [_binding(cache_root / "manifest.json"),
                _binding(cache_root / "COMPLETE.json"),
                _binding(cache_root / "method_input_manifest.json"),
                _binding(_path(record["checkpoint_path"]), record["checkpoint_sha256"]),
                _binding(teacher_root / "manifest.json", record["teacher_manifest_sha256"]),
                _binding(teacher_root / "COMPLETE.json", record["teacher_complete_sha256"])]
    image_ids = list(teacher["image_ids"])
    if len(image_ids) != PILOT_COUNT or len(set(image_ids)) != PILOT_COUNT:
        _fail("teacher image IDs must be the ordered fixed 64")
    for image_id in image_ids:
        if not isinstance(image_id, str) or Path(image_id).name != image_id or image_id in ("", ".", ".."):
            _fail("unsafe Pilot64 image ID")
    return image_ids, bindings


def prepare_run(config_path: Path = DEFAULT_CONFIG, dataset: str = "") -> dict[str, Any]:
    """Read metadata and hash bytes only; do not create an experiment directory."""
    if dataset not in DATASETS:
        _fail(f"unsupported train-side dataset: {dataset}")
    raw = read_config(config_path)
    parent = b4.load_contract(_path(raw["parent_config"]))
    bindings = [_binding(DEFAULT_CONFIG, FROZEN_CONFIG_SHA256), _binding(_path(raw["addendum"]))]
    bindings.extend(_historical_bindings(parent, raw))
    image_ids, inputs = _input_bindings(parent, dataset)
    bindings.extend(inputs)
    for path in raw["implementation_files"]:
        bindings.append(_binding(_path(path)))
    unique: dict[str, dict[str, Any]] = {}
    for binding in bindings:
        if binding["path"] in unique and unique[binding["path"]] != binding:
            _fail("binding changed while preparing")
        unique[binding["path"]] = binding
    bindings = [unique[key] for key in sorted(unique)]
    contract = {"schema_version": 1, "protocol_id": PROTOCOL_ID, "phase": "candidate",
                "dataset": dataset, "config_sha256": FROZEN_CONFIG_SHA256,
                "image_ids": image_ids, "image_count_per_condition": PILOT_COUNT,
                "condition_count": len(CONDITIONS), "episode_count": EPISODE_COUNT,
                "ordered_conditions": [list(condition) for condition in CONDITIONS],
                "bindings": bindings, "scope": dict(raw["scope"]),
                "method_label_accesses": 0, "paper_result": False}
    preflight = {"protocol_id": PROTOCOL_ID, "dataset": dataset,
                 "status": "metadata_and_byte_hashes_verified",
                 "images_deserialized": 0, "checkpoints_deserialized": 0,
                 "probabilities_deserialized": 0, "outer_target_loader_calls": 0,
                 "test_payload_opens": 0, "validation_payload_opens": 0,
                 "no_validation_split": True, "paper_result": False}
    return {"new_config": raw, "parent_contract": parent,
            "output": _path(RESULT_ROOT) / "candidate" / dataset,
            "image_ids": image_ids, "bindings": bindings,
            "contract": contract, "preflight": preflight}


def _assert_prepared_unchanged(prepared: Mapping[str, Any]) -> None:
    raw = read_config(DEFAULT_CONFIG)
    if raw != prepared["new_config"]:
        _fail("prepared config changed")
    expected_output = _path(RESULT_ROOT) / "candidate" / prepared["contract"]["dataset"]
    if prepared["output"] != expected_output:
        _fail("prepared output path changed")
    for binding in prepared["bindings"]:
        if _binding(_path(binding["path"]), binding["sha256"]) != binding:
            _fail("prepared binding changed")
    b4._verify_consumed_payloads(prepared["parent_contract"], prepared["contract"]["dataset"],
                                 include_outer_target=False)


def freeze_run(prepared: Mapping[str, Any]) -> dict[str, Any]:
    """Create a non-replaceable preregistration before model/input decoding."""
    _assert_prepared_unchanged(prepared)
    output = prepared["output"]
    _reject_symlinks(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.mkdir(exist_ok=False)
    _write_json(output / "preflight.json", prepared["preflight"])
    _write_json(output / "manifest.json", prepared["contract"])
    return dict(prepared["contract"])


def _validate_summary(summary: Mapping[str, Any], manifest: Mapping[str, Any]) -> None:
    expected = {"phase": "candidate", "dataset": manifest["dataset"],
                "episode_count": EPISODE_COUNT, "condition_count": len(CONDITIONS),
                "image_count_per_condition": PILOT_COUNT, "image_ids": manifest["image_ids"],
                "method_label_accesses": 0, "outer_target_loader_calls": 0,
                "test_payload_opens": 0, "validation_payload_opens": 0,
                "source_state_restored": True, "predictions_complete": True,
                "no_validation_split": True, "paper_result": False}
    for key, value in expected.items():
        if summary.get(key) != value or type(summary.get(key)) is not type(value):
            _fail(f"candidate summary field is invalid: {key}")


def _file_ledger(root: Path, excluded: frozenset[str] = frozenset()) -> dict[str, Any]:
    ledger = {}
    _reject_symlinks(root)
    for path in sorted(root.rglob("*")):
        _reject_symlinks(path)
        if path.is_dir():
            continue
        relative = path.relative_to(root).as_posix()
        if relative in excluded:
            continue
        binding = _binding(path)
        ledger[relative] = {"sha256": binding["sha256"], "bytes": binding["bytes"]}
    return ledger


def _validate_npy_header(path: Path, expected_shape: tuple[int, ...], expected_dtype: str) -> None:
    """Read only the NPY header, not probability/gradient payload values."""
    from numpy.lib import format as npy_format
    _reject_symlinks(path)
    if not path.is_file():
        _fail(f"missing candidate array: {path}")
    try:
        with path.open("rb") as stream:
            version = npy_format.read_magic(stream)
            if version == (1, 0):
                shape, fortran, dtype = npy_format.read_array_header_1_0(stream)
            elif version == (2, 0):
                shape, fortran, dtype = npy_format.read_array_header_2_0(stream)
            else:
                _fail(f"unsupported candidate NPY header: {version}")
            expected_bytes = stream.tell() + math.prod(expected_shape) * dtype.itemsize
        if (shape != expected_shape or fortran or str(dtype) != expected_dtype
                or dtype.hasobject or path.stat().st_size != expected_bytes):
            _fail(f"candidate array header/length differs: {path}")
    except (OSError, ValueError, EOFError) as exc:
        raise SpatialResidualProtocolError(f"invalid candidate array header: {path}") from exc


def _validate_payload_structure(output: Path, manifest: Mapping[str, Any]) -> None:
    """Require all predictions/receipts; inspect metadata only, never np.load."""
    _json(output / "runtime.json")
    image_ids = manifest["image_ids"]
    for corruption, severity in CONDITIONS:
        condition = b4._condition_key(corruption, severity)
        root = output / "conditions" / condition
        for name, shape, dtype in (
            ("source_probabilities.npy", (PILOT_COUNT, 1, 256, 256), "float32"),
            ("post_probabilities.npy", (PILOT_COUNT, 1, 256, 256), "float32"),
            ("proxy_gradients.npy", (PILOT_COUNT, 144), "float64"),
            ("proposal_directions.npy", (PILOT_COUNT, 144), "float64"),
            ("endpoint_kernels.npy", (PILOT_COUNT, 16, 1, 3, 3), "float32"),
        ):
            _validate_npy_header(root / name, shape, dtype)
        episode_path = root / "episodes.jsonl"
        sha256_file(episode_path)
        try:
            with episode_path.open("r", encoding="utf-8") as stream:
                episodes = [json.loads(line) for line in stream]
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise SpatialResidualProtocolError(f"invalid episode ledger: {episode_path}") from exc
        if len(episodes) != PILOT_COUNT:
            _fail(f"episode ledger is not the complete ordered Pilot64: {condition}")
        for index, (image_id, episode) in enumerate(zip(image_ids, episodes, strict=True)):
            expected = {"dataset": manifest["dataset"], "condition": condition,
                        "image_id": image_id, "image_index": index,
                        "method_label_accesses": 0, "episode_reset_exact": True}
            if not isinstance(episode, dict) or any(
                episode.get(key) != value or type(episode.get(key)) is not type(value)
                for key, value in expected.items()
            ):
                _fail(f"episode identity/order or isolation differs: {condition}/{index}")
        masks = root / "masks"
        _reject_symlinks(masks)
        names = {path.name for path in masks.iterdir()} if masks.is_dir() else set()
        if names != {f"{image_id}.png" for image_id in image_ids}:
            _fail(f"predicted mask roster differs from ordered Pilot64: {condition}")
        for name in names:
            path = masks / name
            _reject_symlinks(path)
            if not path.is_file():
                _fail(f"predicted mask is not a regular file: {path}")
            with path.open("rb") as stream:
                if stream.read(8) != b"\x89PNG\r\n\x1a\n":
                    _fail(f"predicted mask is not PNG: {path}")


def complete_run(prepared: Mapping[str, Any], summary: Mapping[str, Any]) -> dict[str, Any]:
    """Append completion only after the runner's full candidate payload exists."""
    _assert_prepared_unchanged(prepared)
    output = prepared["output"]
    manifest = _json(output / "manifest.json")
    if manifest != prepared["contract"] or _json(output / "preflight.json") != prepared["preflight"]:
        _fail("frozen candidate preregistration changed")
    _validate_summary(summary, manifest)
    _validate_payload_structure(output, manifest)
    if (output / "COMPLETE.json").exists() or (output / "summary.json").exists():
        _fail("candidate completion cannot overwrite existing artifacts")
    _write_json(output / "summary.json", dict(summary))
    ledger = _file_ledger(output, frozenset(("COMPLETE.json",)))
    completion = {"schema_version": 1, "protocol_id": PROTOCOL_ID, "phase": "candidate",
                  "dataset": manifest["dataset"], "complete": True,
                  "manifest_sha256": sha256_file(output / "manifest.json"),
                  "files": ledger, "payload_tree_sha256": hashlib.sha256(_json_bytes(ledger)).hexdigest(),
                  "no_validation_split": True, "paper_result": False}
    _write_json(output / "COMPLETE.json", completion)
    return completion


def verify_candidate(config_path: Path = DEFAULT_CONFIG, dataset: str = "") -> dict[str, Any]:
    """Verify one saved candidate using bytes/metadata only, never target arrays."""
    prepared = prepare_run(config_path, dataset)
    output = prepared["output"]
    manifest = _json(output / "manifest.json")
    complete = _json(output / "COMPLETE.json")
    summary = _json(output / "summary.json")
    if manifest != prepared["contract"] or _json(output / "preflight.json") != prepared["preflight"]:
        _fail("candidate binding/preregistration differs from current inputs")
    if (complete.get("protocol_id") != PROTOCOL_ID or complete.get("phase") != "candidate"
            or complete.get("dataset") != dataset or complete.get("complete") is not True
            or complete.get("manifest_sha256") != sha256_file(output / "manifest.json")
            or complete.get("paper_result") is not False or complete.get("no_validation_split") is not True):
        _fail("candidate completion identity is invalid")
    ledger = _file_ledger(output, frozenset(("COMPLETE.json",)))
    if (ledger != complete.get("files")
            or complete.get("payload_tree_sha256") != hashlib.sha256(_json_bytes(ledger)).hexdigest()):
        _fail("candidate payload bytes/file roster changed after completion")
    _validate_summary(summary, manifest)
    _validate_payload_structure(output, manifest)
    return {**summary, "summary": summary, "manifest": manifest, "complete": complete,
            "boundfiles": ledger, "bound_files": ledger, "bindings": ledger, "output": output,
            "parent_contract": prepared["parent_contract"], "new_config": prepared["new_config"]}


def verify_all_candidates(config_path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    """Global all-three completion barrier; no target loader is imported/called."""
    raw = read_config(config_path)
    for dataset in raw["datasets"]:
        path = _path(RESULT_ROOT) / "candidate" / dataset / "COMPLETE.json"
        if path.is_symlink() or not path.is_file():
            _fail(f"all-candidate barrier: missing verified completion for {dataset}")
    return {dataset: verify_candidate(config_path, dataset) for dataset in raw["datasets"]}
