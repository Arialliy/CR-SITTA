from __future__ import annotations

import argparse
import builtins
import fcntl
import json
import os
from pathlib import Path
import sys

import pytest
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
import materialize_binary_tent_ss_calibration_cache_v2 as cache  # noqa: E402


PROTOCOL = PROJECT_ROOT / "configs/binary_tent_ss_calibration_cache_v2.yaml"
FORMAL_OUTPUT = PROJECT_ROOT / "results/binary_tent/ss_calibration_cache_v2"


def _formal_output_metadata_inventory() -> tuple[tuple[object, ...], ...]:
    """Snapshot enough filesystem identity to prove tests did not publish or edit it."""

    if not FORMAL_OUTPUT.exists():
        return ()
    entries: list[tuple[object, ...]] = []
    for path in sorted((FORMAL_OUTPUT, *FORMAL_OUTPUT.rglob("*"))):
        stat_result = path.lstat()
        entries.append(
            (
                path.relative_to(FORMAL_OUTPUT).as_posix(),
                stat_result.st_mode,
                stat_result.st_ino,
                stat_result.st_size,
                stat_result.st_mtime_ns,
            )
        )
    return tuple(entries)


FORMAL_OUTPUT_METADATA_AT_IMPORT = _formal_output_metadata_inventory()


def _context(dataset: str = "IRSTD-1K") -> dict[str, object]:
    return cache.validate_contract(PROJECT_ROOT, PROTOCOL, dataset)


def _variant(tmp_path: Path, mutate) -> Path:
    value = yaml.safe_load(PROTOCOL.read_text(encoding="utf-8"))
    mutate(value)
    path = tmp_path / "protocol.yaml"
    path.write_text(yaml.safe_dump(value, sort_keys=False), encoding="utf-8")
    return path


def _toy_manifest(context: dict[str, object], staging: Path) -> dict[str, object]:
    conditions_dir = staging / "conditions"
    targets_dir = staging / "outer_evaluator"
    conditions_dir.mkdir()
    targets_dir.mkdir()
    conditions = []
    files = {}
    for index, (corruption, severity) in enumerate(cache.CONDITIONS):
        key = cache._condition_key(corruption, severity)
        relative = f"conditions/{key}.npy"
        path = staging / relative
        path.write_bytes(f"toy-{key}\n".encode())
        digest = cache.secure_io.sha256_file(path)
        files[relative] = {"sha256": digest, "bytes": path.stat().st_size}
        conditions.append(
            {
                "index": index,
                "key": key,
                "corruption": corruption,
                "severity": severity,
                "path": relative,
                "shape": [64, 3, 256, 256],
                "dtype": "little_endian_float32",
                "file_sha256": digest,
                "tensor_sequence_sha256": "1" * 64,
            }
        )
    target = targets_dir / "targets.npy"
    target.write_bytes(b"toy-targets\n")
    target_digest = cache.secure_io.sha256_file(target)
    files["outer_evaluator/targets.npy"] = {
        "sha256": target_digest,
        "bytes": target.stat().st_size,
    }
    manifest = {
        "schema_version": 2,
        "cache_format": cache.CACHE_FORMAT,
        "protocol_id": cache.PROTOCOL_ID,
        "protocol_sha256": context["protocol_sha256"],
        "dataset": context["dataset"],
        "image_ids": list(context["selected_ids"]),
        "ordered_ids_sha256": context["ordered_ids_sha256"],
        "original_sizes": [[256, 256]] * 64,
        "seed": 42,
        "conditions": conditions,
        "targets": {
            "path": "outer_evaluator/targets.npy",
            "role": "outer_evaluator_only",
            "method_facing_access": "forbidden",
            "file_sha256": target_digest,
            "file_bytes": target.stat().st_size,
            "shape": [64, 1, 256, 256],
            "dtype": "little_endian_float32",
            "tensor_sequence_sha256": "3" * 64,
        },
        "files": files,
        "lineage_files_sha256": dict(context["lineage_hashes"]),
        "label_firewall": {
            "method_received_labels": False,
            "targets_for_outer_evaluator_only": True,
        },
        "cache_content_sha256": "",
    }
    for record in manifest["conditions"]:
        record["file_bytes"] = files[record["path"]]["bytes"]
    manifest["cache_content_sha256"] = cache._cache_content_sha256(manifest)
    return manifest


def test_protocol_freezes_independent_train_side_lineage_and_13_conditions() -> None:
    protocol, digest = cache.load_protocol(PROTOCOL)

    assert len(cache._parse_conditions(protocol)) == 13
    assert protocol["input_protocol"]["seed"] == 42
    assert protocol["lineage"]["train_side_pilot_protocol"]["sha256"] == cache.PILOT_PROTOCOL_SHA256
    assert protocol["lineage"]["train_side_pilot_manifest"]["sha256"] == cache.PILOT_MANIFEST_SHA256
    assert protocol["lineage"]["frozen_severity_table"]["sha256"] == cache.FROZEN_SEVERITY_SHA256
    runtime_seal = protocol["lineage"]["runtime_metadata_seal"]
    assert runtime_seal["materializer"]["sha256"] == cache.secure_io.sha256_file(
        PROJECT_ROOT / runtime_seal["materializer"]["path"]
    )
    assert runtime_seal["train_side_pilot_protocol_implementation"]["sha256"] == (
        cache.secure_io.sha256_file(
            PROJECT_ROOT
            / runtime_seal["train_side_pilot_protocol_implementation"]["path"]
        )
    )
    assert protocol["cache"]["root"] == "results/binary_tent/ss_calibration_cache_v2"
    assert "round_02" not in json.dumps(protocol)
    assert len(digest) == 64


@pytest.mark.parametrize("dataset", cache.DATASET_NAMES)
def test_real_validate_only_contract_uses_new_64_and_no_test_pixels(dataset: str) -> None:
    context = _context(dataset)

    assert len(context["selected_ids"]) == 64
    assert len(context["conditions"]) == 13
    assert context["validation"] == {
        "metadata_only": True,
        "numpy_load_calls": 0,
        "torch_imports": 0,
        "pil_imported": False,
        "model_constructed": False,
        "cuda_calls": 0,
        "output_pixels_opened": 0,
        "test_images_opened": 0,
        "test_masks_opened": 0,
        "selected_ids_contained_in_train": True,
        "selected_ids_absent_from_test": True,
        "old_round_02_cache_read": False,
    }


def test_validate_only_never_imports_pixel_model_or_cuda_packages(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_import = builtins.__import__

    def guarded_import(name: str, *args, **kwargs):
        if name.split(".")[0] in {"numpy", "PIL", "torch", "scipy"}:
            raise AssertionError(f"validate-only imported forbidden package: {name}")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    args = argparse.Namespace(
        dataset="IRSTD-1K", protocol=PROTOCOL, validate_only=True, recover_stale=False
    )
    result = cache.run(args)

    assert result["validate_only"] is True
    assert result["validation"]["numpy_load_calls"] == 0


def test_validate_only_reads_only_metadata_and_opaque_lineage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: list[Path] = []
    real_snapshot = cache.secure_io._read_stable_bytes

    def observed_snapshot(path: Path, *, label: str):
        observed.append(path)
        return real_snapshot(path, label=label)

    monkeypatch.setattr(cache.secure_io, "_read_stable_bytes", observed_snapshot)
    _context()

    assert observed
    assert {path.suffix for path in observed} <= {".yaml", ".json", ".txt", ".py"}
    assert not any("source_calibration_cache_v1" in path.as_posix() for path in observed)
    assert not any(path.suffix.lower() in {".png", ".jpg", ".npy"} for path in observed)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda value: value["lineage"]["train_side_pilot_protocol"].__setitem__(
                "sha256", "0" * 64
            ),
            "train_side_pilot_protocol.sha256",
        ),
        (
            lambda value: value["lineage"]["train_side_pilot_manifest"].__setitem__(
                "sha256", "0" * 64
            ),
            "train_side_pilot_manifest.sha256",
        ),
        (
            lambda value: value["datasets"]["IRSTD-1K"].__setitem__(
                "calibration_ids_file_sha256", "0" * 64
            ),
            "Pilot output file SHA256",
        ),
    ],
)
def test_bound_lineage_and_id_hash_drift_fail_closed(
    monkeypatch: pytest.MonkeyPatch, mutation, message: str
) -> None:
    value = yaml.safe_load(PROTOCOL.read_text(encoding="utf-8"))
    mutation(value)
    digest = cache.secure_io.sha256_file(PROTOCOL)
    monkeypatch.setattr(
        cache,
        "load_protocol",
        lambda _path, *, repository=PROJECT_ROOT: (value, digest),
    )
    with pytest.raises(ValueError, match=message):
        cache.validate_contract(PROJECT_ROOT, PROTOCOL, "IRSTD-1K")


def test_protocol_actual_path_must_equal_repo_declared_path(tmp_path: Path) -> None:
    alternate = _variant(tmp_path, lambda _value: None)
    with pytest.raises(ValueError, match="actual/declared"):
        cache.load_protocol(alternate)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda value: value.__setitem__("scope", "validation"), "manifest scope"),
        (
            lambda value: value.__setitem__("no_validation_split", False),
            "no_validation_split",
        ),
        (
            lambda value: value["datasets"]["IRSTD-1K"]["checks"].__setitem__(
                "parent_pilot_output_overlap_count", 1
            ),
            "parent_pilot_output_overlap_count",
        ),
        (
            lambda value: value["datasets"]["IRSTD-1K"]["checks"].__setitem__(
                "output_test_overlap_count", 1
            ),
            "output_test_overlap_count",
        ),
        (
            lambda value: value["datasets"]["IRSTD-1K"]["output"].__setitem__(
                "path", "configs/wrong-parent/IRSTD-1K.txt"
            ),
            "output path",
        ),
    ],
)
def test_pilot_manifest_scope_and_zero_overlap_semantics_fail_closed(
    mutation, message: str
) -> None:
    protocol, _ = cache.load_protocol(PROTOCOL)
    manifest = json.loads(
        (PROJECT_ROOT / "configs/tta_train_side_pilot_v2/manifest.json").read_text(
            encoding="utf-8"
        )
    )
    mutation(manifest)
    with pytest.raises((TypeError, ValueError), match=message):
        cache._validate_pilot_manifest(manifest, protocol)


def test_validate_only_does_not_create_output(tmp_path: Path) -> None:
    output = tmp_path / "cache" / "IRSTD-1K"
    context = cache.validate_contract(
        PROJECT_ROOT, PROTOCOL, "IRSTD-1K", output_override=output
    )

    assert context["final_output"] == output
    assert not output.exists()
    assert not output.parent.exists()


def test_atomic_materialize_writes_complete_firewalled_contract_and_no_overwrite(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "cache" / "IRSTD-1K"
    context = cache.validate_contract(
        PROJECT_ROOT, PROTOCOL, "IRSTD-1K", output_override=output
    )
    monkeypatch.setattr(cache, "_materialize_staging", _toy_manifest)

    complete = cache.materialize(context)

    assert complete["complete"] is True
    assert complete["method_received_labels"] is False
    method = json.loads((output / "method_input_manifest.json").read_text())
    assert method["targets_exposed"] is False
    assert not set(method["sample_fields"]) & cache.FORBIDDEN_METHOD_FIELDS
    assert len(method["conditions"]) == 13
    assert set(method["files"]) == {
        f"conditions/{cache._condition_key(*condition)}.npy"
        for condition in cache.CONDITIONS
    }
    assert all(
        set(record) == {"sha256", "bytes"}
        and len(record["sha256"]) == 64
        and record["bytes"] > 0
        for record in method["files"].values()
    )
    assert "outer_evaluator/targets.npy" not in method["files"]
    outer = json.loads((output / "manifest.json").read_text())
    assert outer["targets"]["method_facing_access"] == "forbidden"
    assert (output / "COMPLETE.json").is_file()
    post = cache.validate_materialization(context)
    assert post["full_file_hashes_verified"] is True
    assert post["verified_payload_count"] == 14
    assert not list(output.parent.glob(".IRSTD-1K.build-*"))
    with pytest.raises(FileExistsError, match="refusing overwrite"):
        cache.materialize(context)


def test_official_consumers_enforce_runtime_label_firewall(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "cache" / "IRSTD-1K"
    context = cache.validate_contract(
        PROJECT_ROOT, PROTOCOL, "IRSTD-1K", output_override=output
    )
    monkeypatch.setattr(cache, "_materialize_staging", _toy_manifest)
    cache.materialize(context)

    class FakeArray:
        def __init__(self, shape):
            self.shape = shape

        def __getitem__(self, _index):
            import numpy as np

            return np.zeros(self.shape[1:], dtype="<f4")

    monkeypatch.setattr(
        cache,
        "_numpy_load",
        lambda path, **_kwargs: FakeArray(
            (64, 1, 256, 256)
            if path.name == "targets.npy"
            else (64, 3, 256, 256)
        ),
    )
    monkeypatch.setattr(
        cache,
        "_validate_numpy_mmap",
        lambda value, *, expected_shape, label: cache._equal(
            tuple(value.shape), expected_shape, label
        ),
    )
    method = cache.SourceCalibrationMethodInputDatasetV2(
        output,
        condition_key="clean_S0",
        expected_protocol_sha256=context["protocol_sha256"],
    )
    sample = method[0]

    assert tuple(sample) == cache.METHOD_FIELDS
    assert not set(sample) & cache.FORBIDDEN_METHOD_FIELDS
    import torch

    assert isinstance(sample["image"], torch.Tensor)
    assert sample["image"].dtype == torch.float32
    assert tuple(sample["image"].shape) == (3, 256, 256)
    with pytest.raises(PermissionError, match="completed method episodes"):
        cache.load_outer_evaluator_targets_v2(
            output,
            expected_protocol_sha256=context["protocol_sha256"],
            episodes_complete=False,
        )
    targets = cache.load_outer_evaluator_targets_v2(
        output,
        expected_protocol_sha256=context["protocol_sha256"],
        episodes_complete=True,
    )
    assert targets.shape == (64, 1, 256, 256)


@pytest.mark.parametrize(
    ("relative", "consumer"),
    [
        (
            "conditions/clean_S0.npy",
            lambda output, protocol_sha: cache.SourceCalibrationMethodInputDatasetV2(
                output,
                condition_key="clean_S0",
                expected_protocol_sha256=protocol_sha,
            ),
        ),
        (
            "outer_evaluator/targets.npy",
            lambda output, protocol_sha: cache.load_outer_evaluator_targets_v2(
                output,
                expected_protocol_sha256=protocol_sha,
                episodes_complete=True,
            ),
        ),
    ],
)
def test_consumers_hash_and_size_verify_payload_before_numpy_load(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    relative: str,
    consumer,
) -> None:
    output = tmp_path / "cache" / "IRSTD-1K"
    context = cache.validate_contract(
        PROJECT_ROOT, PROTOCOL, "IRSTD-1K", output_override=output
    )
    monkeypatch.setattr(cache, "_materialize_staging", _toy_manifest)
    cache.materialize(context)
    path = output / relative
    path.write_bytes(path.read_bytes() + b"tamper")
    calls = 0

    def forbidden_numpy_load(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        raise AssertionError("np.load ran before payload verification")

    monkeypatch.setattr(cache, "_numpy_load", forbidden_numpy_load)
    with pytest.raises(ValueError, match="byte size|SHA256"):
        consumer(output, context["protocol_sha256"])
    assert calls == 0


def test_consumer_rejects_symlink_payload_before_numpy_load(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "cache" / "IRSTD-1K"
    context = cache.validate_contract(
        PROJECT_ROOT, PROTOCOL, "IRSTD-1K", output_override=output
    )
    monkeypatch.setattr(cache, "_materialize_staging", _toy_manifest)
    cache.materialize(context)
    clean = output / "conditions/clean_S0.npy"
    target = output / "conditions/gaussian_noise_S1.npy"
    clean.unlink()
    clean.symlink_to(target)
    monkeypatch.setattr(
        cache,
        "_numpy_load",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("np.load followed a symlink")
        ),
    )
    with pytest.raises((OSError, ValueError), match="symlink|component"):
        cache.SourceCalibrationMethodInputDatasetV2(
            output,
            condition_key="clean_S0",
            expected_protocol_sha256=context["protocol_sha256"],
        )


def test_numpy_payload_schema_requires_read_only_c_little_f4_and_finite(
    tmp_path: Path,
) -> None:
    import numpy as np

    path = tmp_path / "valid.npy"
    writable = np.lib.format.open_memmap(
        path, mode="w+", dtype="<f4", shape=(2, 3), fortran_order=False
    )
    writable[:] = 1.0
    writable.flush()
    del writable
    value = np.load(path, mmap_mode="r", allow_pickle=False)
    cache._validate_numpy_mmap(value, expected_shape=(2, 3), label="valid")
    assert value.dtype.str == "<f4"
    assert value.flags.c_contiguous and not value.flags.writeable

    nonfinite_path = tmp_path / "nonfinite.npy"
    nonfinite = np.lib.format.open_memmap(
        nonfinite_path, mode="w+", dtype="<f4", shape=(2, 3)
    )
    nonfinite[:] = 0.0
    nonfinite[1, 1] = np.nan
    nonfinite.flush()
    del nonfinite
    with pytest.raises(ValueError, match="NaN/Inf"):
        cache._validate_numpy_mmap(
            np.load(nonfinite_path, mmap_mode="r", allow_pickle=False),
            expected_shape=(2, 3),
            label="nonfinite",
        )

    writable_value = np.load(path, mmap_mode="r+", allow_pickle=False)
    with pytest.raises(ValueError, match="read-only"):
        cache._validate_numpy_mmap(
            writable_value, expected_shape=(2, 3), label="writable"
        )


def test_materialization_failure_cleans_current_staging_and_lock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "cache" / "IRSTD-1K"
    context = cache.validate_contract(
        PROJECT_ROOT, PROTOCOL, "IRSTD-1K", output_override=output
    )

    def fail(_context, staging: Path):
        (staging / "partial.bin").write_bytes(b"partial")
        raise RuntimeError("injected crash")

    monkeypatch.setattr(cache, "_materialize_staging", fail)
    with pytest.raises(RuntimeError, match="injected"):
        cache.materialize(context)

    assert not output.exists()
    assert not list(output.parent.glob(".IRSTD-1K.build-*"))
    assert not (output.parent / ".IRSTD-1K.publish.lock").exists()


def test_materializer_holds_flock_and_stable_inode_through_staging(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "cache" / "IRSTD-1K"
    context = cache.validate_contract(
        PROJECT_ROOT, PROTOCOL, "IRSTD-1K", output_override=output
    )
    observed = False

    def inspect_lock(lock_context, staging: Path):
        nonlocal observed
        lock = output.parent / ".IRSTD-1K.publish.lock"
        first = lock.stat()
        competitor = os.open(lock, os.O_RDONLY | os.O_CLOEXEC)
        try:
            with pytest.raises(BlockingIOError):
                fcntl.flock(competitor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        finally:
            os.close(competitor)
        second = lock.stat()
        assert (first.st_dev, first.st_ino) == (second.st_dev, second.st_ino)
        observed = True
        return _toy_manifest(lock_context, staging)

    monkeypatch.setattr(cache, "_materialize_staging", inspect_lock)
    cache.materialize(context)

    assert observed is True
    assert not (output.parent / ".IRSTD-1K.publish.lock").exists()


def test_metadata_drift_during_materialization_refuses_publish_and_cleans(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "cache" / "IRSTD-1K"
    context = cache.validate_contract(
        PROJECT_ROOT, PROTOCOL, "IRSTD-1K", output_override=output
    )
    real_validate = cache.validate_contract
    calls = 0

    def drifting_validate(*args, **kwargs):
        nonlocal calls
        calls += 1
        fresh = real_validate(*args, **kwargs)
        if calls >= 2:
            fresh["lineage_hashes"] = {
                **fresh["lineage_hashes"],
                "materialize_binary_tent_ss_calibration_cache_v2.py": "0" * 64,
            }
        return fresh

    monkeypatch.setattr(cache, "validate_contract", drifting_validate)
    monkeypatch.setattr(cache, "_materialize_staging", _toy_manifest)
    with pytest.raises(RuntimeError, match="runtime seal changed"):
        cache.materialize(context)

    assert not output.exists()
    assert not list(output.parent.glob(".IRSTD-1K.build-*"))
    assert not (output.parent / ".IRSTD-1K.publish.lock").exists()


def test_explicit_crash_recovery_removes_only_owned_staging_and_lock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "cache" / "IRSTD-1K"
    output.parent.mkdir(parents=True)
    stale_pid = 999999
    stale = output.parent / f".IRSTD-1K.build-{stale_pid}-dead"
    stale.mkdir()
    (stale / "partial").write_text("partial", encoding="utf-8")
    lock = output.parent / ".IRSTD-1K.publish.lock"
    lock.write_text(f"pid={stale_pid}\n", encoding="utf-8")
    unrelated = output.parent / "keep.txt"
    unrelated.write_text("keep\n", encoding="utf-8")
    context = {"final_output": output}
    monkeypatch.setattr(cache, "_pid_is_alive", lambda pid: False)

    report = cache.recover_stale(context)

    assert report == {"removed_staging": 1, "removed_lock": 1}
    assert unrelated.read_text() == "keep\n"
    assert not stale.exists()
    assert not lock.exists()


def test_recovery_rejects_live_pid_and_preserves_entries(tmp_path: Path) -> None:
    output = tmp_path / "cache" / "IRSTD-1K"
    output.parent.mkdir(parents=True)
    staging = output.parent / f".IRSTD-1K.build-{os.getpid()}-live"
    staging.mkdir()
    lock = output.parent / ".IRSTD-1K.publish.lock"
    lock.write_text(f"pid={os.getpid()}\n", encoding="ascii")

    with pytest.raises(RuntimeError, match="PID .* live"):
        cache.recover_stale({"final_output": output})

    assert staging.is_dir()
    assert lock.is_file()


def test_recovery_rejects_held_flock_even_for_declared_dead_pid(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "cache" / "IRSTD-1K"
    output.parent.mkdir(parents=True)
    lock = output.parent / ".IRSTD-1K.publish.lock"
    lock.write_text("pid=999999\n", encoding="ascii")
    descriptor = os.open(lock, os.O_RDWR | os.O_CLOEXEC)
    fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
    monkeypatch.setattr(cache, "_pid_is_alive", lambda _pid: False)
    try:
        with pytest.raises(RuntimeError, match="live/held"):
            cache.recover_stale({"final_output": output})
    finally:
        os.close(descriptor)
    assert lock.is_file()


def test_recovery_rejects_symlink_lock(tmp_path: Path) -> None:
    output = tmp_path / "cache" / "IRSTD-1K"
    output.parent.mkdir(parents=True)
    target = output.parent / "real.lock"
    target.write_text("pid=999999\n", encoding="ascii")
    lock = output.parent / ".IRSTD-1K.publish.lock"
    lock.symlink_to(target)

    with pytest.raises((OSError, ValueError), match="symlink|component"):
        cache.recover_stale({"final_output": output})


def test_tests_did_not_publish_or_modify_formal_output() -> None:
    assert _formal_output_metadata_inventory() == FORMAL_OUTPUT_METADATA_AT_IMPORT
