from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import stat
from typing import Any

import pytest

from benchmark import implementation_dependency_seal as dependency_seal


LOCAL_ARTIFACT_TEST_ENV = "NS_FPN_RUN_LOCAL_ARTIFACT_TESTS"
LOCAL_ARTIFACT_TESTS_ENABLED = os.environ.get(LOCAL_ARTIFACT_TEST_ENV) == "1"


def _dependency_path(root: Path, relative: str) -> Path:
    return root.joinpath(*PurePosixPath(relative).parts)


def _materialize_dependency_tree(root: Path) -> None:
    root.mkdir()
    for index, relative in enumerate(
        dependency_seal.IMPLEMENTATION_DEPENDENCY_PATHS
    ):
        path = _dependency_path(root, relative)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f"dependency-{index:03d}:{relative}\n".encode("utf-8"))


@pytest.fixture
def dependency_tree(tmp_path: Path) -> Path:
    root = tmp_path / "project"
    _materialize_dependency_tree(root)
    return root


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


@pytest.mark.skipif(
    not LOCAL_ARTIFACT_TESTS_ENABLED,
    reason=(
        "requires the ignored project-local compiled extension; set "
        f"{LOCAL_ARTIFACT_TEST_ENV}=1 to opt in"
    ),
)
def test_fixed_path_contract_covers_complete_checkpoint_axis_stack() -> None:
    paths = dependency_seal.IMPLEMENTATION_DEPENDENCY_PATHS
    required = {
        "benchmark/implementation_dependency_seal.py",
        "benchmark/checkpoint_axis.py",
        "benchmark/source_corruption_axis_runner_v2.py",
        "benchmark/adabn_axis_runner_v2.py",
        "export_fixed_split_source_axis_v2.py",
        "run_source_corruption_checkpoint_axis_v2.py",
        "run_adabn_corruption_checkpoint_axis_v2.py",
        "scripts/capture_checkpoint_axis_v2_candidate_producer_observation.py",
        "scripts/verify_checkpoint_axis_v2_parity.py",
        "run_adabn_corruption_benchmark.py",
        "configs/adabn_batch_stats_fixed_splits_v1.yaml",
        "dataio/corruption_cache.py",
        "test_source.py",
        "test_fixed_split_source.py",
        "tta/state_manager.py",
        "tta/model_adapter.py",
        "tta/episodic_runner.py",
        "tta/adabn.py",
        "tta/adabn_fast_runner.py",
        "model/MSHNet_NSFPN.py",
        "model/NS_FPN.py",
        "model/diff_cross_attns.py",
        "metrics/irstd_metrics.py",
        "metrics/official_metric_adapter.py",
        "utils/metric.py",
        "SFS_MSDeformAttn/ops/functions/ms_deform_attn_func.py",
        "SFS_MSDeformAttn/ops/modules/ms_deform_attn.py",
    }
    extension = (
        ".conda/lib/python3.10/site-packages/"
        "MultiScaleDeformableAttention.cpython-310-x86_64-linux-gnu.so"
    )

    assert required <= set(paths)
    assert paths == tuple(sorted(paths))
    assert len(paths) == len(set(paths))
    assert tuple(path for path in paths if path.endswith(".so")) == (extension,)
    assert not any(
        path == "results" or path.startswith("results/") for path in paths
    )
    for relative in paths:
        path = dependency_seal.PROJECT_ROOT / relative
        value = path.lstat()
        assert stat.S_ISREG(value.st_mode), relative
        assert not stat.S_ISLNK(value.st_mode), relative


def test_capture_is_deterministic_ordered_and_canonical(
    dependency_tree: Path,
) -> None:
    first = dependency_seal.capture_implementation_dependency_seal(
        dependency_tree
    )
    second = dependency_seal.capture_implementation_dependency_seal(
        dependency_tree
    )

    assert first == second
    assert [record["path"] for record in first["files"]] == list(
        dependency_seal.IMPLEMENTATION_DEPENDENCY_PATHS
    )
    assert first["file_count"] == len(
        dependency_seal.IMPLEMENTATION_DEPENDENCY_PATHS
    )
    assert first["total_bytes"] == sum(
        record["bytes"] for record in first["files"]
    )
    for record in first["files"]:
        payload = _dependency_path(dependency_tree, record["path"]).read_bytes()
        assert record == {
            "path": record["path"],
            "sha256": hashlib.sha256(payload).hexdigest(),
            "bytes": len(payload),
        }
    material = {
        key: value for key, value in first.items() if key != "bundle_sha256"
    }
    assert first["bundle_sha256"] == hashlib.sha256(
        _canonical_json(material)
    ).hexdigest()
    assert dependency_seal.verify_implementation_dependency_seal(
        first, dependency_tree
    ) == first


def test_verify_rejects_missing_and_extra_dependency_records(
    dependency_tree: Path,
) -> None:
    captured = dependency_seal.capture_implementation_dependency_seal(
        dependency_tree
    )
    missing = deepcopy(captured)
    missing["files"].pop()
    extra = deepcopy(captured)
    extra["files"].append(dict(extra["files"][-1]))

    with pytest.raises(
        dependency_seal.ImplementationDependencySealError,
        match="path set differs",
    ):
        dependency_seal.verify_implementation_dependency_seal(
            missing, dependency_tree
        )
    with pytest.raises(
        dependency_seal.ImplementationDependencySealError,
        match="path set differs",
    ):
        dependency_seal.verify_implementation_dependency_seal(
            extra, dependency_tree
        )


def test_verify_rejects_reordered_dependency_records(
    dependency_tree: Path,
) -> None:
    captured = dependency_seal.capture_implementation_dependency_seal(
        dependency_tree
    )
    reordered = deepcopy(captured)
    reordered["files"][0], reordered["files"][1] = (
        reordered["files"][1],
        reordered["files"][0],
    )

    with pytest.raises(
        dependency_seal.ImplementationDependencySealError,
        match="path order/set differs",
    ):
        dependency_seal.verify_implementation_dependency_seal(
            reordered, dependency_tree
        )


def test_verify_rejects_canonical_bundle_tampering(
    dependency_tree: Path,
) -> None:
    captured = dependency_seal.capture_implementation_dependency_seal(
        dependency_tree
    )
    captured["bundle_sha256"] = "0" * 64

    with pytest.raises(
        dependency_seal.ImplementationDependencySealError,
        match="canonical bundle SHA256 differs",
    ):
        dependency_seal.verify_implementation_dependency_seal(
            captured, dependency_tree
        )


def test_verify_rejects_missing_current_dependency(
    dependency_tree: Path,
) -> None:
    captured = dependency_seal.capture_implementation_dependency_seal(
        dependency_tree
    )
    missing_path = _dependency_path(
        dependency_tree, dependency_seal.IMPLEMENTATION_DEPENDENCY_PATHS[0]
    )
    missing_path.unlink()

    with pytest.raises(FileNotFoundError, match="dependency is missing"):
        dependency_seal.verify_implementation_dependency_seal(
            captured, dependency_tree
        )


def test_verify_rejects_same_size_byte_drift(dependency_tree: Path) -> None:
    captured = dependency_seal.capture_implementation_dependency_seal(
        dependency_tree
    )
    relative = "test_source.py"
    path = _dependency_path(dependency_tree, relative)
    original = path.read_bytes()
    path.write_bytes(bytes([original[0] ^ 1]) + original[1:])
    assert path.stat().st_size == len(original)

    with pytest.raises(
        dependency_seal.ImplementationDependencySealError,
        match=rf"SHA256 drifted: {relative}",
    ):
        dependency_seal.verify_implementation_dependency_seal(
            captured, dependency_tree
        )


def test_capture_rejects_leaf_symlink(dependency_tree: Path) -> None:
    relative = dependency_seal.IMPLEMENTATION_DEPENDENCY_PATHS[0]
    path = _dependency_path(dependency_tree, relative)
    outside = dependency_tree.parent / "outside.so"
    outside.write_bytes(path.read_bytes())
    path.unlink()
    path.symlink_to(outside)

    with pytest.raises(
        dependency_seal.ImplementationDependencySealError,
        match="regular non-symlink",
    ):
        dependency_seal.capture_implementation_dependency_seal(dependency_tree)


def test_capture_rejects_ancestor_symlink(dependency_tree: Path) -> None:
    original = dependency_tree / ".conda"
    moved = dependency_tree / ".conda-real"
    original.rename(moved)
    original.symlink_to(moved, target_is_directory=True)

    with pytest.raises(
        dependency_seal.ImplementationDependencySealError,
        match="contains a symlink or non-directory component",
    ):
        dependency_seal.capture_implementation_dependency_seal(dependency_tree)


def test_capture_detects_mutation_during_stable_read(
    dependency_tree: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    relative = dependency_seal.IMPLEMENTATION_DEPENDENCY_PATHS[0]
    path = _dependency_path(dependency_tree, relative)
    original_read = dependency_seal.os.read
    mutated = False

    def mutate_after_first_read(descriptor: int, count: int) -> bytes:
        nonlocal mutated
        payload = original_read(descriptor, count)
        if payload and not mutated:
            mutated = True
            current = path.read_bytes()
            path.write_bytes(bytes([current[0] ^ 1]) + current[1:])
        return payload

    monkeypatch.setattr(dependency_seal.os, "read", mutate_after_first_read)

    with pytest.raises(
        dependency_seal.ImplementationDependencySealError,
        match="changed while being read",
    ):
        dependency_seal.capture_implementation_dependency_seal(dependency_tree)


def test_capture_detects_ancestor_replacement_during_stable_read(
    dependency_tree: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    relative = dependency_seal.IMPLEMENTATION_DEPENDENCY_PATHS[0]
    original_file = _dependency_path(dependency_tree, relative)
    original_payload = original_file.read_bytes()
    original_directory = dependency_tree / ".conda"
    moved_directory = dependency_tree / ".conda-original"
    original_read = dependency_seal.os.read
    replaced = False

    def replace_ancestor_after_first_read(descriptor: int, count: int) -> bytes:
        nonlocal replaced
        payload = original_read(descriptor, count)
        if payload and not replaced:
            replaced = True
            original_directory.rename(moved_directory)
            replacement_file = _dependency_path(dependency_tree, relative)
            replacement_file.parent.mkdir(parents=True)
            replacement_file.write_bytes(original_payload)
        return payload

    monkeypatch.setattr(
        dependency_seal.os, "read", replace_ancestor_after_first_read
    )

    with pytest.raises(
        dependency_seal.ImplementationDependencySealError,
        match="pathname changed",
    ):
        dependency_seal.capture_implementation_dependency_seal(dependency_tree)


def test_verify_rejects_unexpected_seal_or_record_fields(
    dependency_tree: Path,
) -> None:
    captured = dependency_seal.capture_implementation_dependency_seal(
        dependency_tree
    )
    extra_seal_field = deepcopy(captured)
    extra_seal_field["captured_at"] = "nondeterministic"
    extra_record_field = deepcopy(captured)
    extra_record_field["files"][0]["absolute_path"] = "/not-portable"

    with pytest.raises(
        dependency_seal.ImplementationDependencySealError,
        match="seal fields differ",
    ):
        dependency_seal.verify_implementation_dependency_seal(
            extra_seal_field, dependency_tree
        )
    with pytest.raises(
        dependency_seal.ImplementationDependencySealError,
        match="record 0 fields differ",
    ):
        dependency_seal.verify_implementation_dependency_seal(
            extra_record_field, dependency_tree
        )
