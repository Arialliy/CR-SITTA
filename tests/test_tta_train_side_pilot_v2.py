from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import pytest
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATAIO_ROOT = PROJECT_ROOT / "dataio"
if str(DATAIO_ROOT) not in sys.path:
    sys.path.insert(0, str(DATAIO_ROOT))

import train_side_pilot_protocol as pilot  # noqa: E402


PROTOCOL = PROJECT_ROOT / "configs" / "tta_train_side_calibration_pilot_v2.yaml"


def _real_context() -> dict[str, object]:
    return pilot.build_contract(PROJECT_ROOT, PROTOCOL)


def _with_output(context: dict[str, object], output: Path) -> dict[str, object]:
    return {**context, "output_root": output}


def _protocol_variant(tmp_path: Path, mutate) -> Path:
    value = yaml.safe_load(PROTOCOL.read_text(encoding="utf-8"))
    mutate(value)
    path = tmp_path / "protocol.yaml"
    path.write_text(yaml.safe_dump(value, sort_keys=False), encoding="utf-8")
    return path


def test_protocol_explicitly_has_no_validation_split_and_discloses_anchor() -> None:
    protocol, digest = pilot.load_protocol(PROTOCOL)

    assert protocol["scope"] == "source_train_side_method_calibration"
    assert protocol["no_validation_split"] is True
    assert protocol["paper_result"] is False
    assert protocol["data_boundary"]["create_train_core"] is False
    assert protocol["data_boundary"]["create_source_val"] is False
    assert protocol["data_boundary"]["open_images"] is False
    assert protocol["data_boundary"]["open_masks"] is False
    assert protocol["checkpoint_anchor"]["role"] == "best_miou"
    assert protocol["checkpoint_anchor"]["selection"] == "test_selected"
    assert (
        protocol["checkpoint_anchor"]["best_pd_policy"]
        == "reuse_same_frozen_hyperparameters_without_additional_tuning"
    )
    assert digest == pilot.sha256_file(PROTOCOL)


def test_selection_excludes_parent_and_is_order_independent_with_id_tie_break(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    train = ("z", "parent", "b", "a", "c")
    real_sha256 = hashlib.sha256

    class _SameDigest:
        def hexdigest(self) -> str:
            return "0" * 64

    monkeypatch.setattr(
        pilot.hashlib,
        "sha256",
        lambda value=b"": _SameDigest() if value in {b"a", b"b", b"c", b"z"} else real_sha256(value),
    )
    first = pilot.select_train_side_ids(train, ("parent",), limit=3)
    second = pilot.select_train_side_ids(tuple(reversed(train)), ("parent",), limit=3)

    assert first == ("a", "b", "c")
    assert second == first
    assert "parent" not in first


def test_real_contract_proves_uniqueness_containment_and_three_way_disjointness() -> None:
    context = _real_context()

    assert tuple(context["datasets"]) == pilot.DATASET_NAMES
    for report in context["datasets"].values():
        train_ids = set(
            pilot.read_canonical_ids(
                PROJECT_ROOT / report["train"]["path"], label="train"
            )
        )
        test_ids = set(
            pilot.read_canonical_ids(
                PROJECT_ROOT / report["test"]["path"], label="test"
            )
        )
        parent_artifact = json.loads(
            (PROJECT_ROOT / report["parent_pilot"]["artifact"]).read_text(
                encoding="utf-8"
            )
        )
        parent_ids = set(parent_artifact["selection"]["selected_ids"])
        output_ids = set(report["output"]["ids"])

        assert len(parent_ids) == len(output_ids) == 64
        assert parent_ids <= train_ids
        assert output_ids <= train_ids
        assert not train_ids & test_ids
        assert not parent_ids & output_ids
        assert not parent_ids & test_ids
        assert not output_ids & test_ids
        assert all(value is True or value == 0 for value in report["checks"].values())


def test_contract_reads_only_yaml_json_and_id_txt_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    opened: list[Path] = []
    real_snapshot = pilot._read_stable_bytes

    def observed_snapshot(path: Path, *, label: str):
        opened.append(path)
        return real_snapshot(path, label=label)

    monkeypatch.setattr(pilot, "_read_stable_bytes", observed_snapshot)
    _real_context()

    assert opened
    assert len(opened) == len(set(opened)) == 16
    assert {path.suffix for path in opened} <= {".yaml", ".json", ".txt"}
    assert not any("/img/" in path.as_posix() for path in opened)
    assert not any("/label/" in path.as_posix() for path in opened)
    assert not any("/images/" in path.as_posix() for path in opened)
    assert not any("/masks/" in path.as_posix() for path in opened)


@pytest.mark.parametrize(
    ("section", "field", "changed", "message"),
    [
        (None, "protocol_path", "configs/other.yaml", "protocol_path"),
        ("data_boundary", "test_split_use", "pixels", "test_split_use"),
        ("output", "manifest", "configs/other.json", "output.manifest"),
        ("output", "format", "json", "output.format"),
        ("output", "atomic_directory_publish", False, "atomic_directory_publish"),
        ("output", "atomic_directory_publish", 1, "must be a boolean"),
        ("output", "overwrite", True, "output.overwrite"),
        ("output", "overwrite", 0, "must be a boolean"),
    ],
)
def test_loader_enforces_frozen_path_boundary_and_publish_fields(
    tmp_path: Path,
    section: str | None,
    field: str,
    changed: object,
    message: str,
) -> None:
    def mutate(value: dict[str, object]) -> None:
        target = value if section is None else value[section]
        target[field] = changed

    path = _protocol_variant(tmp_path, mutate)
    with pytest.raises((TypeError, ValueError), match=message):
        pilot.load_protocol(path)


@pytest.mark.parametrize("leaf_symlink", [False, True])
def test_stable_snapshot_rejects_intermediate_and_leaf_symlinks(
    tmp_path: Path, leaf_symlink: bool
) -> None:
    real_directory = tmp_path / "real"
    real_directory.mkdir()
    metadata = real_directory / "metadata.json"
    metadata.write_text("{}\n", encoding="utf-8")
    if leaf_symlink:
        unsafe = real_directory / "linked.json"
        unsafe.symlink_to(metadata)
    else:
        linked_directory = tmp_path / "linked-directory"
        linked_directory.symlink_to(real_directory, target_is_directory=True)
        unsafe = linked_directory / "metadata.json"

    with pytest.raises(ValueError, match="symlink|non-directory"):
        pilot._read_stable_bytes(unsafe, label="unsafe metadata")


def test_stable_snapshot_fails_if_pathname_is_swapped_during_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "metadata.json"
    path.write_bytes(b"a" * 32)
    replacement = tmp_path / "replacement.json"
    replacement.write_bytes(b"b" * 32)
    real_read = pilot.os.read
    swapped = False

    def swapping_read(descriptor: int, size: int) -> bytes:
        nonlocal swapped
        chunk = real_read(descriptor, size)
        if chunk and not swapped:
            swapped = True
            pilot.os.replace(replacement, path)
        return chunk

    monkeypatch.setattr(pilot.os, "read", swapping_read)
    with pytest.raises(RuntimeError, match="changed while|pathname changed"):
        pilot._read_stable_bytes(path, label="swapped metadata")


def test_stable_snapshot_fails_if_inode_drifts_during_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "metadata.json"
    path.write_bytes(b"a" * 32)
    real_read = pilot.os.read
    changed = False

    def drifting_read(descriptor: int, size: int) -> bytes:
        nonlocal changed
        chunk = real_read(descriptor, size)
        if chunk and not changed:
            changed = True
            with path.open("r+b") as stream:
                stream.seek(0)
                stream.write(b"b")
                stream.flush()
                pilot.os.fsync(stream.fileno())
        return chunk

    monkeypatch.setattr(pilot.os, "read", drifting_read)
    with pytest.raises(RuntimeError, match="changed while"):
        pilot._read_stable_bytes(path, label="drifting metadata")


@pytest.mark.parametrize(
    ("field", "message"),
    [
        ("train_split_sha256", "train split SHA256"),
        ("test_split_sha256", "test split SHA256"),
        ("output_ordered_ids_sha256", "output ordered ID SHA256"),
        ("output_file_sha256", "output file SHA256"),
    ],
)
def test_contract_fails_closed_on_bound_hash_drift(
    tmp_path: Path, field: str, message: str
) -> None:
    path = _protocol_variant(
        tmp_path,
        lambda value: value["datasets"]["IRSTD-1K"].__setitem__(field, "0" * 64),
    )

    with pytest.raises(ValueError, match=message):
        pilot.build_contract(PROJECT_ROOT, path)


def test_contract_fails_closed_on_parent_pilot_artifact_hash_drift(
    tmp_path: Path,
) -> None:
    path = _protocol_variant(
        tmp_path,
        lambda value: value["datasets"]["IRSTD-1K"][
            "round_02_parent_pilot"
        ].__setitem__("artifact_sha256", "0" * 64),
    )

    with pytest.raises(ValueError, match="parent Pilot artifact SHA256"):
        pilot.build_contract(PROJECT_ROOT, path)


def test_materialize_is_atomic_validated_and_no_overwrite(tmp_path: Path) -> None:
    destination = tmp_path / "configs" / "pilot"
    destination.parent.mkdir()
    context = _with_output(_real_context(), destination)

    result = pilot.materialize(context)

    assert result["valid"] is True
    assert result["read_only_validation"] is True
    assert set(path.name for path in destination.iterdir()) == {
        "IRSTD-1K.txt",
        "NUAA-SIRST.txt",
        "NUDT-SIRST.txt",
        "manifest.json",
    }
    assert not list(destination.parent.glob(f".{destination.name}.build-*"))
    with pytest.raises(FileExistsError, match="refusing overwrite"):
        pilot.materialize(context)


def test_validate_mode_is_read_only(tmp_path: Path) -> None:
    destination = tmp_path / "configs" / "pilot"
    destination.parent.mkdir()
    context = _with_output(_real_context(), destination)
    pilot.materialize(context)
    before = {
        path.name: (path.read_bytes(), path.stat().st_mtime_ns)
        for path in destination.iterdir()
    }

    result = pilot.validate_materialization(context)

    after = {
        path.name: (path.read_bytes(), path.stat().st_mtime_ns)
        for path in destination.iterdir()
    }
    assert result["read_only_validation"] is True
    assert result["images_opened"] == result["masks_opened"] == 0
    assert after == before


def test_validate_fails_on_output_or_manifest_drift(tmp_path: Path) -> None:
    destination = tmp_path / "configs" / "pilot"
    destination.parent.mkdir()
    context = _with_output(_real_context(), destination)
    pilot.materialize(context)
    (destination / "IRSTD-1K.txt").write_text("tampered\n", encoding="utf-8")

    with pytest.raises(ValueError, match="materialized IDs"):
        pilot.validate_materialization(context)


def test_failed_atomic_publish_cleans_staging(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    destination = tmp_path / "configs" / "pilot"
    destination.parent.mkdir()
    context = _with_output(_real_context(), destination)

    def fail_publish(_source: Path, _destination: Path) -> None:
        raise RuntimeError("injected publish failure")

    monkeypatch.setattr(pilot, "_atomic_rename_directory_noreplace", fail_publish)
    with pytest.raises(RuntimeError, match="injected"):
        pilot.materialize(context)

    assert not destination.exists()
    assert not list(destination.parent.glob(f".{destination.name}.build-*"))


def test_generated_repository_artifacts_validate_read_only() -> None:
    result = pilot.validate_materialization(_real_context())

    assert result["valid"] is True
    assert result["scope"] == "source_train_side_method_calibration"
    assert result["images_opened"] == 0
    assert result["masks_opened"] == 0
    assert all(report["count"] == 64 for report in result["datasets"].values())
