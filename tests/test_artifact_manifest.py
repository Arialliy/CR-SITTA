from pathlib import Path

import pytest

from scripts.build_artifact_manifest import artifact_record, artifact_tree, sha256_file


def test_artifact_record_is_portable_and_exact(tmp_path: Path) -> None:
    artifact = tmp_path / "nested" / "value.txt"
    artifact.parent.mkdir()
    artifact.write_bytes(b"stable\n")

    record = artifact_record(artifact, tmp_path)

    assert record == {
        "path": "nested/value.txt",
        "sha256": sha256_file(artifact),
        "size_bytes": 7,
    }
    assert str(tmp_path) not in str(record)


def test_artifact_tree_uses_sorted_relative_member_ledger(tmp_path: Path) -> None:
    directory = tmp_path / "images"
    directory.mkdir()
    (directory / "z.png").write_bytes(b"z")
    (directory / "a.png").write_bytes(b"a")
    (directory / "ignored.txt").write_text("ignored", encoding="utf-8")

    first = artifact_tree(directory, project_root=tmp_path, suffixes=(".png",))
    second = artifact_tree(directory, project_root=tmp_path, suffixes=(".png",))

    assert first == second
    assert first["file_count"] == 2
    assert [item["path"] for item in first["files"]] == [
        "images/a.png",
        "images/z.png",
    ]


def test_artifact_record_rejects_external_path(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    external = tmp_path / "external.txt"
    external.write_text("outside", encoding="utf-8")

    with pytest.raises(ValueError, match="outside the project"):
        artifact_record(external, project)
