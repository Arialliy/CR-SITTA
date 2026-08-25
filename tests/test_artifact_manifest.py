from pathlib import Path

import pytest

from PIL import Image

from scripts.build_artifact_manifest import (
    artifact_record,
    artifact_tree,
    build_contact_sheet,
    sha256_file,
)


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


def test_build_contact_sheet_is_deterministic_and_has_expected_layout(
    tmp_path: Path,
) -> None:
    visualizations = tmp_path / "visualizations"
    visualizations.mkdir()
    for index in range(20):
        image = Image.new("RGB", (40, 10), (index, 0, 0))
        image.save(visualizations / f"{index:02d}.png")
    destination = tmp_path / "contact.png"

    build_contact_sheet(visualizations, destination, columns=2, tile_width=20)
    first_hash = sha256_file(destination)
    build_contact_sheet(visualizations, destination, columns=2, tile_width=20)

    assert sha256_file(destination) == first_hash
    with Image.open(destination) as contact:
        assert contact.size == (40, 50)


def test_build_contact_sheet_requires_exactly_twenty_images(tmp_path: Path) -> None:
    visualizations = tmp_path / "visualizations"
    visualizations.mkdir()
    Image.new("RGB", (10, 10)).save(visualizations / "only.png")

    with pytest.raises(ValueError, match="exactly 20"):
        build_contact_sheet(visualizations, tmp_path / "contact.png")
