from pathlib import Path

import pytest

from scripts.validate_protocol import (
    corpus_manifest_sha256,
    read_split,
    sha256_file,
    validate_frozen_corruption_table,
    validate_frozen_environment,
)


def test_read_split_counts_last_line_without_trailing_newline(tmp_path: Path) -> None:
    split = tmp_path / "split.txt"
    split.write_text("first\nsecond", encoding="utf-8")
    assert read_split(split) == ["first", "second"]


def test_read_split_rejects_duplicate_identifiers(tmp_path: Path) -> None:
    split = tmp_path / "split.txt"
    split.write_text("same\nsame\n", encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate"):
        read_split(split)


def test_sha256_file_is_stable(tmp_path: Path) -> None:
    artifact = tmp_path / "artifact.bin"
    artifact.write_bytes(b"CR-SITTA\n")
    assert sha256_file(artifact) == (
        "39fc969de31e988076e41af0e504360fb58775100a7aff2f5328fa5273e6daf1"
    )


def test_corpus_manifest_hashes_sorted_ids_roles_bytes_and_sizes(
    tmp_path: Path,
) -> None:
    images = tmp_path / "images"
    masks = tmp_path / "masks"
    images.mkdir()
    masks.mkdir()
    (images / "b.png").write_bytes(b"image-b")
    (masks / "b.png").write_bytes(b"mask-b")
    (images / "a.png").write_bytes(b"image-a")
    (masks / "a.png").write_bytes(b"mask-a")

    digest = corpus_manifest_sha256(images, masks, ["b", "a"], ".png")

    import hashlib

    expected = hashlib.sha256()
    for image_id in ("a", "b"):
        image_path = images / f"{image_id}.png"
        mask_path = masks / f"{image_id}.png"
        expected.update(
            (
                f"{image_id}\timage\t{sha256_file(image_path)}\t"
                f"{image_path.stat().st_size}\tmask\t{sha256_file(mask_path)}\t"
                f"{mask_path.stat().st_size}\n"
            ).encode("utf-8")
        )
    assert digest == expected.hexdigest()


def test_frozen_corruption_table_hash_and_calibration_are_validated(
    tmp_path: Path,
) -> None:
    table = tmp_path / "severity.yaml"
    provisional = tmp_path / "severity.provisional.yaml"
    provisional.write_text("frozen: false\n", encoding="utf-8")
    table.write_text(
        "status: frozen_after_source_domain_pilot\n"
        "frozen: true\n"
        "calibration:\n"
        "  completed: true\n"
        "  provisional_table_archive: severity.provisional.yaml\n"
        f"  provisional_table_sha256: {sha256_file(provisional)}\n",
        encoding="utf-8",
    )
    config = {
        "table": "severity.yaml",
        "table_sha256": sha256_file(table),
        "table_frozen": True,
    }

    report = validate_frozen_corruption_table(tmp_path, config)
    assert report["sha256"] == config["table_sha256"]
    assert report["frozen"] is True

    table.write_text(table.read_text() + "# changed\n", encoding="utf-8")
    with pytest.raises(ValueError, match="hash mismatch"):
        validate_frozen_corruption_table(tmp_path, config)


def test_environment_artifact_hashes_are_validated(tmp_path: Path) -> None:
    fields = {
        "conda_explicit_lock": "explicit.txt",
        "python_package_snapshot": "requirements.txt",
        "declarative_environment": "environment.yml",
        "sfs_build_script": "build.sh",
    }
    config = {
        "platform": "linux-64",
        "python": "3.10",
        "torch": "2.1.2",
        "torch_cuda_runtime": "12.1",
        "sfs_host_toolchain": {"cuda_arch_list": "8.6"},
    }
    for index, (path_field, filename) in enumerate(fields.items()):
        path = tmp_path / filename
        path.write_text(f"artifact-{index}\n", encoding="utf-8")
        config[path_field] = filename
        config[f"{path_field}_sha256"] = sha256_file(path)

    report = validate_frozen_environment(tmp_path, config)
    assert report["platform"] == "linux-64"
    assert len(report["artifacts"]) == 4
