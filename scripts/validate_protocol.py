#!/usr/bin/env python3
"""Validate the frozen protocol against repository and local artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
from typing import Any

import yaml


def sha256_file(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def read_split(path: Path) -> list[str]:
    image_ids = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]
    if not image_ids or any(not image_id for image_id in image_ids):
        raise ValueError(f"split is empty or contains blank identifiers: {path}")
    if len(image_ids) != len(set(image_ids)):
        raise ValueError(f"split contains duplicate identifiers: {path}")
    return image_ids


def corpus_manifest_sha256(
    image_directory: Path,
    mask_directory: Path,
    image_ids: list[str] | set[str],
    extension: str,
) -> str:
    """Hash canonical image/mask bytes, sizes, roles, and sorted identifiers."""

    identifiers = sorted(image_ids)
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("corpus manifest image IDs must be globally unique")
    manifest = hashlib.sha256()
    for image_id in identifiers:
        image_path = image_directory / f"{image_id}{extension}"
        mask_path = mask_directory / f"{image_id}{extension}"
        line = (
            f"{image_id}\t"
            f"image\t{sha256_file(image_path)}\t{image_path.stat().st_size}\t"
            f"mask\t{sha256_file(mask_path)}\t{mask_path.stat().st_size}\n"
        )
        manifest.update(line.encode("utf-8"))
    return manifest.hexdigest()


def _load_yaml(path: Path) -> dict[str, Any]:
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError(f"expected a YAML mapping: {path}")
    return loaded


def _resolve_data_root(
    dataset_name: str,
    dataset_config: dict[str, Any],
    local_paths: dict[str, Any],
) -> Path:
    environment_name = str(dataset_config["root_env"])
    environment_value = os.environ.get(environment_name)
    if environment_value:
        return Path(environment_value).expanduser().resolve()
    configured = local_paths.get("datasets", {}).get(dataset_name)
    if configured:
        return Path(configured).expanduser().resolve()
    raise ValueError(
        f"no data root for {dataset_name}; set {environment_name} or local_paths.yaml"
    )


def _resolve_layout(root: Path, candidates: list[str], kind: str) -> Path:
    matches = [root / candidate for candidate in candidates if (root / candidate).is_dir()]
    if not matches:
        raise ValueError(f"{root} has no supported {kind} directory: {candidates}")
    return matches[0]


def validate_frozen_corruption_table(
    repository: Path, corruption_config: dict[str, Any]
) -> dict[str, Any]:
    """Verify that the protocol and severity YAML freeze the same table bytes."""

    table_path = repository / str(corruption_config["table"])
    actual_hash = sha256_file(table_path)
    expected_hash = str(corruption_config["table_sha256"])
    if actual_hash != expected_hash:
        raise ValueError(
            f"corruption severity table hash mismatch: {actual_hash}, "
            f"expected {expected_hash}"
        )
    if corruption_config.get("table_frozen") is not True:
        raise ValueError("protocol must mark the corruption severity table as frozen")

    table = _load_yaml(table_path)
    calibration = table.get("calibration")
    if table.get("frozen") is not True or not isinstance(calibration, dict):
        raise ValueError("severity table must be frozen and contain calibration metadata")
    if calibration.get("completed") is not True:
        raise ValueError("severity table calibration must be completed before freezing")
    provisional_archive = repository / str(calibration["provisional_table_archive"])
    provisional_hash = sha256_file(provisional_archive)
    if provisional_hash != str(calibration["provisional_table_sha256"]):
        raise ValueError(
            "archived provisional severity table does not match its frozen hash"
        )
    return {
        "path": str(table_path),
        "sha256": actual_hash,
        "status": table.get("status"),
        "frozen": True,
        "calibration_completed": True,
        "provisional_table_archive": str(provisional_archive),
        "provisional_table_sha256": provisional_hash,
    }


def validate_frozen_environment(
    repository: Path, environment_config: dict[str, Any]
) -> dict[str, Any]:
    """Verify the declarative, explicit, Python, and SFS build environment files."""

    artifact_fields = (
        ("conda_explicit_lock", "conda_explicit_lock_sha256"),
        ("python_package_snapshot", "python_package_snapshot_sha256"),
        ("declarative_environment", "declarative_environment_sha256"),
        ("sfs_build_script", "sfs_build_script_sha256"),
    )
    artifacts: dict[str, dict[str, Any]] = {}
    for path_field, hash_field in artifact_fields:
        relative_path = str(environment_config[path_field])
        path = repository / relative_path
        actual_hash = sha256_file(path)
        expected_hash = str(environment_config[hash_field])
        if actual_hash != expected_hash:
            raise ValueError(
                f"environment artifact hash mismatch for {relative_path}: "
                f"{actual_hash}, expected {expected_hash}"
            )
        artifacts[path_field] = {
            "path": str(path),
            "sha256": actual_hash,
        }
    return {
        "platform": environment_config["platform"],
        "python": str(environment_config["python"]),
        "torch": str(environment_config["torch"]),
        "torch_cuda_runtime": str(environment_config["torch_cuda_runtime"]),
        "artifacts": artifacts,
        "sfs_host_toolchain": environment_config["sfs_host_toolchain"],
    }


def validate_protocol(
    repository: Path,
    protocol_path: Path,
    local_paths_path: Path | None = None,
) -> dict[str, Any]:
    repository = repository.resolve()
    protocol = _load_yaml(protocol_path)
    local_paths = (
        _load_yaml(local_paths_path)
        if local_paths_path is not None and local_paths_path.exists()
        else {}
    )

    expected_commit = str(protocol["host"]["base_commit"])
    base_file_commit = (repository / "BASE_COMMIT.txt").read_text().strip()
    current_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    dirty_entries = subprocess.run(
        ["git", "status", "--porcelain=v1", "--untracked-files=all"],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    if dirty_entries:
        raise ValueError(
            "frozen protocol validation requires a clean worktree; "
            f"found: {dirty_entries[:10]}"
        )
    is_ancestor = subprocess.run(
        ["git", "merge-base", "--is-ancestor", expected_commit, "HEAD"],
        cwd=repository,
        check=False,
    ).returncode == 0
    if base_file_commit != expected_commit or not is_ancestor:
        raise ValueError(
            "base commit mismatch: "
            f"protocol={expected_commit}, file={base_file_commit}, "
            f"HEAD={current_commit}, base_is_ancestor={is_ancestor}"
        )

    image_directories = list(protocol["data"]["image_directories"])
    mask_directories = list(protocol["data"]["mask_directories"])
    extension = str(protocol["data"]["file_extension"])
    dataset_reports: dict[str, Any] = {}

    for dataset_name, dataset_config in protocol["datasets"].items():
        root = _resolve_data_root(dataset_name, dataset_config, local_paths)
        image_directory = _resolve_layout(root, image_directories, "image")
        mask_directory = _resolve_layout(root, mask_directories, "mask")
        split_report: dict[str, Any] = {}
        split_sets: dict[str, set[str]] = {}

        for split_name in ("trainval", "test"):
            relative_path = Path(dataset_config[f"{split_name}_split"])
            split_path = repository / relative_path
            actual_hash = sha256_file(split_path)
            expected_hash = str(dataset_config[f"{split_name}_split_sha256"])
            if actual_hash != expected_hash:
                raise ValueError(
                    f"{dataset_name} {split_name} split hash mismatch: {actual_hash}"
                )
            identifiers = read_split(split_path)
            expected_count = int(dataset_config[f"{split_name}_images"])
            if len(identifiers) != expected_count:
                raise ValueError(
                    f"{dataset_name} {split_name} count is {len(identifiers)}, "
                    f"expected {expected_count}"
                )
            missing_images = [
                image_id
                for image_id in identifiers
                if not (image_directory / f"{image_id}{extension}").is_file()
            ]
            missing_masks = [
                image_id
                for image_id in identifiers
                if not (mask_directory / f"{image_id}{extension}").is_file()
            ]
            if missing_images or missing_masks:
                raise ValueError(
                    f"{dataset_name} {split_name} is incomplete: "
                    f"missing_images={missing_images[:5]}, missing_masks={missing_masks[:5]}"
                )
            split_sets[split_name] = set(identifiers)
            split_report[split_name] = {
                "count": len(identifiers),
                "sha256": actual_hash,
            }

        overlap = split_sets["trainval"] & split_sets["test"]
        if overlap:
            raise ValueError(
                f"{dataset_name} trainval/test overlap: {sorted(overlap)[:5]}"
            )

        all_image_ids = split_sets["trainval"] | split_sets["test"]
        if dataset_config.get("corpus_manifest_algorithm") != "sorted-id-tab-v1":
            raise ValueError(
                f"{dataset_name} must use corpus_manifest_algorithm=sorted-id-tab-v1"
            )
        corpus_hash = corpus_manifest_sha256(
            image_directory,
            mask_directory,
            all_image_ids,
            extension,
        )
        expected_corpus_hash = str(dataset_config["corpus_manifest_sha256"])
        if corpus_hash != expected_corpus_hash:
            raise ValueError(
                f"{dataset_name} corpus manifest hash mismatch: {corpus_hash}, "
                f"expected {expected_corpus_hash}"
            )

        checkpoint_path = repository / str(dataset_config["checkpoint"])
        checkpoint_hash = sha256_file(checkpoint_path)
        if checkpoint_hash != str(dataset_config["checkpoint_sha256"]):
            raise ValueError(f"{dataset_name} checkpoint hash mismatch: {checkpoint_hash}")

        dataset_reports[dataset_name] = {
            "root": str(root),
            "image_directory": str(image_directory),
            "mask_directory": str(mask_directory),
            "splits": split_report,
            "total_unique_images": len(all_image_ids),
            "corpus_manifest_algorithm": "sorted-id-tab-v1",
            "corpus_manifest_sha256": corpus_hash,
            "checkpoint": str(checkpoint_path),
            "checkpoint_sha256": checkpoint_hash,
        }

    corruption_report = validate_frozen_corruption_table(
        repository, protocol["corruptions"]
    )
    environment_report = validate_frozen_environment(
        repository, protocol["environment"]
    )

    return {
        "protocol_id": protocol["protocol_id"],
        "base_commit": expected_commit,
        "current_commit": current_commit,
        "worktree_clean": True,
        "datasets": dataset_reports,
        "corruptions": corruption_report,
        "environment": environment_report,
        "valid": True,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repository", type=Path, default=Path(__file__).resolve().parents[1]
    )
    parser.add_argument(
        "--protocol", type=Path, default=Path("configs/protocol.yaml")
    )
    parser.add_argument(
        "--local-paths", type=Path, default=Path("configs/local_paths.yaml")
    )
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repository = args.repository.resolve()
    protocol_path = (
        args.protocol if args.protocol.is_absolute() else repository / args.protocol
    )
    local_paths_path = (
        args.local_paths
        if args.local_paths.is_absolute()
        else repository / args.local_paths
    )
    report = validate_protocol(repository, protocol_path, local_paths_path)
    encoded = json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    if args.output:
        output = args.output if args.output.is_absolute() else repository / args.output
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")


if __name__ == "__main__":
    main()
