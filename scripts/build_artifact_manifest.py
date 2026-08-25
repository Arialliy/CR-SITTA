#!/usr/bin/env python3
"""Build a small, versioned manifest for the ignored Steps 0--3 artifacts.

Large per-image records and PNGs remain outside Git.  This manifest binds every
required result file to its SHA-256, records the implementation commit that
generated it, and retains the headline metrics needed to audit the hand-off.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import yaml
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASETS = ("IRSTD-1k", "NUAA-SIRST")


def sha256_file(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    """Return a streaming SHA-256 digest for ``path``."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def artifact_record(path: Path, project_root: Path = PROJECT_ROOT) -> dict[str, Any]:
    """Describe one file without leaking an absolute workspace path."""

    resolved = path.resolve()
    try:
        relative = resolved.relative_to(project_root.resolve()).as_posix()
    except ValueError as error:
        raise ValueError(f"artifact is outside the project: {resolved}") from error
    if not resolved.is_file():
        raise FileNotFoundError(f"required artifact does not exist: {resolved}")
    return {
        "path": relative,
        "sha256": sha256_file(resolved),
        "size_bytes": resolved.stat().st_size,
    }


def artifact_tree(
    directory: Path,
    *,
    project_root: Path = PROJECT_ROOT,
    suffixes: Iterable[str] | None = None,
) -> dict[str, Any]:
    """Hash every selected file in a directory and the ordered member ledger."""

    if not directory.is_dir():
        raise FileNotFoundError(f"required artifact directory does not exist: {directory}")
    allowed = None if suffixes is None else {suffix.lower() for suffix in suffixes}
    paths = sorted(
        path
        for path in directory.rglob("*")
        if path.is_file() and (allowed is None or path.suffix.lower() in allowed)
    )
    if not paths:
        raise RuntimeError(f"artifact directory is empty: {directory}")

    members = [artifact_record(path, project_root) for path in paths]
    ledger = hashlib.sha256()
    for member in members:
        line = (
            f"{member['path']}\t{member['sha256']}\t"
            f"{member['size_bytes']}\n"
        )
        ledger.update(line.encode("utf-8"))
    return {
        "algorithm": "sorted-project-relative-path-tab-sha256-size-lf-v1",
        "sha256": ledger.hexdigest(),
        "file_count": len(members),
        "files": members,
    }


def build_contact_sheet(
    visualization_directory: Path,
    destination: Path,
    *,
    columns: int = 2,
    tile_width: int = 512,
) -> None:
    """Deterministically tile the 20 Source visualizations without ImageMagick."""

    paths = sorted(visualization_directory.glob("*.png"))
    if len(paths) != 20:
        raise ValueError(
            f"contact sheet requires exactly 20 PNGs, found {len(paths)} in "
            f"{visualization_directory}"
        )
    if columns < 1 or tile_width < 1 or len(paths) % columns:
        raise ValueError("contact-sheet columns/width must form complete positive rows")

    tiles: list[Image.Image] = []
    try:
        for path in paths:
            with Image.open(path) as source:
                rgb = source.convert("RGB")
                height = max(1, round(rgb.height * tile_width / rgb.width))
                tiles.append(
                    rgb.resize((tile_width, height), resample=Image.Resampling.LANCZOS)
                )
        row_heights = [
            max(tile.height for tile in tiles[start : start + columns])
            for start in range(0, len(tiles), columns)
        ]
        canvas = Image.new("RGB", (columns * tile_width, sum(row_heights)), "white")
        top = 0
        for row, row_height in enumerate(row_heights):
            for column in range(columns):
                tile = tiles[row * columns + column]
                canvas.paste(tile, (column * tile_width, top))
            top += row_height

        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
        canvas.save(temporary, format="PNG")
        os.replace(temporary, destination)
    finally:
        for tile in tiles:
            tile.close()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid JSON artifact: {path}") from error
    if not isinstance(value, dict):
        raise TypeError(f"JSON artifact must contain an object: {path}")
    return value


def _require_clean_provenance(
    artifact: Mapping[str, Any],
    *,
    label: str,
) -> str:
    provenance = artifact.get("repository_provenance")
    if not isinstance(provenance, Mapping):
        raise ValueError(f"{label} has no repository_provenance")
    commit = provenance.get("head_commit")
    if not isinstance(commit, str) or len(commit) != 40:
        raise ValueError(f"{label} has an invalid head commit")
    if provenance.get("worktree_clean") is not True:
        raise ValueError(f"{label} was not generated from a clean worktree")
    if provenance.get("dirty_entries") != []:
        raise ValueError(f"{label} records dirty worktree entries")
    return commit


def _source_entry(project_root: Path, dataset: str) -> tuple[dict[str, Any], str]:
    root = project_root / "results" / "source_reproduction" / dataset
    metrics_path = root / "metrics.json"
    metrics = _read_json(metrics_path)
    commit = _require_clean_provenance(metrics, label=f"Source/{dataset}")
    if metrics.get("evaluated_images") != metrics.get("available_images"):
        raise ValueError(f"Source/{dataset} is not a complete split run")
    if metrics.get("frozen_reference_comparison", {}).get("passed") is not True:
        raise ValueError(f"Source/{dataset} failed the frozen reference comparison")
    if metrics.get("checks", {}).get("model_state_unchanged") is not True:
        raise ValueError(f"Source/{dataset} changed the model state")
    if metrics.get("visualization_count") != 20:
        raise ValueError(f"Source/{dataset} must contain 20 visualizations")

    return (
        {
            "evaluated_images": metrics["evaluated_images"],
            "split_sha256": metrics["split_sha256"],
            "checkpoint_sha256": metrics["checkpoint_sha256"],
            "official_operating_point": metrics["official_reported_operating_point"],
            "unified_fixed_operating_point": metrics["unified"]["fixed"],
            "frozen_reference_comparison": metrics["frozen_reference_comparison"],
            "runtime_code_sha256": metrics["repository_provenance"]["file_sha256"],
            "checks": metrics["checks"],
            "artifacts": {
                "metrics": artifact_record(metrics_path, project_root),
                "per_image": artifact_record(root / "per_image.jsonl", project_root),
                "contact_sheet": artifact_record(root / "contact_sheet.png", project_root),
                "visualizations": artifact_tree(
                    root / "visualizations",
                    project_root=project_root,
                    suffixes=(".png",),
                ),
            },
        },
        commit,
    )


def _pilot_entry(
    project_root: Path,
    dataset: str,
    *,
    result_group: str,
    require_provenance: bool,
) -> tuple[dict[str, Any], str | None]:
    root = project_root / "results" / result_group / dataset
    pilot_path = root / "pilot.json"
    pilot = _read_json(pilot_path)
    commit = (
        _require_clean_provenance(pilot, label=f"{result_group}/{dataset}")
        if require_provenance
        else None
    )
    checks = pilot.get("checks", {})
    required_checks = (
        "same_ordered_ids_all_conditions",
        "official_test_ids_absent",
        "model_state_unchanged",
        "severity_table_unchanged",
    )
    failed = [name for name in required_checks if checks.get(name) is not True]
    if failed:
        raise ValueError(f"{result_group}/{dataset} failed checks: {failed}")
    if pilot.get("condition_count") != 21:
        raise ValueError(f"{result_group}/{dataset} must contain 21 conditions")

    trend_classes = {
        corruption: {
            metric: details["classification"]
            for metric, details in report["metrics"].items()
        }
        for corruption, report in pilot["trends"].items()
    }
    return (
        {
            "split_role": pilot["split_role"],
            "split_sha256": pilot["split_sha256"],
            "checkpoint_sha256": pilot["checkpoint_sha256"],
            "selected_ids_sha256": pilot["selection"][
                "ordered_selected_ids_sha256"
            ],
            "selected_count": pilot["selection"]["selected_count"],
            "condition_count": pilot["condition_count"],
            "severity_table": {
                key: pilot["severity_table"][key]
                for key in (
                    "status",
                    "frozen",
                    "calibration_required",
                    "calibration_completed",
                )
            },
            "trend_classifications": trend_classes,
            "runtime_code_sha256": (
                pilot["repository_provenance"]["file_sha256"]
                if require_provenance
                else None
            ),
            "checks": checks,
            "artifacts": {
                "pilot": artifact_record(pilot_path, project_root),
                "severity_grids": artifact_tree(
                    root / "severity_grids",
                    project_root=project_root,
                    suffixes=(".png",),
                ),
            },
        },
        commit,
    )


def build_manifest(project_root: Path = PROJECT_ROOT) -> dict[str, Any]:
    """Validate the final artifact set and return its portable manifest."""

    project_root = project_root.resolve()
    protocol_path = project_root / "configs" / "protocol.yaml"
    severity_path = project_root / "corruptions" / "severity_tables.yaml"
    validation_path = project_root / "results" / "protocol_validation.json"
    protocol = yaml.safe_load(protocol_path.read_text(encoding="utf-8"))
    severity = yaml.safe_load(severity_path.read_text(encoding="utf-8"))
    validation = _read_json(validation_path)
    if validation.get("valid") is not True or validation.get("worktree_clean") is not True:
        raise ValueError("protocol validation must be valid and from a clean worktree")

    source: dict[str, Any] = {}
    calibration: dict[str, Any] = {}
    frozen_confirmation: dict[str, Any] = {}
    implementation_commits: set[str] = set()
    for dataset in DATASETS:
        source_root = project_root / "results" / "source_reproduction" / dataset
        build_contact_sheet(
            source_root / "visualizations",
            source_root / "contact_sheet.png",
        )
        source[dataset], source_commit = _source_entry(project_root, dataset)
        implementation_commits.add(source_commit)
        calibration[dataset], _ = _pilot_entry(
            project_root,
            dataset,
            result_group="corruption_pilot",
            require_provenance=False,
        )
        frozen_confirmation[dataset], pilot_commit = _pilot_entry(
            project_root,
            dataset,
            result_group="corruption_pilot_frozen",
            require_provenance=True,
        )
        assert pilot_commit is not None
        implementation_commits.add(pilot_commit)

        expected_pilot_sha = severity["calibration"]["evidence"][dataset][
            "pilot_artifact_sha256"
        ]
        measured_pilot_sha = calibration[dataset]["artifacts"]["pilot"]["sha256"]
        if measured_pilot_sha != expected_pilot_sha:
            raise ValueError(
                f"{dataset} calibration pilot hash differs from frozen severity evidence"
            )

    if len(implementation_commits) != 1:
        raise ValueError(
            "Source and frozen-confirmation artifacts were generated by different commits: "
            f"{sorted(implementation_commits)}"
        )
    implementation_commit = implementation_commits.pop()
    if validation.get("current_commit") != implementation_commit:
        raise ValueError("protocol validation and result artifacts use different commits")

    if severity["status"] != "engineering_v1_frozen_after_source_trainval_pilot":
        raise ValueError("unexpected severity-table status")
    if severity["calibration"]["paper_grade_independent_source_holdout_completed"]:
        raise ValueError("manifest expects the disclosed engineering-v1 limitation")

    environment = protocol["environment"]
    locked_files = (
        environment["conda_explicit_lock"],
        environment["python_package_snapshot"],
        environment["declarative_environment"],
        environment["sfs_build_script"],
    )
    return {
        "schema_version": 1,
        "artifact_contract": "cr-sitta-steps-0-3-v1",
        "implementation_commit": implementation_commit,
        "base_commit": protocol["host"]["base_commit"],
        "scope": {
            "steps": [0, 1, 2, 3],
            "severity_freeze": severity["status"],
            "paper_grade_independent_source_holdout_completed": False,
            "known_limitation": severity["calibration"]["limitation"],
        },
        "contracts": {
            "protocol": artifact_record(protocol_path, project_root),
            "severity_table": artifact_record(severity_path, project_root),
            "manifest_builder": artifact_record(
                project_root / "scripts" / "build_artifact_manifest.py",
                project_root,
            ),
            "provisional_severity_archive": artifact_record(
                project_root / severity["calibration"]["provisional_table_archive"],
                project_root,
            ),
            "environment": {
                path: artifact_record(project_root / path, project_root)
                for path in locked_files
            },
        },
        "protocol_validation": artifact_record(validation_path, project_root),
        "source_reproduction": source,
        "severity_calibration_evidence": calibration,
        "frozen_severity_confirmation": frozen_confirmation,
    }


def write_json_atomic(destination: Path, payload: Mapping[str, Any]) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, destination)


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "steps_0_3_manifest.json",
        help="Tracked output manifest (default: artifacts/steps_0_3_manifest.json).",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    manifest = build_manifest(PROJECT_ROOT)
    write_json_atomic(args.output.expanduser().resolve(), manifest)
    print(f"Artifact manifest: {args.output.expanduser().resolve()}")
    print(f"Implementation commit: {manifest['implementation_commit']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
