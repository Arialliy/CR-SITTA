"""Materialize the frozen 13-condition fixed-test corruption input cache."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import hashlib
import json
import os
from pathlib import Path
import time
from typing import Any

import numpy as np
import yaml

from corruptions.corruption_protocol import load_severity_table
from corruptions.infrared_corruptions import apply_corruption
from dataio.corruption_cache import (
    TensorSequenceHasher,
    condition_key,
    ordered_ids_sha256,
    sha256_file,
)
from dataio.research_dataset import IRSTDResearchDataset, read_split_ids
import test_source as source_runner


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_PROTOCOL = PROJECT_ROOT / "configs" / "source_corruption_benchmark_fixed_splits.yaml"
PROVENANCE_PATHS = (
    "configs/source_corruption_benchmark_fixed_splits.yaml",
    "corruptions/corruption_protocol.py",
    "corruptions/infrared_corruptions.py",
    "corruptions/severity_tables.yaml",
    "corruptions/severity_tables_round_02_candidate.yaml",
    "dataio/corruption_cache.py",
    "dataio/research_dataset.py",
    "materialize_corruption_cache.py",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset", required=True, choices=("IRSTD-1K", "NUAA-SIRST", "NUDT-SIRST")
    )
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser


def _project_path(raw: str | Path) -> Path:
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def _load_protocol(path: Path, dataset_name: str) -> tuple[dict[str, Any], dict[str, Any]]:
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, Mapping):
        raise TypeError("benchmark protocol must be a YAML mapping")
    protocol = dict(loaded)
    if protocol.get("protocol_id") != "nsfpn-source-corruption-benchmark-fixed-splits-v1":
        raise ValueError("unexpected source corruption benchmark protocol_id")
    datasets = protocol.get("datasets", {})
    if dataset_name not in datasets:
        raise ValueError(f"dataset {dataset_name!r} is absent from benchmark protocol")
    return protocol, dict(datasets[dataset_name])


def _conditions(protocol: Mapping[str, Any]) -> tuple[tuple[str, int], ...]:
    conditions = tuple(
        (str(corruption), int(severity))
        for corruption, severity in protocol["corruption"]["ordered_conditions"]
    )
    expected = (
        ("clean", 0),
        ("gaussian_noise", 1),
        ("gaussian_noise", 3),
        ("gaussian_noise", 5),
        ("gaussian_blur", 1),
        ("gaussian_blur", 3),
        ("gaussian_blur", 5),
        ("low_contrast", 1),
        ("low_contrast", 3),
        ("low_contrast", 5),
        ("stripe_noise", 1),
        ("stripe_noise", 3),
        ("stripe_noise", 5),
    )
    if conditions != expected:
        raise ValueError("formal cache requires the exact ordered 13-condition contract")
    return conditions


def _source_manifests(root: Path, image_ids: Sequence[str]) -> dict[str, str]:
    combined = hashlib.sha256()
    images = hashlib.sha256()
    masks = hashlib.sha256()
    for image_id in sorted(image_ids):
        image_path = root / "images" / f"{image_id}.png"
        mask_path = root / "masks" / f"{image_id}.png"
        image_hash = sha256_file(image_path)
        mask_hash = sha256_file(mask_path)
        combined.update(f"{image_id}\t{image_hash}\t{mask_hash}\n".encode("utf-8"))
        images.update(f"{image_id}\t{image_hash}\n".encode("utf-8"))
        masks.update(f"{image_id}\t{mask_hash}\n".encode("utf-8"))
    return {
        "combined_sha256": combined.hexdigest(),
        "images_sha256": images.hexdigest(),
        "masks_sha256": masks.hexdigest(),
        "algorithm": "sorted-id-image_sha256-mask_sha256-lf-v1",
    }


def _write_memmap_atomic(
    destination: Path,
    *,
    shape: tuple[int, ...],
) -> tuple[Path, np.memmap]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(f".{destination.name}.{os.getpid()}.partial")
    if partial.exists():
        raise FileExistsError(f"partial cache file already exists: {partial}")
    mapped = np.lib.format.open_memmap(
        partial,
        mode="w+",
        dtype=np.dtype("<f4"),
        shape=shape,
        fortran_order=False,
        version=(2, 0),
    )
    return partial, mapped


def _hash_memmap_records(
    path: Path,
    image_ids: Sequence[str],
) -> str:
    values = np.load(path, mmap_mode="r", allow_pickle=False)
    if len(values) != len(image_ids):
        raise ValueError("materialized tensor first dimension does not match IDs")
    digest = TensorSequenceHasher()
    for image_id, value in zip(image_ids, values):
        digest.update(image_id, value)
    return digest.hexdigest()


def materialize(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    protocol_path = args.protocol.expanduser().resolve()
    protocol, dataset_contract = _load_protocol(protocol_path, args.dataset)
    protocol_hash = sha256_file(protocol_path)
    conditions = _conditions(protocol)
    root = _project_path(dataset_contract["root"])
    split = _project_path(dataset_contract["test_split"])
    output_root = _project_path(protocol["materialized_cache"]["root"])
    final_output_dir = (
        args.output_dir or output_root / args.dataset
    ).expanduser().resolve()
    if final_output_dir.exists():
        raise FileExistsError(
            f"cache destination already exists; refusing overwrite: {final_output_dir}"
        )
    final_output_dir.parent.mkdir(parents=True, exist_ok=True)
    output_dir = final_output_dir.with_name(
        f".{final_output_dir.name}.build-{os.getpid()}"
    )
    if output_dir.exists():
        raise FileExistsError(f"cache staging directory already exists: {output_dir}")
    output_dir.mkdir(parents=False)
    if sha256_file(split) != dataset_contract["test_split_sha256"]:
        raise ValueError("fixed test split SHA256 drifted")
    raw_ids = read_split_ids(split)
    image_ids = tuple(Path(value).with_suffix("").as_posix() for value in raw_ids)
    if len(image_ids) != int(dataset_contract["test_images"]):
        raise ValueError("fixed test split count drifted")
    if ordered_ids_sha256(image_ids) != dataset_contract["ordered_test_ids_sha256"]:
        raise ValueError("fixed test ordered ID hash drifted")
    source_manifests = _source_manifests(root, image_ids)
    for actual_key, expected_key in (
        ("combined_sha256", "test_source_manifest_sha256"),
        ("images_sha256", "test_image_manifest_sha256"),
        ("masks_sha256", "test_mask_manifest_sha256"),
    ):
        if source_manifests[actual_key] != dataset_contract[expected_key]:
            raise ValueError(f"fixed test source bytes drifted: {actual_key}")

    corruption_contract = protocol["corruption"]
    severity_path = _project_path(corruption_contract["severity_table"])
    if sha256_file(severity_path) != corruption_contract["severity_table_sha256"]:
        raise ValueError("frozen severity table SHA256 drifted")
    severity_table = load_severity_table(severity_path)
    if not severity_table.frozen or not severity_table.calibration_completed:
        raise ValueError("formal cache requires a frozen, calibrated severity table")
    pilot_report_path = _project_path(corruption_contract["pilot_report"])
    if sha256_file(pilot_report_path) != corruption_contract["pilot_report_sha256"]:
        raise ValueError("Pilot calibration report SHA256 drifted")
    pilot_report = json.loads(pilot_report_path.read_text(encoding="utf-8"))
    if pilot_report.get("decision") != "freeze":
        raise ValueError("Pilot report did not authorize severity freeze")

    seed = int(corruption_contract["seed"])
    image_size = int(protocol["preprocessing"]["image_resize"]["size"][0])
    if image_size != 256:
        raise ValueError("formal cache requires 256x256 preprocessing")
    target_path = output_dir / "targets.npy"
    target_partial: Path | None = None
    target_map: np.memmap | None = None
    target_tensor_hash: str | None = None
    original_sizes: list[list[int]] = []
    condition_records = []
    files: dict[str, dict[str, Any]] = {}
    reference_mask_hash: str | None = None

    for condition_index, (corruption, severity) in enumerate(conditions):
        key = condition_key(corruption, severity)
        destination = output_dir / "conditions" / f"{key}.npy"
        partial, image_map = _write_memmap_atomic(
            destination, shape=(len(image_ids), 3, image_size, image_size)
        )
        if condition_index == 0:
            target_partial, target_map = _write_memmap_atomic(
                target_path, shape=(len(image_ids), 1, image_size, image_size)
            )
        dataset = IRSTDResearchDataset(
            root,
            split_file=split,
            image_size=image_size,
            dataset_name=args.dataset,
            corruption=corruption,
            severity=severity,
            seed=seed,
            corruption_transform=None if corruption == "clean" else apply_corruption,
        )
        image_hasher = TensorSequenceHasher()
        mask_hasher = TensorSequenceHasher()
        observed_ids = []
        condition_start = time.perf_counter()
        for index in range(len(dataset)):
            sample = dataset[index]
            image_id = str(sample["image_id"])
            if image_id != image_ids[index]:
                raise RuntimeError("dataset order changed during cache materialization")
            image = sample["image"].detach().cpu().contiguous().numpy()
            mask = sample["mask"].detach().cpu().contiguous().numpy()
            if image.dtype != np.float32 or mask.dtype != np.float32:
                raise TypeError("cache materialization requires exact float32 tensors")
            image_map[index] = image
            image_hasher.update(image_id, image)
            mask_hasher.update(image_id, mask)
            observed_ids.append(image_id)
            if condition_index == 0:
                assert target_map is not None
                target_map[index] = mask
                original_sizes.append([int(value) for value in sample["original_size"]])
        if tuple(observed_ids) != image_ids:
            raise RuntimeError("condition did not materialize the exact fixed test IDs")
        image_map.flush()
        del image_map
        os.replace(partial, destination)
        generated_mask_hash = mask_hasher.hexdigest()
        if reference_mask_hash is None:
            reference_mask_hash = generated_mask_hash
            assert target_map is not None and target_partial is not None
            target_map.flush()
            del target_map
            target_map = None
            os.replace(target_partial, target_path)
            target_tensor_hash = _hash_memmap_records(target_path, image_ids)
            if target_tensor_hash != reference_mask_hash:
                raise RuntimeError("target cache bytes differ from generated target tensors")
            relative_target = str(target_path.relative_to(output_dir))
            files[relative_target] = {
                "sha256": sha256_file(target_path),
                "bytes": target_path.stat().st_size,
            }
        elif generated_mask_hash != reference_mask_hash:
            raise RuntimeError(f"GT masks changed under condition {key}")
        stored_tensor_hash = _hash_memmap_records(destination, image_ids)
        if stored_tensor_hash != image_hasher.hexdigest():
            raise RuntimeError(f"stored cache tensor bytes changed for {key}")
        replay_dataset = IRSTDResearchDataset(
            root,
            split_file=split,
            image_size=image_size,
            dataset_name=args.dataset,
            corruption=corruption,
            severity=severity,
            seed=seed,
            corruption_transform=None if corruption == "clean" else apply_corruption,
        )
        replay_image_hasher = TensorSequenceHasher()
        replay_mask_hasher = TensorSequenceHasher()
        for index in range(len(replay_dataset)):
            sample = replay_dataset[index]
            image_id = str(sample["image_id"])
            if image_id != image_ids[index]:
                raise RuntimeError("replay dataset order changed")
            replay_image_hasher.update(image_id, sample["image"])
            replay_mask_hasher.update(image_id, sample["mask"])
        if replay_image_hasher.hexdigest() != stored_tensor_hash:
            raise RuntimeError(f"condition {key} was not bit-exact on independent replay")
        if replay_mask_hasher.hexdigest() != reference_mask_hash:
            raise RuntimeError(f"condition {key} replay changed GT masks")
        relative = str(destination.relative_to(output_dir))
        file_hash = sha256_file(destination)
        files[relative] = {"sha256": file_hash, "bytes": destination.stat().st_size}
        condition_records.append(
            {
                "index": condition_index,
                "key": key,
                "corruption": corruption,
                "severity": severity,
                "path": relative,
                "shape": [len(image_ids), 3, image_size, image_size],
                "dtype": "float32",
                "tensor_sequence_sha256": stored_tensor_hash,
                "gt_mask_tensor_sequence_sha256": generated_mask_hash,
                "file_sha256": file_hash,
                "independent_replay_exact": True,
                "runtime_seconds": time.perf_counter() - condition_start,
            }
        )

    assert target_tensor_hash is not None and reference_mask_hash is not None
    content_payload = {
        "protocol_sha256": protocol_hash,
        "dataset": args.dataset,
        "ordered_ids_sha256": ordered_ids_sha256(image_ids),
        "targets_sha256": files["targets.npy"]["sha256"],
        "conditions": [
            [record["key"], record["file_sha256"]] for record in condition_records
        ],
    }
    content_hash = hashlib.sha256(
        json.dumps(content_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    manifest = {
        "schema_version": 1,
        "cache_format": "nsfpn-materialized-corruption-cache-v1",
        "protocol_id": protocol["protocol_id"],
        "protocol_path": str(protocol_path),
        "protocol_sha256": protocol_hash,
        "repository_provenance": source_runner.repository_provenance(PROVENANCE_PATHS),
        "dataset": args.dataset,
        "dataset_root": str(root),
        "split_file": str(split),
        "split_sha256": sha256_file(split),
        "source_manifests": source_manifests,
        "seed": seed,
        "image_ids": list(image_ids),
        "ordered_ids_sha256": ordered_ids_sha256(image_ids),
        "original_sizes": original_sizes,
        "condition_count": len(condition_records),
        "conditions": condition_records,
        "targets": {
            "path": "targets.npy",
            "shape": [len(image_ids), 1, image_size, image_size],
            "dtype": "float32",
            "tensor_sequence_sha256": target_tensor_hash,
            "file_sha256": files["targets.npy"]["sha256"],
        },
        "files": files,
        "checks": {
            "full_fixed_test_split": True,
            "all_conditions_same_ordered_ids": True,
            "gt_mask_hash_identical_across_conditions": True,
            "stored_tensors_match_generated_tensors": True,
            "independent_regeneration_exact_all_conditions": True,
            "corruption_before_normalization": True,
            "frozen_severity_table_verified": True,
            "consumer_corruption_regeneration_forbidden": True,
        },
        "cache_content_sha256": content_hash,
        "runtime_seconds": time.perf_counter() - started,
    }
    manifest_path = output_dir / "manifest.json"
    source_runner.write_json_atomic(manifest_path, manifest)
    completion = {
        "complete": True,
        "dataset": args.dataset,
        "protocol_sha256": protocol_hash,
        "cache_content_sha256": content_hash,
        "manifest_sha256": sha256_file(manifest_path),
    }
    source_runner.write_json_atomic(output_dir / "COMPLETE.json", completion)
    os.replace(output_dir, final_output_dir)
    manifest["published_output_dir"] = str(final_output_dir)
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    manifest = materialize(args)
    print(
        f"{manifest['dataset']}: materialized {manifest['condition_count']} conditions "
        f"for {len(manifest['image_ids'])} fixed-test images in "
        f"{manifest['runtime_seconds']:.2f}s"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
