#!/usr/bin/env python3
"""Issue the global, exact best_miou parity gate for checkpoint-axis v2.

The formal best_pd development axis is blocked until all three v2 producers
(clean Source, Source corruption, and AdaBN) reproduce their immutable v1
best_miou scientific payloads exactly. This signer validates the frozen v1
references, recursively verifies each v2 artifact, compares probabilities and
masks bit-for-bit, and atomically publishes one global receipt envelope.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
from typing import Any, Mapping, Sequence

import numpy as np

_IMPORT_ROOT = Path(__file__).resolve().parents[1]
if str(_IMPORT_ROOT) not in sys.path:
    sys.path.insert(0, str(_IMPORT_ROOT))

from benchmark import checkpoint_axis as checkpoint_axis_api
from benchmark.implementation_dependency_seal import (
    capture_implementation_dependency_seal,
)
from tta.d0_secure_io import (
    ensure_directory_chain_nofollow,
    publish_directory_noreplace,
    read_stable_regular_file,
)


PROJECT_ROOT = _IMPORT_ROOT
DEFAULT_AXIS_CONFIG = PROJECT_ROOT / "configs" / "checkpoint_axis_best_pd_v1.yaml"
DATASETS = checkpoint_axis_api.SUPPORTED_DATASETS
CONDITIONS = (
    "clean_S0",
    "gaussian_noise_S1",
    "gaussian_noise_S3",
    "gaussian_noise_S5",
    "gaussian_blur_S1",
    "gaussian_blur_S3",
    "gaussian_blur_S5",
    "low_contrast_S1",
    "low_contrast_S3",
    "low_contrast_S5",
    "stripe_noise_S1",
    "stripe_noise_S3",
    "stripe_noise_S5",
)


class ParityError(RuntimeError):
    """Raised when a v2 payload differs from its immutable v1 reference."""


def _path(raw: str | Path) -> Path:
    value = Path(raw).expanduser()
    if not value.is_absolute():
        value = PROJECT_ROOT / value
    return value.resolve()


def _sha256(path: Path) -> str:
    return read_stable_regular_file(path).sha256


def _load_json(path: Path) -> dict[str, Any]:
    loaded = json.loads(read_stable_regular_file(path).data)
    if not isinstance(loaded, Mapping):
        raise ParityError(f"expected JSON object: {path}")
    return dict(loaded)


def _load_jsonl(path: Path) -> tuple[dict[str, Any], ...]:
    snapshot = read_stable_regular_file(path)
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(
        snapshot.data.decode("utf-8").splitlines(), start=1
    ):
        if not line.strip():
            raise ParityError(f"blank JSONL line: {path}:{line_number}")
        value = json.loads(line)
        if not isinstance(value, Mapping):
            raise ParityError(f"expected JSON object: {path}:{line_number}")
        records.append(dict(value))
    return tuple(records)


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _require_equal(actual: Any, expected: Any, label: str) -> None:
    if _canonical(actual) != _canonical(expected):
        raise ParityError(f"{label} differs")


def _safe_child(root: Path, raw: str, label: str) -> Path:
    relative = Path(raw)
    if relative.is_absolute() or ".." in relative.parts:
        raise ParityError(f"unsafe {label} path: {raw!r}")
    resolved = (root / relative).resolve()
    if not resolved.is_relative_to(root.resolve()):
        raise ParityError(f"{label} escapes artifact root: {raw!r}")
    return resolved


def _compare_array_files(reference: Path, candidate: Path, label: str) -> int:
    left = np.load(reference, mmap_mode="r", allow_pickle=False)
    right = np.load(candidate, mmap_mode="r", allow_pickle=False)
    if left.shape != right.shape or left.dtype != right.dtype:
        raise ParityError(
            f"{label} shape/dtype differs: {left.shape}/{left.dtype} vs "
            f"{right.shape}/{right.dtype}"
        )
    if not np.array_equal(left, right):
        raise ParityError(f"{label} values are not bit-exact")
    if _sha256(reference) != _sha256(candidate):
        raise ParityError(f"{label} file SHA256 differs despite equal arrays")
    return int(left.size)


def _compare_mask_files(reference: Path, candidate: Path, label: str) -> None:
    if read_stable_regular_file(reference).data != read_stable_regular_file(candidate).data:
        raise ParityError(f"{label} PNG bytes differ")


def _compare_records(
    reference_root: Path,
    candidate_root: Path,
    *,
    probability_mode: str,
    label: str,
) -> dict[str, int]:
    left = _load_jsonl(reference_root / "per_image.jsonl")
    right = _load_jsonl(candidate_root / "per_image.jsonl")
    if len(left) != len(right):
        raise ParityError(f"{label} per-image record count differs")
    probability_values = 0
    mask_files = 0
    integer_records = 0
    for index, (old, new) in enumerate(zip(left, right)):
        for key in ("index", "image_id"):
            _require_equal(new.get(key), old.get(key), f"{label}[{index}].{key}")
        old_mask = _safe_child(reference_root, str(old["prediction_mask"]), "mask")
        new_mask = _safe_child(candidate_root, str(new["prediction_mask"]), "mask")
        _compare_mask_files(old_mask, new_mask, f"{label}[{index}]")
        mask_files += 1
        if "pixel_metrics_at_probability_gt_0_5" in old:
            _require_equal(
                new.get("pixel_metrics_at_probability_gt_0_5"),
                old["pixel_metrics_at_probability_gt_0_5"],
                f"{label}[{index}] integer pixel statistics",
            )
            integer_records += 1
        if probability_mode == "per_image":
            old_probability = _safe_child(
                reference_root, str(old["probability_map"]), "probability map"
            )
            new_probability = _safe_child(
                candidate_root, str(new["probability_map"]), "probability map"
            )
            probability_values += _compare_array_files(
                old_probability, new_probability, f"{label}[{index}] probability"
            )
        if "probability_tensor_raw_sha256" in old:
            _require_equal(
                new.get("probability_tensor_raw_sha256"),
                old["probability_tensor_raw_sha256"],
                f"{label}[{index}].probability_tensor_raw_sha256",
            )
    return {
        "images": len(left),
        "mask_files": mask_files,
        "probability_values": probability_values,
        "integer_records": integer_records,
    }


def _artifact_seals(
    root: Path,
    *,
    expected_axis: checkpoint_axis_api.CheckpointAxis,
) -> dict[str, Any]:
    verified = checkpoint_axis_api.verify_published_artifact(
        root, expected_axis=expected_axis
    )
    return {
        "artifact_manifest.json": verified["manifest_sha256"],
        "COMPLETE.json": verified["complete_sha256"],
        "payload_tree_sha256": verified["payload_tree"]["sha256"],
        "payload_file_count": verified["payload_tree"]["file_count"],
    }


def compare_clean(
    reference_root: Path,
    candidate_root: Path,
    axes: Mapping[str, checkpoint_axis_api.CheckpointAxis],
) -> dict[str, Any]:
    datasets: dict[str, Any] = {}
    totals = {
        "images": 0,
        "mask_files": 0,
        "probability_values": 0,
        "integer_records": 0,
    }
    for dataset in DATASETS:
        old = reference_root / dataset / "best_miou"
        new = candidate_root / dataset
        old_metrics = _load_json(old / "metrics.json")
        new_metrics = _load_json(new / "metrics.json")
        for key in (
            "dataset",
            "evaluated_images",
            "official",
            "official_reported_operating_point",
            "unified",
        ):
            _require_equal(
                new_metrics.get(key), old_metrics.get(key), f"clean {dataset} {key}"
            )
        counts = _compare_records(
            old, new, probability_mode="per_image", label=f"clean/{dataset}"
        )
        datasets[dataset] = {
            **counts,
            "candidate_seals": _artifact_seals(new, expected_axis=axes[dataset]),
        }
        for key in totals:
            totals[key] += counts[key]
    return {"passed": True, "datasets": datasets, "totals": totals}


def _compare_condition(
    old: Path,
    new: Path,
    *,
    label: str,
    metric_keys: Sequence[str],
) -> dict[str, int]:
    old_metrics = _load_json(old / "metrics.json")
    new_metrics = _load_json(new / "metrics.json")
    for key in metric_keys:
        _require_equal(new_metrics.get(key), old_metrics.get(key), f"{label} {key}")
    counts = _compare_records(old, new, probability_mode="shard", label=label)
    counts["probability_values"] = _compare_array_files(
        old / "probabilities_256.npy",
        new / "probabilities_256.npy",
        f"{label} probability shard",
    )
    return counts


def compare_source(
    reference_root: Path,
    candidate_root: Path,
    axes: Mapping[str, checkpoint_axis_api.CheckpointAxis],
) -> dict[str, Any]:
    datasets: dict[str, Any] = {}
    totals = {
        "conditions": 0,
        "images": 0,
        "mask_files": 0,
        "probability_values": 0,
        "integer_records": 0,
    }
    for dataset in DATASETS:
        old_dataset = reference_root / dataset / "best_miou"
        new_dataset = candidate_root / dataset
        old_benchmark = _load_json(old_dataset / "benchmark.json")
        new_benchmark = _load_json(new_dataset / "benchmark.json")
        _require_equal(
            new_benchmark.get("condition_count"), 13, f"Source {dataset} condition count"
        )
        _require_equal(
            new_benchmark.get("evaluated_images_per_condition"),
            old_benchmark.get("evaluated_images_per_condition"),
            f"Source {dataset} evaluated images",
        )
        condition_results: dict[str, Any] = {}
        for key in CONDITIONS:
            counts = _compare_condition(
                old_dataset / "conditions" / key,
                new_dataset / "conditions" / key,
                label=f"Source/{dataset}/{key}",
                metric_keys=("summary", "official", "unified"),
            )
            condition_results[key] = counts
            totals["conditions"] += 1
            for count_key in (
                "images",
                "mask_files",
                "probability_values",
                "integer_records",
            ):
                totals[count_key] += counts[count_key]
        datasets[dataset] = {
            "conditions": condition_results,
            "candidate_seals": _artifact_seals(
                new_dataset, expected_axis=axes[dataset]
            ),
        }
    return {"passed": True, "datasets": datasets, "totals": totals}


def compare_adabn(
    reference_root: Path,
    candidate_root: Path,
    axes: Mapping[str, checkpoint_axis_api.CheckpointAxis],
) -> dict[str, Any]:
    datasets: dict[str, Any] = {}
    totals = {
        "conditions": 0,
        "images": 0,
        "mask_files": 0,
        "probability_values": 0,
        "integer_records": 0,
    }
    for dataset in DATASETS:
        old_dataset = reference_root / dataset
        new_dataset = candidate_root / dataset
        condition_results: dict[str, Any] = {}
        for key in CONDITIONS:
            counts = _compare_condition(
                old_dataset / "conditions" / key,
                new_dataset / "conditions" / key,
                label=f"AdaBN/{dataset}/{key}",
                metric_keys=(
                    "summary",
                    "official",
                    "unified",
                    "source_summary",
                    "deltas_from_source",
                    "source_pre_parity",
                ),
            )
            condition_results[key] = counts
            totals["conditions"] += 1
            for count_key in (
                "images",
                "mask_files",
                "probability_values",
                "integer_records",
            ):
                totals[count_key] += counts[count_key]
        datasets[dataset] = {
            "conditions": condition_results,
            "candidate_seals": _artifact_seals(
                new_dataset, expected_axis=axes[dataset]
            ),
        }
    return {"passed": True, "datasets": datasets, "totals": totals}


def _verify_tree(root: Path, expected: Mapping[str, Any], label: str) -> None:
    ledger = checkpoint_axis_api.artifact_tree_ledger(root)
    _require_equal(
        ledger["algorithm"],
        expected.get("tree_algorithm", checkpoint_axis_api.TREE_ALGORITHM),
        f"{label} tree algorithm",
    )
    _require_equal(
        ledger["file_count"], expected["tree_file_count"], f"{label} file count"
    )
    _require_equal(ledger["sha256"], expected["tree_sha256"], f"{label} tree SHA256")


def _verify_frozen_references(config: Mapping[str, Any]) -> dict[str, Any]:
    references = config["parity_gate"]["references"]
    for kind in ("clean", "source"):
        for dataset in DATASETS:
            record = references[kind][dataset]
            root = _path(record["root"])
            _verify_tree(root, record, f"frozen {kind}/{dataset}")
            if kind == "clean":
                for name, field in (
                    ("metrics.json", "metrics_sha256"),
                    ("per_image.jsonl", "per_image_sha256"),
                ):
                    _require_equal(
                        _sha256(root / name),
                        record[field],
                        f"frozen clean/{dataset}/{name}",
                    )
            else:
                for name, field in (
                    (record["summary_file"], "summary_sha256"),
                    ("artifact_manifest.json", "artifact_manifest_sha256"),
                    ("COMPLETE.json", "complete_sha256"),
                ):
                    _require_equal(
                        _sha256(root / name),
                        record[field],
                        f"frozen Source/{dataset}/{name}",
                    )

    adabn = references["adabn"]
    adabn_root = _path(adabn["root"])
    for path_field, digest_field in (
        ("aggregate_metrics_file", "aggregate_metrics_sha256"),
        ("artifact_manifest_file", "artifact_manifest_sha256"),
        ("complete_file", "complete_sha256"),
    ):
        path = _safe_child(adabn_root, str(adabn[path_field]), path_field)
        _require_equal(_sha256(path), adabn[digest_field], f"frozen AdaBN/{path_field}")
    for dataset in DATASETS:
        _verify_tree(
            adabn_root / dataset,
            adabn["datasets"][dataset],
            f"frozen AdaBN/{dataset}",
        )
    return checkpoint_axis_api.frozen_reference_seal(config)


def _configured_roots(config: Mapping[str, Any], section: str) -> dict[str, Path]:
    raw = config["parity_gate"][section]
    return {
        kind: _path(raw[kind]) for kind in checkpoint_axis_api.ARTIFACT_KINDS
    }


def _axes(
    config: Mapping[str, Any],
    *,
    kind: str,
    candidate_root: Path,
) -> dict[str, checkpoint_axis_api.CheckpointAxis]:
    return {
        dataset: checkpoint_axis_api.resolve_axis(
            config,
            dataset=dataset,
            role="best_miou",
            artifact_kind=kind,
            output_override=candidate_root / dataset,
        )
        for dataset in DATASETS
    }


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        indent=2,
        allow_nan=False,
    ) + "\n"
    with path.open("x", encoding="utf-8") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def _publish_receipt_envelope(
    *,
    receipt: Mapping[str, Any],
    output: Path,
    config: Mapping[str, Any],
    verifier_sha256: str,
    candidates: Mapping[str, Path],
    axes_by_kind: Mapping[
        str, Mapping[str, checkpoint_axis_api.CheckpointAxis]
    ],
) -> dict[str, Any]:
    configured = _path(config["parity_gate"]["receipt_path"])
    if output.resolve() != configured:
        raise ParityError(f"receipt output must be the frozen path: {configured}")
    final = configured.parent
    results_root = (PROJECT_ROOT / "results").resolve()
    if not final.is_relative_to(results_root) or final == results_root:
        raise ParityError(f"unsafe parity envelope destination: {final}")
    if final.exists() or final.is_symlink():
        raise FileExistsError(f"parity envelope already exists: {final}")
    relative_parent = final.parent.relative_to(results_root)
    parent = ensure_directory_chain_nofollow(results_root, relative_parent.parts)
    staging = Path(tempfile.mkdtemp(prefix=f".{final.name}.build-", dir=parent))
    try:
        _write_json(staging / "PARITY_RECEIPT.json", receipt)
        payload_tree = checkpoint_axis_api.artifact_tree_ledger(
            staging, exclude=("artifact_manifest.json", "COMPLETE.json")
        )
        if (
            payload_tree["file_count"] != 1
            or payload_tree["files"][0]["path"] != "PARITY_RECEIPT.json"
        ):
            raise ParityError(
                "parity envelope payload must contain only PARITY_RECEIPT.json"
            )
        manifest = {
            "schema_version": 1,
            "artifact_contract": checkpoint_axis_api.PARITY_CONTRACT,
            "receipt_type": receipt["receipt_type"],
            "axis_config_sha256": receipt["axis_config_sha256"],
            "verifier_sha256": verifier_sha256,
            "payload_tree": payload_tree,
        }
        _write_json(staging / "artifact_manifest.json", manifest)
        manifest_sha256 = _sha256(staging / "artifact_manifest.json")
        complete = {
            "schema_version": 1,
            "complete": True,
            "artifact_contract": checkpoint_axis_api.PARITY_CONTRACT,
            "axis_config_sha256": receipt["axis_config_sha256"],
            "manifest_sha256": manifest_sha256,
            "parity_receipt_sha256": _sha256(staging / "PARITY_RECEIPT.json"),
            "payload_tree_sha256": payload_tree["sha256"],
            "payload_file_count": payload_tree["file_count"],
        }
        _write_json(staging / "COMPLETE.json", complete)

        def guard() -> None:
            current = checkpoint_axis_api.load_axis_config(
                DEFAULT_AXIS_CONFIG, project_root=PROJECT_ROOT
            )
            _require_equal(
                current["_runtime"]["config_sha256"],
                receipt["axis_config_sha256"],
                "axis config at parity publication boundary",
            )
            _require_equal(
                _sha256(Path(__file__).resolve()),
                verifier_sha256,
                "parity verifier at publication boundary",
            )
            _require_equal(
                capture_implementation_dependency_seal(PROJECT_ROOT),
                receipt["authorized_best_pd_live_implementation"]["seal"],
                "authorized best_pd implementation at publication boundary",
            )
            _require_equal(
                checkpoint_axis_api.verify_candidate_producer_observation(
                    project_root=PROJECT_ROOT
                ),
                receipt["candidate_producer_provenance"]["observation"],
                "candidate producer observation at publication boundary",
            )
            _require_equal(
                checkpoint_axis_api.checkpoint_axis_guard_only_patch_audit(
                    project_root=PROJECT_ROOT
                ),
                receipt["authorized_best_pd_live_implementation"][
                    "guard_only_patch_audit"
                ],
                "guard-only patch audit at publication boundary",
            )
            _require_equal(
                _verify_frozen_references(current),
                receipt["frozen_reference_seal"],
                "frozen references at parity publication boundary",
            )
            for kind in checkpoint_axis_api.ARTIFACT_KINDS:
                for dataset in DATASETS:
                    live = _artifact_seals(
                        candidates[kind] / dataset,
                        expected_axis=axes_by_kind[kind][dataset],
                    )
                    _require_equal(
                        live,
                        receipt[kind]["datasets"][dataset]["candidate_seals"],
                        f"candidate {kind}/{dataset} at publication boundary",
                    )

        publish_directory_noreplace(staging, final, pre_rename_guard=guard)
        return checkpoint_axis_api.verify_parity_receipt(config=config)
    except BaseException:
        if (
            staging.exists()
            and staging.parent == final.parent
            and staging.name.startswith(f".{final.name}.build-")
        ):
            shutil.rmtree(staging)
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--axis-config", type=Path, default=DEFAULT_AXIS_CONFIG)
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT
        / "results"
        / "checkpoint_axis_v2_parity"
        / "best_miou"
        / "PARITY_RECEIPT.json",
    )
    return parser


def run(args: argparse.Namespace) -> dict[str, Any]:
    config = checkpoint_axis_api.load_axis_config(
        args.axis_config, project_root=PROJECT_ROOT
    )
    configured_receipt = _path(config["parity_gate"]["receipt_path"])
    output = _path(args.output)
    if output != configured_receipt:
        raise ParityError(f"--output must equal frozen receipt path {configured_receipt}")
    references = {
        "clean": PROJECT_ROOT / "results" / "baseline",
        "source": PROJECT_ROOT / "results" / "source_corruption_benchmark_fixed_split_v1",
        "adabn": PROJECT_ROOT / "results" / "adabn" / "adabn_batch_stats_v1",
    }
    candidates = _configured_roots(config, "candidate_roots")
    verifier_sha256 = _sha256(Path(__file__).resolve())
    implementation_seal = capture_implementation_dependency_seal(PROJECT_ROOT)
    implementation_records = {
        str(record["path"]): record for record in implementation_seal["files"]
    }
    _require_equal(
        implementation_records["scripts/verify_checkpoint_axis_v2_parity.py"][
            "sha256"
        ],
        verifier_sha256,
        "parity verifier implementation dependency",
    )
    candidate_observation = (
        checkpoint_axis_api.verify_candidate_producer_observation(
            project_root=PROJECT_ROOT
        )
    )
    guard_only_patch_audit = (
        checkpoint_axis_api.checkpoint_axis_guard_only_patch_audit(
            project_root=PROJECT_ROOT
        )
    )
    receipt: dict[str, Any] = {
        "schema_version": 1,
        "artifact_contract": checkpoint_axis_api.PARITY_CONTRACT,
        "receipt_type": config["parity_gate"]["required_receipt_type"],
        "status": "passed",
        "passed": True,
        "checkpoint_role": "best_miou",
        "axis_config_path": str(Path(args.axis_config).resolve()),
        "axis_config_sha256": config["_runtime"]["config_sha256"],
        "numeric_tolerance_used": False,
        "comparison_contract": {
            "ordered_ids_exact": True,
            "float32_probability_arrays_bit_exact": True,
            "probability_file_sha256_exact": True,
            "binary_png_bytes_exact": True,
            "integer_sufficient_statistics_exact": True,
            "official_metrics_exact": True,
            "unified_metrics_exact": True,
        },
        "reference_roots": {
            key: str(value.resolve()) for key, value in references.items()
        },
        "candidate_roots": {
            key: str(value.resolve()) for key, value in candidates.items()
        },
        "producer": {
            "path": str(Path(__file__).resolve().relative_to(PROJECT_ROOT)),
            "sha256": verifier_sha256,
        },
        "candidate_producer_provenance": {
            "capture_timing": "post_run_pre_patch",
            "full_runtime_dependency_sealed": False,
            "implementation_identity_asserted": False,
            "adabn_v2_orchestrator_runtime_bound": False,
            "observation": candidate_observation,
        },
        "parity_assertion": {
            "subject": "legacy_best_miou_candidate_scientific_payloads",
            "scientific_payload_bit_exact": True,
            "implementation_identity_asserted": False,
            "applies_to_best_pd_runtime": False,
        },
        "receipt_verifier_implementation": {
            "purpose": "verify_and_publish_exact_output_parity_receipt",
            "seal": implementation_seal,
        },
        "authorized_best_pd_live_implementation": {
            "purpose": "authorize_frozen_best_pd_development_axis_without_retuning",
            "direct_parity_status": "not_run",
            "guard_only_patch_continuity": True,
            "guard_only_patch_audit": guard_only_patch_audit,
            "seal": implementation_seal,
        },
    }
    receipt["frozen_reference_seal"] = _verify_frozen_references(config)
    axes_by_kind = {
        kind: _axes(config, kind=kind, candidate_root=candidates[kind])
        for kind in checkpoint_axis_api.ARTIFACT_KINDS
    }
    receipt["clean"] = compare_clean(
        references["clean"],
        candidates["clean"],
        axes_by_kind["clean"],
    )
    receipt["source"] = compare_source(
        references["source"],
        candidates["source"],
        axes_by_kind["source"],
    )
    receipt["adabn"] = compare_adabn(
        references["adabn"],
        candidates["adabn"],
        axes_by_kind["adabn"],
    )
    return _publish_receipt_envelope(
        receipt=receipt,
        output=output,
        config=config,
        verifier_sha256=verifier_sha256,
        candidates=candidates,
        axes_by_kind=axes_by_kind,
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        receipt = run(args)
    except Exception as error:
        failure = {
            "passed": False,
            "status": "failed_not_published",
            "error_type": type(error).__name__,
            "error": str(error),
        }
        print(json.dumps(failure, ensure_ascii=False, sort_keys=True, allow_nan=False))
        return 2
    print(json.dumps(receipt, ensure_ascii=False, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
