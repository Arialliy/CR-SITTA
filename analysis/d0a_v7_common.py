"""Small shared, train-only data and artifact helpers for v7 diagnostics.

This is not a training runner. No test ID file or test payload is opened.
The existing Pilot64 manifest supplies the previously verified disjointness.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random
from typing import Any, Iterable

import yaml

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs/cr_sitta_d0a_v7_diagnostics_v1.yaml"


def absolute(path: str | Path) -> Path:
    path = Path(path)
    return (path if path.is_absolute() else ROOT / path).resolve()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with absolute(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def binding(path: str | Path) -> dict[str, str]:
    path = absolute(path)
    return {"path": str(path), "sha256": sha256_file(path)}


def read_config(path: str | Path = DEFAULT_CONFIG) -> dict[str, Any]:
    config = yaml.safe_load(absolute(path).read_text(encoding="utf-8"))
    if config["protocol_id"] != "cr-sitta-d0a-v7-diagnostics-v1":
        raise ValueError("unexpected diagnostics protocol")
    for key in ("resume_full_training", "train_ipma", "authorize_d0b_or_ipma",
                "new_test_payload_access", "paper_result"):
        if config["scope"].get(key) is not False:
            raise ValueError(f"forbidden diagnostic scope: {key}")
    if config["sampling"]["count"] != 64 or config["sampling"]["workers"] != 0:
        raise ValueError("diagnostics require frozen Pilot64 with workers=0")
    return config


def _ids(path: str | Path) -> list[str]:
    identifiers = absolute(path).read_text(encoding="utf-8").splitlines()
    if not identifiers or len(identifiers) != len(set(identifiers)):
        raise ValueError("empty or duplicate IDs")
    for identifier in identifiers:
        p = Path(identifier)
        if not identifier or p.is_absolute() or ".." in p.parts or p.suffix:
            raise ValueError(f"unsafe or noncanonical ID: {identifier!r}")
    return identifiers


def load_pilot_records(dataset: str, config: dict[str, Any]) -> list[dict[str, str]]:
    spec = config["datasets"][dataset]
    for key in ("train_split", "pilot_ids"):
        if sha256_file(spec[key]) != spec[f"{key}_sha256"]:
            raise ValueError(f"{dataset} {key} hash mismatch")
    train, pilot = _ids(spec["train_split"]), _ids(spec["pilot_ids"])
    if len(train) != spec["train_images"] or len(pilot) != 64 or not set(pilot) <= set(train):
        raise ValueError("Pilot64 must be a unique subset of the fixed official train split")
    sampling = config["sampling"]
    if sha256_file(sampling["pilot_manifest"]) != sampling["pilot_manifest_sha256"]:
        raise ValueError("frozen Pilot64 manifest hash mismatch")
    manifest = json.loads(absolute(sampling["pilot_manifest"]).read_text())
    entry = manifest["datasets"][dataset]
    if entry["checks"]["output_test_overlap_count"] != 0:
        raise ValueError("Pilot64 manifest reports test overlap")
    if entry["output"]["file_sha256"] != spec["pilot_ids_sha256"]:
        raise ValueError("pilot manifest does not bind the selected IDs")
    if entry["train_split"]["sha256"] != spec["train_split_sha256"]:
        raise ValueError("pilot manifest does not bind official train IDs")
    root = absolute(spec["root"])
    records = []
    for identifier in pilot:
        record = {"dataset": dataset, "image_id": identifier}
        for field, directory in (("image_path", "images"), ("mask_path", "masks")):
            path = (root / directory / f"{identifier}.png").resolve(strict=True)
            if not path.is_relative_to((root / directory).resolve()):
                raise ValueError("train payload escapes its declared directory")
            record[field] = str(path)
        records.append(record)
    return records


def tensor_sha256(tensor: Any) -> str:
    tensor = tensor.detach().cpu().contiguous()
    digest = hashlib.sha256(str(tensor.dtype).encode())
    digest.update(json.dumps(list(tensor.shape)).encode())
    digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def load_sample(record: dict[str, str], view: str, config: dict[str, Any],
                access: dict[str, int] | None = None) -> tuple[Any, Any, dict[str, Any]]:
    """Load only an allowlisted training ID, with one deterministic crop per ID."""
    import numpy as np
    from PIL import Image
    import torch
    from torchvision.transforms.functional import to_tensor
    from train_fixed_split import FixedSplitIRSTDDataset, IMAGENET_NORMALIZE

    if view not in config["sampling"]["views"]:
        raise ValueError("unknown diagnostic view")
    spec = config["datasets"][record["dataset"]]
    identifier = record["image_id"]
    if identifier not in _ids(spec["pilot_ids"]) or identifier not in _ids(spec["train_split"]):
        raise ValueError("attempted non-Pilot64 train payload access")
    for field, directory in (("image_path", "images"), ("mask_path", "masks")):
        expected = absolute(spec["root"]) / directory / f"{identifier}.png"
        if absolute(record[field]) != expected.resolve():
            raise ValueError("payload path does not match allowlisted training ID")
    if access is not None:
        for name in ("train_image_opens", "train_mask_opens", "test_image_opens",
                     "test_mask_opens", "validation_image_opens", "validation_mask_opens"):
            access.setdefault(name, 0)
    with Image.open(record["image_path"]) as handle:
        image = handle.convert("RGB")
        if access is not None:
            access["train_image_opens"] += 1
    with Image.open(record["mask_path"]) as handle:
        mask = handle.convert("L")
        if access is not None:
            access["train_mask_opens"] += 1
    descriptor = json.dumps([config["augmentation_seed_namespace"],
                             config["global_seed"], record["dataset"], identifier],
                            separators=(",", ":"))
    seed = int.from_bytes(hashlib.sha256(descriptor.encode()).digest()[:8], "big") % (2**32)
    if view == "train_crop_224":
        state = random.getstate()
        try:
            random.seed(seed)
            # The original transform only requires base_size and crop_size;
            # constructing the dataset does not open any image or test path.
            transform = FixedSplitIRSTDDataset(absolute(spec["root"]), absolute(spec["pilot_ids"]),
                training=True, base_size=config["sampling"]["base_size"],
                crop_size=config["sampling"]["crop_size"])
            image, mask = transform._train_transform(image, mask)
        finally:
            random.setstate(state)
    else:
        size = config["sampling"]["base_size"]
        image = image.resize((size, size), Image.Resampling.BILINEAR)
        mask = mask.resize((size, size), Image.Resampling.NEAREST)
    normalized = IMAGENET_NORMALIZE(to_tensor(image))
    target = torch.from_numpy(np.asarray(mask, dtype=np.float32).copy())[None] / 255.0
    return normalized, target, {"augmentation_seed": seed if view == "train_crop_224" else None,
        "input_tensor_sha256": tensor_sha256(normalized), "target_tensor_sha256": tensor_sha256(target),
        "view": view, "historical_training_random_stream_replay": False}


def runtime_bindings(config_path: str | Path, extra_paths: Iterable[str | Path],
                     records: Iterable[dict[str, str]]) -> list[dict[str, str]]:
    config = read_config(config_path)
    paths = [absolute(config_path), Path(__file__), absolute(config["design_document"]),
        absolute(config["sampling"]["pilot_manifest"]),
        ROOT / "configs/cr_sitta_d0a_train_v2.yaml", ROOT / "train_fixed_split.py",
        ROOT / "train_cr_sitta_d0a.py", ROOT / "tta/deteriorations/fourier_low_mask.py",
        ROOT / "tta/deteriorations/high_frequency_noise.py", ROOT / "tta/deteriorations/image_space.py"]
    paths.extend(absolute(p) for p in extra_paths)
    datasets = set()
    for record in records:
        paths.extend([absolute(record["image_path"]), absolute(record["mask_path"])])
        datasets.add(record["dataset"])
    for dataset in datasets:
        paths.extend(absolute(config["datasets"][dataset][key]) for key in ("train_split", "pilot_ids"))
    return [binding(path) for path in sorted(set(paths))]


def reserve_output(path: str | Path) -> None:
    path = absolute(path)
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"refusing to overwrite full or partial diagnostics: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.mkdir()


def write_json_new(path: str | Path, obj: Any) -> None:
    with absolute(path).open("x", encoding="utf-8") as handle:
        json.dump(obj, handle, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        handle.write("\n")


def write_jsonl_new(path: str | Path, rows: Iterable[Any]) -> None:
    with absolute(path).open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n")


def freeze_run(output: str | Path, contract: dict[str, Any],
               bindings: list[dict[str, str]]) -> None:
    output = absolute(output)
    if not output.exists():
        reserve_output(output)
    elif any(output.iterdir()):
        raise FileExistsError("freeze requires a new empty output directory")
    write_json_new(output / "RUN_CONTRACT.json", {**contract,
        "created_at_utc": datetime.now(timezone.utc).isoformat()})
    write_json_new(output / "PRE_RUN_FREEZE.json", {
        "contract": binding(output / "RUN_CONTRACT.json"), "input_bindings": bindings,
        "frozen_before_payload_diagnostics": True})
    (output / "RUN_CONTRACT.json").chmod(0o444)
    (output / "PRE_RUN_FREEZE.json").chmod(0o444)


def complete_run(output: str | Path, summary: dict[str, Any]) -> None:
    output = absolute(output)
    freeze = json.loads((output / "PRE_RUN_FREEZE.json").read_text())
    for item in freeze["input_bindings"] + [freeze["contract"]]:
        if sha256_file(item["path"]) != item["sha256"]:
            raise RuntimeError(f"frozen input changed: {item['path']}")
    write_json_new(output / "summary.json", summary)
    files = {str(p.relative_to(output)): sha256_file(p)
             for p in sorted(output.rglob("*")) if p.is_file()}
    write_json_new(output / "artifact_manifest.json", {"schema_version": 1, "files": files})
    write_json_new(output / "COMPLETE.json", {"schema_version": 1, "complete": True,
        "development_only": True, "paper_result": False, "new_training_authorized": False,
        "artifact_manifest": binding(output / "artifact_manifest.json"),
        "completed_at_utc": datetime.now(timezone.utc).isoformat()})
