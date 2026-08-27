"""Reproduce the frozen NS-FPN source baseline without test-time adaptation.

The official ``main.py`` remains untouched.  This entry point adds the
research dataset metadata, a single canonical model adapter, both official
and research evaluators, deterministic safety checks, and machine-readable
artifacts required by the CR-SITTA protocol.

The NS-FPN model and research dataset are imported lazily so importing this
module (and running its pure unit tests) does not require the custom SFS CUDA
extension.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import hashlib
import json
import os
from pathlib import Path
import random
import re
import struct
import subprocess
from typing import Any
import warnings

import numpy as np
from PIL import Image, ImageDraw
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader
import yaml

from metrics.irstd_metrics import IRSTDEvaluationProtocol, UnifiedResearchEvaluator
from metrics.official_metric_adapter import OfficialMetricAdapter
from tta.model_adapter import IRSTDModelAdapter


PROJECT_ROOT = Path(__file__).resolve().parent
SEED = 42
IMAGE_SIZE = 256
IMAGENET_MEAN = np.asarray((0.485, 0.456, 0.406), dtype=np.float32)
IMAGENET_STD = np.asarray((0.229, 0.224, 0.225), dtype=np.float32)

DATASET_DEFAULTS = {
    "IRSTD-1k": {
        "split": PROJECT_ROOT / "dataset" / "IRSTD-1k" / "test.txt",
        "checkpoint": PROJECT_ROOT / "weights" / "IRSTD-1k_MSHNet_NSFPN.pkl",
    },
    "NUAA-SIRST": {
        "split": PROJECT_ROOT / "dataset" / "NUAA-SIRST" / "test.txt",
        "checkpoint": PROJECT_ROOT / "weights" / "NUAA-SIRST_MSHNet_NSFPN.pkl",
    },
}

SOURCE_PROVENANCE_PATHS = (
    "configs/protocol.yaml",
    "dataio/research_dataset.py",
    "environment.cr-sitta.yml",
    "environment.linux-64.explicit.txt",
    "metrics/connected_components.py",
    "metrics/irstd_metrics.py",
    "metrics/official_metric_adapter.py",
    "metrics/target_matching.py",
    "requirements.lock.txt",
    "test_source.py",
    "tta/model_adapter.py",
)

STATE_DICT_WRAPPER_KEYS = (
    "state_dict",
    "model_state_dict",
    "model_state",
    "net",
    "model",
)


def positive_integer(raw_value: str) -> int:
    """Argparse converter accepting integers greater than zero."""

    try:
        value = int(raw_value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("value must be an integer") from error
    if value < 1:
        raise argparse.ArgumentTypeError("value must be greater than zero")
    return value


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Deterministic NS-FPN source-baseline reproduction."
    )
    parser.add_argument(
        "--dataset",
        required=True,
        choices=tuple(DATASET_DEFAULTS),
        help="Frozen dataset protocol to evaluate.",
    )
    parser.add_argument(
        "--root",
        required=True,
        type=Path,
        help="Dataset root containing img/label or images/masks.",
    )
    parser.add_argument(
        "--split",
        type=Path,
        default=None,
        help="Explicit split file (default: the repository's official test split).",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Trusted state-dict checkpoint (default: official local weight).",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="Torch device, for example cuda, cuda:1, or cpu.",
    )
    parser.add_argument(
        "--max-images",
        type=positive_integer,
        default=None,
        help="Optional deterministic prefix length for smoke reproduction.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Artifact directory (default: "
            "results/source_reproduction/<dataset>)."
        ),
    )
    parser.add_argument(
        "--image-size",
        type=positive_integer,
        default=IMAGE_SIZE,
        help="Square official evaluation size (default: 256).",
    )
    return parser


def seed_everything(seed: int = SEED) -> None:
    """Fix Python, NumPy, and Torch inference randomness."""

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)


def sha256_file(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    """Return the SHA-256 of a file without reading it all into memory."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def repository_provenance(
    relative_paths: Sequence[str] = SOURCE_PROVENANCE_PATHS,
) -> dict[str, Any]:
    """Bind an artifact to the current Git commit and exact runtime code bytes."""

    head_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    status_lines = subprocess.run(
        ["git", "status", "--porcelain=v1", "--untracked-files=all"],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    file_hashes: dict[str, str] = {}
    for relative_path in relative_paths:
        path = PROJECT_ROOT / relative_path
        if not path.is_file():
            raise FileNotFoundError(f"Provenance file does not exist: {path}")
        file_hashes[relative_path] = sha256_file(path)
    base_commit = (PROJECT_ROOT / "BASE_COMMIT.txt").read_text(encoding="utf-8").strip()
    return {
        "base_commit": base_commit,
        "head_commit": head_commit,
        "worktree_clean": not status_lines,
        "dirty_entries": status_lines,
        "file_sha256": file_hashes,
    }


def compare_with_frozen_source_reference(
    dataset_name: str,
    split_file: Path,
    max_images: int | None,
    measured: Mapping[str, float],
) -> dict[str, Any]:
    """Check a complete official-test run against protocol tolerances."""

    is_full_official_split = (
        split_file == Path(DATASET_DEFAULTS[dataset_name]["split"]).resolve()
        and max_images is None
    )
    if not is_full_official_split:
        return {
            "evaluated": False,
            "passed": None,
            "reason": "comparison requires the complete frozen official test split",
        }

    protocol_path = PROJECT_ROOT / "configs" / "protocol.yaml"
    protocol = yaml.safe_load(protocol_path.read_text(encoding="utf-8"))
    dataset_protocol = protocol["datasets"][dataset_name]
    expected = dataset_protocol["upstream_reported_metrics"]
    tolerance = protocol["source_inference"]["reproduction_tolerance"]
    comparisons = {
        "iou": {
            "measured": float(measured["mean_iou"]),
            "expected": float(expected["iou"]),
            "absolute_tolerance": float(tolerance["iou_absolute"]),
        },
        "pd": {
            "measured": float(measured["detection_probability"]),
            "expected": float(expected["pd"]),
            "absolute_tolerance": float(tolerance["pd_absolute"]),
        },
        "fa_per_pixel_x1e6": {
            "measured": float(measured["false_alarm_per_million_pixels"]),
            "expected": float(expected["fa_per_pixel_x1e6"]),
            "absolute_tolerance": float(
                tolerance["fa_per_pixel_x1e6_absolute"]
            ),
        },
    }
    for comparison in comparisons.values():
        comparison["absolute_error"] = abs(
            comparison["measured"] - comparison["expected"]
        )
        comparison["passed"] = (
            comparison["absolute_error"] <= comparison["absolute_tolerance"]
        )
    passed = all(comparison["passed"] for comparison in comparisons.values())
    return {
        "evaluated": True,
        "passed": passed,
        "protocol_sha256": sha256_file(protocol_path),
        "metrics": comparisons,
    }


def build_unified_evaluation_protocol() -> IRSTDEvaluationProtocol:
    """Construct the evaluator from the versioned protocol instead of defaults."""

    protocol = yaml.safe_load(
        (PROJECT_ROOT / "configs" / "protocol.yaml").read_text(encoding="utf-8")
    )
    postprocessing = protocol["postprocessing"]
    connected = postprocessing["connected_components"]
    matching = postprocessing["target_matching"]
    neighbour_count = int(connected["foreground_connectivity_2d"])
    connectivity = {4: 1, 8: 2}.get(neighbour_count)
    if connectivity is None:
        raise ValueError("protocol connectivity must be 4 or 8 neighbours")
    if matching["assignment"] != "hungarian_minimum_centroid_distance":
        raise ValueError("unsupported protocol target assignment")
    if matching["comparison"] != "strict_less_than" or not matching["one_to_one"]:
        raise ValueError("protocol requires strict one-to-one centroid matching")
    if protocol["data"]["normalization"]["order"] != "after_corruption":
        raise ValueError("protocol requires normalization after corruption")
    if (
        protocol["data"]["mask_handling"]["unified_foreground_rule"]
        != "strict_greater_than_zero"
    ):
        raise ValueError("unsupported unified target-mask foreground rule")
    if not np.array_equal(
        np.asarray(protocol["data"]["normalization"]["mean"], dtype=np.float32),
        IMAGENET_MEAN,
    ) or not np.array_equal(
        np.asarray(protocol["data"]["normalization"]["std"], dtype=np.float32),
        IMAGENET_STD,
    ):
        raise ValueError("runtime ImageNet normalization constants drifted from protocol")
    return IRSTDEvaluationProtocol(
        fixed_probability_threshold=float(postprocessing["operating_threshold"]),
        froc_probability_thresholds=tuple(
            float(value) for value in postprocessing["froc"]["probability_thresholds"]
        ),
        connectivity=connectivity,
        max_centroid_distance=float(matching["centroid_distance_pixels"]),
    )


def state_dict_sha256(state_dict: Mapping[str, Tensor]) -> str:
    """Hash state names, dtypes, shapes, and exact tensor bytes deterministically."""

    digest = hashlib.sha256()
    for name, value in state_dict.items():
        if not isinstance(name, str) or not isinstance(value, Tensor):
            raise TypeError("state_dict must map string names to torch tensors")
        tensor = value.detach().cpu()
        if tensor.layout != torch.strided:
            tensor = tensor.to_dense()
        tensor = tensor.contiguous()

        name_bytes = name.encode("utf-8")
        dtype_bytes = str(tensor.dtype).encode("ascii")
        digest.update(struct.pack(">Q", len(name_bytes)))
        digest.update(name_bytes)
        digest.update(struct.pack(">Q", len(dtype_bytes)))
        digest.update(dtype_bytes)
        digest.update(struct.pack(">Q", tensor.ndim))
        for dimension in tensor.shape:
            digest.update(struct.pack(">q", int(dimension)))
        # Torch cannot reinterpret a zero-dimensional multi-byte tensor as
        # uint8 directly.  Preserve the original shape in the hash above, then
        # temporarily expose one element for byte-level hashing (notably for
        # BatchNorm ``num_batches_tracked`` scalar buffers).
        byte_tensor = tensor.reshape(1) if tensor.ndim == 0 else tensor
        raw_bytes = byte_tensor.view(torch.uint8).numpy().tobytes(order="C")
        digest.update(struct.pack(">Q", len(raw_bytes)))
        digest.update(raw_bytes)
    return digest.hexdigest()


def is_tensor_state_dict(candidate: Any) -> bool:
    """Return whether ``candidate`` is a non-empty tensor state mapping."""

    return (
        isinstance(candidate, Mapping)
        and bool(candidate)
        and all(isinstance(key, str) for key in candidate)
        and all(isinstance(value, Tensor) for value in candidate.values())
    )


def extract_state_dict(checkpoint: Any) -> tuple[Mapping[str, Tensor], str]:
    """Extract a direct or conventionally wrapped state dict without key rewriting."""

    if is_tensor_state_dict(checkpoint):
        return checkpoint, "direct"
    if isinstance(checkpoint, Mapping):
        for wrapper_key in STATE_DICT_WRAPPER_KEYS:
            candidate = checkpoint.get(wrapper_key)
            if is_tensor_state_dict(candidate):
                return candidate, wrapper_key
    raise TypeError(
        "checkpoint must be a direct tensor state_dict or contain one under "
        f"one of {STATE_DICT_WRAPPER_KEYS}"
    )


def load_trusted_checkpoint(
    model: nn.Module,
    checkpoint_path: Path,
) -> str:
    """Load a trusted checkpoint with exact model keys and shapes.

    Returns the wrapper name (``direct`` for the official checkpoints).  No
    ``module.`` prefix stripping or partial loading is performed, so a wrong
    architecture cannot silently pass.
    """

    checkpoint_path = checkpoint_path.expanduser()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint_path}")
    try:
        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=True,
        )
    except TypeError:  # Compatibility with older supported Torch versions.
        checkpoint = torch.load(checkpoint_path, map_location="cpu")

    state_dict, wrapper = extract_state_dict(checkpoint)
    expected_state = model.state_dict()
    expected_keys = set(expected_state)
    supplied_keys = set(state_dict)
    missing = sorted(expected_keys - supplied_keys)
    unexpected = sorted(supplied_keys - expected_keys)
    if missing or unexpected:
        raise RuntimeError(
            "Checkpoint keys do not exactly match the model; "
            f"missing={missing}, unexpected={unexpected}."
        )

    shape_mismatches = {
        key: (tuple(state_dict[key].shape), tuple(expected_state[key].shape))
        for key in expected_state
        if tuple(state_dict[key].shape) != tuple(expected_state[key].shape)
    }
    if shape_mismatches:
        raise RuntimeError(
            "Checkpoint tensor shapes do not exactly match the model: "
            f"{shape_mismatches}."
        )

    model.load_state_dict(state_dict, strict=True)
    return wrapper


def resolve_device(device_name: str) -> torch.device:
    """Resolve a requested device and fail early for unavailable CUDA devices."""

    try:
        device = torch.device(device_name)
    except (RuntimeError, ValueError) as error:
        raise ValueError(f"Invalid Torch device {device_name!r}.") from error
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false.")
        index = torch.cuda.current_device() if device.index is None else device.index
        if index < 0 or index >= torch.cuda.device_count():
            raise RuntimeError(
                f"CUDA device index {index} is unavailable; "
                f"found {torch.cuda.device_count()} device(s)."
            )
        # The upstream SFS extension calls getCurrentCUDAStream() without an
        # explicit device guard. Keep the process-wide CUDA context aligned
        # with the requested tensor/model device, especially for cuda:1+.
        torch.cuda.set_device(index)
    return device


def build_nsfpn_model() -> nn.Module:
    """Import and instantiate NS-FPN only when a real run is requested."""

    from model.MSHNet_NSFPN import MSHNet_NSFPN

    return MSHNet_NSFPN(3)


def build_research_dataset(
    dataset_root: Path,
    split_file: Path,
    dataset_name: str,
    image_size: int,
):
    """Import and construct the standard clean research dataset lazily."""

    from dataio.research_dataset import IRSTDResearchDataset

    return IRSTDResearchDataset(
        dataset_root,
        split_file=split_file,
        image_size=image_size,
        dataset_name=dataset_name,
        corruption="clean",
        severity=0,
        seed=SEED,
    )


def unbatch_metadata(value: Any) -> Any:
    """Undo batch-size-one default collation for JSON metadata."""

    if isinstance(value, Tensor):
        value = value.detach().cpu()
        if value.numel() == 1:
            return value.item()
        if value.ndim >= 1 and value.shape[0] == 1:
            return unbatch_metadata(value[0])
        return value.tolist()
    if isinstance(value, np.ndarray):
        if value.size == 1:
            return value.reshape(-1)[0].item()
        if value.ndim >= 1 and value.shape[0] == 1:
            return unbatch_metadata(value[0])
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): unbatch_metadata(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        converted = [unbatch_metadata(item) for item in value]
        return converted[0] if len(converted) == 1 else converted
    return value


def metadata_from_batch(batch: Mapping[str, Any]) -> dict[str, Any]:
    """Extract the frozen non-tensor sample contract from a collated batch."""

    keys = (
        "image_id",
        "original_size",
        "dataset",
        "corruption",
        "severity",
        "seed",
    )
    missing = [key for key in keys if key not in batch]
    if missing:
        raise KeyError(f"Dataset batch is missing metadata fields: {missing}")
    return {key: unbatch_metadata(batch[key]) for key in keys}


def checked_source_forward(
    adapter: IRSTDModelAdapter,
    image: Tensor,
    mask: Tensor,
    *,
    repeat_exact: bool = False,
) -> tuple[Tensor, bool | None]:
    """Run source inference while enforcing the Step 1 tensor invariants."""

    if image.ndim != 4 or image.shape[0] != 1:
        raise ValueError(f"image must have batch-one [1,C,H,W] shape, got {image.shape}")
    if mask.ndim != 4 or mask.shape[:2] != (1, 1):
        raise ValueError(f"mask must have [1,1,H,W] shape, got {mask.shape}")
    if image.shape[-2:] != mask.shape[-2:]:
        raise ValueError("image and mask spatial sizes differ")
    if not torch.isfinite(image).all():
        raise ValueError("input image contains NaN or Inf")
    if not torch.isfinite(mask).all():
        raise ValueError("input mask contains NaN or Inf")

    with torch.no_grad():
        logits = adapter.forward_logits(image)
        repeat_logits = adapter.forward_logits(image) if repeat_exact else None

    if logits.shape[-2:] != image.shape[-2:]:
        raise ValueError(
            "model output spatial size differs from input: "
            f"input={tuple(image.shape[-2:])}, output={tuple(logits.shape[-2:])}"
        )
    if not torch.isfinite(logits).all():
        raise ValueError("model output contains NaN or Inf")

    repeat_ok: bool | None = None
    if repeat_logits is not None:
        repeat_ok = bool(torch.equal(logits, repeat_logits))
        if not repeat_ok:
            maximum_difference = float(
                (logits - repeat_logits).abs().max().detach().cpu()
            )
            raise RuntimeError(
                "Repeated source logits are not exactly equal; "
                f"maximum absolute difference={maximum_difference}."
            )
    return logits, repeat_ok


def normalised_image_to_uint8(image: Tensor) -> np.ndarray:
    """Convert a single ImageNet-normalised CHW tensor to an RGB uint8 array."""

    if image.ndim == 4 and image.shape[0] == 1:
        image = image[0]
    if image.ndim != 3 or image.shape[0] != 3:
        raise ValueError("visualized image must have shape [3,H,W]")
    array = image.detach().cpu().float().numpy().transpose(1, 2, 0)
    array = array * IMAGENET_STD + IMAGENET_MEAN
    return np.rint(np.clip(array, 0.0, 1.0) * 255.0).astype(np.uint8)


def _labelled_panel(array: np.ndarray, label: str) -> Image.Image:
    panel = Image.fromarray(array, mode="RGB")
    header_height = 22
    canvas = Image.new("RGB", (panel.width, panel.height + header_height), "white")
    canvas.paste(panel, (0, header_height))
    ImageDraw.Draw(canvas).text((5, 4), label, fill="black")
    return canvas


def save_prediction_visualization(
    image: Tensor,
    mask: Tensor,
    logits: Tensor,
    destination: Path,
    *,
    threshold: float = 0.5,
) -> None:
    """Save input, GT, probability, and prediction overlays as one PNG."""

    rgb = normalised_image_to_uint8(image)
    target = mask.detach().cpu().squeeze().numpy() > 0
    probability = torch.sigmoid(logits).detach().cpu().squeeze().numpy()
    prediction = probability > threshold
    if target.shape != rgb.shape[:2] or probability.shape != rgb.shape[:2]:
        raise ValueError("visualization arrays must share one spatial size")

    target_overlay = rgb.copy()
    target_overlay[target] = (
        0.35 * target_overlay[target] + 0.65 * np.array([0, 255, 0])
    ).astype(np.uint8)

    heat = np.empty_like(rgb)
    heat[..., 0] = np.rint(probability * 255.0).astype(np.uint8)
    heat[..., 1] = np.rint(np.sqrt(probability) * 180.0).astype(np.uint8)
    heat[..., 2] = np.rint((1.0 - probability) * 255.0).astype(np.uint8)

    prediction_overlay = rgb.copy()
    prediction_overlay[prediction] = (
        0.35 * prediction_overlay[prediction] + 0.65 * np.array([255, 0, 0])
    ).astype(np.uint8)
    overlap = np.logical_and(prediction, target)
    prediction_overlay[overlap] = np.array([255, 255, 0], dtype=np.uint8)

    panels = (
        _labelled_panel(rgb, "normalized input (restored)"),
        _labelled_panel(target_overlay, "ground truth (green)"),
        _labelled_panel(heat, "sigmoid probability"),
        _labelled_panel(prediction_overlay, "prediction red / overlap yellow"),
    )
    canvas = Image.new(
        "RGB",
        (sum(panel.width for panel in panels), max(panel.height for panel in panels)),
        "white",
    )
    left = 0
    for panel in panels:
        canvas.paste(panel, (left, 0))
        left += panel.width
    destination.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(destination, format="PNG")


def safe_artifact_stem(image_id: Any) -> str:
    """Create a stable filename component from an image identifier."""

    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", str(image_id)).strip("._")
    return stem or "image"


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Tensor):
        return value.detach().cpu().tolist()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serialisable")


def write_json_atomic(destination: Path, payload: Any) -> None:
    """Atomically replace a JSON artifact after fully serialising it."""

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(
            payload,
            handle,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
            default=_json_default,
        )
        handle.write("\n")
    os.replace(temporary, destination)


def write_jsonl_atomic(destination: Path, records: Sequence[Mapping[str, Any]]) -> None:
    """Atomically replace a newline-delimited per-image artifact."""

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                    sort_keys=True,
                    allow_nan=False,
                    default=_json_default,
                )
            )
            handle.write("\n")
    os.replace(temporary, destination)


def _update_official_evaluator(
    evaluator: OfficialMetricAdapter,
    logits: Tensor,
    mask: Tensor,
) -> None:
    """Run unchanged legacy metrics without NumPy 2.x bridge-warning spam."""

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r"__array__ implementation doesn't accept a copy keyword.*",
            category=DeprecationWarning,
        )
        evaluator.update(logits, mask)


def run_source_reproduction(args: argparse.Namespace) -> dict[str, Any]:
    """Execute one deterministic clean Source run and write its artifacts."""

    defaults = DATASET_DEFAULTS[args.dataset]
    dataset_root = args.root.expanduser().resolve()
    split_file = (args.split or defaults["split"]).expanduser().resolve()
    checkpoint_path = (args.checkpoint or defaults["checkpoint"]).expanduser().resolve()
    output_dir = (
        args.output_dir
        or PROJECT_ROOT / "results" / "source_reproduction" / args.dataset
    ).expanduser().resolve()

    if not dataset_root.is_dir():
        raise FileNotFoundError(f"Dataset root does not exist: {dataset_root}")
    if not split_file.is_file():
        raise FileNotFoundError(f"Split file does not exist: {split_file}")
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint_path}")

    seed_everything(SEED)
    device = resolve_device(args.device)
    dataset = build_research_dataset(
        dataset_root,
        split_file,
        args.dataset,
        args.image_size,
    )
    available_images = len(dataset)
    evaluated_images = min(available_images, args.max_images or available_images)
    if evaluated_images < 1:
        raise RuntimeError("Source reproduction requires at least one image.")
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        drop_last=False,
        num_workers=0,
    )

    model = build_nsfpn_model()
    checkpoint_wrapper = load_trusted_checkpoint(model, checkpoint_path)
    model.to(device)
    adapter = IRSTDModelAdapter(model, warm_flag=False)
    adapter.set_source_eval_mode()
    state_hash_before = state_dict_sha256(model.state_dict())

    evaluation_protocol = build_unified_evaluation_protocol()
    official_evaluator = OfficialMetricAdapter(image_size=args.image_size)
    unified_evaluator = UnifiedResearchEvaluator(evaluation_protocol)
    records: list[dict[str, Any]] = []
    visualization_files: list[str] = []
    visualization_target = min(20, evaluated_images)
    repeat_logit_exact = False

    for index, batch in enumerate(loader):
        if index >= evaluated_images:
            break
        if not isinstance(batch, Mapping):
            raise TypeError("Research dataset must yield a dictionary batch.")
        if "image" not in batch or "mask" not in batch:
            raise KeyError("Research dataset batch must contain image and mask.")

        image = batch["image"].to(device, non_blocking=False)
        mask_device = batch["mask"].to(device, non_blocking=False)
        metadata = metadata_from_batch(batch)

        logits, repeat_result = checked_source_forward(
            adapter,
            image,
            mask_device,
            repeat_exact=index == 0,
        )
        if index == 0:
            repeat_logit_exact = bool(repeat_result)

        logits_cpu = logits.detach().cpu()
        mask_cpu = mask_device.detach().cpu()
        _update_official_evaluator(official_evaluator, logits_cpu, mask_cpu)
        unified_evaluator.update_logits(logits_cpu, mask_cpu)

        per_image_official = OfficialMetricAdapter(image_size=args.image_size)
        _update_official_evaluator(per_image_official, logits_cpu, mask_cpu)
        per_image_unified = UnifiedResearchEvaluator(evaluation_protocol)
        per_image_unified.update_logits(logits_cpu, mask_cpu)

        record = {
            "index": index,
            **metadata,
            "input_shape": list(image.shape),
            "logit_shape": list(logits.shape),
            "repeat_logit_exact": repeat_result,
            "official": per_image_official.compute().to_dict(),
            "unified": per_image_unified.compute().to_dict(),
        }
        records.append(record)

        if index < visualization_target:
            visualization_name = (
                f"{index:04d}_{safe_artifact_stem(metadata['image_id'])}.png"
            )
            visualization_path = output_dir / "visualizations" / visualization_name
            save_prediction_visualization(
                image.detach().cpu(),
                mask_cpu,
                logits_cpu,
                visualization_path,
            )
            visualization_files.append(str(visualization_path.relative_to(output_dir)))

    state_hash_after = state_dict_sha256(model.state_dict())
    state_unchanged = state_hash_before == state_hash_after
    if not state_unchanged:
        raise RuntimeError(
            "Source inference mutated model parameters or buffers: "
            f"before={state_hash_before}, after={state_hash_after}."
        )
    if len(records) != evaluated_images:
        raise RuntimeError(
            f"Expected {evaluated_images} images, but processed {len(records)}."
        )
    if len(visualization_files) < visualization_target:
        raise RuntimeError(
            f"Expected {visualization_target} visualizations, saved "
            f"{len(visualization_files)}."
        )

    official_result = official_evaluator.compute()
    unified_result = unified_evaluator.compute()
    reported_operating_point = {
        "mean_iou": official_result.mean_iou,
        "detection_probability": float(official_result.detection_probability[0]),
        "false_alarm_per_million_pixels": float(
            official_result.false_alarm_pixel_rate[0] * 1_000_000.0
        ),
        "raw_logit_threshold": 0.0,
    }
    reproduction_check = compare_with_frozen_source_reference(
        args.dataset,
        split_file,
        args.max_images,
        reported_operating_point,
    )
    if reproduction_check["evaluated"] and not reproduction_check["passed"]:
        raise RuntimeError(
            "Complete Source run falls outside the frozen reproduction tolerances: "
            f"{reproduction_check['metrics']}"
        )
    aggregate = {
        "schema_version": 1,
        "method": "source",
        "repository_provenance": repository_provenance(),
        "dataset": args.dataset,
        "dataset_root": str(dataset_root),
        "split_file": str(split_file),
        "split_sha256": sha256_file(split_file),
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "checkpoint_wrapper": checkpoint_wrapper,
        "device": str(device),
        "seed": SEED,
        "image_size": args.image_size,
        "available_images": available_images,
        "evaluated_images": evaluated_images,
        "max_images": args.max_images,
        "official": official_result.to_dict(),
        "official_reported_operating_point": reported_operating_point,
        "frozen_reference_comparison": reproduction_check,
        "unified": unified_result.to_dict(),
        "checks": {
            "warm_flag": False,
            "batch_size": 1,
            "repeat_logit_exact_first_image": repeat_logit_exact,
            "input_output_spatial_match_all": True,
            "finite_inputs_outputs_all": True,
            "model_state_sha256_before": state_hash_before,
            "model_state_sha256_after": state_hash_after,
            "model_state_unchanged": state_unchanged,
        },
        "visualization_count": len(visualization_files),
        "visualizations": visualization_files,
    }

    write_jsonl_atomic(output_dir / "per_image.jsonl", records)
    write_json_atomic(output_dir / "metrics.json", aggregate)
    return aggregate


def main(argv: Sequence[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    result = run_source_reproduction(args)
    output_dir = (
        args.output_dir
        or PROJECT_ROOT / "results" / "source_reproduction" / args.dataset
    ).expanduser().resolve()
    reported = result["official_reported_operating_point"]
    print(f"Artifacts: {output_dir}")
    print(
        "Official source: "
        f"mIoU={reported['mean_iou']:.6f}, "
        f"Pd={reported['detection_probability']:.6f}, "
        f"Fa(x1e-6)={reported['false_alarm_per_million_pixels']:.6f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
