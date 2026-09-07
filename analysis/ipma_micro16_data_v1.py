"""Separated image-only/fit-label access and frozen D0 extraction for v9 R4.

Only the first 16 IDs of the original NUDT train Pilot64 are image-eligible.
The first eight can supply outer fit labels; the following eight cannot open
labels through this module.  No test/validation ID file is consulted.  Original
v7 data helpers and the original training transform are reused unchanged.
"""
from __future__ import annotations

from collections.abc import Mapping, MutableMapping
import hashlib
import json
import random
from typing import Any

import numpy as np
from PIL import Image
import torch
from torch import Tensor, nn
from torchvision.transforms.functional import to_tensor

from analysis import d0a_v7_common as legacy
from train_fixed_split import FixedSplitIRSTDDataset, IMAGENET_NORMALIZE


DATASET = "NUDT-SIRST"
VIEW = "train_crop_224"
ACCESS_FIELDS = (
    "train_image_opens", "train_mask_opens", "fit_mask_opens", "check_mask_opens",
    "test_split_reads", "test_image_opens", "test_mask_opens",
    "validation_split_reads", "validation_image_opens", "validation_mask_opens",
)
FORBIDDEN_ACCESS_FIELDS = ("check_mask_opens", *ACCESS_FIELDS[4:])


def _access(access: MutableMapping[str, int]) -> None:
    if not isinstance(access, MutableMapping):
        raise TypeError("an explicit mutable access ledger is required")
    for name in ACCESS_FIELDS:
        access.setdefault(name, 0)
        if type(access[name]) is not int or access[name] < 0:
            raise ValueError(f"invalid access count: {name}")
    if any(access[name] != 0 for name in FORBIDDEN_ACCESS_FIELDS):
        raise ValueError("forbidden check/test/validation payload access in ledger")


def _validated_record(
    record: Mapping[str, str], config: Mapping[str, Any], *, fit_only: bool,
) -> tuple[dict[str, str], int]:
    if not isinstance(record, Mapping) or record.get("dataset") != DATASET:
        raise ValueError("IPMA micro16 only allows the frozen NUDT Pilot64")
    if config.get("protocol_id") != "cr-sitta-d0a-v7-diagnostics-v1":
        raise ValueError("original v7 input replay config is required")
    if config.get("global_seed") != 42 or config.get("augmentation_seed_namespace") != "cr-sitta-d0a-v7-diagnostic-per-image-crop-v1":
        raise ValueError("original v7 crop seed policy changed")
    sampling = config["sampling"]
    if (sampling.get("count"), sampling.get("workers"), sampling.get("base_size"), sampling.get("crop_size")) != (64, 0, 256, 224):
        raise ValueError("original Pilot64/view dimensions changed")
    if VIEW not in sampling.get("views", ()):
        raise ValueError("original crop view missing")
    # This checks train/Pilot hashes, their counts, the sealed disjointness
    # receipt, canonical paths, and path containment. It never opens PNGs.
    records = legacy.load_pilot_records(DATASET, config)
    count = 8 if fit_only else 16
    candidates = {entry["image_id"]: (entry, index) for index, entry in enumerate(records[:count])}
    if record.get("image_id") not in candidates:
        raise ValueError("not an allowed meta-fit train8 ID" if fit_only else "not an allowed micro16 train ID")
    canonical, index = candidates[record["image_id"]]
    for name in ("image_path", "mask_path"):
        if name not in record or legacy.absolute(record[name]) != legacy.absolute(canonical[name]):
            raise ValueError(f"payload path does not match official train/Pilot allowlist: {name}")
    return canonical, index


def _augmentation_seed(record: Mapping[str, str], config: Mapping[str, Any]) -> int:
    descriptor = json.dumps(
        [config["augmentation_seed_namespace"], config["global_seed"], record["dataset"], record["image_id"]],
        separators=(",", ":"),
    )
    return int.from_bytes(hashlib.sha256(descriptor.encode()).digest()[:8], "big") % (2**32)


def _replay_transform(image: Image.Image, mask: Image.Image,
                      record: Mapping[str, str], config: Mapping[str, Any]):
    spec, sampling = config["datasets"][DATASET], config["sampling"]
    seed = _augmentation_seed(record, config)
    state = random.getstate()
    try:
        random.seed(seed)
        transform = FixedSplitIRSTDDataset(
            legacy.absolute(spec["root"]), legacy.absolute(spec["pilot_ids"]),
            training=True, base_size=sampling["base_size"], crop_size=sampling["crop_size"],
        )
        transformed_image, transformed_mask = transform._train_transform(image, mask)
    finally:
        random.setstate(state)
    return transformed_image, transformed_mask, seed


def load_observation(
    record: Mapping[str, str], legacy_config: Mapping[str, Any], access: MutableMapping[str, int],
) -> tuple[Tensor, dict[str, Any]]:
    """Read one allowed image and reproduce v7 image geometry without a label.

    The all-zero PIL mask is only a transform-interface placeholder. It is
    neither loaded from disk nor returned, and cannot be used as an outer GT.
    """
    _access(access)
    canonical, index = _validated_record(record, legacy_config, fit_only=False)
    with Image.open(canonical["image_path"]) as handle:
        access["train_image_opens"] += 1
        image = handle.convert("RGB")
    native_size = image.size
    placeholder = Image.new("L", native_size, color=0)
    transformed, _, seed = _replay_transform(image, placeholder, canonical, legacy_config)
    normalized = IMAGENET_NORMALIZE(to_tensor(transformed))
    return normalized, {
        "input_tensor_sha256": legacy.tensor_sha256(normalized),
        "augmentation_seed": seed, "view": VIEW,
        "native_image_size": list(native_size),
        "source_role": "meta_fit_train8" if index < 8 else "meta_check_train8",
        "historical_training_random_stream_replay": False,
        "ground_truth_loaded": False,
    }


def load_fit_target(
    record: Mapping[str, str], legacy_config: Mapping[str, Any], access: MutableMapping[str, int],
    *, native_image_size: tuple[int, int] | list[int],
) -> tuple[Tensor, dict[str, Any]]:
    """Load only fit8 labels and replay their geometry using a blank RGB image.

    ``native_image_size`` must come from the corresponding image-only load.
    The real image is not opened again; no check8 label is opened, even to
    inspect its size.  The runner verifies the result against sealed P2 GT hash.
    """
    _access(access)
    canonical, _ = _validated_record(record, legacy_config, fit_only=True)
    if (not isinstance(native_image_size, (tuple, list)) or len(native_image_size) != 2
            or any(type(size) is not int or size <= 0 for size in native_image_size)):
        raise ValueError("native_image_size must be the positive integer [width,height] from observation")
    with Image.open(canonical["mask_path"]) as handle:
        access["train_mask_opens"] += 1
        access["fit_mask_opens"] += 1
        mask = handle.convert("L")
    dummy_image = Image.new("RGB", tuple(native_image_size), color=0)
    _, transformed_mask, seed = _replay_transform(dummy_image, mask, canonical, legacy_config)
    target = torch.from_numpy(np.asarray(transformed_mask, dtype=np.float32).copy())[None] / 255.0
    return target, {
        "target_tensor_sha256": legacy.tensor_sha256(target),
        "augmentation_seed": seed, "view": VIEW,
        "native_image_size": list(native_image_size), "source_role": "meta_fit_train8",
        "historical_training_random_stream_replay": False, "ground_truth_loaded": True,
    }


def _tensor_digest(tensor: Tensor) -> dict[str, Any]:
    value = tensor.detach().cpu().contiguous()
    raw = value.reshape(-1).view(torch.uint8).numpy().tobytes()
    return {"shape": list(tensor.shape), "dtype": str(tensor.dtype), "device": str(tensor.device),
            "sha256": hashlib.sha256(raw).hexdigest()}


def state_digest(model: nn.Module) -> dict[str, Any]:
    """Fingerprint parameters, all buffers, requires-grad flags and every mode."""
    if not isinstance(model, nn.Module):
        raise TypeError("model must be nn.Module")
    state = {
        "parameters": {name: _tensor_digest(value) for name, value in model.named_parameters()},
        "buffers": {name: _tensor_digest(value) for name, value in model.named_buffers()},
        "requires_grad": {name: value.requires_grad for name, value in model.named_parameters()},
        "training_modes": {name: value.training for name, value in model.named_modules()},
    }
    digest = hashlib.sha256(json.dumps(state, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return {"schema": "ipma_micro16_host_state_v1", "sha256": digest, **state}


def assert_state_unchanged(model: nn.Module, before: Mapping[str, Any]) -> None:
    if state_digest(model) != before:
        raise RuntimeError("frozen host parameters/buffers/modes/requires_grad changed")


def extract_d0_source_features(model: nn.Module, image: Tensor) -> tuple[Tensor, Tensor]:
    """Cache detached D0 and verify exact head replay without SFS backprop."""
    if not isinstance(model, nn.Module) or not isinstance(getattr(model, "output_0", None), nn.Module):
        raise TypeError("a host exposing the actual output_0 module is required")
    if any(module.training for module in model.modules()):
        raise ValueError("all host modules must already be in eval mode")
    if any(parameter.requires_grad for parameter in model.parameters()):
        raise ValueError("all source host parameters must already be frozen")
    if (not isinstance(image, Tensor) or image.ndim != 4 or image.shape[0] != 1
            or not image.is_floating_point() or not bool(torch.isfinite(image).all())):
        raise ValueError("one finite floating BCHW image is required")
    if any(value.device != image.device for value in (*model.parameters(), *model.buffers())):
        raise ValueError("host state and image must be on the same device")
    before = state_digest(model)
    captured: list[Tensor] = []

    def capture_input(_module: nn.Module, args: tuple[Any, ...]) -> None:
        if len(args) != 1 or not isinstance(args[0], Tensor):
            raise RuntimeError("unexpected output_0 input signature")
        captured.append(args[0].detach().clone())

    handle = model.output_0.register_forward_pre_hook(capture_input)
    try:
        with torch.no_grad():
            result = model(image, False)
    finally:
        handle.remove()
        assert_state_unchanged(model, before)
    if not isinstance(result, (tuple, list)) or len(result) != 2 or not isinstance(result[1], Tensor):
        raise RuntimeError("host did not return the original two-part NS-FPN output")
    if len(captured) != 1:
        raise RuntimeError("expected exactly one output_0 capture")
    feature, logits = captured[0], result[1]
    if feature.device != image.device or logits.device != image.device:
        raise RuntimeError("D0/head output moved to a different device")
    if not bool(torch.isfinite(feature).all()) or not bool(torch.isfinite(logits).all()):
        raise RuntimeError("non-finite frozen D0/head output")
    try:
        with torch.no_grad():
            replay = model.output_0(feature)
    finally:
        assert_state_unchanged(model, before)
    if not torch.equal(logits, replay):
        raise RuntimeError("D0 replay differs from frozen host logits")
    return feature, logits.detach().clone()


__all__ = [
    "load_observation", "load_fit_target", "extract_d0_source_features",
    "state_digest", "assert_state_unchanged", "ACCESS_FIELDS", "FORBIDDEN_ACCESS_FIELDS",
]
