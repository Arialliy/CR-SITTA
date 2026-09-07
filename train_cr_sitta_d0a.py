"""Train the Stage-D0A CR-SITTA source model without opening test data.

Stage-D0A adds one deterministic supervised LF/HF deterioration branch to
the original NS-FPN training objective.  It is a training-compatibility
hypothesis, not evidence that the later label-free TTA proxy is aligned.  The
frozen train Pilot64 gradient gate must establish that separately.

The baseline entry point remains untouched.  This runner intentionally has no
test loader and emits a fixed-final-epoch checkpoint instead of a test-selected
``best`` checkpoint.
"""

from __future__ import annotations

import argparse
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shutil
import time
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn
from torch.optim import Adagrad
from torch.utils.data import DataLoader
import yaml

from model.loss import AverageMeter, SLSIoULoss
from train_fixed_split import (
    FixedSplitIRSTDDataset,
    _cpu_state_dict,
    append_jsonl,
    corpus_manifest,
    make_train_loader,
    positive_integer,
    read_split,
    repository_state,
    resolve_device,
    save_torch_atomic,
    seed_everything,
    sha256_file,
    write_json_atomic,
)
from tta.deteriorations import (
    imagenet_denormalize,
    imagenet_normalize,
    inject_high_frequency_noise,
    mask_low_frequency_amplitude,
)


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_PROTOCOL = PROJECT_ROOT / "configs" / "cr_sitta_d0a_train_v2.yaml"
PROBE_IDS = ("lf_mask", "hf_noise")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train CR-SITTA D0-A on a frozen official train split only."
    )
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--root", type=Path, default=None)
    parser.add_argument("--train-split", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--epochs", type=positive_integer, default=None)
    parser.add_argument("--batch-size", type=positive_integer, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument(
        "--train-only-smoke",
        action="store_true",
        help="Run an isolated one-epoch/two-batch engineering smoke with no test path.",
    )
    parser.add_argument(
        "--max-train-batches",
        type=positive_integer,
        default=None,
        help="Allowed only with --train-only-smoke (default: 2).",
    )
    return parser


def _resolve_project_path(raw_path: str | Path) -> Path:
    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def _finite_nonnegative_float(raw: Any, *, name: str) -> float:
    if isinstance(raw, bool) or not isinstance(raw, (int, float)):
        raise TypeError(f"{name} must be a real number")
    value = float(raw)
    if not math.isfinite(value) or value < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")
    return value


def load_run_configuration(args: argparse.Namespace) -> dict[str, Any]:
    protocol_path = args.protocol.expanduser().resolve()
    protocol = yaml.safe_load(protocol_path.read_text(encoding="utf-8"))
    if args.dataset not in protocol["datasets"]:
        available = ", ".join(sorted(protocol["datasets"]))
        raise ValueError(f"unknown dataset {args.dataset!r}; expected one of {available}")
    scope = protocol["scope"]
    if bool(scope["use_test_payload"]) or bool(scope["use_validation_payload"]):
        raise ValueError("D0-A protocol must forbid validation and test payload access")

    training = protocol["training"]
    objective = protocol["objective"]
    dataset = protocol["datasets"][args.dataset]
    smoke = bool(args.train_only_smoke)
    protocol_epochs = int(training["epochs"])
    requested_epochs = protocol_epochs if args.epochs is None else int(args.epochs)
    requested_batch_size = (
        int(training["batch_size"])
        if args.batch_size is None
        else int(args.batch_size)
    )
    if not smoke and requested_epochs != protocol_epochs:
        raise ValueError("full D0-A training must use the frozen 1000-epoch endpoint")
    if not smoke and requested_batch_size != int(training["batch_size"]):
        raise ValueError("full D0-A training must use the frozen batch size")
    if not smoke and args.max_train_batches is not None:
        raise ValueError("partial epochs are allowed only in train-only smoke mode")
    if args.num_workers is not None and int(args.num_workers) < 0:
        raise ValueError("num-workers must be non-negative")

    if smoke:
        epochs = 1 if args.epochs is None else requested_epochs
        max_train_batches = 2 if args.max_train_batches is None else args.max_train_batches
        num_workers = 0 if args.num_workers is None else int(args.num_workers)
        default_root = protocol["smoke_gate"]["separate_result_root"]
    else:
        epochs = protocol_epochs
        max_train_batches = None
        num_workers = (
            int(training["num_workers"])
            if args.num_workers is None
            else int(args.num_workers)
        )
        default_root = scope["result_root"]

    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else _resolve_project_path(default_root) / args.dataset
    )
    probes = protocol["deterioration_views"]["probes"]
    config = {
        "schema_version": int(protocol["schema_version"]),
        "protocol_id": str(protocol["protocol_id"]),
        "protocol_path": str(protocol_path),
        "protocol_sha256": sha256_file(protocol_path),
        "method_name": "CR-SITTA",
        "method_stage": "D0-A",
        "host_architecture": str(training["architecture"]),
        "development_only": True,
        "data_mode": "train_only_smoke" if smoke else "full_train_only",
        "dataset": str(args.dataset),
        "root": str(_resolve_project_path(args.root or dataset["root"])),
        "train_split": str(
            _resolve_project_path(args.train_split or dataset["train_split"])
        ),
        "expected_train_split_sha256": str(dataset["train_split_sha256"]),
        "expected_train_images": int(dataset["train_images"]),
        "expected_train_corpus_manifest_sha256": str(
            dataset["train_corpus_manifest_sha256"]
        ),
        "known_train_size_mismatches": list(
            dataset.get("known_train_size_mismatches", [])
        ),
        "epochs": int(epochs),
        "batch_size": int(requested_batch_size),
        "learning_rate": float(training["learning_rate"]),
        "warm_epochs": int(training["warm_epochs"]),
        "base_size": int(training["base_size"]),
        "crop_size": int(training["crop_size"]),
        "seed": int(training["seed"] if args.seed is None else args.seed),
        "num_workers": num_workers,
        "device": str(args.device),
        "output_dir": str(output_dir),
        "max_train_batches": max_train_batches,
        "train_only_smoke": smoke,
        "lambda_degraded": _finite_nonnegative_float(
            objective["lambda_degraded"], name="lambda_degraded"
        ),
        "probe_schedule": "deterministic_alternating_per_optimizer_step",
        "lf_mask_ratio": float(probes["lf_mask"]["mask_ratio"]),
        "lf_keep_probability": float(probes["lf_mask"]["keep_probability"]),
        "hf_target_rms": float(probes["hf_noise"]["target_rms"]),
        "hf_low_cut_ratio": float(probes["hf_noise"]["low_cut_ratio"]),
        "degraded_branch_batchnorm": str(training["degraded_branch_batchnorm"]),
        "expected_state_dict_keys": int(training["expected_state_dict_keys"]),
        "checkpoint_selection": "fixed_final_epoch_train_only",
        "validation_payload_access_allowed": False,
        "test_payload_access_allowed": False,
    }
    if config["degraded_branch_batchnorm"] not in {
        "eval_no_running_stat_update",
        "train_batch_stats_no_running_update",
    }:
        raise ValueError("unsupported degraded-branch BatchNorm policy")
    return config


def train_corpus_manifest(
    root: Path, identifiers: Sequence[str]
) -> tuple[str, list[dict[str, Any]]]:
    """Public train-only alias used by access-firewall tests."""

    return corpus_manifest(root, list(identifiers))


def validate_train_only_split(config: Mapping[str, Any]) -> dict[str, Any]:
    """Validate only the official train split; never resolve a test path."""

    root = Path(config["root"])
    split = Path(config["train_split"])
    if not root.is_dir():
        raise FileNotFoundError(f"dataset root does not exist: {root}")
    identifiers = read_split(split)
    split_hash = sha256_file(split)
    if split_hash != config["expected_train_split_sha256"]:
        raise ValueError(
            "train split hash drift: expected "
            f"{config['expected_train_split_sha256']}, got {split_hash}"
        )
    if len(identifiers) != int(config["expected_train_images"]):
        raise ValueError("train split count drifted from protocol")
    for directory_name in ("images", "masks"):
        directory = root / directory_name
        if not directory.is_dir():
            raise FileNotFoundError(f"required directory does not exist: {directory}")
    missing = [
        str(root / directory_name / f"{identifier}.png")
        for identifier in identifiers
        for directory_name in ("images", "masks")
        if not (root / directory_name / f"{identifier}.png").is_file()
    ]
    if missing:
        raise FileNotFoundError(
            f"{len(missing)} train image/mask files are missing; first: {missing[:10]}"
        )
    corpus_hash, size_mismatches = train_corpus_manifest(root, identifiers)
    if corpus_hash != config["expected_train_corpus_manifest_sha256"]:
        raise ValueError(
            "train corpus hash drift: expected "
            f"{config['expected_train_corpus_manifest_sha256']}, got {corpus_hash}"
        )
    expected_mismatches = sorted(
        config.get("known_train_size_mismatches", []),
        key=lambda item: item["image_id"],
    )
    if size_mismatches != expected_mismatches:
        raise ValueError(
            f"train image/mask size mismatch audit drifted: {size_mismatches}"
        )
    return {
        "role": "official_train_only",
        "train_count": len(identifiers),
        "train_split_sha256": split_hash,
        "train_corpus_manifest_sha256": corpus_hash,
        "known_train_size_mismatches": size_mismatches,
        "validation_split_reads": 0,
        "validation_image_opens": 0,
        "validation_mask_opens": 0,
        "test_split_reads": 0,
        "test_image_opens": 0,
        "test_mask_opens": 0,
    }


def probe_kind_for_step(global_optimizer_step: int) -> str:
    if isinstance(global_optimizer_step, bool) or not isinstance(
        global_optimizer_step, int
    ):
        raise TypeError("global_optimizer_step must be an integer")
    if global_optimizer_step < 0:
        raise ValueError("global_optimizer_step must be non-negative")
    return PROBE_IDS[global_optimizer_step % len(PROBE_IDS)]


def derive_probe_seed(
    protocol_id: str,
    global_seed: int,
    dataset_id: str,
    human_epoch: int,
    image_id: str,
    probe_id: str,
) -> int:
    """Derive one order-independent, auditable per-image probe seed."""

    if probe_id not in PROBE_IDS:
        raise ValueError(f"unknown probe_id {probe_id!r}")
    descriptor = json.dumps(
        {
            "dataset_id": str(dataset_id),
            "global_seed": int(global_seed),
            "human_epoch": int(human_epoch),
            "image_id": str(image_id),
            "probe_id": probe_id,
            "protocol_id": str(protocol_id),
        },
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return int.from_bytes(hashlib.sha256(descriptor).digest()[:8], "big") % (2**63 - 1)


def _cpu_generator(seed: int) -> torch.Generator:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    return generator


def build_deteriorated_view(
    normalized_image: Tensor,
    identifiers: Sequence[str],
    probe_id: str,
    protocol_id: str,
    global_seed: int,
    dataset_id: str,
    human_epoch: int,
    lf_mask_ratio: float,
    lf_keep_probability: float,
    hf_target_rms: float,
    hf_low_cut_ratio: float,
) -> Tensor:
    """Build a deterministic LF/HF training view without accepting GT metadata."""

    if probe_id not in PROBE_IDS:
        raise ValueError(f"unknown probe_id {probe_id!r}")
    if normalized_image.ndim != 4 or normalized_image.shape[1] != 3:
        raise ValueError("normalized_image must have shape [B,3,H,W]")
    if len(identifiers) != int(normalized_image.shape[0]):
        raise ValueError("identifier count must equal image batch size")
    physical = imagenet_denormalize(normalized_image)
    outputs: list[Tensor] = []
    for index, identifier in enumerate(identifiers):
        sample = physical[index : index + 1]
        seed = derive_probe_seed(
            protocol_id,
            global_seed,
            dataset_id,
            human_epoch,
            str(identifier),
            probe_id,
        )
        generator = _cpu_generator(seed)
        if probe_id == "lf_mask":
            deteriorated = mask_low_frequency_amplitude(
                sample,
                mask_ratio=lf_mask_ratio,
                keep_probability=lf_keep_probability,
                generator=generator,
            ).image
        else:
            deteriorated = inject_high_frequency_noise(
                sample,
                target_rms=hf_target_rms,
                low_cut_ratio=hf_low_cut_ratio,
                generator=generator,
            )
        outputs.append(deteriorated)
    result = imagenet_normalize(torch.cat(outputs, dim=0))
    if result.shape != normalized_image.shape:
        raise RuntimeError("deteriorated view changed tensor shape")
    if not bool(torch.isfinite(result.detach()).all().item()):
        raise RuntimeError("deteriorated normalized view contains NaN or Inf")
    return result


@contextmanager
def batchnorm_eval_only(model: nn.Module) -> Iterator[None]:
    """Temporarily freeze BN running statistics while preserving all parameters."""

    states: list[tuple[nn.modules.batchnorm._BatchNorm, bool]] = []
    for module in model.modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            states.append((module, bool(module.training)))
            module.eval()
    try:
        yield
    finally:
        for module, was_training in states:
            module.train(was_training)


@contextmanager
def batchnorm_batch_stats_no_running_update(model: nn.Module) -> Iterator[None]:
    """Use BN batch statistics without persisting a second branch update."""

    states: list[
        tuple[
            nn.modules.batchnorm._BatchNorm,
            bool,
            bool,
            Tensor | None,
            Tensor | None,
            Tensor | None,
        ]
    ] = []
    for module in model.modules():
        if not isinstance(module, nn.modules.batchnorm._BatchNorm):
            continue
        states.append(
            (
                module,
                bool(module.training),
                bool(module.track_running_stats),
                None if module.running_mean is None else module.running_mean.detach().clone(),
                None if module.running_var is None else module.running_var.detach().clone(),
                None
                if module.num_batches_tracked is None
                else module.num_batches_tracked.detach().clone(),
            )
        )
        module.train(True)
        module.track_running_stats = False
    try:
        yield
    finally:
        for module, was_training, tracked, mean, variance, count in states:
            module.track_running_stats = tracked
            module.train(was_training)
            for current, expected, label in (
                (module.running_mean, mean, "running_mean"),
                (module.running_var, variance, "running_var"),
                (module.num_batches_tracked, count, "num_batches_tracked"),
            ):
                if current is None or expected is None:
                    if current is not expected:
                        raise RuntimeError(f"BatchNorm {label} presence changed")
                elif not torch.equal(current.detach(), expected):
                    raise RuntimeError(f"degraded branch changed BatchNorm {label}")


def compute_segmentation_loss(
    model_outputs: tuple[Sequence[Tensor], Tensor],
    target: Tensor,
    loss_function: SLSIoULoss,
    *,
    warm_epochs: int,
    epoch_index: int,
) -> Tensor:
    """Reproduce the baseline final+deep-supervision loss exactly."""

    auxiliary, prediction = model_outputs
    loss = loss_function(prediction, target, warm_epochs, epoch_index)
    auxiliary_target = target
    downsample = nn.MaxPool2d(2, 2)
    for index, auxiliary_prediction in enumerate(auxiliary):
        if index > 0:
            auxiliary_target = downsample(auxiliary_target)
        loss = loss + loss_function(
            auxiliary_prediction, auxiliary_target, warm_epochs, epoch_index
        )
    return loss / (len(auxiliary) + 1)


def _tensor_sha256(tensor: Tensor) -> str:
    contiguous = tensor.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(contiguous.dtype).encode("ascii"))
    digest.update(json.dumps(list(contiguous.shape)).encode("ascii"))
    digest.update(contiguous.numpy().tobytes(order="C"))
    return digest.hexdigest()


def _bn_buffer_snapshot(model: nn.Module) -> dict[str, Tensor]:
    return {
        name: buffer.detach().cpu().clone()
        for name, buffer in model.named_buffers()
        if any(
            marker in name
            for marker in ("running_mean", "running_var", "num_batches_tracked")
        )
    }


def _assert_tensor_mapping_equal(
    first: Mapping[str, Tensor], second: Mapping[str, Tensor], *, message: str
) -> None:
    if first.keys() != second.keys() or any(
        not torch.equal(first[name], second[name]) for name in first
    ):
        raise RuntimeError(message)


def _gradient_l2_and_validate(model: nn.Module) -> float:
    squared_norm = 0.0
    has_nonzero = False
    for parameter in model.parameters():
        if parameter.grad is None:
            continue
        gradient = parameter.grad.detach()
        if not bool(torch.isfinite(gradient).all().item()):
            raise RuntimeError("model gradient contains NaN or Inf")
        squared_norm += float(gradient.double().square().sum().cpu())
        has_nonzero = has_nonzero or bool((gradient != 0).any().item())
    if not has_nonzero:
        raise RuntimeError("training batch produced no nonzero model gradient")
    return math.sqrt(squared_norm)


def train_one_epoch_d0a(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    loader: DataLoader[Any],
    loss_function: SLSIoULoss,
    device: torch.device,
    config: Mapping[str, Any],
    *,
    human_epoch: int,
    starting_optimizer_step: int,
) -> dict[str, Any]:
    model.train()
    epoch_index = human_epoch - 1
    warm_flag = epoch_index < int(config["warm_epochs"])
    clean_losses = AverageMeter()
    degraded_losses = AverageMeter()
    combined_losses = AverageMeter()
    probe_counts = {probe_id: 0 for probe_id in PROBE_IDS}
    probe_hashes: list[dict[str, Any]] = []
    batches = 0
    global_step = int(starting_optimizer_step)
    started = time.monotonic()
    lambda_degraded = float(config["lambda_degraded"])
    denominator = 1.0 + lambda_degraded
    audit_contracts = bool(config["train_only_smoke"])

    for image, target, identifiers in loader:
        image = image.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        target_before = target.detach().clone() if audit_contracts else None

        optimizer.zero_grad()
        clean_outputs = model(image, warm_flag)
        clean_loss = compute_segmentation_loss(
            clean_outputs,
            target,
            loss_function,
            warm_epochs=int(config["warm_epochs"]),
            epoch_index=epoch_index,
        )
        if not bool(torch.isfinite(clean_loss.detach()).item()):
            raise RuntimeError(
                f"non-finite clean loss at epoch {human_epoch}, batch {batches}"
            )
        (clean_loss / denominator).backward()

        degraded_loss_value = 0.0
        probe_id: str | None = None
        if lambda_degraded > 0.0:
            probe_id = probe_kind_for_step(global_step)
            with torch.no_grad():
                degraded_image = build_deteriorated_view(
                    image.detach(),
                    list(identifiers),
                    probe_id=probe_id,
                    protocol_id=str(config["protocol_id"]),
                    global_seed=int(config["seed"]),
                    dataset_id=str(config["dataset"]),
                    human_epoch=human_epoch,
                    lf_mask_ratio=float(config["lf_mask_ratio"]),
                    lf_keep_probability=float(config["lf_keep_probability"]),
                    hf_target_rms=float(config["hf_target_rms"]),
                    hf_low_cut_ratio=float(config["hf_low_cut_ratio"]),
                )
                if audit_contracts:
                    repeated = build_deteriorated_view(
                        image.detach(),
                        list(identifiers),
                        probe_id=probe_id,
                        protocol_id=str(config["protocol_id"]),
                        global_seed=int(config["seed"]),
                        dataset_id=str(config["dataset"]),
                        human_epoch=human_epoch,
                        lf_mask_ratio=float(config["lf_mask_ratio"]),
                        lf_keep_probability=float(config["lf_keep_probability"]),
                        hf_target_rms=float(config["hf_target_rms"]),
                        hf_low_cut_ratio=float(config["hf_low_cut_ratio"]),
                    )
                    if not torch.equal(degraded_image, repeated):
                        raise RuntimeError("same probe seed did not reproduce bit exactly")

            bn_after_clean = _bn_buffer_snapshot(model) if audit_contracts else {}
            batchnorm_policy = str(config["degraded_branch_batchnorm"])
            context = (
                batchnorm_batch_stats_no_running_update(model)
                if batchnorm_policy == "train_batch_stats_no_running_update"
                else batchnorm_eval_only(model)
            )
            with context:
                degraded_outputs = model(degraded_image, warm_flag)
                degraded_loss = compute_segmentation_loss(
                    degraded_outputs,
                    target,
                    loss_function,
                    warm_epochs=int(config["warm_epochs"]),
                    epoch_index=epoch_index,
                )
            if not bool(torch.isfinite(degraded_loss.detach()).item()):
                raise RuntimeError(
                    f"non-finite degraded loss at epoch {human_epoch}, batch {batches}"
                )
            (lambda_degraded * degraded_loss / denominator).backward()
            degraded_loss_value = float(degraded_loss.detach().cpu())
            probe_counts[probe_id] += 1
            if audit_contracts:
                _assert_tensor_mapping_equal(
                    bn_after_clean,
                    _bn_buffer_snapshot(model),
                    message="degraded branch changed BatchNorm running buffers",
                )
                probe_hashes.append(
                    {
                        "global_optimizer_step": global_step,
                        "probe_id": probe_id,
                        "tensor_sha256": _tensor_sha256(degraded_image),
                    }
                )

        gradient_l2 = _gradient_l2_and_validate(model)
        optimizer.step()
        if target_before is not None and not torch.equal(target, target_before):
            raise RuntimeError("D0-A modified the source train target")

        clean_value = float(clean_loss.detach().cpu())
        combined_value = (
            clean_value + lambda_degraded * degraded_loss_value
        ) / denominator
        batch_size = int(image.shape[0])
        clean_losses.update(clean_value, batch_size)
        if lambda_degraded > 0.0:
            degraded_losses.update(degraded_loss_value, batch_size)
        combined_losses.update(combined_value, batch_size)
        batches += 1
        global_step += 1
        if config["max_train_batches"] is not None and batches >= int(
            config["max_train_batches"]
        ):
            break

    if batches == 0:
        raise RuntimeError("training loader produced zero full batches")
    if lambda_degraded > 0.0 and sum(probe_counts.values()) != batches:
        raise RuntimeError("probe accounting does not equal optimizer-step count")
    return {
        "epoch": human_epoch,
        "epoch_index": epoch_index,
        "warm_flag": warm_flag,
        "batches": batches,
        "starting_optimizer_step": int(starting_optimizer_step),
        "ending_optimizer_step": global_step,
        "mean_clean_loss": float(clean_losses.avg),
        "mean_degraded_loss": (
            float(degraded_losses.avg) if lambda_degraded > 0.0 else None
        ),
        "mean_combined_loss": float(combined_losses.avg),
        "last_gradient_l2": gradient_l2,
        "probe_counts": probe_counts,
        "probe_tensor_sha256s": probe_hashes,
        "learning_rate": float(optimizer.param_groups[0]["lr"]),
        "duration_seconds": time.monotonic() - started,
    }


def runtime_artifact_hashes() -> dict[str, str]:
    relative_paths = (
        "train_cr_sitta_d0a.py",
        "configs/cr_sitta_d0a_train_v2.yaml",
        "train_fixed_split.py",
        "model/loss.py",
        "model/MSHNet_NSFPN.py",
        "model/NS_FPN.py",
        "model/diff_cross_attns.py",
        "tta/deteriorations/fourier_low_mask.py",
        "tta/deteriorations/high_frequency_noise.py",
        "tta/deteriorations/image_space.py",
        "SFS_MSDeformAttn/ops/functions/ms_deform_attn_func.py",
        "SFS_MSDeformAttn/ops/modules/ms_deform_attn.py",
    )
    hashes = {path: sha256_file(PROJECT_ROOT / path) for path in relative_paths}
    import MultiScaleDeformableAttention as extension

    extension_path = Path(extension.__file__).resolve()
    hashes[str(extension_path)] = sha256_file(extension_path)
    return hashes


def _checkpoint_payload(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    config: Mapping[str, Any],
    split_manifest: Mapping[str, Any],
    train_metrics: Mapping[str, Any],
    *,
    human_epoch: int,
    global_optimizer_step: int,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "architecture": "MSHNet_NSFPN",
        "method_name": "CR-SITTA",
        "method_stage": "D0-A",
        "development_only": True,
        "test_selected": False,
        "selection_rule": "fixed_final_epoch_train_only",
        "epoch": human_epoch,
        "global_optimizer_step": global_optimizer_step,
        "state_dict": _cpu_state_dict(model),
        "optimizer": optimizer.state_dict(),
        "latest_train_metrics": dict(train_metrics),
        "split_manifest": dict(split_manifest),
        "run_config": dict(config),
        "rng_state": {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch_cpu": torch.get_rng_state(),
            "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        },
    }


def _restore_rng_state(payload: Mapping[str, Any]) -> None:
    state = payload.get("rng_state")
    if not state:
        return
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if torch.cuda.is_available() and state["torch_cuda"]:
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def run_training(args: argparse.Namespace) -> dict[str, Any]:
    config = load_run_configuration(args)
    split_manifest = validate_train_only_split(config)
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    last_path = output_dir / "last.pth.tar"
    if last_path.exists() and args.resume is None:
        raise FileExistsError(
            f"{last_path} already exists; pass --resume explicitly or use a new output dir"
        )

    seed_everything(int(config["seed"]))
    device = resolve_device(str(config["device"]))
    train_dataset = FixedSplitIRSTDDataset(
        Path(config["root"]),
        Path(config["train_split"]),
        training=True,
        base_size=int(config["base_size"]),
        crop_size=int(config["crop_size"]),
    )
    from model.MSHNet_NSFPN import MSHNet_NSFPN

    model = MSHNet_NSFPN(3).to(device)
    actual_key_count = len(model.state_dict())
    if actual_key_count != int(config["expected_state_dict_keys"]):
        raise RuntimeError(
            "MSHNet_NSFPN state_dict schema drift: expected "
            f"{config['expected_state_dict_keys']} keys, got {actual_key_count}"
        )
    optimizer = Adagrad(model.parameters(), lr=float(config["learning_rate"]))
    loss_function = SLSIoULoss()

    start_epoch = 1
    global_optimizer_step = 0
    if args.resume is not None:
        resume_path = args.resume.expanduser().resolve()
        checkpoint = torch.load(resume_path, map_location=device)
        if checkpoint.get("method_stage") != "D0-A":
            raise ValueError("resume checkpoint is not a CR-SITTA D0-A checkpoint")
        if checkpoint["run_config"] != config:
            raise ValueError(
                "resume checkpoint run_config does not exactly match this run; "
                "smoke/full or protocol mixing is forbidden"
            )
        if checkpoint["split_manifest"] != split_manifest:
            raise ValueError("resume checkpoint train manifest does not match this run")
        model.load_state_dict(checkpoint["state_dict"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = int(checkpoint["epoch"]) + 1
        global_optimizer_step = int(checkpoint["global_optimizer_step"])
        _restore_rng_state(checkpoint)

    contract = {
        "run_config": config,
        "split_manifest": split_manifest,
        "access_firewall": {
            "implementation_has_test_loader": False,
            "implementation_has_validation_loader": False,
            "test_split_reads": 0,
            "test_image_opens": 0,
            "test_mask_opens": 0,
            "validation_split_reads": 0,
            "validation_image_opens": 0,
            "validation_mask_opens": 0,
        },
        "repository": repository_state(),
        "runtime_sha256": runtime_artifact_hashes(),
        "torch_version": torch.__version__,
        "torch_cuda_runtime": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "started_at_unix": time.time(),
    }
    contract_path = output_dir / "run_contract.json"
    if contract_path.exists():
        existing = json.loads(contract_path.read_text(encoding="utf-8"))
        for key in ("run_config", "split_manifest", "runtime_sha256", "access_firewall"):
            if existing[key] != contract[key]:
                raise ValueError(f"existing run contract {key} drifted")
    else:
        write_json_atomic(contract_path, contract)
    split_archive = output_dir / "splits"
    split_archive.mkdir(parents=True, exist_ok=True)
    shutil.copy2(config["train_split"], split_archive / "train.txt")

    started = time.monotonic()
    latest_metrics: dict[str, Any] | None = None
    for human_epoch in range(start_epoch, int(config["epochs"]) + 1):
        train_loader = make_train_loader(train_dataset, config, human_epoch)
        latest_metrics = train_one_epoch_d0a(
            model,
            optimizer,
            train_loader,
            loss_function,
            device,
            config,
            human_epoch=human_epoch,
            starting_optimizer_step=global_optimizer_step,
        )
        global_optimizer_step = int(latest_metrics["ending_optimizer_step"])
        append_jsonl(output_dir / "train_metrics.jsonl", latest_metrics)
        payload = _checkpoint_payload(
            model,
            optimizer,
            config,
            split_manifest,
            latest_metrics,
            human_epoch=human_epoch,
            global_optimizer_step=global_optimizer_step,
        )
        save_torch_atomic(last_path, payload)
        print(
            f"dataset={config['dataset']} epoch={human_epoch}/{config['epochs']} "
            f"loss={latest_metrics['mean_combined_loss']:.6f} "
            f"clean={latest_metrics['mean_clean_loss']:.6f} "
            f"degraded={latest_metrics['mean_degraded_loss']}",
            flush=True,
        )

    if latest_metrics is None:
        raise RuntimeError("resume checkpoint is already beyond configured endpoint")
    final_checkpoint: str | None = None
    if not bool(config["train_only_smoke"]):
        expected_epoch = int(config["epochs"])
        final_path = output_dir / f"epoch_{expected_epoch}_train_only.pth.tar"
        save_torch_atomic(
            final_path,
            _checkpoint_payload(
                model,
                optimizer,
                config,
                split_manifest,
                latest_metrics,
                human_epoch=expected_epoch,
                global_optimizer_step=global_optimizer_step,
            ),
        )
        final_checkpoint = str(final_path)

    summary = {
        "dataset": config["dataset"],
        "method_name": "CR-SITTA",
        "method_stage": "D0-A",
        "data_mode": config["data_mode"],
        "completed_epochs": int(config["epochs"]),
        "global_optimizer_steps": global_optimizer_step,
        "latest_train_metrics": latest_metrics,
        "fixed_final_checkpoint": final_checkpoint,
        "test_selected": False,
        "validation_payload_opens": 0,
        "test_payload_opens": 0,
        "next_gate": "frozen_train_Pilot64_Stage-C0_gradient_gate",
        "duration_seconds": time.monotonic() - started,
        "output_dir": str(output_dir),
    }
    write_json_atomic(output_dir / "summary.json", summary)
    return summary


def main() -> None:
    args = build_parser().parse_args()
    summary = run_training(args)
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
