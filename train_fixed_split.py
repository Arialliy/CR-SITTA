"""Train NS-FPN on immutable train/test ``img_idx`` splits.

This entry point preserves the upstream architecture, augmentation, loss,
optimizer, and warm-stage semantics.  Unlike the upstream ``main.py``, paths
and output locations are explicit, checkpoints are atomic and resumable, and
the two requested test-selected exports are recorded independently.

The configured protocol deliberately evaluates the fixed test set every epoch
from epoch 500 onward.  Consequently ``best_miou.pth.tar`` and
``best_pd.pth.tar`` are *test-selected* checkpoints; the run metadata states
that limitation rather than presenting the test set as untouched.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import subprocess
import time
from typing import Any

import numpy as np
from PIL import Image, ImageFilter, ImageOps
import torch
from torch import Tensor, nn
from torch.optim import Adagrad
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import yaml

from metrics.official_metric_adapter import OfficialMetricAdapter
from model.loss import AverageMeter, SLSIoULoss


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_PROTOCOL = PROJECT_ROOT / "configs" / "retrain_fixed_splits.yaml"
IMAGENET_NORMALIZE = transforms.Normalize(
    mean=(0.485, 0.456, 0.406),
    std=(0.229, 0.224, 0.225),
)


def positive_integer(raw: str) -> int:
    value = int(raw)
    if value < 1:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return value


def nonnegative_integer(raw: str) -> int:
    value = int(raw)
    if value < 0:
        raise argparse.ArgumentTypeError("value must be a non-negative integer")
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train NS-FPN on one frozen train/test img_idx split."
    )
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--root", type=Path, default=None)
    parser.add_argument("--train-split", type=Path, default=None)
    parser.add_argument("--test-split", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--epochs", type=positive_integer, default=None)
    parser.add_argument("--eval-start-epoch", type=positive_integer, default=None)
    parser.add_argument("--batch-size", type=positive_integer, default=None)
    parser.add_argument("--num-workers", type=nonnegative_integer, default=8)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument(
        "--max-train-batches",
        type=positive_integer,
        default=None,
        help="Smoke-test only: stop each training epoch after this many batches.",
    )
    parser.add_argument(
        "--max-test-images",
        type=positive_integer,
        default=None,
        help="Smoke-test only: evaluate only this deterministic test prefix.",
    )
    return parser


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_split(path: Path) -> list[str]:
    if not path.is_file():
        raise FileNotFoundError(f"split does not exist: {path}")
    identifiers = [line.strip() for line in path.read_text(encoding="utf-8-sig").splitlines()]
    if not identifiers or any(not identifier for identifier in identifiers):
        raise ValueError(f"split is empty or contains a blank identifier: {path}")
    if len(identifiers) != len(set(identifiers)):
        raise ValueError(f"split contains duplicate identifiers: {path}")
    return identifiers


def _resolve_project_path(raw_path: str | Path) -> Path:
    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def load_run_configuration(args: argparse.Namespace) -> dict[str, Any]:
    protocol_path = args.protocol.expanduser().resolve()
    protocol = yaml.safe_load(protocol_path.read_text(encoding="utf-8"))
    if args.dataset not in protocol["datasets"]:
        available = ", ".join(sorted(protocol["datasets"]))
        raise ValueError(f"unknown dataset {args.dataset!r}; expected one of {available}")

    training = protocol["training"]
    dataset = protocol["datasets"][args.dataset]
    config = {
        "schema_version": int(protocol["schema_version"]),
        "protocol_id": str(protocol["protocol_id"]),
        "protocol_path": str(protocol_path),
        "protocol_sha256": sha256_file(protocol_path),
        "dataset": args.dataset,
        "root": str(_resolve_project_path(args.root or dataset["root"])),
        "train_split": str(
            _resolve_project_path(args.train_split or dataset["train_split"])
        ),
        "test_split": str(
            _resolve_project_path(args.test_split or dataset["test_split"])
        ),
        "expected_train_split_sha256": str(dataset["train_split_sha256"]),
        "expected_test_split_sha256": str(dataset["test_split_sha256"]),
        "expected_corpus_manifest_sha256": str(
            dataset["corpus_manifest_sha256"]
        ),
        "known_size_mismatches": list(dataset.get("known_size_mismatches", [])),
        "expected_train_images": int(dataset["train_images"]),
        "expected_test_images": int(dataset["test_images"]),
        "epochs": int(args.epochs or training["epochs"]),
        "eval_start_epoch": int(
            args.eval_start_epoch or training["evaluation_start_epoch"]
        ),
        "batch_size": int(args.batch_size or training["batch_size"]),
        "learning_rate": float(training["learning_rate"]),
        "warm_epochs": int(training["warm_epochs"]),
        "base_size": int(training["base_size"]),
        "crop_size": int(training["crop_size"]),
        "seed": int(training["seed"] if args.seed is None else args.seed),
        "num_workers": int(args.num_workers),
        "device": str(args.device),
        "output_dir": str(args.output_dir.expanduser().resolve()),
        "max_train_batches": args.max_train_batches,
        "max_test_images": args.max_test_images,
        "test_selected_checkpoints": True,
        "selection_metrics": ["miou", "pd"],
    }
    if config["eval_start_epoch"] > config["epochs"]:
        raise ValueError("eval-start-epoch cannot be greater than epochs")
    if (args.max_train_batches is None) != (args.max_test_images is None):
        raise ValueError(
            "smoke mode requires both --max-train-batches and --max-test-images"
        )
    return config


def corpus_manifest(root: Path, identifiers: list[str]) -> tuple[str, list[dict[str, Any]]]:
    """Hash exact canonical image/mask bytes and audit spatial mismatches."""

    digest = hashlib.sha256()
    size_mismatches: list[dict[str, Any]] = []
    for identifier in sorted(identifiers):
        image_path = root / "images" / f"{identifier}.png"
        mask_path = root / "masks" / f"{identifier}.png"
        image_hash = sha256_file(image_path)
        mask_hash = sha256_file(mask_path)
        digest.update(
            f"{identifier}\t{image_hash}\t{mask_hash}\n".encode("utf-8")
        )
        with Image.open(image_path) as image_handle:
            image_size = list(image_handle.size)
        with Image.open(mask_path) as mask_handle:
            mask_size = list(mask_handle.size)
        if image_size != mask_size:
            size_mismatches.append(
                {
                    "image_id": identifier,
                    "image_size": image_size,
                    "mask_size": mask_size,
                }
            )
    return digest.hexdigest(), size_mismatches


def validate_fixed_splits(config: Mapping[str, Any]) -> dict[str, Any]:
    root = Path(config["root"])
    train_split = Path(config["train_split"])
    test_split = Path(config["test_split"])
    if not root.is_dir():
        raise FileNotFoundError(f"dataset root does not exist: {root}")

    train_ids = read_split(train_split)
    test_ids = read_split(test_split)
    overlap = sorted(set(train_ids) & set(test_ids))
    if overlap:
        raise ValueError(f"train/test overlap ({len(overlap)} IDs): {overlap[:10]}")

    train_hash = sha256_file(train_split)
    test_hash = sha256_file(test_split)
    if train_hash != config["expected_train_split_sha256"]:
        raise ValueError(
            f"train split hash drift: expected {config['expected_train_split_sha256']}, "
            f"got {train_hash}"
        )
    if test_hash != config["expected_test_split_sha256"]:
        raise ValueError(
            f"test split hash drift: expected {config['expected_test_split_sha256']}, "
            f"got {test_hash}"
        )
    if len(train_ids) != config["expected_train_images"]:
        raise ValueError("train split count drifted from protocol")
    if len(test_ids) != config["expected_test_images"]:
        raise ValueError("test split count drifted from protocol")

    for directory_name in ("images", "masks"):
        directory = root / directory_name
        if not directory.is_dir():
            raise FileNotFoundError(f"required directory does not exist: {directory}")
    missing: list[str] = []
    for identifier in train_ids + test_ids:
        for directory_name in ("images", "masks"):
            path = root / directory_name / f"{identifier}.png"
            if not path.is_file():
                missing.append(str(path))
    if missing:
        raise FileNotFoundError(
            f"{len(missing)} split-referenced image/mask files are missing; "
            f"first entries: {missing[:10]}"
        )
    corpus_hash, size_mismatches = corpus_manifest(
        root, sorted(set(train_ids) | set(test_ids))
    )
    if corpus_hash != config["expected_corpus_manifest_sha256"]:
        raise ValueError(
            "dataset corpus hash drift: expected "
            f"{config['expected_corpus_manifest_sha256']}, got {corpus_hash}"
        )
    expected_mismatches = sorted(
        config["known_size_mismatches"], key=lambda item: item["image_id"]
    )
    if size_mismatches != expected_mismatches:
        raise ValueError(
            "image/mask size mismatch audit drifted: "
            f"expected {expected_mismatches}, got {size_mismatches}"
        )
    return {
        "train_count": len(train_ids),
        "test_count": len(test_ids),
        "overlap_count": 0,
        "train_split_sha256": train_hash,
        "test_split_sha256": test_hash,
        "corpus_manifest_algorithm": "sorted-id-image_sha256-mask_sha256-lf-v1",
        "corpus_manifest_sha256": corpus_hash,
        "known_size_mismatches": size_mismatches,
    }


class FixedSplitIRSTDDataset(Dataset[tuple[Tensor, Tensor, str]]):
    """Faithful NS-FPN data path with an explicit immutable split file."""

    def __init__(
        self,
        root: Path,
        split_file: Path,
        *,
        training: bool,
        base_size: int,
        crop_size: int,
    ) -> None:
        self.root = root
        self.identifiers = read_split(split_file)
        self.training = bool(training)
        self.base_size = int(base_size)
        self.crop_size = int(crop_size)

    def __len__(self) -> int:
        return len(self.identifiers)

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor, str]:
        identifier = self.identifiers[index]
        with Image.open(self.root / "images" / f"{identifier}.png") as handle:
            image = handle.convert("RGB")
        with Image.open(self.root / "masks" / f"{identifier}.png") as handle:
            mask = handle.copy()
        if self.training:
            image, mask = self._train_transform(image, mask)
        else:
            image = image.resize((self.base_size, self.base_size), Image.BILINEAR)
            mask = mask.resize((self.base_size, self.base_size), Image.NEAREST)
        image_tensor = IMAGENET_NORMALIZE(transforms.functional.to_tensor(image))
        mask_tensor = transforms.functional.to_tensor(mask)
        return image_tensor, mask_tensor, identifier

    def _train_transform(
        self, image: Image.Image, mask: Image.Image
    ) -> tuple[Image.Image, Image.Image]:
        if random.random() < 0.5:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)

        long_size = random.randint(
            int(self.base_size * 0.5), int(self.base_size * 2.0)
        )
        width, height = image.size
        if height > width:
            out_height = long_size
            out_width = int(width * long_size / height + 0.5)
            short_size = out_width
        else:
            out_width = long_size
            out_height = int(height * long_size / width + 0.5)
            short_size = out_height
        image = image.resize((out_width, out_height), Image.BILINEAR)
        mask = mask.resize((out_width, out_height), Image.NEAREST)

        if short_size < self.crop_size:
            pad_height = max(self.crop_size - out_height, 0)
            pad_width = max(self.crop_size - out_width, 0)
            image = ImageOps.expand(image, border=(0, 0, pad_width, pad_height), fill=0)
            mask = ImageOps.expand(mask, border=(0, 0, pad_width, pad_height), fill=0)

        width, height = image.size
        left = random.randint(0, width - self.crop_size)
        top = random.randint(0, height - self.crop_size)
        box = (left, top, left + self.crop_size, top + self.crop_size)
        image = image.crop(box)
        mask = mask.crop(box)
        if random.random() < 0.5:
            image = image.filter(ImageFilter.GaussianBlur(radius=random.random()))
        return image, mask


def seed_everything(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def seed_worker(worker_id: int) -> None:
    del worker_id
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def resolve_device(name: str) -> torch.device:
    device = torch.device(name)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
        index = torch.cuda.current_device() if device.index is None else device.index
        if index >= torch.cuda.device_count():
            raise RuntimeError(f"CUDA device {index} is unavailable")
        torch.cuda.set_device(index)
    return device


def make_train_loader(
    dataset: Dataset[Any], config: Mapping[str, Any], human_epoch: int
) -> DataLoader[Any]:
    generator = torch.Generator()
    generator.manual_seed(int(config["seed"]) + human_epoch)
    return DataLoader(
        dataset,
        batch_size=int(config["batch_size"]),
        shuffle=True,
        drop_last=True,
        num_workers=int(config["num_workers"]),
        pin_memory=True,
        persistent_workers=False,
        worker_init_fn=seed_worker,
        generator=generator,
    )


def make_test_loader(dataset: Dataset[Any], config: Mapping[str, Any]) -> DataLoader[Any]:
    return DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        drop_last=False,
        num_workers=int(config["num_workers"]),
        pin_memory=True,
        persistent_workers=int(config["num_workers"]) > 0,
    )


def train_one_epoch(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    loader: DataLoader[Any],
    loss_function: SLSIoULoss,
    device: torch.device,
    *,
    human_epoch: int,
    warm_epochs: int,
    max_batches: int | None,
) -> dict[str, float | int]:
    model.train()
    epoch_index = human_epoch - 1
    warm_flag = epoch_index < warm_epochs
    downsample = nn.MaxPool2d(2, 2)
    losses = AverageMeter()
    batches = 0
    started = time.monotonic()

    for image, target, _ in loader:
        image = image.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        auxiliary, prediction = model(image, warm_flag)
        loss = loss_function(prediction, target, warm_epochs, epoch_index)
        auxiliary_target = target
        for index, auxiliary_prediction in enumerate(auxiliary):
            if index > 0:
                auxiliary_target = downsample(auxiliary_target)
            loss = loss + loss_function(
                auxiliary_prediction, auxiliary_target, warm_epochs, epoch_index
            )
        loss = loss / (len(auxiliary) + 1)
        if not torch.isfinite(loss):
            raise RuntimeError(f"non-finite loss at epoch {human_epoch}, batch {batches}")
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.update(float(loss.detach().cpu()), prediction.shape[0])
        batches += 1
        if max_batches is not None and batches >= max_batches:
            break

    if batches == 0:
        raise RuntimeError("training loader produced zero full batches")
    return {
        "epoch": human_epoch,
        "epoch_index": epoch_index,
        "warm_flag": warm_flag,
        "batches": batches,
        "mean_loss": float(losses.avg),
        "learning_rate": float(optimizer.param_groups[0]["lr"]),
        "duration_seconds": time.monotonic() - started,
    }


def evaluate(
    model: nn.Module,
    loader: DataLoader[Any],
    device: torch.device,
    *,
    human_epoch: int,
    image_size: int,
    max_images: int | None,
) -> dict[str, float | int]:
    model.eval()
    evaluator = OfficialMetricAdapter(image_size=image_size)
    image_count = 0
    started = time.monotonic()
    with torch.no_grad():
        for image, target, _ in loader:
            image = image.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            _, logits = model(image, False)
            if not torch.isfinite(logits).all():
                raise RuntimeError(f"non-finite test logits at epoch {human_epoch}")
            evaluator.update(logits, target)
            image_count += 1
            if max_images is not None and image_count >= max_images:
                break
    result = evaluator.compute()
    return {
        "epoch": human_epoch,
        "images": image_count,
        "miou": float(result.mean_iou),
        "pd": float(result.detection_probability[0]),
        "fa_per_pixel_x1e6": float(result.false_alarm_pixel_rate[0] * 1e6),
        "duration_seconds": time.monotonic() - started,
    }


def _cpu_state_dict(model: nn.Module) -> dict[str, Tensor]:
    return {
        name: tensor.detach().cpu().clone()
        for name, tensor in model.state_dict().items()
    }


def save_torch_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def write_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def append_jsonl(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(dict(payload), sort_keys=True, ensure_ascii=False))
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def repository_state() -> dict[str, Any]:
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    dirty = subprocess.run(
        ["git", "status", "--porcelain=v1", "--untracked-files=normal"],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    return {"head_commit": head, "dirty_entries": dirty}


def runtime_artifact_hashes() -> dict[str, str]:
    """Bind a run to code bytes even when the working tree is not committed."""

    relative_paths = (
        "train_fixed_split.py",
        "configs/retrain_fixed_splits.yaml",
        "metrics/official_metric_adapter.py",
        "utils/metric.py",
        "model/loss.py",
        "model/MSHNet_NSFPN.py",
        "model/NS_FPN.py",
        "model/diff_cross_attns.py",
        "SFS_MSDeformAttn/ops/functions/ms_deform_attn_func.py",
        "SFS_MSDeformAttn/ops/modules/ms_deform_attn.py",
    )
    hashes = {path: sha256_file(PROJECT_ROOT / path) for path in relative_paths}
    import MultiScaleDeformableAttention as extension

    extension_path = Path(extension.__file__).resolve()
    hashes[str(extension_path)] = sha256_file(extension_path)
    return hashes


def _best_payload(
    model: nn.Module,
    config: Mapping[str, Any],
    split_manifest: Mapping[str, Any],
    metrics: Mapping[str, Any],
    selection_metric: str,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "architecture": "MSHNet_NSFPN",
        "dataset": config["dataset"],
        "epoch": metrics["epoch"],
        "selection_metric": selection_metric,
        "selection_rule": (
            "maximize_miou_then_pd_then_minimize_fa"
            if selection_metric == "miou"
            else "maximize_pd_then_minimize_fa_then_miou"
        ),
        "selection_value": metrics[selection_metric],
        "test_metrics": dict(metrics),
        "test_selected": True,
        "state_dict": _cpu_state_dict(model),
        "split_manifest": dict(split_manifest),
        "run_config": dict(config),
    }


def selection_key(metric_name: str, metrics: Mapping[str, Any]) -> tuple[float, ...]:
    """Return a frozen lexicographic key while preserving the primary metric."""

    miou = float(metrics["miou"])
    pd = float(metrics["pd"])
    negative_fa = -float(metrics["fa_per_pixel_x1e6"])
    if metric_name == "miou":
        return (miou, pd, negative_fa)
    if metric_name == "pd":
        return (pd, negative_fa, miou)
    raise ValueError(f"unsupported selection metric: {metric_name}")


def run_training(args: argparse.Namespace) -> dict[str, Any]:
    config = load_run_configuration(args)
    split_manifest = validate_fixed_splits(config)
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
    test_dataset = FixedSplitIRSTDDataset(
        Path(config["root"]),
        Path(config["test_split"]),
        training=False,
        base_size=int(config["base_size"]),
        crop_size=int(config["crop_size"]),
    )
    test_loader = make_test_loader(test_dataset, config)
    # Keep the custom SFS/CUDA extension lazy so configuration and split
    # validation remain usable on machines that only inspect the protocol.
    from model.MSHNet_NSFPN import MSHNet_NSFPN

    model = MSHNet_NSFPN(3).to(device)
    optimizer = Adagrad(model.parameters(), lr=float(config["learning_rate"]))
    loss_function = SLSIoULoss()

    start_epoch = 1
    best = {
        "miou": {
            "value": float("-inf"),
            "key": (float("-inf"),) * 3,
            "epoch": None,
            "metrics": None,
        },
        "pd": {
            "value": float("-inf"),
            "key": (float("-inf"),) * 3,
            "epoch": None,
            "metrics": None,
        },
    }
    if args.resume is not None:
        resume_path = args.resume.expanduser().resolve()
        checkpoint = torch.load(resume_path, map_location=device)
        if checkpoint["run_config"] != config:
            raise ValueError(
                "resume checkpoint run_config does not exactly match this run; "
                "smoke-to-full, changed hyperparameters, changed output dirs, and "
                "protocol mixing are forbidden"
            )
        if checkpoint["split_manifest"] != split_manifest:
            raise ValueError("resume checkpoint split manifest does not match this run")
        model.load_state_dict(checkpoint["state_dict"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best = checkpoint["best"]
        for metric_name in ("miou", "pd"):
            if best[metric_name]["epoch"] is not None:
                best_path = output_dir / f"best_{metric_name}.pth.tar"
                if not best_path.is_file():
                    raise FileNotFoundError(
                        f"resume checkpoint refers to a missing historical best: {best_path}"
                    )

    contract = {
        "run_config": config,
        "split_manifest": split_manifest,
        "repository": repository_state(),
        "runtime_sha256": runtime_artifact_hashes(),
        "torch_version": torch.__version__,
        "torch_cuda_runtime": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "started_at_unix": time.time(),
    }
    contract_path = output_dir / "run_contract.json"
    if contract_path.exists():
        existing_contract = json.loads(contract_path.read_text(encoding="utf-8"))
        if existing_contract["run_config"] != config:
            raise ValueError("existing run contract does not match the requested run")
        if existing_contract["split_manifest"] != split_manifest:
            raise ValueError("existing run contract split manifest drifted")
        if existing_contract["runtime_sha256"] != contract["runtime_sha256"]:
            raise ValueError("runtime code changed since the original run started")
    else:
        write_json_atomic(contract_path, contract)
    split_archive = output_dir / "splits"
    split_archive.mkdir(parents=True, exist_ok=True)
    shutil.copy2(config["train_split"], split_archive / "train.txt")
    shutil.copy2(config["test_split"], split_archive / "test.txt")

    started = time.monotonic()
    for human_epoch in range(start_epoch, int(config["epochs"]) + 1):
        train_loader = make_train_loader(train_dataset, config, human_epoch)
        train_metrics = train_one_epoch(
            model,
            optimizer,
            train_loader,
            loss_function,
            device,
            human_epoch=human_epoch,
            warm_epochs=int(config["warm_epochs"]),
            max_batches=config["max_train_batches"],
        )
        append_jsonl(output_dir / "train_metrics.jsonl", train_metrics)

        test_metrics = None
        if human_epoch >= int(config["eval_start_epoch"]):
            test_metrics = evaluate(
                model,
                test_loader,
                device,
                human_epoch=human_epoch,
                image_size=int(config["base_size"]),
                max_images=config["max_test_images"],
            )
            append_jsonl(output_dir / "test_metrics.jsonl", test_metrics)
            for metric_name in ("miou", "pd"):
                value = float(test_metrics[metric_name])
                key = selection_key(metric_name, test_metrics)
                previous_key = tuple(best[metric_name]["key"])
                if key > previous_key:
                    best[metric_name] = {
                        "value": value,
                        "key": key,
                        "epoch": human_epoch,
                        "metrics": dict(test_metrics),
                    }
                    save_torch_atomic(
                        output_dir / f"best_{metric_name}.pth.tar",
                        _best_payload(
                            model,
                            config,
                            split_manifest,
                            test_metrics,
                            metric_name,
                        ),
                    )

        last_payload = {
            "schema_version": 1,
            "architecture": "MSHNet_NSFPN",
            "epoch": human_epoch,
            "state_dict": _cpu_state_dict(model),
            "optimizer": optimizer.state_dict(),
            "best": best,
            "latest_train_metrics": train_metrics,
            "latest_test_metrics": test_metrics,
            "split_manifest": split_manifest,
            "run_config": config,
        }
        save_torch_atomic(last_path, last_payload)
        status = (
            f"dataset={config['dataset']} epoch={human_epoch}/{config['epochs']} "
            f"loss={train_metrics['mean_loss']:.6f}"
        )
        if test_metrics is not None:
            status += (
                f" miou={test_metrics['miou']:.6f} pd={test_metrics['pd']:.6f} "
                f"fa_x1e6={test_metrics['fa_per_pixel_x1e6']:.6f}"
            )
        print(status, flush=True)

    summary = {
        "dataset": config["dataset"],
        "completed_epochs": int(config["epochs"]),
        "best": best,
        "duration_seconds": time.monotonic() - started,
        "output_dir": str(output_dir),
        "test_selected_checkpoints": True,
    }
    write_json_atomic(output_dir / "summary.json", summary)
    return summary


def main() -> None:
    args = build_parser().parse_args()
    summary = run_training(args)
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
