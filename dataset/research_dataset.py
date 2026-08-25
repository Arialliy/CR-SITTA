"""Standardised, metadata-rich test dataset for CR-SITTA experiments.

The dataset deliberately requires an explicit split file and resolves either
the official ``img/label`` layout or the common ``images/masks`` layout.  It
does not import a corruption implementation.  Instead, callers may inject a
callable with this positional signature::

    transform(image_01, corruption, severity, rng) -> image_01

``image_01`` is an ``H x W x 3`` RGB NumPy array in the physical intensity
range ``[0, 1]``.  The callable must return a finite array with the same shape
and range.  ``rng`` is deterministically derived from ``(image_id,
corruption, severity, seed)``.  The transform runs after spatial resizing and
before ImageNet normalisation.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Protocol, Sequence, Tuple, Union

import numpy as np
from numpy.typing import NDArray
from PIL import Image
import torch
from torch.utils.data import Dataset

from corruptions.corruption_protocol import make_sample_rng, validate_corruption_request


ImageSize = Union[int, Tuple[int, int]]
DEFAULT_EXTENSIONS: Tuple[str, ...] = (
    ".png",
    ".jpg",
    ".jpeg",
    ".bmp",
    ".tif",
    ".tiff",
)
IMAGENET_MEAN = np.asarray((0.485, 0.456, 0.406), dtype=np.float32)
IMAGENET_STD = np.asarray((0.229, 0.224, 0.225), dtype=np.float32)


class CorruptionTransform(Protocol):
    """Injected physical-domain transform; no concrete corruption API assumed."""

    def __call__(
        self,
        image_01: NDArray[np.float32],
        corruption: str,
        severity: int,
        rng: np.random.Generator,
    ) -> Any:
        ...


@dataclass(frozen=True)
class DatasetLayout:
    images_dir: Path
    masks_dir: Path


@dataclass(frozen=True)
class _SampleRecord:
    image_id: str
    image_path: Path
    mask_path: Path


def _normalise_identifier(raw_identifier: str, *, line_number: int) -> str:
    identifier = raw_identifier.strip()
    if not identifier:
        raise ValueError(f"Split line {line_number} contains an empty identifier.")
    relative_path = Path(identifier)
    if relative_path.is_absolute() or ".." in relative_path.parts:
        raise ValueError(
            f"Split line {line_number} is not a safe relative identifier: "
            f"{identifier!r}."
        )
    return relative_path.as_posix()


def read_split_ids(split_file: Union[str, Path]) -> Tuple[str, ...]:
    """Read non-blank IDs from an explicit split file and reject duplicates."""

    split_path = Path(split_file).expanduser()
    if not split_path.is_file():
        raise FileNotFoundError(f"Split file does not exist: {split_path}")

    identifiers: List[str] = []
    first_line_by_identifier: Dict[str, int] = {}
    with split_path.open("r", encoding="utf-8-sig") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            if not raw_line.strip():
                continue
            identifier = _normalise_identifier(raw_line, line_number=line_number)
            if identifier in first_line_by_identifier:
                first_line = first_line_by_identifier[identifier]
                raise ValueError(
                    f"Duplicate image ID {identifier!r} in {split_path} at lines "
                    f"{first_line} and {line_number}."
                )
            first_line_by_identifier[identifier] = line_number
            identifiers.append(identifier)

    if not identifiers:
        raise ValueError(f"Split file contains no image IDs: {split_path}")
    return tuple(identifiers)


def resolve_dataset_layout(dataset_root: Union[str, Path]) -> DatasetLayout:
    """Resolve exactly one supported image/mask directory pair."""

    root = Path(dataset_root).expanduser()
    if not root.is_dir():
        raise FileNotFoundError(f"Dataset root does not exist: {root}")

    complete_layouts = []
    for image_directory_name, mask_directory_name in (
        ("img", "label"),
        ("images", "masks"),
    ):
        images_dir = root / image_directory_name
        masks_dir = root / mask_directory_name
        if images_dir.is_dir() and masks_dir.is_dir():
            complete_layouts.append(
                DatasetLayout(images_dir=images_dir, masks_dir=masks_dir)
            )

    if not complete_layouts:
        raise FileNotFoundError(
            f"Could not find either 'img/label' or 'images/masks' under {root}."
        )
    if len(complete_layouts) > 1:
        raise ValueError(
            f"Dataset root {root} contains both supported layouts; keep one layout "
            "or pass a root containing an unambiguous dataset."
        )
    return complete_layouts[0]


def _normalise_extensions(extensions: Sequence[str]) -> Tuple[str, ...]:
    if not extensions:
        raise ValueError("extensions cannot be empty.")
    normalised = []
    for extension in extensions:
        if not isinstance(extension, str) or not extension.strip():
            raise ValueError("Every extension must be a non-empty string.")
        extension = extension.strip().lower()
        if not extension.startswith("."):
            extension = f".{extension}"
        if extension not in normalised:
            normalised.append(extension)
    return tuple(normalised)


def _resolve_file(
    directory: Path,
    identifier: str,
    *,
    extensions: Tuple[str, ...],
    role: str,
) -> Path:
    identifier_path = Path(identifier)
    if identifier_path.suffix:
        candidates = [directory / identifier_path]
    else:
        candidates = [directory / f"{identifier}{extension}" for extension in extensions]
    existing = [candidate for candidate in candidates if candidate.is_file()]
    if not existing:
        searched = ", ".join(str(candidate) for candidate in candidates)
        raise FileNotFoundError(
            f"Missing {role} for image ID {identifier!r}; searched: {searched}"
        )
    if len(existing) > 1:
        matches = ", ".join(str(path) for path in existing)
        raise ValueError(
            f"Ambiguous {role} for image ID {identifier!r}; multiple files match: "
            f"{matches}"
        )
    return existing[0]


def _normalise_image_size(image_size: ImageSize) -> Tuple[int, int]:
    if isinstance(image_size, bool):
        raise ValueError("image_size must be a positive integer or (height, width).")
    if isinstance(image_size, (int, np.integer)):
        height = width = int(image_size)
    else:
        if not isinstance(image_size, (tuple, list)) or len(image_size) != 2:
            raise ValueError("image_size must contain exactly (height, width).")
        height, width = image_size
        if (
            isinstance(height, bool)
            or isinstance(width, bool)
            or not isinstance(height, (int, np.integer))
            or not isinstance(width, (int, np.integer))
        ):
            raise ValueError("image_size entries must be positive integers.")
        height, width = int(height), int(width)
    if height < 1 or width < 1:
        raise ValueError("image_size entries must be positive integers.")
    return height, width


def _validate_protocol_metadata(corruption: str, severity: int, seed: int) -> None:
    validate_corruption_request(corruption, severity)
    if (
        isinstance(seed, bool)
        or not isinstance(seed, (int, np.integer))
        or not 0 <= int(seed) < 2**64
    ):
        raise ValueError("seed must be an integer in [0, 2**64).")


def _sample_rng(
    image_id: str,
    corruption: str,
    severity: int,
    seed: int,
) -> np.random.Generator:
    return make_sample_rng(image_id, corruption, severity, seed)


def _validate_physical_image(
    image: Any,
    *,
    expected_shape: Tuple[int, int, int],
    source: str,
) -> NDArray[np.float32]:
    array = np.asarray(image)
    if array.shape != expected_shape:
        raise ValueError(
            f"{source} must return shape {expected_shape}, got {array.shape}."
        )
    if not np.issubdtype(array.dtype, np.number):
        raise TypeError(f"{source} must return a numeric array.")
    array = array.astype(np.float32, copy=False)
    if not np.isfinite(array).all():
        raise ValueError(f"{source} returned NaN or Inf values.")
    minimum = float(array.min())
    maximum = float(array.max())
    if minimum < 0.0 or maximum > 1.0:
        raise ValueError(
            f"{source} must stay in [0, 1], got range [{minimum}, {maximum}]."
        )
    return array


class IRSTDResearchDataset(Dataset):
    """Deterministic evaluation dataset with a standard CR-SITTA sample dict."""

    def __init__(
        self,
        dataset_root: Union[str, Path],
        *,
        split_file: Union[str, Path],
        image_size: ImageSize,
        dataset_name: str | None = None,
        corruption: str = "clean",
        severity: int = 0,
        seed: int = 42,
        corruption_transform: CorruptionTransform | None = None,
        extensions: Sequence[str] = DEFAULT_EXTENSIONS,
    ) -> None:
        root = Path(dataset_root).expanduser()
        self.layout = resolve_dataset_layout(root)
        split_path = Path(split_file).expanduser()
        if not split_path.is_absolute():
            split_path = root / split_path
        self.split_file = split_path
        identifiers = read_split_ids(split_path)
        self.output_size = _normalise_image_size(image_size)
        _validate_protocol_metadata(corruption, severity, seed)
        self.dataset_name = root.name if dataset_name is None else dataset_name
        if not isinstance(self.dataset_name, str) or not self.dataset_name.strip():
            raise ValueError("dataset_name must be a non-empty string.")
        if corruption_transform is not None and not callable(corruption_transform):
            raise TypeError("corruption_transform must be callable or None.")
        if corruption != "clean" and corruption_transform is None:
            raise ValueError(
                "a non-clean corruption requires corruption_transform; refusing "
                "to emit clean pixels with corrupted metadata"
            )
        self.corruption = corruption
        self.severity = int(severity)
        self.seed = int(seed)
        self.corruption_transform = corruption_transform
        self.extensions = _normalise_extensions(extensions)

        records = []
        seen_image_paths: Dict[Path, str] = {}
        seen_mask_paths: Dict[Path, str] = {}
        for identifier in identifiers:
            image_path = _resolve_file(
                self.layout.images_dir,
                identifier,
                extensions=self.extensions,
                role="image",
            )
            mask_path = _resolve_file(
                self.layout.masks_dir,
                identifier,
                extensions=self.extensions,
                role="mask",
            )
            resolved_image_path = image_path.resolve()
            resolved_mask_path = mask_path.resolve()
            if resolved_image_path in seen_image_paths:
                other_identifier = seen_image_paths[resolved_image_path]
                raise ValueError(
                    f"Image IDs {other_identifier!r} and {identifier!r} resolve to "
                    f"the same image file: {image_path}"
                )
            if resolved_mask_path in seen_mask_paths:
                other_identifier = seen_mask_paths[resolved_mask_path]
                raise ValueError(
                    f"Image IDs {other_identifier!r} and {identifier!r} resolve to "
                    f"the same mask file: {mask_path}"
                )
            seen_image_paths[resolved_image_path] = identifier
            seen_mask_paths[resolved_mask_path] = identifier
            records.append(
                _SampleRecord(
                    image_id=Path(identifier).with_suffix("").as_posix(),
                    image_path=image_path,
                    mask_path=mask_path,
                )
            )
        self._records = tuple(records)

    def __len__(self) -> int:
        return len(self._records)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        record = self._records[index]
        with Image.open(record.image_path) as image_handle:
            image = image_handle.convert("RGB")
            original_width, original_height = image.size
            output_height, output_width = self.output_size
            resized_image = image.resize(
                (output_width, output_height), resample=Image.Resampling.BILINEAR
            )
            image_01 = np.asarray(resized_image, dtype=np.float32) / 255.0

        with Image.open(record.mask_path) as mask_handle:
            mask = mask_handle.convert("L")
            # Resize image and mask independently, matching the official loader.
            # NUAA-SIRST/Misc_111 is a known upstream sample whose source image
            # and annotation dimensions differ, although both map to the same
            # frozen 256 x 256 evaluation grid.
            resized_mask = mask.resize(
                (output_width, output_height), resample=Image.Resampling.NEAREST
            )
            # Preserve the official ToTensor mask values in [0,1].  Most masks
            # are exactly binary, but IRSTD-1K contains a handful of non-255
            # boundary pixels whose legacy mIoU semantics must remain visible
            # to OfficialMetricAdapter.  UnifiedResearchEvaluator owns its own
            # explicit target binarisation rule.
            mask_01 = np.asarray(resized_mask, dtype=np.float32) / 255.0

        expected_shape = (output_height, output_width, 3)
        image_01 = _validate_physical_image(
            image_01, expected_shape=expected_shape, source="loaded image"
        )
        if self.corruption_transform is not None:
            rng = _sample_rng(
                record.image_id,
                self.corruption,
                self.severity,
                self.seed,
            )
            transformed = self.corruption_transform(
                image_01.copy(), self.corruption, self.severity, rng
            )
            image_01 = _validate_physical_image(
                transformed,
                expected_shape=expected_shape,
                source="corruption_transform",
            )

        normalised = (image_01 - IMAGENET_MEAN) / IMAGENET_STD
        image_tensor = torch.from_numpy(
            np.ascontiguousarray(normalised.transpose(2, 0, 1), dtype=np.float32)
        )
        mask_tensor = torch.from_numpy(
            np.ascontiguousarray(mask_01[None, ...], dtype=np.float32)
        )

        return {
            "image": image_tensor,
            "mask": mask_tensor,
            "image_id": record.image_id,
            "original_size": (original_height, original_width),
            "dataset": self.dataset_name,
            "corruption": self.corruption,
            "severity": self.severity,
            "seed": self.seed,
        }


__all__ = [
    "CorruptionTransform",
    "DEFAULT_EXTENSIONS",
    "DatasetLayout",
    "IMAGENET_MEAN",
    "IMAGENET_STD",
    "IRSTDResearchDataset",
    "read_split_ids",
    "resolve_dataset_layout",
]
