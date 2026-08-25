"""Connected-component extraction shared by all research evaluators.

The protocol uses two-dimensional masks.  ``connectivity=1`` means 4-neighbour
connectivity and ``connectivity=2`` means 8-neighbour connectivity.  Keeping
this choice in the immutable evaluation protocol prevents methods from using
different post-processing rules.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Tuple

import numpy as np
from numpy.typing import NDArray
from scipy import ndimage


Centroid = Tuple[float, float]
BoundingBox = Tuple[int, int, int, int]


@dataclass(frozen=True)
class ConnectedComponent:
    """A foreground component in row/column image coordinates.

    ``bbox`` follows NumPy slicing semantics: ``(row_min, col_min,
    row_max_exclusive, col_max_exclusive)``.
    """

    label: int
    area: int
    centroid: Centroid
    bbox: BoundingBox


@dataclass(frozen=True)
class ComponentExtraction:
    """The labelled mask and its components, in ascending label order."""

    labels: NDArray[np.int32]
    components: Tuple[ConnectedComponent, ...]


def _as_binary_2d(mask: Any) -> NDArray[np.bool_]:
    if hasattr(mask, "detach"):
        mask = mask.detach().cpu().numpy()
    array = np.asarray(mask)
    if array.ndim != 2:
        raise ValueError(f"Expected a 2-D mask, got shape {array.shape}.")
    if not np.isfinite(array).all():
        raise ValueError("Mask contains NaN or Inf values.")
    return np.asarray(array > 0, dtype=np.bool_)


def _validate_connectivity(connectivity: int) -> None:
    if connectivity not in (1, 2):
        raise ValueError(
            "For 2-D masks, connectivity must be 1 (4-neighbour) or "
            "2 (8-neighbour)."
        )


def label_connected_components(
    mask: Any,
    *,
    connectivity: int = 2,
    min_area: int = 1,
) -> ComponentExtraction:
    """Label foreground components and return their geometry.

    Components smaller than ``min_area`` are removed before contiguous labels
    are assigned.  A zero mask returns an all-zero label image and an empty
    component tuple.
    """

    _validate_connectivity(connectivity)
    if isinstance(min_area, bool) or not isinstance(min_area, (int, np.integer)):
        raise TypeError("min_area must be an integer.")
    if min_area < 1:
        raise ValueError("min_area must be at least 1.")

    binary = _as_binary_2d(mask)
    structure = ndimage.generate_binary_structure(rank=2, connectivity=connectivity)
    raw_labels, component_count = ndimage.label(binary, structure=structure)
    labels = np.zeros(binary.shape, dtype=np.int32)
    components = []

    next_label = 1
    for raw_label in range(1, component_count + 1):
        coordinates = np.argwhere(raw_labels == raw_label)
        area = int(coordinates.shape[0])
        if area < min_area:
            continue

        labels[raw_labels == raw_label] = next_label
        row_min, col_min = coordinates.min(axis=0)
        row_max, col_max = coordinates.max(axis=0) + 1
        row_centroid, col_centroid = coordinates.mean(axis=0)
        components.append(
            ConnectedComponent(
                label=next_label,
                area=area,
                centroid=(float(row_centroid), float(col_centroid)),
                bbox=(int(row_min), int(col_min), int(row_max), int(col_max)),
            )
        )
        next_label += 1

    return ComponentExtraction(labels=labels, components=tuple(components))


def extract_connected_components(
    mask: Any,
    *,
    connectivity: int = 2,
    min_area: int = 1,
) -> Tuple[ConnectedComponent, ...]:
    """Return only component records for callers that do not need labels."""

    return label_connected_components(
        mask, connectivity=connectivity, min_area=min_area
    ).components


__all__ = [
    "BoundingBox",
    "Centroid",
    "ComponentExtraction",
    "ConnectedComponent",
    "extract_connected_components",
    "label_connected_components",
]
