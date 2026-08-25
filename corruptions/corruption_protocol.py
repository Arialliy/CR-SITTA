"""Protocol helpers for reproducible, per-image corruption sampling.

The low-level :func:`corruptions.infrared_corruptions.apply_corruption` API
requires a caller-owned ``numpy.random.Generator``.  Dataset runners should
derive that generator with :func:`make_sample_rng`, which deliberately uses a
cryptographic digest instead of Python's process-randomised ``hash()``.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import hashlib
import json
import math
from numbers import Integral, Real
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np
import yaml


NON_CLEAN_CORRUPTIONS: tuple[str, ...] = (
    "gaussian_noise",
    "gaussian_blur",
    "low_contrast",
    "stripe_noise",
)
SUPPORTED_CORRUPTIONS: tuple[str, ...] = ("clean", *NON_CLEAN_CORRUPTIONS)
NON_CLEAN_SEVERITIES: tuple[int, ...] = (1, 2, 3, 4, 5)
_DEFAULT_TABLE_PATH = Path(__file__).with_name("severity_tables.yaml")
_SEED_PERSONALISATION = b"CRSITTA-seed-v1"


def _freeze_mapping(values: Mapping[Any, Any]) -> Mapping[Any, Any]:
    frozen: dict[Any, Any] = {}
    for key, value in values.items():
        frozen[key] = _freeze_mapping(value) if isinstance(value, Mapping) else value
    return MappingProxyType(frozen)


@dataclass(frozen=True)
class SeverityTable:
    """Validated corruption severity configuration.

    ``levels`` is deeply read-only so one experiment cannot accidentally alter
    the process-wide default table after it has been loaded.
    """

    schema_version: int
    status: str
    frozen: bool
    calibration_required: bool
    calibration_completed: bool
    calibration_scope: str
    levels: Mapping[str, Mapping[int, Mapping[str, float]]]
    strength_parameters: Mapping[str, str | None]
    monotonic_directions: Mapping[str, str]
    source_path: Path

    def parameters(self, corruption: str, severity: int) -> Mapping[str, float]:
        corruption, severity = validate_corruption_request(corruption, severity)
        return self.levels[corruption][severity]


def validate_corruption_request(corruption: str, severity: int) -> tuple[str, int]:
    """Validate and canonicalise a corruption/severity pair.

    Clean is intentionally represented only by severity 0.  Every stochastic
    or deterministic degradation is represented only by severities 1--5.
    """

    if not isinstance(corruption, str):
        raise TypeError("corruption must be a string")
    if corruption not in SUPPORTED_CORRUPTIONS:
        choices = ", ".join(SUPPORTED_CORRUPTIONS)
        raise ValueError(f"unsupported corruption {corruption!r}; choose one of: {choices}")
    if isinstance(severity, bool) or not isinstance(severity, Integral):
        raise TypeError("severity must be an integer")

    severity = int(severity)
    valid = (0,) if corruption == "clean" else NON_CLEAN_SEVERITIES
    if severity not in valid:
        raise ValueError(
            f"invalid severity {severity} for {corruption!r}; valid values are {valid}"
        )
    return corruption, severity


def _require_mapping(value: Any, context: str) -> Mapping[Any, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{context} must be a mapping")
    return value


def _validate_numeric_parameters(
    corruption: str, severity: int, parameters: Mapping[Any, Any]
) -> dict[str, float]:
    converted: dict[str, float] = {}
    for name, value in parameters.items():
        if not isinstance(name, str):
            raise ValueError(f"{corruption} severity {severity} has a non-string parameter")
        if isinstance(value, bool) or not isinstance(value, Real):
            raise ValueError(
                f"{corruption} severity {severity} parameter {name!r} must be numeric"
            )
        value = float(value)
        if not math.isfinite(value):
            raise ValueError(
                f"{corruption} severity {severity} parameter {name!r} must be finite"
            )
        converted[name] = value
    return converted


def _validate_algorithm_parameters(
    corruption: str, severity: int, parameters: Mapping[str, float]
) -> None:
    expected = {
        "clean": set(),
        "gaussian_noise": {"sigma"},
        "gaussian_blur": {"sigma"},
        "low_contrast": {"contrast_factor"},
        "stripe_noise": {"amplitude", "smooth_sigma_pixels"},
    }[corruption]
    if set(parameters) != expected:
        raise ValueError(
            f"{corruption} severity {severity} parameters must be exactly {sorted(expected)}"
        )

    if corruption in {"gaussian_noise", "gaussian_blur"}:
        if parameters["sigma"] <= 0.0:
            raise ValueError(f"{corruption} sigma must be positive")
    elif corruption == "low_contrast":
        if not 0.0 < parameters["contrast_factor"] <= 1.0:
            raise ValueError("low_contrast contrast_factor must be in (0, 1]")
    elif corruption == "stripe_noise":
        if parameters["amplitude"] <= 0.0:
            raise ValueError("stripe_noise amplitude must be positive")
        if parameters["smooth_sigma_pixels"] < 0.0:
            raise ValueError("stripe_noise smooth_sigma_pixels cannot be negative")


def load_severity_table(path: str | Path | None = None) -> SeverityTable:
    """Load and validate a severity table from YAML.

    The shipped table is versioned and frozen after its source-domain pilot.
    Target-test labels are never a valid calibration source.
    """

    source_path = _DEFAULT_TABLE_PATH if path is None else Path(path)
    with source_path.open("r", encoding="utf-8") as stream:
        raw = yaml.safe_load(stream)
    raw = _require_mapping(raw, "severity table")

    schema_version = raw.get("schema_version")
    if schema_version != 1:
        raise ValueError(f"unsupported severity table schema_version: {schema_version!r}")

    status = raw.get("status")
    if not isinstance(status, str) or not status:
        raise ValueError("severity table status must be a non-empty string")
    frozen = raw.get("frozen")
    if not isinstance(frozen, bool):
        raise ValueError("severity table frozen must be boolean")

    calibration = _require_mapping(raw.get("calibration"), "calibration")
    required = calibration.get("required")
    completed = calibration.get("completed")
    scope = calibration.get("scope")
    if not isinstance(required, bool) or not isinstance(completed, bool):
        raise ValueError("calibration required/completed must be boolean")
    if not isinstance(scope, str) or not scope:
        raise ValueError("calibration scope must be a non-empty string")
    if frozen and (required and not completed):
        raise ValueError("a table pending required calibration cannot be marked frozen")

    raw_corruptions = _require_mapping(raw.get("corruptions"), "corruptions")
    if set(raw_corruptions) != set(SUPPORTED_CORRUPTIONS):
        raise ValueError(
            "severity table must define exactly: " + ", ".join(SUPPORTED_CORRUPTIONS)
        )

    levels: dict[str, dict[int, Mapping[str, float]]] = {}
    strength_parameters: dict[str, str | None] = {}
    directions: dict[str, str] = {}

    for corruption in SUPPORTED_CORRUPTIONS:
        entry = _require_mapping(raw_corruptions[corruption], corruption)
        strength_parameter = entry.get("strength_parameter")
        direction = entry.get("monotonic")
        if corruption == "clean":
            if strength_parameter is not None or direction != "identity":
                raise ValueError("clean must use strength_parameter=null and monotonic=identity")
        else:
            if not isinstance(strength_parameter, str) or not strength_parameter:
                raise ValueError(f"{corruption} strength_parameter must be a string")
            if direction not in {"increasing", "decreasing"}:
                raise ValueError(f"{corruption} monotonic must be increasing or decreasing")

        raw_levels = _require_mapping(entry.get("levels"), f"{corruption}.levels")
        expected_severities = {0} if corruption == "clean" else set(NON_CLEAN_SEVERITIES)
        normalised_keys: set[int] = set()
        for key in raw_levels:
            if isinstance(key, bool) or not isinstance(key, Integral):
                raise ValueError(f"{corruption} severity keys must be integers")
            normalised_keys.add(int(key))
        if normalised_keys != expected_severities:
            raise ValueError(
                f"{corruption} severity levels must be exactly {sorted(expected_severities)}"
            )

        corruption_levels: dict[int, Mapping[str, float]] = {}
        for severity in sorted(expected_severities):
            parameters = _validate_numeric_parameters(
                corruption,
                severity,
                _require_mapping(raw_levels[severity], f"{corruption}.levels.{severity}"),
            )
            _validate_algorithm_parameters(corruption, severity, parameters)
            corruption_levels[severity] = MappingProxyType(parameters)

        if corruption != "clean":
            assert isinstance(strength_parameter, str)
            if any(
                strength_parameter not in corruption_levels[severity]
                for severity in NON_CLEAN_SEVERITIES
            ):
                raise ValueError(
                    f"{corruption} strength_parameter {strength_parameter!r} is absent"
                )
            values = [
                corruption_levels[severity][strength_parameter]
                for severity in NON_CLEAN_SEVERITIES
            ]
            pairs = zip(values, values[1:])
            is_monotonic = (
                all(left < right for left, right in pairs)
                if direction == "increasing"
                else all(left > right for left, right in pairs)
            )
            if not is_monotonic:
                raise ValueError(
                    f"{corruption} strength values must be strictly {direction}"
                )

        levels[corruption] = corruption_levels
        strength_parameters[corruption] = strength_parameter
        directions[corruption] = direction

    return SeverityTable(
        schema_version=1,
        status=status,
        frozen=frozen,
        calibration_required=required,
        calibration_completed=completed,
        calibration_scope=scope,
        levels=_freeze_mapping(levels),
        strength_parameters=MappingProxyType(strength_parameters),
        monotonic_directions=MappingProxyType(directions),
        source_path=source_path.resolve(),
    )


@lru_cache(maxsize=1)
def get_default_severity_table() -> SeverityTable:
    """Return the validated package-local severity table."""

    return load_severity_table(_DEFAULT_TABLE_PATH)


def derive_sample_seed(
    image_id: str,
    corruption: str,
    severity: int,
    base_seed: int = 42,
) -> int:
    """Derive a stable unsigned 64-bit seed for one corruption sample."""

    corruption, severity = validate_corruption_request(corruption, severity)
    if not isinstance(image_id, str) or not image_id:
        raise ValueError("image_id must be a non-empty string")
    if isinstance(base_seed, bool) or not isinstance(base_seed, Integral):
        raise TypeError("base_seed must be an integer")
    base_seed = int(base_seed)
    if not 0 <= base_seed < 2**64:
        raise ValueError("base_seed must be in [0, 2**64)")

    payload = json.dumps(
        ["cr-sitta-corruption-seed-v1", image_id, corruption, severity, base_seed],
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    digest = hashlib.blake2b(
        payload,
        digest_size=8,
        person=_SEED_PERSONALISATION,
    ).digest()
    return int.from_bytes(digest, byteorder="little", signed=False)


def make_sample_rng(
    image_id: str,
    corruption: str,
    severity: int,
    base_seed: int = 42,
) -> np.random.Generator:
    """Construct the canonical per-sample NumPy generator."""

    seed = derive_sample_seed(image_id, corruption, severity, base_seed)
    return np.random.default_rng(seed)


def apply_sample_corruption(
    image_01: np.ndarray,
    *,
    image_id: str,
    corruption: str,
    severity: int,
    base_seed: int = 42,
) -> np.ndarray:
    """Apply a corruption using the canonical per-sample seed derivation."""

    from .infrared_corruptions import apply_corruption

    rng = make_sample_rng(image_id, corruption, severity, base_seed)
    return apply_corruption(image_01, corruption, severity, rng)
