"""Deterministic physical-intensity corruptions for CR-SITTA experiments."""

from .corruption_protocol import (
    NON_CLEAN_CORRUPTIONS,
    SUPPORTED_CORRUPTIONS,
    SeverityTable,
    apply_sample_corruption,
    derive_sample_seed,
    get_default_severity_table,
    load_severity_table,
    make_sample_rng,
    validate_corruption_request,
)
from .infrared_corruptions import apply_corruption

__all__ = [
    "NON_CLEAN_CORRUPTIONS",
    "SUPPORTED_CORRUPTIONS",
    "SeverityTable",
    "apply_corruption",
    "apply_sample_corruption",
    "derive_sample_seed",
    "get_default_severity_table",
    "load_severity_table",
    "make_sample_rng",
    "validate_corruption_request",
]
