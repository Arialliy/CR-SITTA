"""Candidate mechanisms evaluated before CR-SITTA parameter adaptation."""

from .source_multiview_teacher import (
    CONTEXT_TILE_CROP_SIZE,
    CONTEXT_TILE_INPUT_SIZE,
    CONTEXT_TILE_ORIGINS,
    GEOMETRIC_VIEWS,
    AggregationMethod,
    GeometricView,
    LogitsAdapter,
    aggregate_aligned_probabilities,
    apply_geometric_view,
    build_aligned_multiview_teacher,
    build_aligned_view_probabilities,
    extract_context_tiles,
    frozen_probability_forward,
    infer_context_tile_probability,
    inverse_geometric_view,
    stitch_context_tile_probabilities,
)

__all__ = [
    "AggregationMethod",
    "CONTEXT_TILE_CROP_SIZE",
    "CONTEXT_TILE_INPUT_SIZE",
    "CONTEXT_TILE_ORIGINS",
    "GEOMETRIC_VIEWS",
    "GeometricView",
    "LogitsAdapter",
    "aggregate_aligned_probabilities",
    "apply_geometric_view",
    "build_aligned_multiview_teacher",
    "build_aligned_view_probabilities",
    "extract_context_tiles",
    "frozen_probability_forward",
    "infer_context_tile_probability",
    "inverse_geometric_view",
    "stitch_context_tile_probabilities",
]
