"""Validated, label-free view primitives for CR-SITTA Stage-B."""

from .irstd_views import (
    GEOMETRIC_VIEW_NAMES,
    IMAGENET_MEAN,
    IMAGENET_STD,
    INVERTIBLE_VIEW_REGISTRY,
    MILD_CONTRAST_FACTOR,
    NORMALIZED_RGB_INPUT_CONTRACT,
    STUDENT_PERTURBATION_REGISTRY,
    DetachedRegionWeights,
    InvertibleView,
    StudentPerturbation,
    apply_fixed_mild_contrast,
    build_detached_region_weights,
    validated_student_perturbations,
    validated_weak_views,
)

__all__ = [
    "DetachedRegionWeights",
    "GEOMETRIC_VIEW_NAMES",
    "IMAGENET_MEAN",
    "IMAGENET_STD",
    "INVERTIBLE_VIEW_REGISTRY",
    "InvertibleView",
    "MILD_CONTRAST_FACTOR",
    "NORMALIZED_RGB_INPUT_CONTRACT",
    "STUDENT_PERTURBATION_REGISTRY",
    "StudentPerturbation",
    "apply_fixed_mild_contrast",
    "build_detached_region_weights",
    "validated_student_perturbations",
    "validated_weak_views",
]
