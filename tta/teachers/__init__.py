"""Detached teacher builders used by CR-SITTA Stage-C."""

from .strong_view_teacher import (
    StrongTeacherOutput,
    build_strong_teacher,
    build_strong_teacher_from_aligned_probabilities,
)


__all__ = [
    "StrongTeacherOutput",
    "build_strong_teacher",
    "build_strong_teacher_from_aligned_probabilities",
]
