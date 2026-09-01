"""Strict, CPU-only parser for the frozen Phase-D0 execution contract.

The D0 YAML used to contain fields that the diagnostic runner did not
necessarily consume.  Treating those fields as documentation would make it
possible for the YAML and the execution to disagree silently.  This module
therefore validates every root field and every nested value, including scope,
dataset/checkpoint bindings, ordered conditions, execution settings and the
complete negative-archive comparison contract, against frozen D0-v1 values.
Execution-facing values are returned as immutable dataclasses/read-only
mappings.

The supervised gradient named by this contract is deliberately narrow: it is
the post-warm SLSIoU loss evaluated on the Source *evaluation* graph with
frozen Source BN running statistics.  It is not claimed to reproduce the full
source-training runtime.  Likewise, the zero no-op floors below are an exact
bitwise-zero policy for retrospective D0 diagnosis; they are not a proposed
activity margin for a future v3 selector.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

import yaml


class D0ProtocolContractError(ValueError):
    """The D0 configuration differs from the frozen execution contract."""


@dataclass(frozen=True, order=True)
class CandidateContract:
    optimizer: str
    learning_rate: float

    @property
    def slug(self) -> str:
        learning_rate = format(self.learning_rate, ".0e").replace("e-0", "e-")
        return f"{self.optimizer}_lr_{learning_rate.replace('-', 'm').replace('+', 'p')}"


@dataclass(frozen=True)
class CacheContract:
    root: str
    protocol_path: str
    protocol_sha256: str
    subset_size_per_dataset: int
    condition_count: int
    target_path: str
    target_role: str


@dataclass(frozen=True)
class StrictForwardDeterminismContract:
    deterministic_algorithms_enabled: bool
    warn_only: bool
    cudnn_deterministic: bool
    cudnn_benchmark: bool


@dataclass(frozen=True)
class DeterminismContract:
    policy: str
    strict_forward: StrictForwardDeterminismContract
    temporary_backward_disable_scopes: tuple[str, str]
    restore_strict_policy_before_optimizer_step: bool
    restore_strict_policy_before_post_forward: bool
    restore_strict_policy_after_backward_exception: bool


@dataclass(frozen=True)
class MethodContract:
    name: str
    bn_protocol: str
    adaptable_parameters: str
    optimizer_steps_per_image: int
    entropy_backward_passes_per_image: int
    gradient_reuse: str
    entropy_eps: float
    determinism: DeterminismContract
    candidates: tuple[CandidateContract, ...]


@dataclass(frozen=True)
class SourceTaskGradientContract:
    contract: str
    implementation: str
    loss_reference: str
    human_epoch: int
    epoch_index: int
    warm_epochs: int
    warm_flag: bool
    expected_auxiliary_count: int
    output: str
    with_shape: bool
    loss_terms: int
    averaging_denominator: int
    model_runtime: str
    bn_protocol_during_diagnostic: str
    full_source_training_runtime_exact: bool
    role: str
    adaptation_gradient_uses_labels: bool


@dataclass(frozen=True)
class NullFloorContract:
    policy: str
    scope: str
    is_future_v3_activity_margin: bool
    parameter: float
    logit: float
    probability: float


@dataclass(frozen=True)
class OptimizerGeometryContract:
    runtime_hard_gate_reference: str
    cross_backend_cpu_storage_replay_reference: str
    continuous_ideal_reference: str
    small_gradient_threshold: float
    adam_near_sign_step_threshold: float
    verification_rtol: float
    verification_atol: float


@dataclass(frozen=True)
class AlignmentContract:
    first_order_zero_tolerance: float


@dataclass(frozen=True)
class EvaluationContract:
    probability_transform: str
    prediction_threshold: float
    threshold_rule: str
    near_threshold_interval: tuple[float, float]
    connectivity: int
    max_centroid_distance: float
    min_component_area: int
    no_op_null_floors: NullFloorContract
    optimizer_geometry: OptimizerGeometryContract
    alignment: AlignmentContract


@dataclass(frozen=True, order=True)
class EquivalenceSampleContract:
    dataset: str
    condition: str
    image_index: int
    image_id: str


@dataclass(frozen=True)
class EquivalenceContract:
    role: str
    samples: tuple[EquivalenceSampleContract, ...]
    candidate_scope: str
    pre_logits: str
    post_logits: str
    entropy_gradient_bundle_sha256: str
    parameter_delta_bundle_sha256: str
    changed_parameter_tensor_count: str
    step_norm_abs_tolerance: float
    historical_reset: str
    fresh_process_repetitions: int
    fresh_process_identity: str
    cross_process_pair_count: int
    method_label_accesses: int
    outer_evaluator_label_accesses: int
    target_payload_deserialized: bool
    test_images_opened: int
    test_labels_opened: int

    def sample_for(self, dataset: str) -> EquivalenceSampleContract:
        matches = tuple(sample for sample in self.samples if sample.dataset == dataset)
        if len(matches) != 1:
            raise D0ProtocolContractError(
                f"equivalence sample is not uniquely frozen for {dataset!r}"
            )
        return matches[0]


@dataclass(frozen=True)
class PerTensorParameterDiagnosticsContract:
    formal: str
    smoke: str


@dataclass(frozen=True)
class OutputContract:
    root: str
    canonical_shards: str
    smoke_shards: str
    equivalence: str
    aggregate: str
    refuse_overwrite: bool
    atomic_publish: bool
    raw_logits_written: bool
    raw_parameters_written: bool
    raw_gradients_written: bool
    per_tensor_parameter_diagnostics: PerTensorParameterDiagnosticsContract


@dataclass(frozen=True)
class ComparisonContract:
    frozen_v2_negative_archive: str
    archive_source_inventory_sha256: str
    archive_file_count: int
    stage1_records: str
    stage1_records_sha256: str
    direct_equality_required_only_for_full_64_image_cells: bool
    smoke_comparison_role: str


@dataclass(frozen=True)
class D0ProtocolContract:
    """Fully validated, immutable D0-v1 execution contract."""

    schema_version: int
    protocol_id: str
    cache: CacheContract
    method: MethodContract
    source_task_gradient: SourceTaskGradientContract
    evaluation: EvaluationContract
    equivalence: EquivalenceContract
    output: OutputContract
    comparison: ComparisonContract

    def as_frozen_mapping(self) -> Mapping[str, Any]:
        """Return the execution-facing contract as a deeply read-only mapping."""

        value: dict[str, Any] = {
            "schema_version": self.schema_version,
            "protocol_id": self.protocol_id,
            "cache": {
                "root": self.cache.root,
                "protocol_path": self.cache.protocol_path,
                "protocol_sha256": self.cache.protocol_sha256,
                "subset_size_per_dataset": self.cache.subset_size_per_dataset,
                "condition_count": self.cache.condition_count,
                "target_path": self.cache.target_path,
                "target_role": self.cache.target_role,
            },
            "method": {
                "name": self.method.name,
                "bn_protocol": self.method.bn_protocol,
                "adaptable_parameters": self.method.adaptable_parameters,
                "optimizer_steps_per_image": self.method.optimizer_steps_per_image,
                "entropy_backward_passes_per_image": (
                    self.method.entropy_backward_passes_per_image
                ),
                "gradient_reuse": self.method.gradient_reuse,
                "entropy_eps": self.method.entropy_eps,
                "determinism": {
                    "policy": self.method.determinism.policy,
                    "strict_forward": {
                        "deterministic_algorithms_enabled": (
                            self.method.determinism.strict_forward
                            .deterministic_algorithms_enabled
                        ),
                        "warn_only": (
                            self.method.determinism.strict_forward.warn_only
                        ),
                        "cudnn_deterministic": (
                            self.method.determinism.strict_forward
                            .cudnn_deterministic
                        ),
                        "cudnn_benchmark": (
                            self.method.determinism.strict_forward.cudnn_benchmark
                        ),
                    },
                    "temporary_backward_disable_scopes": (
                        self.method.determinism.temporary_backward_disable_scopes
                    ),
                    "restore_strict_policy_before_optimizer_step": (
                        self.method.determinism
                        .restore_strict_policy_before_optimizer_step
                    ),
                    "restore_strict_policy_before_post_forward": (
                        self.method.determinism
                        .restore_strict_policy_before_post_forward
                    ),
                    "restore_strict_policy_after_backward_exception": (
                        self.method.determinism
                        .restore_strict_policy_after_backward_exception
                    ),
                },
                "candidates": tuple(
                    {
                        "optimizer": candidate.optimizer,
                        "learning_rate": candidate.learning_rate,
                    }
                    for candidate in self.method.candidates
                ),
            },
            "source_task_gradient": {
                "contract": self.source_task_gradient.contract,
                "implementation": self.source_task_gradient.implementation,
                "loss_reference": self.source_task_gradient.loss_reference,
                "human_epoch": self.source_task_gradient.human_epoch,
                "epoch_index": self.source_task_gradient.epoch_index,
                "warm_epochs": self.source_task_gradient.warm_epochs,
                "warm_flag": self.source_task_gradient.warm_flag,
                "expected_auxiliary_count": (
                    self.source_task_gradient.expected_auxiliary_count
                ),
                "output": self.source_task_gradient.output,
                "with_shape": self.source_task_gradient.with_shape,
                "loss_terms": self.source_task_gradient.loss_terms,
                "averaging_denominator": (
                    self.source_task_gradient.averaging_denominator
                ),
                "model_runtime": self.source_task_gradient.model_runtime,
                "bn_protocol_during_diagnostic": (
                    self.source_task_gradient.bn_protocol_during_diagnostic
                ),
                "full_source_training_runtime_exact": (
                    self.source_task_gradient.full_source_training_runtime_exact
                ),
                "role": self.source_task_gradient.role,
                "adaptation_gradient_uses_labels": (
                    self.source_task_gradient.adaptation_gradient_uses_labels
                ),
            },
            "evaluation": {
                "probability_transform": self.evaluation.probability_transform,
                "prediction_threshold": self.evaluation.prediction_threshold,
                "threshold_rule": self.evaluation.threshold_rule,
                "near_threshold_interval": self.evaluation.near_threshold_interval,
                "connectivity": self.evaluation.connectivity,
                "max_centroid_distance": self.evaluation.max_centroid_distance,
                "min_component_area": self.evaluation.min_component_area,
                "no_op_null_floors": {
                    "policy": self.evaluation.no_op_null_floors.policy,
                    "scope": self.evaluation.no_op_null_floors.scope,
                    "is_future_v3_activity_margin": (
                        self.evaluation.no_op_null_floors.is_future_v3_activity_margin
                    ),
                    "parameter": self.evaluation.no_op_null_floors.parameter,
                    "logit": self.evaluation.no_op_null_floors.logit,
                    "probability": self.evaluation.no_op_null_floors.probability,
                },
                "optimizer_geometry": {
                    "runtime_hard_gate_reference": (
                        self.evaluation.optimizer_geometry
                        .runtime_hard_gate_reference
                    ),
                    "cross_backend_cpu_storage_replay_reference": (
                        self.evaluation.optimizer_geometry
                        .cross_backend_cpu_storage_replay_reference
                    ),
                    "continuous_ideal_reference": (
                        self.evaluation.optimizer_geometry.continuous_ideal_reference
                    ),
                    "small_gradient_threshold": (
                        self.evaluation.optimizer_geometry.small_gradient_threshold
                    ),
                    "adam_near_sign_step_threshold": (
                        self.evaluation.optimizer_geometry.adam_near_sign_step_threshold
                    ),
                    "verification_rtol": (
                        self.evaluation.optimizer_geometry.verification_rtol
                    ),
                    "verification_atol": (
                        self.evaluation.optimizer_geometry.verification_atol
                    ),
                },
                "alignment": {
                    "first_order_zero_tolerance": (
                        self.evaluation.alignment.first_order_zero_tolerance
                    )
                },
            },
            "equivalence": {
                "role": self.equivalence.role,
                "samples": {
                    sample.dataset: {
                        "condition": sample.condition,
                        "image_index": sample.image_index,
                        "image_id": sample.image_id,
                    }
                    for sample in self.equivalence.samples
                },
                "candidate_scope": self.equivalence.candidate_scope,
                "pre_logits": self.equivalence.pre_logits,
                "post_logits": self.equivalence.post_logits,
                "entropy_gradient_bundle_sha256": (
                    self.equivalence.entropy_gradient_bundle_sha256
                ),
                "parameter_delta_bundle_sha256": (
                    self.equivalence.parameter_delta_bundle_sha256
                ),
                "changed_parameter_tensor_count": (
                    self.equivalence.changed_parameter_tensor_count
                ),
                "step_norm_abs_tolerance": (
                    self.equivalence.step_norm_abs_tolerance
                ),
                "historical_reset": self.equivalence.historical_reset,
                "fresh_process_repetitions": (
                    self.equivalence.fresh_process_repetitions
                ),
                "fresh_process_identity": self.equivalence.fresh_process_identity,
                "cross_process_pair_count": self.equivalence.cross_process_pair_count,
                "method_label_accesses": self.equivalence.method_label_accesses,
                "outer_evaluator_label_accesses": (
                    self.equivalence.outer_evaluator_label_accesses
                ),
                "target_payload_deserialized": (
                    self.equivalence.target_payload_deserialized
                ),
                "test_images_opened": self.equivalence.test_images_opened,
                "test_labels_opened": self.equivalence.test_labels_opened,
            },
            "output": {
                "root": self.output.root,
                "canonical_shards": self.output.canonical_shards,
                "smoke_shards": self.output.smoke_shards,
                "equivalence": self.output.equivalence,
                "aggregate": self.output.aggregate,
                "refuse_overwrite": self.output.refuse_overwrite,
                "atomic_publish": self.output.atomic_publish,
                "raw_logits_written": self.output.raw_logits_written,
                "raw_parameters_written": self.output.raw_parameters_written,
                "raw_gradients_written": self.output.raw_gradients_written,
                "per_tensor_parameter_diagnostics": {
                    "formal": (
                        self.output.per_tensor_parameter_diagnostics.formal
                    ),
                    "smoke": self.output.per_tensor_parameter_diagnostics.smoke,
                },
            },
            "comparison": {
                "frozen_v2_negative_archive": (
                    self.comparison.frozen_v2_negative_archive
                ),
                "archive_source_inventory_sha256": (
                    self.comparison.archive_source_inventory_sha256
                ),
                "archive_file_count": self.comparison.archive_file_count,
                "stage1_records": self.comparison.stage1_records,
                "stage1_records_sha256": self.comparison.stage1_records_sha256,
                "direct_equality_required_only_for_full_64_image_cells": (
                    self.comparison
                    .direct_equality_required_only_for_full_64_image_cells
                ),
                "smoke_comparison_role": self.comparison.smoke_comparison_role,
            },
        }
        return _deep_freeze(value)


_EXPECTED_CANDIDATES: tuple[Mapping[str, Any], ...] = tuple(
    {"optimizer": optimizer, "learning_rate": learning_rate}
    for optimizer in ("Adam", "SGD")
    for learning_rate in (1.0e-5, 3.0e-5, 1.0e-4, 3.0e-4, 1.0e-3)
)

_EXPECTED_SCOPE: Mapping[str, Any] = {
    "phase": "D0_existing_result_diagnosis",
    "source_train_derived": True,
    "paper_result": False,
    "paper_test_result": False,
    "oracle_analysis": True,
    "no_validation_split": True,
    "allowed_split_roles": ("train", "test_metadata_only_for_leakage_guard"),
    "use_test_images": False,
    "use_test_labels": False,
    "method_label_accesses": 0,
    "disclosure": (
        "This is an outer-oracle diagnosis of the frozen Binary TENT Stage-1 "
        "negative result. Images and masks come only from the frozen 64-image "
        "source-train cache. Train masks are unavailable to the method and are "
        "deserialized only after every selected label-free episode in a cell "
        "has completed. This artifact is not a paper test result and cannot "
        "select or authorize a Stage-2 candidate."
    ),
}

_EXPECTED_DATASETS: Mapping[str, Any] = {
    "IRSTD-1K": {
        "train_split_sha256": (
            "689a5f30a394ad47315ebe0f6df2d7f12429aa314ffb2cdf86f7fbd7be4ee744"
        ),
        "checkpoint": {
            "role": "best_miou",
            "path": "results/retraining_fixed_split/IRSTD-1K/best_miou.pth.tar",
            "sha256": (
                "ee8d5c3af67d7c93ff6be25b169bd8fd59afcd776afd173b4e230ad8dc8aa8b2"
            ),
        },
    },
    "NUAA-SIRST": {
        "train_split_sha256": (
            "324e5dadcb6cc9fc2a99a5f5dedd06ad4de77b2ed826e4ceffda8b6a784da0b4"
        ),
        "checkpoint": {
            "role": "best_miou",
            "path": "results/retraining_fixed_split/NUAA-SIRST/best_miou.pth.tar",
            "sha256": (
                "23feea50a847cdfc9874fe67d338a3884d9afa75d7a27b9b71a62ff2d00379cf"
            ),
        },
    },
    "NUDT-SIRST": {
        "train_split_sha256": (
            "e0a79f7c3d42548ba7d7dad9d2d336012b63a6bc5081e89e286f0f45036f8ec3"
        ),
        "checkpoint": {
            "role": "best_miou",
            "path": "results/retraining_fixed_split/NUDT-SIRST/best_miou.pth.tar",
            "sha256": (
                "a67d3a7e2597d201e1fecf32b7f218aefa8274ad0443d2726197d84f44d1343e"
            ),
        },
    },
}

_EXPECTED_CONDITIONS: tuple[str, ...] = (
    "clean_S0",
    "gaussian_noise_S1",
    "gaussian_noise_S3",
    "gaussian_noise_S5",
    "gaussian_blur_S1",
    "gaussian_blur_S3",
    "gaussian_blur_S5",
    "low_contrast_S1",
    "low_contrast_S3",
    "low_contrast_S5",
    "stripe_noise_S1",
    "stripe_noise_S3",
    "stripe_noise_S5",
)

_EXPECTED_CACHE: Mapping[str, Any] = {
    "root": "results/binary_tent/ss_calibration_cache_v2",
    "protocol_path": "configs/binary_tent_ss_calibration_cache_v2.yaml",
    "protocol_sha256": (
        "e225311cce252125eaf3a2b47eeeea4cde60d0363c8c71f48494d70e41c2aa3e"
    ),
    "subset_size_per_dataset": 64,
    "condition_count": 13,
    "target_path": "outer_evaluator/targets.npy",
    "target_role": "outer_evaluator_only_after_cell_label_free_completion",
}

_EXPECTED_METHOD: Mapping[str, Any] = {
    "name": "Binary Episodic TENT",
    "bn_protocol": "source_running_statistics",
    "adaptable_parameters": "all_batchnorm2d_affine",
    "optimizer_steps_per_image": 1,
    "entropy_backward_passes_per_image": 1,
    "gradient_reuse": (
        "one_backward_cloned_to_each_fresh_empty_optimizer_with_bit_exact_"
        "parameter_reset"
    ),
    "entropy_eps": 1.0e-6,
    "determinism": {
        "policy": "strict_forwards_temporary_backward_disable",
        "strict_forward": {
            "deterministic_algorithms_enabled": True,
            "warn_only": False,
            "cudnn_deterministic": True,
            "cudnn_benchmark": False,
        },
        "temporary_backward_disable_scopes": (
            "entropy_backward",
            "supervised_task_backward",
        ),
        "restore_strict_policy_before_optimizer_step": True,
        "restore_strict_policy_before_post_forward": True,
        "restore_strict_policy_after_backward_exception": True,
    },
    "candidates": _EXPECTED_CANDIDATES,
}

_EXPECTED_SOURCE_TASK: Mapping[str, Any] = {
    "contract": "post_warm_SLSIoU_on_source_eval_graph",
    "implementation": "model.loss.SLSIoULoss",
    "loss_reference": "train_fixed_split.py::train_one_epoch",
    "human_epoch": 1000,
    "epoch_index": 999,
    "warm_epochs": 5,
    "warm_flag": False,
    "expected_auxiliary_count": 0,
    "output": "output_0_x_d0",
    "with_shape": True,
    "loss_terms": 1,
    "averaging_denominator": 1,
    "model_runtime": "source_eval",
    "bn_protocol_during_diagnostic": "source_running_statistics",
    "full_source_training_runtime_exact": False,
    "role": "outer_oracle_train_labels_only",
    "adaptation_gradient_uses_labels": False,
}

_EXPECTED_EVALUATION: Mapping[str, Any] = {
    "probability_transform": "sigmoid_once",
    "prediction_threshold": 0.5,
    "threshold_rule": "strict_greater_than",
    "near_threshold_interval": (0.45, 0.55),
    "connectivity": 2,
    "max_centroid_distance": 3.0,
    "min_component_area": 1,
    "no_op_null_floors": {
        "policy": "exact_bitwise_zero",
        "scope": "D0_existing_result_diagnosis_only",
        "is_future_v3_activity_margin": False,
        "parameter": 0.0,
        "logit": 0.0,
        "probability": 0.0,
    },
    "optimizer_geometry": {
        "runtime_hard_gate_reference": (
            "pytorch_2_1_2_single_tensor_same_device_native_dtype_"
            "bit_exact_after_v1"
        ),
        "cross_backend_cpu_storage_replay_reference": (
            "pytorch_2_1_2_cpu_snapshot_native_dtype_cross_backend_"
            "storage_replay_diagnostic_v1"
        ),
        "continuous_ideal_reference": (
            "continuous_float64_empty_state_first_step_v1"
        ),
        "small_gradient_threshold": 1.0e-8,
        "adam_near_sign_step_threshold": 0.9,
        "verification_rtol": 1.0e-5,
        "verification_atol": 1.0e-7,
    },
    "alignment": {"first_order_zero_tolerance": 0.0},
}

_EXPECTED_EQUIVALENCE: Mapping[str, Any] = {
    "role": "shared_gradient_vs_frozen_v2_independent_candidate_execution",
    "samples": {
        "IRSTD-1K": {
            "condition": "clean_S0",
            "image_index": 0,
            "image_id": "XDU102",
        },
        "NUAA-SIRST": {
            "condition": "clean_S0",
            "image_index": 0,
            "image_id": "Misc_421",
        },
        "NUDT-SIRST": {
            "condition": "clean_S0",
            "image_index": 0,
            "image_id": "000891",
        },
    },
    "candidate_scope": "all_10_frozen_candidates",
    "pre_logits": "bit_exact",
    "post_logits": "bit_exact",
    "entropy_gradient_bundle_sha256": "exact_equal",
    "parameter_delta_bundle_sha256": "exact_equal",
    "changed_parameter_tensor_count": "exact_equal",
    "step_norm_abs_tolerance": 1.0e-12,
    "historical_reset": "bit_exact_source",
    "fresh_process_repetitions": 3,
    "fresh_process_identity": "linux_pid_and_proc_start_time_ticks",
    "cross_process_pair_count": 3,
    "method_label_accesses": 0,
    "outer_evaluator_label_accesses": 0,
    "target_payload_deserialized": False,
    "test_images_opened": 0,
    "test_labels_opened": 0,
}

_EXPECTED_OUTPUT: Mapping[str, Any] = {
    "root": "results/cr_sitta/tent_failure_diagnostics_v1",
    "canonical_shards": "shards",
    "smoke_shards": "smoke",
    "equivalence": "equivalence",
    "aggregate": "aggregate",
    "refuse_overwrite": True,
    "atomic_publish": True,
    "raw_logits_written": False,
    "raw_parameters_written": False,
    "raw_gradients_written": False,
    "per_tensor_parameter_diagnostics": {
        "formal": "omitted_after_global_and_semantic_group_aggregation",
        "smoke": "retained",
    },
}

_EXPECTED_COMPARISON: Mapping[str, Any] = {
    "frozen_v2_negative_archive": (
        "results/binary_tent/ss_calibration_v2_negative_archive"
    ),
    "archive_source_inventory_sha256": (
        "bf8954dd2692f23e6b65761e4337f1daa77a64439386b05c6bb8df716800dfbf"
    ),
    "archive_file_count": 113,
    "stage1_records": "stage1/aggregate/stage1_records.jsonl",
    "stage1_records_sha256": (
        "ee6a21b29ceee32fa6f9f70c6509436971133f81e815596fd148716807bbe3d3"
    ),
    "direct_equality_required_only_for_full_64_image_cells": True,
    "smoke_comparison_role": (
        "contextual_full_cell_reference_not_direct_equality"
    ),
}

_ROOT_KEYS = frozenset(
    {
        "schema_version",
        "protocol_id",
        "created_at",
        "scope",
        "cache",
        "datasets",
        "conditions",
        "method",
        "source_task_gradient",
        "evaluation",
        "equivalence",
        "output",
        "comparison",
    }
)


def _deep_freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType(
            {str(key): _deep_freeze(child) for key, child in value.items()}
        )
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return tuple(_deep_freeze(child) for child in value)
    return value


def _require_mapping(value: Any, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise D0ProtocolContractError(f"{path} must be a mapping")
    if not all(isinstance(key, str) for key in value):
        raise D0ProtocolContractError(f"{path} keys must all be strings")
    return value


def _assert_exact(actual: Any, expected: Any, path: str) -> None:
    if isinstance(expected, Mapping):
        observed = _require_mapping(actual, path)
        missing = sorted(set(expected) - set(observed))
        unknown = sorted(set(observed) - set(expected))
        if missing or unknown:
            raise D0ProtocolContractError(
                f"{path} fields are not exact; missing={missing}, unknown={unknown}"
            )
        for key, expected_value in expected.items():
            _assert_exact(observed[key], expected_value, f"{path}.{key}")
        return
    if isinstance(expected, tuple):
        if not isinstance(actual, list):
            raise D0ProtocolContractError(f"{path} must be a YAML sequence")
        if len(actual) != len(expected):
            raise D0ProtocolContractError(
                f"{path} length drifted: expected {len(expected)}, got {len(actual)}"
            )
        for index, (observed, expected_value) in enumerate(
            zip(actual, expected, strict=True)
        ):
            _assert_exact(observed, expected_value, f"{path}[{index}]")
        return
    if type(actual) is not type(expected) or actual != expected:
        raise D0ProtocolContractError(
            f"{path} drifted: expected {expected!r}, got {actual!r}"
        )


def parse_d0_protocol_contract(value: Mapping[str, Any]) -> D0ProtocolContract:
    """Validate a loaded YAML mapping and return its frozen D0 contract."""

    root = _require_mapping(value, "root")
    missing = sorted(_ROOT_KEYS - set(root))
    unknown = sorted(set(root) - _ROOT_KEYS)
    if missing or unknown:
        raise D0ProtocolContractError(
            f"root fields are not exact; missing={missing}, unknown={unknown}"
        )
    _assert_exact(root["schema_version"], 1, "schema_version")
    _assert_exact(
        root["protocol_id"],
        "cr-sitta-tent-failure-diagnostics-v1",
        "protocol_id",
    )
    _assert_exact(root["created_at"], "2026-09-01", "created_at")
    _assert_exact(root["scope"], _EXPECTED_SCOPE, "scope")
    _assert_exact(root["datasets"], _EXPECTED_DATASETS, "datasets")
    _assert_exact(root["conditions"], _EXPECTED_CONDITIONS, "conditions")
    _assert_exact(root["cache"], _EXPECTED_CACHE, "cache")
    _assert_exact(root["method"], _EXPECTED_METHOD, "method")
    _assert_exact(
        root["source_task_gradient"],
        _EXPECTED_SOURCE_TASK,
        "source_task_gradient",
    )
    _assert_exact(root["evaluation"], _EXPECTED_EVALUATION, "evaluation")
    _assert_exact(root["equivalence"], _EXPECTED_EQUIVALENCE, "equivalence")
    _assert_exact(root["output"], _EXPECTED_OUTPUT, "output")
    _assert_exact(root["comparison"], _EXPECTED_COMPARISON, "comparison")

    method = _require_mapping(root["method"], "method")
    determinism = _require_mapping(method["determinism"], "method.determinism")
    strict_forward = _require_mapping(
        determinism["strict_forward"], "method.determinism.strict_forward"
    )
    candidates = tuple(
        CandidateContract(
            optimizer=str(candidate["optimizer"]),
            learning_rate=float(candidate["learning_rate"]),
        )
        for candidate in method["candidates"]
    )
    source = _require_mapping(root["source_task_gradient"], "source_task_gradient")
    evaluation = _require_mapping(root["evaluation"], "evaluation")
    floors = _require_mapping(
        evaluation["no_op_null_floors"], "evaluation.no_op_null_floors"
    )
    geometry = _require_mapping(
        evaluation["optimizer_geometry"], "evaluation.optimizer_geometry"
    )
    alignment = _require_mapping(
        evaluation["alignment"], "evaluation.alignment"
    )
    equivalence = _require_mapping(root["equivalence"], "equivalence")
    samples = _require_mapping(equivalence["samples"], "equivalence.samples")
    output = _require_mapping(root["output"], "output")
    per_tensor = _require_mapping(
        output["per_tensor_parameter_diagnostics"],
        "output.per_tensor_parameter_diagnostics",
    )
    cache = _require_mapping(root["cache"], "cache")
    comparison = _require_mapping(root["comparison"], "comparison")

    return D0ProtocolContract(
        schema_version=1,
        protocol_id="cr-sitta-tent-failure-diagnostics-v1",
        cache=CacheContract(**cache),
        method=MethodContract(
            name=str(method["name"]),
            bn_protocol=str(method["bn_protocol"]),
            adaptable_parameters=str(method["adaptable_parameters"]),
            optimizer_steps_per_image=int(method["optimizer_steps_per_image"]),
            entropy_backward_passes_per_image=int(
                method["entropy_backward_passes_per_image"]
            ),
            gradient_reuse=str(method["gradient_reuse"]),
            entropy_eps=float(method["entropy_eps"]),
            determinism=DeterminismContract(
                policy=str(determinism["policy"]),
                strict_forward=StrictForwardDeterminismContract(**strict_forward),
                temporary_backward_disable_scopes=tuple(
                    determinism["temporary_backward_disable_scopes"]
                ),
                restore_strict_policy_before_optimizer_step=bool(
                    determinism["restore_strict_policy_before_optimizer_step"]
                ),
                restore_strict_policy_before_post_forward=bool(
                    determinism["restore_strict_policy_before_post_forward"]
                ),
                restore_strict_policy_after_backward_exception=bool(
                    determinism["restore_strict_policy_after_backward_exception"]
                ),
            ),
            candidates=candidates,
        ),
        source_task_gradient=SourceTaskGradientContract(**source),
        evaluation=EvaluationContract(
            probability_transform=str(evaluation["probability_transform"]),
            prediction_threshold=float(evaluation["prediction_threshold"]),
            threshold_rule=str(evaluation["threshold_rule"]),
            near_threshold_interval=tuple(evaluation["near_threshold_interval"]),
            connectivity=int(evaluation["connectivity"]),
            max_centroid_distance=float(evaluation["max_centroid_distance"]),
            min_component_area=int(evaluation["min_component_area"]),
            no_op_null_floors=NullFloorContract(**floors),
            optimizer_geometry=OptimizerGeometryContract(**geometry),
            alignment=AlignmentContract(**alignment),
        ),
        equivalence=EquivalenceContract(
            role=str(equivalence["role"]),
            samples=tuple(
                EquivalenceSampleContract(dataset=dataset, **sample)
                for dataset, sample in samples.items()
            ),
            candidate_scope=str(equivalence["candidate_scope"]),
            pre_logits=str(equivalence["pre_logits"]),
            post_logits=str(equivalence["post_logits"]),
            entropy_gradient_bundle_sha256=str(
                equivalence["entropy_gradient_bundle_sha256"]
            ),
            parameter_delta_bundle_sha256=str(
                equivalence["parameter_delta_bundle_sha256"]
            ),
            changed_parameter_tensor_count=str(
                equivalence["changed_parameter_tensor_count"]
            ),
            step_norm_abs_tolerance=float(
                equivalence["step_norm_abs_tolerance"]
            ),
            historical_reset=str(equivalence["historical_reset"]),
            fresh_process_repetitions=int(
                equivalence["fresh_process_repetitions"]
            ),
            fresh_process_identity=str(equivalence["fresh_process_identity"]),
            cross_process_pair_count=int(
                equivalence["cross_process_pair_count"]
            ),
            method_label_accesses=int(equivalence["method_label_accesses"]),
            outer_evaluator_label_accesses=int(
                equivalence["outer_evaluator_label_accesses"]
            ),
            target_payload_deserialized=bool(
                equivalence["target_payload_deserialized"]
            ),
            test_images_opened=int(equivalence["test_images_opened"]),
            test_labels_opened=int(equivalence["test_labels_opened"]),
        ),
        output=OutputContract(
            root=str(output["root"]),
            canonical_shards=str(output["canonical_shards"]),
            smoke_shards=str(output["smoke_shards"]),
            equivalence=str(output["equivalence"]),
            aggregate=str(output["aggregate"]),
            refuse_overwrite=bool(output["refuse_overwrite"]),
            atomic_publish=bool(output["atomic_publish"]),
            raw_logits_written=bool(output["raw_logits_written"]),
            raw_parameters_written=bool(output["raw_parameters_written"]),
            raw_gradients_written=bool(output["raw_gradients_written"]),
            per_tensor_parameter_diagnostics=(
                PerTensorParameterDiagnosticsContract(**per_tensor)
            ),
        ),
        comparison=ComparisonContract(**comparison),
    )


def load_d0_protocol_contract(path: str | Path) -> D0ProtocolContract:
    """Load and strictly validate a non-symlink D0 YAML file."""

    unresolved = Path(path).expanduser()
    if unresolved.is_symlink():
        raise D0ProtocolContractError(
            f"D0 config missing or symlink: {unresolved.absolute()}"
        )
    source = unresolved.resolve()
    if not source.is_file():
        raise D0ProtocolContractError(f"D0 config missing or symlink: {source}")
    try:
        value = yaml.safe_load(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, yaml.YAMLError) as exc:
        raise D0ProtocolContractError(f"invalid D0 YAML: {source}") from exc
    if not isinstance(value, Mapping):
        raise D0ProtocolContractError("D0 YAML root must be a mapping")
    return parse_d0_protocol_contract(value)


__all__ = [
    "AlignmentContract",
    "CacheContract",
    "CandidateContract",
    "ComparisonContract",
    "DeterminismContract",
    "D0ProtocolContract",
    "D0ProtocolContractError",
    "EquivalenceContract",
    "EquivalenceSampleContract",
    "EvaluationContract",
    "MethodContract",
    "NullFloorContract",
    "OptimizerGeometryContract",
    "OutputContract",
    "PerTensorParameterDiagnosticsContract",
    "SourceTaskGradientContract",
    "StrictForwardDeterminismContract",
    "load_d0_protocol_contract",
    "parse_d0_protocol_contract",
]
