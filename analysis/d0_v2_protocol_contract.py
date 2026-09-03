"""Strict parser for the engineering-only D0-v2 protocol.

The canonical YAML is intentionally compared against an independently coded
literal contract.  Unknown, missing, reordered sequence, type-drifted, or
value-drifted fields all fail closed.  Loading this contract performs no
directory creation and grants no scientific or Stage-2 authorization.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
import hashlib
import json
from pathlib import Path
from types import MappingProxyType
from typing import Any

import yaml

from analysis.d0_v2_independent_candidate_contract import (
    FROZEN_CANDIDATES,
    PROTOCOL_ID,
    CandidateSpec,
)
from analysis.d0_v2_task_loss import D0V2TaskLossConfig
from tta.d0_secure_io import read_stable_regular_file
from tta.d0_v2_parameter_groups import (
    ELIGIBLE_GROUP_IDS,
    FINAL_HEAD_REASON,
    FROZEN_D0_V2_FINE_INVENTORY_SHA256,
)


class D0V2ProtocolContractError(ValueError):
    """The supplied configuration differs from the frozen D0-v2 contract."""


CONDITIONS = (
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


_EXPECTED: dict[str, Any] = {
    "schema_version": 2,
    "protocol_id": PROTOCOL_ID,
    "created_at": "2026-09-02",
    "scope": {
        "phase": "D0_v2_independent_candidate_engineering",
        "source_train_derived": True,
        "paper_result": False,
        "paper_test_result": False,
        "no_validation_split": True,
        "use_validation": False,
        "allowed_split_roles": [
            "train_pilot64",
            "test_metadata_only_for_leakage_guard",
        ],
        "use_test_images": False,
        "use_test_labels": False,
        "method_label_accesses": 0,
        "scientific_gate_status": "unresolved",
        "scientific_selection_performed": False,
        "stage2_authorized": False,
        "disclosure": (
            "Engineering-only D0-v2 diagnosis on the frozen 64-image "
            "source-train Pilot. It performs no validation, opens no test image "
            "or mask, makes no scientific selection, and cannot authorize Stage 2."
        ),
    },
    "cache": {
        "root": "results/binary_tent/ss_calibration_cache_v2",
        "protocol_path": "configs/binary_tent_ss_calibration_cache_v2.yaml",
        "protocol_sha256": (
            "e225311cce252125eaf3a2b47eeeea4cde60d0363c8c71f48494d70e41c2aa3e"
        ),
        "execution_protocol_path": (
            "configs/binary_tent_ss_calibration_execution_v2.yaml"
        ),
        "execution_protocol_sha256": (
            "d5ca162ad41e1d18df96ee235da1ff1fec282db1ae6068eef3e1466166b31264"
        ),
        "split_name": "train",
        "split_role": "frozen_pilot64",
        "subset_size_per_dataset": 64,
        "condition_count": 13,
        "target_path": "outer_evaluator/targets.npy",
        "target_load_timing": "after_all_10_label_free_candidates_in_cell",
    },
    "datasets": {
        "IRSTD-1K": {
            "train_split_sha256": (
                "689a5f30a394ad47315ebe0f6df2d7f12429aa314ffb2cdf86f7fbd7be4ee744"
            ),
            "checkpoint_role": "best_miou",
            "checkpoint_path": (
                "results/retraining_fixed_split/IRSTD-1K/best_miou.pth.tar"
            ),
            "checkpoint_sha256": (
                "ee8d5c3af67d7c93ff6be25b169bd8fd59afcd776afd173b4e230ad8dc8aa8b2"
            ),
        },
        "NUAA-SIRST": {
            "train_split_sha256": (
                "324e5dadcb6cc9fc2a99a5f5dedd06ad4de77b2ed826e4ceffda8b6a784da0b4"
            ),
            "checkpoint_role": "best_miou",
            "checkpoint_path": (
                "results/retraining_fixed_split/NUAA-SIRST/best_miou.pth.tar"
            ),
            "checkpoint_sha256": (
                "23feea50a847cdfc9874fe67d338a3884d9afa75d7a27b9b71a62ff2d00379cf"
            ),
        },
        "NUDT-SIRST": {
            "train_split_sha256": (
                "e0a79f7c3d42548ba7d7dad9d2d336012b63a6bc5081e89e286f0f45036f8ec3"
            ),
            "checkpoint_role": "best_miou",
            "checkpoint_path": (
                "results/retraining_fixed_split/NUDT-SIRST/best_miou.pth.tar"
            ),
            "checkpoint_sha256": (
                "a67d3a7e2597d201e1fecf32b7f218aefa8274ad0443d2726197d84f44d1343e"
            ),
        },
    },
    "conditions": list(CONDITIONS),
    "independent_candidate_execution": {
        "candidate_count": 10,
        "isolation_unit": "fresh_model_method_optimizer_per_candidate_per_image",
        "gradient_reuse": "forbidden",
        "shared_autograd_graph_across_candidates": False,
        "shared_gradient_buffers_across_candidates": False,
        "optimizer_state_entry_count_before_step": 0,
        "entropy_backward_passes_per_candidate_image": 1,
        "optimizer_steps_per_candidate_image": 1,
        "source_parameter_reset_exact": True,
        "source_runtime_reset_exact": True,
        "source_rng_reset_exact": True,
        "only_selected_parameters_trainable": True,
        "bn_protocol": "source_running_statistics",
        "amp_enabled": False,
        "candidates": [
            {
                "optimizer": candidate.optimizer,
                "learning_rate": candidate.learning_rate,
            }
            for candidate in FROZEN_CANDIDATES
        ],
    },
    "diagnostics": {
        "prediction_threshold": 0.5,
        "threshold_rule": "strict_probability_greater_than_0_5",
        "four_level_noop_order": [
            "parameter_numeric_noop",
            "functional_logit_noop",
            "threshold_noop",
            "metric_noop",
        ],
        "required_extensions": [
            "p95_abs_delta_logit",
            "p95_abs_delta_probability",
            "near_threshold_pixel_count_by_margin_bin",
            "foreground_probability_mass_sum_pre_post_delta",
            "largest_component_area_pre_post_delta",
        ],
        "margin_bins": [
            {
                "key": "ge_0_lt_1e_minus_4",
                "lower_inclusive": 0.0,
                "upper_exclusive": 1.0e-4,
            },
            {
                "key": "ge_1e_minus_4_lt_1e_minus_3",
                "lower_inclusive": 1.0e-4,
                "upper_exclusive": 1.0e-3,
            },
            {
                "key": "ge_1e_minus_3_lt_1e_minus_2",
                "lower_inclusive": 1.0e-3,
                "upper_exclusive": 1.0e-2,
            },
            {
                "key": "ge_1e_minus_2_lt_5e_minus_2",
                "lower_inclusive": 1.0e-2,
                "upper_exclusive": 5.0e-2,
            },
        ],
        "outside_margin_bin": {
            "key": "ge_5e_minus_2",
            "lower_inclusive": 5.0e-2,
        },
        "connectivity": 2,
        "min_component_area": 1,
    },
    "outer_oracle_task_loss": {
        "role": "train_only_outer_evaluator_gradient_alignment",
        "used_by_adaptation": False,
        "scientific_weight_status": "provisional_engineering_not_selected",
        "lambda_bce": 1.0,
        "lambda_soft_iou": 1.0,
        "eps": 1.0e-6,
        "bce_reduction": "mean_over_all_pixels",
        "soft_iou_reduction": "mean_over_sample",
        "empty_target_convention": (
            "epsilon_smoothed_union_penalizes_foreground"
        ),
    },
    "fine_parameter_groups": {
        "parameter_kind": "batchnorm2d_affine_only",
        "inventory_sha256": FROZEN_D0_V2_FINE_INVENTORY_SHA256,
        "eligible": list(ELIGIBLE_GROUP_IDS),
        "structurally_ineligible": {"final_head": FINAL_HEAD_REASON},
        "single_group_update_runner_implemented": False,
        "scientific_group_selection_performed": False,
    },
    "engineering_smoke": {
        "role": "engineering_smoke",
        "paper_result": False,
        "paper_test_result": False,
        "scientific_selection_performed": False,
        "formal_p3_complete": False,
        "stage2_authorized": False,
        "sample": {
            "dataset": "NUAA-SIRST",
            "condition": "clean_S0",
            "image_index": 0,
            "image_id": "Misc_421",
            "original_size": [225, 334],
            "split_name": "train",
            "split_role": "frozen_pilot64",
        },
        "execution": {
            "fresh_process_count": 3,
            "candidate_count_per_process": 10,
            "candidate_process_order": (
                "canonical_sequential_within_each_fresh_process"
            ),
            "subprocess_order": "sequential",
            "fresh_process_identity": "linux_pid_and_proc_start_time_ticks",
            "device_type": "cuda",
            "logical_device": "cuda:0",
            "visible_cuda_devices_per_worker": 1,
            "seed": 42,
            "model_method_optimizer_builds_per_process": 10,
            "backward_passes_per_candidate": 1,
            "optimizer_steps_per_candidate": 1,
            "gradient_reuse": "forbidden",
            "same_device_native_step_endpoint": (
                "bit_exact_per_parameter_and_optimizer_state"
            ),
            "full_source_reset_audit_per_candidate": True,
            "method_label_accesses": 0,
            "outer_evaluator_label_accesses": 0,
            "target_payload_deserialized": False,
            "test_images_opened": 0,
            "test_labels_opened": 0,
            "amp_enabled": False,
        },
        "method": {
            "name": "Binary Episodic TENT",
            "bn_protocol": "source_running_statistics",
            "adaptable_parameters": "all_batchnorm2d_affine",
            "entropy_eps": 1.0e-6,
            "diagnostic_detail": "global",
            "determinism": {
                "policy": "strict_forwards_temporary_backward_disable",
                "strict_forward": {
                    "deterministic_algorithms_enabled": True,
                    "warn_only": False,
                    "cudnn_deterministic": True,
                    "cudnn_benchmark": False,
                },
                "temporary_backward_disable_scopes": [
                    "entropy_backward",
                    "supervised_task_backward",
                ],
                "restore_strict_policy_before_optimizer_step": True,
                "restore_strict_policy_before_post_forward": True,
                "restore_strict_policy_after_backward_exception": True,
            },
        },
        "cross_process_gate": {
            "frozen_source_state_sha256": "exact_equal",
            "frozen_source_state_hash_contract": (
                "cr-sitta-d0-v2-source-state-excluding-candidate-optimizer-v1"
            ),
            "frozen_source_state_components": (
                "model_runtime_topology_gradients_extras"
            ),
            "candidate_optimizer_excluded_from_shared_hash": True,
            "candidate_optimizer_exact_reset_verified_per_candidate": True,
            "input_sha256": "exact_equal",
            "pre_logits_sha256": "exact_equal",
            "candidate_structure_and_integer_gates": "exact_equal",
            "raw_cuda_post_logits_sha256": "not_required_equal",
            "raw_cuda_entropy_gradient_bundle_sha256": "not_required_equal",
            "raw_cuda_parameter_delta_bundle_sha256": "not_required_equal",
            "floating_step_norm": "not_required_equal",
        },
        "runtime_binding": {
            "method_facing_input_seal": True,
            "seal_scope": (
                "nuaa_clean_method_input_without_target_or_test_payload"
            ),
            "frozen_v2_full_payload_seal_reused": False,
            "target_payload_bytes_opened": 0,
            "target_payload_deserialized": False,
            "test_split_files_opened": 0,
            "test_images_opened": 0,
            "test_labels_opened": 0,
            "full_byte_hash_at_process_entry_and_exit": True,
            "critical_code_paths": [
                "scripts/run_d0_v2_engineering_smoke.py",
                "analysis/d0_v2_protocol_contract.py",
                "analysis/d0_v2_independent_candidate_contract.py",
                "analysis/d0_v2_smoke_repro_contract.py",
                "analysis/d0_v2_task_loss.py",
                "analysis/analyze_tent_optimizer_geometry.py",
                "analysis/d0_determinism_runtime.py",
                "tta/d0_v2_candidate_worker.py",
                "tta/d0_v2_native_step.py",
                "tta/d0_v2_parameter_groups.py",
                "tta/binary_tent.py",
                "tta/binary_tent_fast_runner.py",
                "tta/binary_tent_fast_runner_v2.py",
                "tta/state_manager.py",
                "tta/model_adapter.py",
                "tta/d0_secure_io.py",
                "materialize_binary_tent_ss_calibration_cache_v2.py",
                "metrics/irstd_metrics.py",
                "metrics/connected_components.py",
                "metrics/target_matching.py",
                "metrics/official_metric_adapter.py",
                "model/MSHNet_NSFPN.py",
                "model/NS_FPN.py",
                "model/diff_cross_attns.py",
                "SFS_MSDeformAttn/ops/__init__.py",
                "SFS_MSDeformAttn/ops/functions/__init__.py",
                "SFS_MSDeformAttn/ops/functions/ms_deform_attn_func.py",
                "SFS_MSDeformAttn/ops/modules/__init__.py",
                "SFS_MSDeformAttn/ops/modules/ms_deform_attn.py",
                (
                    ".conda/lib/python3.10/site-packages/"
                    "MultiScaleDeformableAttention.cpython-310-x86_64-linux-gnu.so"
                ),
                "test_source.py",
            ],
        },
        "publication": {
            "relative_directory": "engineering_smoke",
            "artifact_role": "engineering_smoke_only",
            "atomic_no_replace": True,
            "refuse_overwrite": True,
            "publish_only_after_three_receipts_verify": True,
        },
    },
    "implementation_status": {
        "strict_config_contract": "implemented",
        "independent_candidate_ownership_and_receipt_contract": "implemented",
        "four_level_noop_extensions": "implemented",
        "outer_oracle_task_loss_pure_function": "implemented",
        "fine_group_inventory": "implemented",
        "gpu_engineering_smoke_runner": "implemented_not_executed",
        "fresh_process_smoke_aggregate": "implemented_not_executed",
        "formal_gpu_runner": "not_implemented",
        "formal_p3_complete": False,
    },
    "output": {
        "root": (
            "results/cr_sitta/tent_failure_diagnostics_v2_independent_candidates"
        ),
        "engineering_smoke": "engineering_smoke",
        "registration_status": "path_only_not_materialized",
        "create_on_contract_load": False,
        "formal_result_publication_authorized": False,
        "refuse_overwrite": True,
        "atomic_publish_required": True,
    },
    "sealed_v1_inputs": {
        "read_only_reference": True,
        "files": [
            {
                "path": "configs/tent_failure_diagnostics_v1.yaml",
                "sha256": (
                    "6f6cabbc9d4b43b54c3d2dc97539eb368daabe934aee0c7f0ef9a4f5735ef815"
                ),
            },
            {
                "path": "scripts/run_tent_failure_diagnostics.py",
                "sha256": (
                    "38df90a01938771eba705219fa632abb47bbd3f47b05186459153024bc970f1e"
                ),
            },
            {
                "path": "analysis/d0_protocol_contract.py",
                "sha256": (
                    "2bfbb2b58cab9419cbb6086d9e5b5b2f84221f8aac5565a5db8f7aa75d4498c5"
                ),
            },
            {
                "path": "analysis/d0_determinism_runtime.py",
                "sha256": (
                    "185a650076b4abcb63748558ea4fac637aa4d1e86896b66e3384d2995a63a41e"
                ),
            },
            {
                "path": "analysis/d0_equivalence_repro_contract.py",
                "sha256": (
                    "18560bec3467e0a286a2ae33dfddf209ef484d1eb9dbc0a059c280fc62e42615"
                ),
            },
        ],
    },
}


def _assert_exact(observed: Any, expected: Any, *, path: str) -> None:
    if isinstance(expected, Mapping):
        if not isinstance(observed, Mapping):
            raise D0V2ProtocolContractError(f"{path} must be a mapping")
        missing = sorted(set(expected) - set(observed))
        unknown = sorted(set(observed) - set(expected), key=str)
        if missing or unknown:
            raise D0V2ProtocolContractError(
                f"{path} schema drifted; missing={missing}, unknown={unknown}"
            )
        for key, expected_value in expected.items():
            _assert_exact(
                observed[key], expected_value, path=f"{path}.{key}"
            )
        return
    if isinstance(expected, list):
        if not isinstance(observed, list):
            raise D0V2ProtocolContractError(f"{path} must be a YAML list")
        if len(observed) != len(expected):
            raise D0V2ProtocolContractError(
                f"{path} length drifted: expected={len(expected)}, "
                f"observed={len(observed)}"
            )
        for index, (observed_value, expected_value) in enumerate(
            zip(observed, expected)
        ):
            _assert_exact(
                observed_value, expected_value, path=f"{path}[{index}]"
            )
        return
    if type(observed) is not type(expected) or observed != expected:
        raise D0V2ProtocolContractError(
            f"{path} drifted: expected={expected!r} ({type(expected).__name__}), "
            f"observed={observed!r} ({type(observed).__name__})"
        )


def _deep_freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({key: _deep_freeze(child) for key, child in value.items()})
    if isinstance(value, list):
        return tuple(_deep_freeze(child) for child in value)
    return value


def _deep_thaw(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _deep_thaw(child) for key, child in value.items()}
    if isinstance(value, tuple):
        return [_deep_thaw(child) for child in value]
    return value


@dataclass(frozen=True)
class D0V2ProtocolContract:
    schema_version: int
    protocol_id: str
    created_at: str
    config_file_sha256: str | None
    conditions: tuple[str, ...]
    candidates: tuple[CandidateSpec, ...]
    eligible_group_ids: tuple[str, ...]
    fine_inventory_sha256: str
    task_loss: D0V2TaskLossConfig
    output_root: str
    raw: Mapping[str, Any]

    @property
    def scientific_gate_status(self) -> str:
        return str(self.raw["scope"]["scientific_gate_status"])

    @property
    def stage2_authorized(self) -> bool:
        return bool(self.raw["scope"]["stage2_authorized"])

    @property
    def formal_p3_complete(self) -> bool:
        return bool(self.raw["implementation_status"]["formal_p3_complete"])

    def as_frozen_mapping(self) -> Mapping[str, Any]:
        return self.raw

    def canonical_mapping_sha256(self) -> str:
        payload = json.dumps(
            _deep_thaw(self.raw),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()


def parse_d0_v2_protocol_contract(
    value: Mapping[str, Any],
) -> D0V2ProtocolContract:
    """Validate every nested field against the pre-registered protocol."""

    if not isinstance(value, Mapping):
        raise D0V2ProtocolContractError("D0-v2 config must be a mapping")
    _assert_exact(value, _EXPECTED, path="config")
    task_section = value["outer_oracle_task_loss"]
    task_loss = D0V2TaskLossConfig.from_mapping(
        {
            key: task_section[key]
            for key in (
                "lambda_bce",
                "lambda_soft_iou",
                "eps",
                "bce_reduction",
                "soft_iou_reduction",
                "empty_target_convention",
            )
        }
    )
    return D0V2ProtocolContract(
        schema_version=2,
        protocol_id=PROTOCOL_ID,
        created_at="2026-09-02",
        config_file_sha256=None,
        conditions=CONDITIONS,
        candidates=FROZEN_CANDIDATES,
        eligible_group_ids=ELIGIBLE_GROUP_IDS,
        fine_inventory_sha256=FROZEN_D0_V2_FINE_INVENTORY_SHA256,
        task_loss=task_loss,
        output_root=_EXPECTED["output"]["root"],
        raw=_deep_freeze(value),
    )


def load_d0_v2_protocol_contract(
    path: str | Path,
) -> D0V2ProtocolContract:
    """Read a stable non-symlink YAML file; never create an output path."""

    try:
        snapshot = read_stable_regular_file(path)
    except (OSError, RuntimeError, ValueError) as exc:
        raise D0V2ProtocolContractError(
            f"cannot securely read D0-v2 contract: {path}"
        ) from exc
    try:
        decoded = snapshot.data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise D0V2ProtocolContractError("D0-v2 contract must be UTF-8") from exc
    try:
        value = yaml.safe_load(decoded)
    except yaml.YAMLError as exc:
        raise D0V2ProtocolContractError("D0-v2 contract is invalid YAML") from exc
    contract = parse_d0_v2_protocol_contract(value)
    return replace(contract, config_file_sha256=snapshot.sha256)


def verify_sealed_v1_inputs(
    contract: D0V2ProtocolContract,
    *,
    repository_root: str | Path,
) -> tuple[tuple[str, str], ...]:
    """Read-only verification that every referenced D0-v1 seal is unchanged."""

    if not isinstance(contract, D0V2ProtocolContract):
        raise D0V2ProtocolContractError(
            "contract must be D0V2ProtocolContract"
        )
    # Reparse to reject a deliberately mutated frozen instance.
    reparsed = parse_d0_v2_protocol_contract(_deep_thaw(contract.raw))
    root = Path(repository_root)
    verified: list[tuple[str, str]] = []
    files: Sequence[Mapping[str, str]] = reparsed.raw["sealed_v1_inputs"]["files"]
    for entry in files:
        relative = entry["path"]
        expected_sha256 = entry["sha256"]
        try:
            snapshot = read_stable_regular_file(root / relative)
        except (OSError, RuntimeError, ValueError) as exc:
            raise D0V2ProtocolContractError(
                f"cannot securely read sealed D0-v1 input: {relative}"
            ) from exc
        if snapshot.sha256 != expected_sha256:
            raise D0V2ProtocolContractError(
                f"sealed D0-v1 input drifted: {relative}; "
                f"expected={expected_sha256}, observed={snapshot.sha256}"
            )
        verified.append((relative, snapshot.sha256))
    return tuple(verified)


__all__ = [
    "CONDITIONS",
    "D0V2ProtocolContract",
    "D0V2ProtocolContractError",
    "load_d0_v2_protocol_contract",
    "parse_d0_v2_protocol_contract",
    "verify_sealed_v1_inputs",
]
