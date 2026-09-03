from __future__ import annotations

import copy
import hashlib
import json
import os

import pytest

from analysis.d0_v2_independent_candidate_contract import (
    FROZEN_CANDIDATES,
    IndependentCandidateReceipt,
)
from analysis.d0_v2_smoke_repro_contract import (
    D0V2SmokeReproContractError,
    FreshSubprocessIdentity,
    aggregate_three,
    build_d0_v2_smoke_process_receipt,
    canonical_d0_v2_smoke_process_receipt_bytes,
    canonical_d0_v2_smoke_repro_aggregate_bytes,
    d0_v2_smoke_process_receipt_sha256,
    d0_v2_smoke_repro_aggregate_sha256,
    read_proc_process_start_time_ticks,
    validate_d0_v2_smoke_process_receipt,
    validate_d0_v2_smoke_repro_aggregate,
)


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _candidate_receipt(
    process_index: int,
    candidate_index: int,
    *,
    config_label: str = "config",
    pre_label: str = "pre",
    gradient_tensor_count: int = 106,
) -> dict:
    candidate = FROZEN_CANDIDATES[candidate_index]
    return IndependentCandidateReceipt(
        candidate_index=candidate_index,
        candidate=candidate,
        config_sha256=_sha(config_label),
        dataset="NUAA-SIRST",
        condition="clean_S0",
        sample_index=0,
        sample_id="Misc_421",
        split_sha256=_sha("split"),
        checkpoint_sha256=_sha("checkpoint"),
        source_state_sha256=_sha("source-state"),
        runtime_sha256=_sha("runtime"),
        determinism_sha256=_sha("determinism"),
        input_sha256=_sha("input"),
        selected_parameter_names_sha256=_sha("selected-parameters"),
        pre_logits_sha256=_sha(pre_label),
        # CUDA-backward-dependent evidence intentionally differs by process.
        post_logits_sha256=_sha(f"post:{process_index}:{candidate_index}"),
        entropy_gradient_bundle_sha256=_sha(
            f"gradient:{process_index}:{candidate_index}"
        ),
        parameter_delta_bundle_sha256=_sha(
            f"delta:{process_index}:{candidate_index}"
        ),
        model_instance_id=f"process-{process_index}:model:{candidate_index}",
        method_instance_id=f"process-{process_index}:method:{candidate_index}",
        optimizer_instance_id=f"process-{process_index}:optimizer:{candidate_index}",
        autograd_graph_id=f"process-{process_index}:graph:{candidate_index}",
        backward_execution_id=f"process-{process_index}:backward:{candidate_index}",
        gradient_buffer_owner_id=(
            f"process-{process_index}:gradient-owner:{candidate_index}"
        ),
        gradient_tensor_count=gradient_tensor_count,
        changed_parameter_tensor_count=100,
        optimizer_state_entry_count_after_step=gradient_tensor_count,
        native_reference_parameter_tensor_count=gradient_tensor_count,
        native_reference_optimizer_state_tensor_count=(
            3 * gradient_tensor_count
            if candidate.optimizer == "Adam"
            else gradient_tensor_count
        ),
        step_norm_l2=(process_index + 1) * (candidate_index + 1) * 1.0e-4,
    ).to_dict()


def _candidate_receipts(
    process_index: int,
    *,
    config_label: str = "config",
    pre_label: str = "pre",
    gradient_tensor_count: int = 106,
) -> list[dict]:
    return [
        _candidate_receipt(
            process_index,
            index,
            config_label=config_label,
            pre_label=pre_label,
            gradient_tensor_count=gradient_tensor_count,
        )
        for index in range(len(FROZEN_CANDIDATES))
    ]


def _process_receipt(
    process_index: int,
    *,
    pid: int | None = None,
    start_ticks: int | None = None,
    config_label: str = "config",
    pre_label: str = "pre",
    gradient_tensor_count: int = 106,
) -> dict:
    return build_d0_v2_smoke_process_receipt(
        # Builder must canonicalize candidate order.
        list(
            reversed(
                _candidate_receipts(
                    process_index,
                    config_label=config_label,
                    pre_label=pre_label,
                    gradient_tensor_count=gradient_tensor_count,
                )
            )
        ),
        parent_run_nonce=_sha("parent-run"),
        child_launch_nonce=_sha(f"child:{process_index}"),
        command_sha256=_sha(f"command:{process_index}"),
        process_identity=FreshSubprocessIdentity(
            process_id=f"fresh-process-{process_index}",
            os_process_id=pid if pid is not None else 1000 + process_index,
            process_start_time_ticks=(
                start_ticks if start_ticks is not None else 5000 + process_index
            ),
        ),
    )


def _process_receipts() -> list[dict]:
    return [_process_receipt(index) for index in range(3)]


def test_process_receipt_round_trip_binds_fresh_identity_and_cell() -> None:
    receipt = _process_receipt(0)
    validated = validate_d0_v2_smoke_process_receipt(receipt)

    assert validated == receipt
    assert validated["fresh_subprocess"] is True
    assert validated["engineering_smoke"] is True
    assert validated["paper_result"] is False
    assert validated["formal_p3_complete"] is False
    assert validated["stage2_authorized"] is False
    assert validated["process_identity"] == {
        "process_id": "fresh-process-0",
        "os_process_id": 1000,
        "process_start_time_ticks": 5000,
    }
    assert validated["candidate_count"] == 10
    assert len(validated["candidate_receipts"]) == 10
    assert validated["candidate_slugs"] == [
        candidate.slug for candidate in FROZEN_CANDIDATES
    ]
    assert validated["cell_aggregate"]["independent_execution"][
        "engineering_gate_passed"
    ] is True
    assert validated["authorization"] == {
        "scientific_gate_status": "unresolved",
        "scientific_selection_performed": False,
    }
    assert validated["data_boundary"]["method_label_accesses"] == 0
    assert validated["data_boundary"][
        "train_target_payload_bytes_opened"
    ] == 0
    assert validated["data_boundary"]["test_split_files_opened"] == 0
    assert validated["data_boundary"]["test_images_opened"] == 0
    assert validated["data_boundary"]["test_masks_opened"] == 0
    assert validated["data_boundary"]["test_labels_opened"] == 0


def test_three_process_aggregate_allows_cuda_backward_numeric_differences() -> None:
    receipts = list(reversed(_process_receipts()))
    # Each process intentionally has different post/gradient/delta hashes and
    # floating step norms, while source/pre/shared bindings remain identical.
    assert len(
        {
            receipt["candidate_receipts"][0]["numeric_evidence"][
                "post_logits_sha256"
            ]
            for receipt in receipts
        }
    ) == 3
    assert len(
        {
            receipt["candidate_receipts"][0]["numeric_evidence"]["step_norm_l2"]
            for receipt in receipts
        }
    ) == 3

    aggregate = aggregate_three(receipts)

    assert aggregate["process_ids"] == [
        "fresh-process-0",
        "fresh-process-1",
        "fresh-process-2",
    ]
    assert aggregate["fresh_process_count"] == 3
    assert aggregate["candidate_receipts_per_process"] == 10
    assert aggregate["total_candidate_receipt_count"] == 30
    assert aggregate["engineering_smoke"] is True
    assert aggregate["paper_result"] is False
    assert aggregate["formal_p3_complete"] is False
    assert aggregate["stage2_authorized"] is False
    assert aggregate["cross_process_gates"][
        "candidate_integer_structure_gates_exact"
    ] is True
    assert aggregate["cross_process_gates"][
        "same_device_native_step_endpoint_exact_all_candidates"
    ] is True
    assert aggregate["cross_process_gates"][
        "same_device_native_optimizer_state_exact_all_candidates"
    ] is True
    assert aggregate["cross_process_policy"] == {
        "required_exact_shared_cell_fields": [
            "config_sha256",
            "dataset",
            "condition",
            "sample_index",
            "sample_id",
            "split_sha256",
            "checkpoint_sha256",
            "source_state_sha256",
            "runtime_sha256",
            "determinism_sha256",
            "input_sha256",
            "selected_parameter_names_sha256",
        ],
        "source_state_sha256_contract": (
            "cr-sitta-d0-v2-source-state-excluding-candidate-optimizer-v1"
        ),
        "source_state_components": (
            "model_runtime_topology_gradients_extras"
        ),
        "candidate_optimizer_excluded_from_shared_source_hash": True,
        "candidate_optimizer_exact_reset_required_per_candidate": True,
        "required_exact_pre_logits_sha256": True,
        "required_exact_candidate_integer_fields": [
            "gradient_tensor_count",
            "changed_parameter_tensor_count",
            "optimizer_state_entry_count_after_step",
            "native_reference_parameter_tensor_count",
            "native_reference_optimizer_state_tensor_count",
        ],
        "cuda_backward_bit_exact_required": False,
        "post_logits_sha256_cross_process_exact_required": False,
        "entropy_gradient_bundle_sha256_cross_process_exact_required": False,
        "parameter_delta_bundle_sha256_cross_process_exact_required": False,
        "step_norm_l2_cross_process_exact_required": False,
    }
    json.dumps(aggregate, allow_nan=False)


def test_process_and_aggregate_canonical_bytes_and_hashes_are_stable() -> None:
    process = _process_receipt(0)
    aggregate = aggregate_three(_process_receipts())

    process_bytes = canonical_d0_v2_smoke_process_receipt_bytes(process)
    aggregate_bytes = canonical_d0_v2_smoke_repro_aggregate_bytes(aggregate)
    assert process_bytes.endswith(b"\n")
    assert aggregate_bytes.endswith(b"\n")
    assert d0_v2_smoke_process_receipt_sha256(process) == hashlib.sha256(
        process_bytes
    ).hexdigest()
    assert d0_v2_smoke_repro_aggregate_sha256(aggregate) == hashlib.sha256(
        aggregate_bytes
    ).hexdigest()
    assert validate_d0_v2_smoke_repro_aggregate(aggregate) == aggregate


@pytest.mark.parametrize(
    ("field", "replacement", "message"),
    [
        ("engineering_smoke", False, "engineering_smoke"),
        ("paper_result", True, "paper_result"),
        ("formal_p3_complete", True, "formal_p3_complete"),
        ("stage2_authorized", True, "stage2_authorized"),
        ("fresh_subprocess", False, "fresh_subprocess"),
        ("parent_run_nonce", "bad", "parent_run_nonce"),
        ("candidate_count", 9, "candidate_count"),
    ],
)
def test_process_status_identity_and_count_gates_fail_closed(
    field: str, replacement: object, message: str
) -> None:
    receipt = _process_receipt(0)
    receipt[field] = replacement
    with pytest.raises(D0V2SmokeReproContractError, match=message):
        validate_d0_v2_smoke_process_receipt(receipt)


def test_process_receipt_requires_exact_recomputed_cell_aggregate() -> None:
    receipt = _process_receipt(0)
    receipt["cell_aggregate"]["candidate_count"] = 9
    with pytest.raises(D0V2SmokeReproContractError, match="cell_aggregate"):
        validate_d0_v2_smoke_process_receipt(receipt)

    missing = _process_receipt(0)
    missing["candidate_receipts"].pop()
    with pytest.raises(D0V2SmokeReproContractError, match="independent candidate cell"):
        validate_d0_v2_smoke_process_receipt(missing)

    unknown = _process_receipt(0)
    unknown["unsafe"] = True
    with pytest.raises(D0V2SmokeReproContractError, match="unknown"):
        validate_d0_v2_smoke_process_receipt(unknown)


@pytest.mark.parametrize(
    ("variant", "message"),
    [
        ("config", "config_sha256"),
        ("pre", "pre_logits_sha256"),
        ("integer", "integer structure"),
    ],
)
def test_cross_process_shared_and_integer_gate_drift_fails_closed(
    variant: str, message: str
) -> None:
    receipts = _process_receipts()
    if variant == "config":
        receipts[2] = _process_receipt(2, config_label="drifted-config")
    elif variant == "pre":
        receipts[2] = _process_receipt(2, pre_label="drifted-pre")
    else:
        receipts[2] = _process_receipt(2, gradient_tensor_count=107)
    with pytest.raises(D0V2SmokeReproContractError, match=message):
        aggregate_three(receipts)


def test_aggregate_requires_exactly_three_unique_fresh_processes() -> None:
    with pytest.raises(D0V2SmokeReproContractError, match="exactly three"):
        aggregate_three(_process_receipts()[:2])

    duplicate_logical = _process_receipts()
    duplicate_logical[2] = _process_receipt(2)
    duplicate_logical[2]["process_identity"]["process_id"] = "fresh-process-0"
    # Rebuild is necessary because exact process validation binds the identity.
    duplicate_logical[2] = build_d0_v2_smoke_process_receipt(
        duplicate_logical[2]["candidate_receipts"],
        parent_run_nonce=_sha("parent-run"),
        child_launch_nonce=_sha("child:2"),
        command_sha256=_sha("command:2"),
        process_identity={
            "process_id": "fresh-process-0",
            "os_process_id": 1002,
            "process_start_time_ticks": 5002,
        },
    )
    with pytest.raises(D0V2SmokeReproContractError, match="logical process IDs"):
        aggregate_three(duplicate_logical)

    duplicate_linux = _process_receipts()
    duplicate_linux[2] = _process_receipt(2, pid=1000, start_ticks=5000)
    with pytest.raises(D0V2SmokeReproContractError, match="Linux process identities"):
        aggregate_three(duplicate_linux)

    duplicate_nonce = _process_receipts()
    duplicate_nonce[2] = build_d0_v2_smoke_process_receipt(
        duplicate_nonce[2]["candidate_receipts"],
        parent_run_nonce=_sha("parent-run"),
        child_launch_nonce=_sha("child:0"),
        command_sha256=_sha("command:2"),
        process_identity=duplicate_nonce[2]["process_identity"],
    )
    with pytest.raises(D0V2SmokeReproContractError, match="child_launch_nonce"):
        aggregate_three(duplicate_nonce)


def test_cross_process_execution_identity_reuse_is_forbidden() -> None:
    receipts = _process_receipts()
    candidates = copy.deepcopy(receipts[2]["candidate_receipts"])
    candidates[0]["execution_identity"]["model_instance_id"] = receipts[0][
        "candidate_receipts"
    ][0]["execution_identity"]["model_instance_id"]
    receipts[2] = build_d0_v2_smoke_process_receipt(
        candidates,
        parent_run_nonce=_sha("parent-run"),
        child_launch_nonce=_sha("child:2"),
        command_sha256=_sha("command:2"),
        process_identity={
            "process_id": "fresh-process-2",
            "os_process_id": 1002,
            "process_start_time_ticks": 5002,
        },
    )
    with pytest.raises(D0V2SmokeReproContractError, match="model_instance_id reuse"):
        aggregate_three(receipts)


def test_native_reference_integer_drift_fails_cross_process_gate() -> None:
    receipts = _process_receipts()
    candidates = copy.deepcopy(receipts[2]["candidate_receipts"])
    candidates[0]["numeric_evidence"][
        "native_reference_optimizer_state_tensor_count"
    ] += 1
    receipts[2] = build_d0_v2_smoke_process_receipt(
        candidates,
        parent_run_nonce=_sha("parent-run"),
        child_launch_nonce=_sha("child:2"),
        command_sha256=_sha("command:2"),
        process_identity=receipts[2]["process_identity"],
    )
    with pytest.raises(D0V2SmokeReproContractError, match="integer structure"):
        aggregate_three(receipts)


@pytest.mark.parametrize(
    "gate",
    [
        "same_device_native_step_endpoint_exact",
        "same_device_native_optimizer_state_exact",
    ],
)
def test_same_device_native_candidate_gate_is_inherited(gate: str) -> None:
    candidates = _candidate_receipts(0)
    candidates[0]["execution_gates"][gate] = False
    with pytest.raises(
        D0V2SmokeReproContractError, match="independent candidate cell"
    ):
        build_d0_v2_smoke_process_receipt(
            candidates,
            parent_run_nonce=_sha("parent-run"),
            child_launch_nonce=_sha("child:0"),
            command_sha256=_sha("command:0"),
            process_identity={
                "process_id": "fresh-process-0",
                "os_process_id": 1000,
                "process_start_time_ticks": 5000,
            },
        )


def test_parent_nonce_must_bind_all_three_processes() -> None:
    receipts = _process_receipts()
    receipts[2] = build_d0_v2_smoke_process_receipt(
        receipts[2]["candidate_receipts"],
        parent_run_nonce=_sha("different-parent"),
        child_launch_nonce=_sha("child:2"),
        command_sha256=_sha("command:2"),
        process_identity=receipts[2]["process_identity"],
    )
    with pytest.raises(D0V2SmokeReproContractError, match="parent_run_nonce"):
        aggregate_three(receipts)


def test_aggregate_mutation_cannot_authorize_science_or_stage2() -> None:
    aggregate = aggregate_three(_process_receipts())
    for field in ("paper_result", "formal_p3_complete", "stage2_authorized"):
        mutated = copy.deepcopy(aggregate)
        mutated[field] = True
        with pytest.raises(D0V2SmokeReproContractError, match=field):
            validate_d0_v2_smoke_repro_aggregate(mutated)

    unknown = copy.deepcopy(aggregate)
    unknown["scientific_result"] = True
    with pytest.raises(D0V2SmokeReproContractError, match="unknown"):
        validate_d0_v2_smoke_repro_aggregate(unknown)


def test_current_process_identity_uses_pid_and_proc_start_ticks() -> None:
    ticks = read_proc_process_start_time_ticks(os.getpid())
    identity = FreshSubprocessIdentity.capture_current("cpu-test-process")
    assert ticks > 0
    assert identity == FreshSubprocessIdentity(
        process_id="cpu-test-process",
        os_process_id=os.getpid(),
        process_start_time_ticks=ticks,
    )
