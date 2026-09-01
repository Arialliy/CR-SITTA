from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Callable

import pytest

from analysis.d0_equivalence_repro_contract import (
    AGGREGATE_ARTIFACT_TYPE,
    D0EquivalenceReproContractError,
    EquivalenceReceiptBinding,
    LOGITS_TENSOR_SHA256_CONTRACT,
    SINGLE_PROCESS_ARTIFACT_TYPE,
    SUPPORTED_RUNNER_ATTESTATION_SCOPE,
    build_d0_equivalence_repro_receipt,
    canonical_d0_equivalence_repro_receipt_bytes,
    d0_equivalence_repro_receipt_sha256,
    validate_d0_equivalence_repro_receipt,
)


FROZEN_CANDIDATES = (
    ("Adam", 1.0e-5, "Adam_lr_1em5"),
    ("Adam", 3.0e-5, "Adam_lr_3em5"),
    ("Adam", 1.0e-4, "Adam_lr_1em4"),
    ("Adam", 3.0e-4, "Adam_lr_3em4"),
    ("Adam", 1.0e-3, "Adam_lr_1em3"),
    ("SGD", 1.0e-5, "SGD_lr_1em5"),
    ("SGD", 3.0e-5, "SGD_lr_3em5"),
    ("SGD", 1.0e-4, "SGD_lr_1em4"),
    ("SGD", 3.0e-4, "SGD_lr_3em4"),
    ("SGD", 1.0e-3, "SGD_lr_1em3"),
)

DETERMINISM = {
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
}
PARENT_RUN_NONCE = hashlib.sha256(b"test-parent-run").hexdigest()


def _hash(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _command_hash(command: list[str]) -> str:
    payload = json.dumps(
        command,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _proc_start_ticks(pid: int) -> int:
    value = Path(f"/proc/{pid}/stat").read_text(encoding="ascii")
    close = value.rfind(")")
    return int(value[close + 2 :].split()[19])


def _canonical(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        + b"\n"
    )


def _process_instance(process_id: str) -> tuple[int, int]:
    number = int(_hash(process_id)[:8], 16)
    return 1000 + number % 100_000, 1_000_000 + number


def _single_receipt(
    process_id: str,
    *,
    os_process_id: int,
    process_start_time_ticks: int,
    step_offset: float = 0.0,
    parent_run_nonce: str = PARENT_RUN_NONCE,
    child_launch_nonce: str | None = None,
    command_sha256: str | None = None,
) -> dict:
    child_launch_nonce = child_launch_nonce or _hash(f"child-nonce:{process_id}")
    command_sha256 = command_sha256 or _hash(f"command:{process_id}")
    runtime_sha = _hash("runtime")
    source_hashes = {
        "configs/tent_failure_diagnostics_v1.yaml": _hash("config-source"),
        "scripts/run_tent_failure_diagnostics.py": _hash("runner-source"),
    }
    comparisons: list[dict[str, Any]] = []
    for index, (optimizer, learning_rate, slug) in enumerate(FROZEN_CANDIDATES):
        step = float(index + 1) * 1.0e-4 + step_offset
        comparisons.append(
            {
                "candidate": {
                    "optimizer": optimizer,
                    "learning_rate": learning_rate,
                },
                "candidate_slug": slug,
                "pre_logits_bit_exact": True,
                "post_logits_bit_exact": True,
                "post_logits_max_abs_difference": 0.0,
                "shared_post_logits_tensor_sha256": _hash(f"post:{slug}"),
                "shared_step_norm": step,
                "historical_step_norm": step,
                "step_norm_abs_difference": 0.0,
                "shared_changed_parameter_tensors": index + 1,
                "historical_changed_parameter_tensors": index + 1,
                "changed_parameter_tensor_count_equal": True,
                "shared_entropy_gradient_bundle_sha256": _hash(
                    f"gradient:{slug}"
                ),
                "historical_entropy_gradient_bundle_sha256": _hash(
                    f"gradient:{slug}"
                ),
                "entropy_gradient_bundle_sha256_equal": True,
                "shared_parameter_delta_bundle_sha256": _hash(f"delta:{slug}"),
                "historical_parameter_delta_bundle_sha256": _hash(
                    f"delta:{slug}"
                ),
                "parameter_delta_bundle_sha256_equal": True,
                "historical_reset_exact_source": True,
            }
        )
    return {
        "schema_version": 1,
        "artifact_type": SINGLE_PROCESS_ARTIFACT_TYPE,
        "paper_test_result": False,
        "source_train_derived": True,
        "oracle_analysis": False,
        "fresh_process": True,
        "parent_run_nonce": parent_run_nonce,
        "child_launch_nonce": child_launch_nonce,
        "command_sha256": command_sha256,
        "process_id": process_id,
        "os_process_id": os_process_id,
        "process_start_time_ticks": process_start_time_ticks,
        "dataset": "IRSTD-1K",
        "condition": "clean_S0",
        "image_index": 0,
        "image_id": "XDU102",
        "config_sha256": _hash("config"),
        "global_runtime_seal_sha256": runtime_sha,
        "diagnostic_source_code_sha256": source_hashes,
        "determinism_contract": copy.deepcopy(DETERMINISM),
        "logits_tensor_sha256_contract": LOGITS_TENSOR_SHA256_CONTRACT,
        "shared_pre_logits_tensor_sha256": _hash("pre-logits"),
        "runtime_audits": [
            {
                "stage": "d0_equivalence_entry",
                "verified": True,
                "global_runtime_seal_sha256": runtime_sha,
                "bound_file_count": 5,
                "rehashed_file_count": 3,
                "all_bound_file_identity_and_metadata_verified": True,
                "full_byte_rehash": False,
                "active_paths_rehashed": [],
            },
            {
                "stage": "d0_equivalence_pre_receipt",
                "verified": True,
                "global_runtime_seal_sha256": runtime_sha,
                "bound_file_count": 5,
                "rehashed_file_count": 5,
                "all_bound_file_identity_and_metadata_verified": True,
                "full_byte_rehash": True,
                "active_paths_rehashed": [],
            },
        ],
        "diagnostic_source_audits": [
            {
                "stage": "d0_equivalence_entry",
                "verified": True,
                "bound_file_count": len(source_hashes),
                "all_files_rehashed": True,
            },
            {
                "stage": "d0_equivalence_pre_receipt",
                "verified": True,
                "bound_file_count": len(source_hashes),
                "all_files_rehashed": True,
            },
        ],
        "candidate_slugs": [value[2] for value in FROZEN_CANDIDATES],
        "candidate_count": 10,
        "method_label_accesses": 0,
        "outer_evaluator_label_accesses": 0,
        "test_images_opened": 0,
        "test_labels_opened": 0,
        "target_payload_deserialized": False,
        "pre_logits_all_bit_exact": True,
        "post_logits_all_bit_exact": True,
        "step_norm_all_within_1e_minus_12": True,
        "changed_parameter_tensor_counts_all_equal": True,
        "entropy_gradient_hashes_all_equal": True,
        "parameter_delta_hashes_all_equal": True,
        "historical_resets_all_exact_source": True,
        "cache_zero_test_opens_verified": True,
        "comparisons": comparisons,
        "passed": True,
    }


def _write_receipt(
    root: Path,
    process_id: str,
    *,
    step_offset: float = 0.0,
    receipt: dict | None = None,
    os_process_id: int | None = None,
    process_start_time_ticks: int | None = None,
) -> EquivalenceReceiptBinding:
    default_pid, default_ticks = _process_instance(process_id)
    os_process_id = default_pid if os_process_id is None else os_process_id
    process_start_time_ticks = (
        default_ticks
        if process_start_time_ticks is None
        else process_start_time_ticks
    )
    value = (
        _single_receipt(
            process_id,
            os_process_id=os_process_id,
            process_start_time_ticks=process_start_time_ticks,
            step_offset=step_offset,
        )
        if receipt is None
        else receipt
    )
    path = root / f"{process_id}.json"
    payload = _canonical(value)
    path.write_bytes(payload)
    return EquivalenceReceiptBinding(
        path=path,
        sha256=hashlib.sha256(payload).hexdigest(),
        process_id=process_id,
        os_process_id=os_process_id,
        process_start_time_ticks=process_start_time_ticks,
    )


def _bindings(tmp_path: Path) -> list[EquivalenceReceiptBinding]:
    return [
        _write_receipt(tmp_path, "proc-c", step_offset=4.0e-13),
        _write_receipt(tmp_path, "proc-a", step_offset=0.0),
        _write_receipt(tmp_path, "proc-b", step_offset=2.0e-13),
    ]


def _parent_run(
    bindings: list[EquivalenceReceiptBinding],
    *,
    parent_run_nonce: str = PARENT_RUN_NONCE,
) -> dict[str, Any]:
    launches = []
    for binding in bindings:
        receipt = json.loads(binding.path.read_text(encoding="utf-8"))
        launches.append(
            {
                "process_id": binding.process_id,
                "parent_run_nonce": parent_run_nonce,
                "child_launch_nonce": receipt["child_launch_nonce"],
                "command_sha256": receipt["command_sha256"],
                "os_process_id": binding.os_process_id,
                "process_start_time_ticks": binding.process_start_time_ticks,
                "returncode": 0,
                "stdout_sha256": _hash(f"stdout:{binding.process_id}"),
                "stderr_sha256": _hash(f"stderr:{binding.process_id}"),
                "receipt_path": str(binding.path.absolute()),
                "receipt_sha256": binding.sha256,
            }
        )
    return {
        "parent_run_nonce": parent_run_nonce,
        "parent_os_process_id": 999_999,
        "parent_process_start_time_ticks": 999_999_999,
        "launches": launches,
    }


def _build(
    bindings: list[Any],
    *,
    parent_run: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return build_d0_equivalence_repro_receipt(
        bindings,
        parent_run=_parent_run(bindings) if parent_run is None else parent_run,
    )


def _mutate_binding(
    binding: EquivalenceReceiptBinding,
    mutator: Callable[[dict], None],
) -> EquivalenceReceiptBinding:
    value = json.loads(binding.path.read_text(encoding="utf-8"))
    mutator(value)
    payload = _canonical(value)
    binding.path.write_bytes(payload)
    return EquivalenceReceiptBinding(
        path=binding.path,
        sha256=hashlib.sha256(payload).hexdigest(),
        process_id=binding.process_id,
        os_process_id=binding.os_process_id,
        process_start_time_ticks=binding.process_start_time_ticks,
    )


def test_three_fresh_processes_build_canonical_pairwise_receipt(
    tmp_path: Path,
) -> None:
    bindings = _bindings(tmp_path)
    receipt = _build(bindings)

    assert receipt["artifact_type"] == AGGREGATE_ARTIFACT_TYPE
    assert receipt["passed"] is True
    assert receipt["process_ids"] == ["proc-a", "proc-b", "proc-c"]
    assert len(
        {
            (value["os_process_id"], value["process_start_time_ticks"])
            for value in receipt["process_instances"]
        }
    ) == 3
    assert receipt["fresh_process_count"] == 3
    assert receipt["attestation_scope"] == SUPPORTED_RUNNER_ATTESTATION_SCOPE
    assert "without_external_trust_root" in receipt["attestation_scope"]
    assert "artifact_and_checksum_corewrite_out_of_scope" in receipt[
        "attestation_scope"
    ]
    assert receipt["parent_run"]["parent_run_nonce"] == PARENT_RUN_NONCE
    assert len(receipt["parent_run"]["launches"]) == 3
    assert receipt["pair_count"] == 3
    assert receipt["candidate_count"] == 10
    assert [value["process_ids"] for value in receipt["pairwise_comparisons"]] == [
        ["proc-a", "proc-b"],
        ["proc-a", "proc-c"],
        ["proc-b", "proc-c"],
    ]
    assert all(
        pair["pre_logits_bit_exact"]
        and pair["post_logits_all_bit_exact"]
        and pair["entropy_gradient_hashes_all_exact"]
        and pair["parameter_delta_hashes_all_exact"]
        and pair["changed_parameter_tensor_counts_all_exact"]
        and pair["step_norms_all_within_1e_minus_12"]
        for pair in receipt["pairwise_comparisons"]
    )
    assert all(
        value["max_step_norm_abs_difference"] <= 1.0e-12
        for pair in receipt["pairwise_comparisons"]
        for value in pair["comparisons"]
    )

    # Input order cannot alter the aggregate value or bytes.
    reverse = _build(list(reversed(bindings)))
    assert reverse == receipt
    assert canonical_d0_equivalence_repro_receipt_bytes(receipt).endswith(b"\n")
    assert d0_equivalence_repro_receipt_sha256(receipt) == (
        hashlib.sha256(canonical_d0_equivalence_repro_receipt_bytes(receipt)).hexdigest()
    )
    assert validate_d0_equivalence_repro_receipt(receipt) == (
        d0_equivalence_repro_receipt_sha256(receipt)
    )


@pytest.mark.parametrize("count", [0, 1, 2, 4])
def test_requires_exactly_three_bindings(tmp_path: Path, count: int) -> None:
    values = [
        _write_receipt(tmp_path, f"proc-{index}") for index in range(count)
    ]
    with pytest.raises(D0EquivalenceReproContractError, match="exactly three"):
        _build(values)


def test_binding_schema_hash_and_fresh_process_identity_fail_closed(
    tmp_path: Path,
) -> None:
    bindings = _bindings(tmp_path)
    mappings = [
        {
            "path": str(value.path),
            "sha256": value.sha256,
            "process_id": value.process_id,
            "os_process_id": value.os_process_id,
            "process_start_time_ticks": value.process_start_time_ticks,
        }
        for value in bindings
    ]
    mappings[0]["unexpected"] = True
    with pytest.raises(D0EquivalenceReproContractError, match="schema is not exact"):
        _build(mappings, parent_run=_parent_run(bindings))

    wrong_sha = list(bindings)
    wrong_sha[0] = EquivalenceReceiptBinding(
        wrong_sha[0].path,
        "0" * 64,
        wrong_sha[0].process_id,
        wrong_sha[0].os_process_id,
        wrong_sha[0].process_start_time_ticks,
    )
    with pytest.raises(D0EquivalenceReproContractError, match="binding mismatch"):
        _build(wrong_sha)

    not_fresh = list(bindings)
    not_fresh[0] = _mutate_binding(
        not_fresh[0], lambda value: value.__setitem__("fresh_process", False)
    )
    with pytest.raises(D0EquivalenceReproContractError, match="protocol gates"):
        _build(not_fresh)

    mismatch_root = tmp_path / "mismatch"
    mismatch_root.mkdir()
    mismatch = _bindings(mismatch_root)
    mismatch[0] = EquivalenceReceiptBinding(
        mismatch[0].path,
        mismatch[0].sha256,
        "external-id",
        mismatch[0].os_process_id,
        mismatch[0].process_start_time_ticks,
    )
    with pytest.raises(D0EquivalenceReproContractError, match="protocol gates"):
        _build(mismatch)


def test_requires_unique_process_ids_paths_and_receipt_hashes(tmp_path: Path) -> None:
    bindings = _bindings(tmp_path)
    duplicate_id = list(bindings)
    duplicate_id[2] = EquivalenceReceiptBinding(
        duplicate_id[2].path,
        duplicate_id[2].sha256,
        duplicate_id[1].process_id,
        duplicate_id[2].os_process_id,
        duplicate_id[2].process_start_time_ticks,
    )
    with pytest.raises(D0EquivalenceReproContractError, match="unique fresh process"):
        _build(duplicate_id)

    duplicate_path = list(bindings)
    duplicate_path[2] = EquivalenceReceiptBinding(
        duplicate_path[1].path,
        duplicate_path[2].sha256,
        duplicate_path[2].process_id,
        duplicate_path[2].os_process_id,
        duplicate_path[2].process_start_time_ticks,
    )
    with pytest.raises(D0EquivalenceReproContractError, match="unique receipt paths"):
        _build(duplicate_path)

    duplicate_sha = list(bindings)
    duplicate_sha[2] = EquivalenceReceiptBinding(
        duplicate_sha[2].path,
        duplicate_sha[1].sha256,
        duplicate_sha[2].process_id,
        duplicate_sha[2].os_process_id,
        duplicate_sha[2].process_start_time_ticks,
    )
    with pytest.raises(D0EquivalenceReproContractError, match="unique receipt SHA"):
        _build(duplicate_sha)

    duplicate_instance_root = tmp_path / "duplicate-instance"
    duplicate_instance_root.mkdir()
    duplicate_instance = _bindings(duplicate_instance_root)
    duplicate_instance[2] = _mutate_binding(
        duplicate_instance[2],
        lambda value: (
            value.__setitem__(
                "os_process_id", duplicate_instance[1].os_process_id
            ),
            value.__setitem__(
                "process_start_time_ticks",
                duplicate_instance[1].process_start_time_ticks,
            ),
        ),
    )
    duplicate_instance[2] = EquivalenceReceiptBinding(
        duplicate_instance[2].path,
        duplicate_instance[2].sha256,
        duplicate_instance[2].process_id,
        duplicate_instance[1].os_process_id,
        duplicate_instance[1].process_start_time_ticks,
    )
    with pytest.raises(D0EquivalenceReproContractError, match="Linux process"):
        _build(duplicate_instance)


@pytest.mark.parametrize("field", ["os_process_id", "process_start_time_ticks"])
def test_linux_process_identity_must_be_positive(
    tmp_path: Path, field: str
) -> None:
    bindings = _bindings(tmp_path)
    bindings[0] = _mutate_binding(
        bindings[0], lambda value: value.__setitem__(field, 0)
    )
    kwargs = {
        "path": bindings[0].path,
        "sha256": bindings[0].sha256,
        "process_id": bindings[0].process_id,
        "os_process_id": bindings[0].os_process_id,
        "process_start_time_ticks": bindings[0].process_start_time_ticks,
    }
    kwargs[field] = 0
    bindings[0] = EquivalenceReceiptBinding(**kwargs)
    with pytest.raises(D0EquivalenceReproContractError, match=field):
        _build(bindings)


@pytest.mark.parametrize(
    ("field", "replacement", "message"),
    [
        ("config_sha256", _hash("different-config"), "config_sha256"),
        (
            "diagnostic_source_code_sha256",
            {
                "configs/tent_failure_diagnostics_v1.yaml": _hash("different"),
                "scripts/run_tent_failure_diagnostics.py": _hash("runner-source"),
            },
            "diagnostic_source_code_sha256",
        ),
        ("global_runtime_seal_sha256", _hash("different-runtime"), "runtime"),
    ],
)
def test_config_source_and_runtime_must_match(
    tmp_path: Path, field: str, replacement: Any, message: str
) -> None:
    bindings = _bindings(tmp_path)

    def mutate(value: dict) -> None:
        value[field] = replacement
        if field == "global_runtime_seal_sha256":
            for audit in value["runtime_audits"]:
                audit["global_runtime_seal_sha256"] = replacement
        if field == "diagnostic_source_code_sha256":
            for audit in value["diagnostic_source_audits"]:
                audit["bound_file_count"] = len(replacement)

    bindings[0] = _mutate_binding(bindings[0], mutate)
    with pytest.raises(D0EquivalenceReproContractError, match=message):
        _build(bindings)


def test_dataset_sample_and_determinism_are_frozen_and_identical(
    tmp_path: Path,
) -> None:
    bindings = _bindings(tmp_path)

    def valid_other_sample(value: dict) -> None:
        value["dataset"] = "NUDT-SIRST"
        value["condition"] = "clean_S0"
        value["image_index"] = 0
        value["image_id"] = "000891"

    bindings[0] = _mutate_binding(bindings[0], valid_other_sample)
    with pytest.raises(D0EquivalenceReproContractError, match="dataset"):
        _build(bindings)

    determinism_root = tmp_path / "determinism"
    determinism_root.mkdir()
    determinism = _bindings(determinism_root)
    determinism[0] = _mutate_binding(
        determinism[0],
        lambda value: value["determinism_contract"]["strict_forward"].__setitem__(
            "cudnn_benchmark", True
        ),
    )
    with pytest.raises(D0EquivalenceReproContractError, match="determinism_contract"):
        _build(determinism)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda value: value.__setitem__(
                "shared_pre_logits_tensor_sha256", _hash("different-pre")
            ),
            "pre logits",
        ),
        (
            lambda value: value["comparisons"][0].__setitem__(
                "shared_post_logits_tensor_sha256", _hash("different-post")
            ),
            "post logits",
        ),
        (
            lambda value: (
                value["comparisons"][0].__setitem__(
                    "shared_entropy_gradient_bundle_sha256",
                    _hash("different-gradient"),
                ),
                value["comparisons"][0].__setitem__(
                    "historical_entropy_gradient_bundle_sha256",
                    _hash("different-gradient"),
                ),
            ),
            "entropy gradients",
        ),
        (
            lambda value: (
                value["comparisons"][0].__setitem__(
                    "shared_parameter_delta_bundle_sha256", _hash("different-delta")
                ),
                value["comparisons"][0].__setitem__(
                    "historical_parameter_delta_bundle_sha256",
                    _hash("different-delta"),
                ),
            ),
            "parameter deltas",
        ),
        (
            lambda value: (
                value["comparisons"][0].__setitem__(
                    "shared_changed_parameter_tensors", 99
                ),
                value["comparisons"][0].__setitem__(
                    "historical_changed_parameter_tensors", 99
                ),
            ),
            "changed tensor counts",
        ),
        (
            lambda value: (
                value["comparisons"][0].__setitem__(
                    "shared_step_norm",
                    value["comparisons"][0]["shared_step_norm"] + 2.0e-12,
                ),
                value["comparisons"][0].__setitem__(
                    "historical_step_norm",
                    value["comparisons"][0]["historical_step_norm"] + 2.0e-12,
                ),
            ),
            "step norm",
        ),
    ],
)
def test_every_cross_process_evidence_dimension_fails_closed(
    tmp_path: Path,
    mutation: Callable[[dict], Any],
    message: str,
) -> None:
    bindings = _bindings(tmp_path)
    bindings[0] = _mutate_binding(bindings[0], mutation)
    with pytest.raises(D0EquivalenceReproContractError, match=message):
        _build(bindings)


def test_single_receipt_and_comparison_schema_are_exact(tmp_path: Path) -> None:
    bindings = _bindings(tmp_path)
    bindings[0] = _mutate_binding(
        bindings[0], lambda value: value.__setitem__("decorative", True)
    )
    with pytest.raises(D0EquivalenceReproContractError, match="schema is not exact"):
        _build(bindings)

    comparison_root = tmp_path / "comparison"
    comparison_root.mkdir()
    comparisons = _bindings(comparison_root)
    comparisons[0] = _mutate_binding(
        comparisons[0],
        lambda value: value["comparisons"][0].__setitem__("decorative", True),
    )
    with pytest.raises(D0EquivalenceReproContractError, match="schema is not exact"):
        _build(comparisons)


def test_noncanonical_and_symlink_receipts_are_rejected(tmp_path: Path) -> None:
    bindings = _bindings(tmp_path)
    value = json.loads(bindings[0].path.read_text(encoding="utf-8"))
    payload = json.dumps(value, indent=2).encode("utf-8")
    bindings[0].path.write_bytes(payload)
    bindings[0] = EquivalenceReceiptBinding(
        bindings[0].path,
        hashlib.sha256(payload).hexdigest(),
        bindings[0].process_id,
        bindings[0].os_process_id,
        bindings[0].process_start_time_ticks,
    )
    with pytest.raises(D0EquivalenceReproContractError, match="not canonical"):
        _build(bindings)

    symlink_root = tmp_path / "symlink"
    symlink_root.mkdir()
    clean = _bindings(symlink_root)
    link = symlink_root / "linked.json"
    link.symlink_to(clean[0].path)
    clean[0] = EquivalenceReceiptBinding(
        link,
        clean[0].sha256,
        clean[0].process_id,
        clean[0].os_process_id,
        clean[0].process_start_time_ticks,
    )
    with pytest.raises(D0EquivalenceReproContractError, match="stably read"):
        _build(clean)


@pytest.mark.parametrize(
    "mutator",
    [
        lambda parent: parent["launches"][0].__setitem__("returncode", 1),
        lambda parent: parent["launches"][0].__setitem__(
            "parent_run_nonce", _hash("wrong-parent-nonce")
        ),
        lambda parent: parent["launches"][0].__setitem__(
            "child_launch_nonce", parent["launches"][1]["child_launch_nonce"]
        ),
        lambda parent: parent["launches"][0].__setitem__(
            "command_sha256", parent["launches"][1]["command_sha256"]
        ),
        lambda parent: parent["launches"][0].__setitem__(
            "receipt_sha256", _hash("wrong-receipt")
        ),
        lambda parent: parent["launches"][0].__setitem__("stdout_sha256", "bad"),
        lambda parent: parent["launches"][0].__setitem__("decorative", True),
    ],
)
def test_parent_launch_transcript_tamper_fails_closed(
    tmp_path: Path, mutator: Callable[[dict[str, Any]], Any]
) -> None:
    bindings = _bindings(tmp_path)
    parent = _parent_run(bindings)
    mutator(parent)
    with pytest.raises(D0EquivalenceReproContractError):
        _build(bindings, parent_run=parent)


def test_aggregate_parent_transcript_is_revalidated(tmp_path: Path) -> None:
    receipt = _build(_bindings(tmp_path))
    receipt["parent_run"]["launches"][0]["returncode"] = 1
    with pytest.raises(D0EquivalenceReproContractError):
        validate_d0_equivalence_repro_receipt(receipt, revalidate_inputs=False)


@pytest.mark.skipif(
    sys.platform != "linux",
    reason="requires Linux /proc process identity",
)
def test_real_subprocesses_bind_parent_nonce_command_and_linux_instances(
    tmp_path: Path,
) -> None:
    """Exercise the supported-runner transcript with three real exec children."""

    parent_nonce = _hash("real-subprocess-parent-run")
    child_code = r'''
import hashlib
import json
import os
from pathlib import Path
import sys
import time

time.sleep(0.05)
command = [sys.executable, *sys.argv]
command_sha = hashlib.sha256(json.dumps(
    command, ensure_ascii=False, sort_keys=True,
    separators=(",", ":"), allow_nan=False,
).encode("utf-8")).hexdigest()
if command_sha != os.environ["D0_TEST_COMMAND_SHA256"]:
    raise SystemExit(7)
stat = Path(f"/proc/{os.getpid()}/stat").read_text(encoding="ascii")
close = stat.rfind(")")
ticks = int(stat[close + 2:].split()[19])
template_path = Path(sys.argv[1])
output_path = Path(sys.argv[2])
value = json.loads(template_path.read_text(encoding="utf-8"))
value.update({
    "process_id": os.environ["D0_TEST_PROCESS_ID"],
    "parent_run_nonce": os.environ["D0_TEST_PARENT_NONCE"],
    "child_launch_nonce": os.environ["D0_TEST_CHILD_NONCE"],
    "command_sha256": command_sha,
    "os_process_id": os.getpid(),
    "process_start_time_ticks": ticks,
})
payload = json.dumps(
    value, ensure_ascii=False, sort_keys=True,
    separators=(",", ":"), allow_nan=False,
).encode("utf-8") + b"\n"
output_path.write_bytes(payload)
print(json.dumps({"process_id": value["process_id"], "pid": os.getpid()}))
'''
    helper_path = tmp_path / "real_subprocess_child.py"
    helper_path.write_text(child_code, encoding="utf-8")
    bindings: list[EquivalenceReceiptBinding] = []
    launches: list[dict[str, Any]] = []
    for index in range(3):
        process_id = f"real-proc-{index}"
        child_nonce = _hash(f"real-child-nonce:{index}")
        template_path = tmp_path / f"template-{index}.json"
        output_path = tmp_path / f"receipt-{index}.json"
        command = [
            sys.executable,
            str(helper_path),
            str(template_path),
            str(output_path),
        ]
        command_sha = _command_hash(command)
        template = _single_receipt(
            process_id,
            os_process_id=1,
            process_start_time_ticks=1,
            parent_run_nonce=parent_nonce,
            child_launch_nonce=child_nonce,
            command_sha256=command_sha,
        )
        template_path.write_bytes(_canonical(template))
        environment = dict(os.environ)
        environment.update(
            {
                "D0_TEST_PROCESS_ID": process_id,
                "D0_TEST_PARENT_NONCE": parent_nonce,
                "D0_TEST_CHILD_NONCE": child_nonce,
                "D0_TEST_COMMAND_SHA256": command_sha,
            }
        )
        process = subprocess.Popen(
            command,
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        observed_ticks = _proc_start_ticks(process.pid)
        stdout, stderr = process.communicate(timeout=10)
        assert process.returncode == 0, stderr.decode("utf-8", errors="replace")
        payload = output_path.read_bytes()
        receipt = json.loads(payload)
        assert receipt["os_process_id"] == process.pid
        assert receipt["process_start_time_ticks"] == observed_ticks
        binding = EquivalenceReceiptBinding(
            path=output_path,
            sha256=hashlib.sha256(payload).hexdigest(),
            process_id=process_id,
            os_process_id=process.pid,
            process_start_time_ticks=observed_ticks,
        )
        bindings.append(binding)
        launches.append(
            {
                "process_id": process_id,
                "parent_run_nonce": parent_nonce,
                "child_launch_nonce": child_nonce,
                "command_sha256": command_sha,
                "os_process_id": process.pid,
                "process_start_time_ticks": observed_ticks,
                "returncode": process.returncode,
                "stdout_sha256": hashlib.sha256(stdout).hexdigest(),
                "stderr_sha256": hashlib.sha256(stderr).hexdigest(),
                "receipt_path": str(output_path.absolute()),
                "receipt_sha256": binding.sha256,
            }
        )
    parent_run = {
        "parent_run_nonce": parent_nonce,
        "parent_os_process_id": os.getpid(),
        "parent_process_start_time_ticks": _proc_start_ticks(os.getpid()),
        "launches": launches,
    }
    aggregate = _build(bindings, parent_run=parent_run)
    assert aggregate["passed"] is True
    assert aggregate["parent_run"] == {
        **parent_run,
        "launches": sorted(launches, key=lambda value: value["process_id"]),
    }
    assert len(
        {
            (value["os_process_id"], value["process_start_time_ticks"])
            for value in aggregate["process_instances"]
        }
    ) == 3
    validate_d0_equivalence_repro_receipt(aggregate)


def test_aggregate_validation_rejects_schema_or_pair_tamper(tmp_path: Path) -> None:
    receipt = _build(_bindings(tmp_path))
    unknown = copy.deepcopy(receipt)
    unknown["decorative"] = True
    with pytest.raises(D0EquivalenceReproContractError, match="schema is not exact"):
        validate_d0_equivalence_repro_receipt(unknown, revalidate_inputs=False)

    pair = copy.deepcopy(receipt)
    pair["pairwise_comparisons"][0]["comparisons"][0][
        "post_logits_bit_exact"
    ] = False
    with pytest.raises(D0EquivalenceReproContractError, match="did not pass"):
        validate_d0_equivalence_repro_receipt(pair, revalidate_inputs=False)

    inconsistent = copy.deepcopy(receipt)
    inconsistent["pairwise_comparisons"][0]["comparisons"][0][
        "post_logits_tensor_sha256"
    ] = _hash("internally-inconsistent-post")
    with pytest.raises(D0EquivalenceReproContractError, match="disagree"):
        validate_d0_equivalence_repro_receipt(
            inconsistent, revalidate_inputs=False
        )

    rebound = copy.deepcopy(receipt)
    rebound["input_receipts"][0]["sha256"] = _hash("rebound")
    with pytest.raises(D0EquivalenceReproContractError):
        validate_d0_equivalence_repro_receipt(rebound)
