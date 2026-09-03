from __future__ import annotations

import copy
import json
import math

import pytest

from analysis.d0_v2_independent_candidate_contract import (
    D0V2IndependentCandidateError,
    FROZEN_CANDIDATES,
    IndependentCandidateExecutionLedger,
    IndependentCandidateReceipt,
    parse_independent_candidate_receipt,
    validate_independent_candidate_cell,
)


def _sha(character: str) -> str:
    assert len(character) == 1 and character in "0123456789abcdef"
    return character * 64


def _receipt(index: int) -> dict:
    candidate = FROZEN_CANDIDATES[index]
    return IndependentCandidateReceipt(
        candidate_index=index,
        candidate=candidate,
        config_sha256=_sha("1"),
        dataset="NUAA-SIRST",
        condition="clean_S0",
        sample_index=0,
        sample_id="Misc_421",
        split_sha256=_sha("2"),
        checkpoint_sha256=_sha("3"),
        source_state_sha256=_sha("4"),
        runtime_sha256=_sha("5"),
        determinism_sha256=_sha("6"),
        input_sha256=_sha("7"),
        selected_parameter_names_sha256=_sha("8"),
        pre_logits_sha256=_sha("9"),
        post_logits_sha256=f"{index + 1:064x}",
        # Equal numeric hashes are legal: equality is not evidence of object reuse.
        entropy_gradient_bundle_sha256=_sha("a"),
        parameter_delta_bundle_sha256=f"{index + 21:064x}",
        model_instance_id=f"model:{index}",
        method_instance_id=f"method:{index}",
        optimizer_instance_id=f"optimizer:{index}",
        autograd_graph_id=f"graph:{index}",
        backward_execution_id=f"backward:{index}",
        gradient_buffer_owner_id=f"gradient-owner:{index}",
        gradient_tensor_count=106,
        changed_parameter_tensor_count=100,
        optimizer_state_entry_count_after_step=106,
        native_reference_parameter_tensor_count=106,
        native_reference_optimizer_state_tensor_count=(318 if candidate.optimizer == "Adam" else 106),
        step_norm_l2=1.0e-4 * (index + 1),
    ).to_dict()


def _receipts() -> list[dict]:
    return [_receipt(index) for index in range(len(FROZEN_CANDIDATES))]


def _set_nested(value: dict, path: tuple[str, ...], replacement: object) -> None:
    cursor = value
    for key in path[:-1]:
        child = cursor[key]
        assert isinstance(child, dict)
        cursor = child
    cursor[path[-1]] = replacement


def test_exact_ten_candidate_cell_builds_canonical_non_authorizing_receipt() -> None:
    # Input order is irrelevant; the aggregate is canonicalized by frozen index.
    aggregate = validate_independent_candidate_cell(list(reversed(_receipts())))
    value = aggregate.to_dict()

    assert value["candidate_count"] == 10
    assert value["candidate_slugs"] == [
        candidate.slug for candidate in FROZEN_CANDIDATES
    ]
    assert len(value["candidate_receipt_sha256s"]) == 10
    assert len(set(value["candidate_receipt_sha256s"])) == 10
    assert value["independent_execution"] == {
        "fresh_model_count": 10,
        "fresh_method_count": 10,
        "fresh_optimizer_count": 10,
        "own_autograd_graph_count": 10,
        "own_backward_execution_count": 10,
        "own_gradient_buffer_owner_count": 10,
        "gradient_reuse": False,
        "shared_autograd_graph_across_candidates": False,
        "shared_gradient_buffers_across_candidates": False,
        "engineering_gate_passed": True,
    }
    assert value["authorization"] == {
        "scientific_gate_status": "unresolved",
        "scientific_selection_performed": False,
        "stage2_authorized": False,
    }
    assert value["data_boundary"]["use_validation"] is False
    assert value["data_boundary"]["use_test_images"] is False
    assert value["data_boundary"]["use_test_labels"] is False
    assert value["data_boundary"]["train_target_payload_bytes_opened"] == 0
    assert value["data_boundary"]["test_split_files_opened"] == 0
    assert value["data_boundary"]["test_masks_opened"] == 0
    json.dumps(value, allow_nan=False)


def test_one_receipt_is_strictly_round_trippable() -> None:
    original = _receipt(0)
    parsed = parse_independent_candidate_receipt(original)
    assert parsed.to_dict() == original
    assert parsed.candidate == FROZEN_CANDIDATES[0]


@pytest.mark.parametrize(
    ("path", "replacement", "message"),
    [
        (("execution_gates", "gradient_reuse"), True, "gradient_reuse"),
        (("execution_gates", "own_forward_backward"), False, "own_forward_backward"),
        (
            ("execution_gates", "shared_autograd_graph_across_candidates"),
            True,
            "shared_autograd_graph",
        ),
        (
            ("execution_gates", "shared_gradient_buffers_across_candidates"),
            True,
            "shared_gradient_buffers",
        ),
        (
            ("execution_gates", "optimizer_state_entry_count_before_step"),
            1,
            "optimizer_state_entry_count_before_step",
        ),
        (("execution_gates", "entropy_backward_pass_count"), 0, "backward"),
        (("execution_gates", "optimizer_step_count"), 2, "optimizer_step_count"),
        (("data_boundary", "use_validation"), True, "use_validation"),
        (("data_boundary", "use_test_images"), True, "use_test_images"),
        (("data_boundary", "use_test_labels"), True, "use_test_labels"),
        (("data_boundary", "method_label_accesses"), 1, "method_label_accesses"),
        (
            ("data_boundary", "train_target_payload_bytes_opened"),
            1,
            "train_target_payload_bytes_opened",
        ),
        (
            ("data_boundary", "test_split_files_opened"),
            1,
            "test_split_files_opened",
        ),
        (
            ("data_boundary", "target_payload_deserialized_during_candidate"),
            True,
            "target_payload",
        ),
        (
            ("authorization", "scientific_gate_status"),
            "passed",
            "unresolved",
        ),
        (("authorization", "stage2_authorized"), True, "stage2_authorized"),
        (("numeric_evidence", "step_norm_l2"), math.inf, "finite"),
        (("cell_binding", "split_name"), "test", "Pilot64"),
    ],
)
def test_receipt_safety_and_state_drift_fail_closed(
    path: tuple[str, ...], replacement: object, message: str
) -> None:
    value = _receipt(0)
    _set_nested(value, path, replacement)
    with pytest.raises(D0V2IndependentCandidateError, match=message):
        parse_independent_candidate_receipt(value)


def test_unknown_missing_and_wrong_candidate_fields_fail_closed() -> None:
    unknown = _receipt(0)
    unknown["unsafe"] = True
    with pytest.raises(D0V2IndependentCandidateError, match="unknown"):
        parse_independent_candidate_receipt(unknown)

    missing = _receipt(0)
    del missing["execution_identity"]["backward_execution_id"]
    with pytest.raises(D0V2IndependentCandidateError, match="missing"):
        parse_independent_candidate_receipt(missing)

    wrong_candidate = _receipt(0)
    wrong_candidate["candidate"]["learning_rate"] = 3.0e-5
    with pytest.raises(D0V2IndependentCandidateError, match="frozen candidate"):
        parse_independent_candidate_receipt(wrong_candidate)

    integer_learning_rate = _receipt(0)
    integer_learning_rate["candidate"]["learning_rate"] = 0
    with pytest.raises(D0V2IndependentCandidateError, match="frozen candidate"):
        parse_independent_candidate_receipt(integer_learning_rate)


def test_cell_requires_complete_unique_grid_and_shared_bindings() -> None:
    with pytest.raises(D0V2IndependentCandidateError, match="exactly 10"):
        validate_independent_candidate_cell(_receipts()[:-1])

    duplicate_candidate = _receipts()
    duplicate_candidate[-1] = copy.deepcopy(duplicate_candidate[0])
    with pytest.raises(D0V2IndependentCandidateError, match="candidate_index"):
        validate_independent_candidate_cell(duplicate_candidate)

    changed_input = _receipts()
    changed_input[-1]["cell_binding"]["input_sha256"] = _sha("f")
    with pytest.raises(D0V2IndependentCandidateError, match="shared input_sha256"):
        validate_independent_candidate_cell(changed_input)

    changed_pre = _receipts()
    changed_pre[-1]["numeric_evidence"]["pre_logits_sha256"] = _sha("f")
    with pytest.raises(D0V2IndependentCandidateError, match="pre_logits_sha256"):
        validate_independent_candidate_cell(changed_pre)


@pytest.mark.parametrize(
    "identity_field",
    [
        "model_instance_id",
        "method_instance_id",
        "optimizer_instance_id",
        "autograd_graph_id",
        "backward_execution_id",
        "gradient_buffer_owner_id",
    ],
)
def test_cross_candidate_execution_identity_reuse_is_forbidden(
    identity_field: str,
) -> None:
    values = _receipts()
    values[1]["execution_identity"][identity_field] = values[0][
        "execution_identity"
    ][identity_field]
    with pytest.raises(D0V2IndependentCandidateError, match=f"{identity_field} reuse"):
        validate_independent_candidate_cell(values)


def test_equal_gradient_hashes_are_not_mistaken_for_gradient_object_reuse() -> None:
    values = _receipts()
    assert len(
        {
            value["numeric_evidence"]["entropy_gradient_bundle_sha256"]
            for value in values
        }
    ) == 1
    aggregate = validate_independent_candidate_cell(values)
    assert len(aggregate.candidate_receipts) == 10


def test_live_ledger_proves_unique_objects_graphs_and_gradient_buffers() -> None:
    ledger = IndependentCandidateExecutionLedger()
    objects: list[object] = []
    for candidate in FROZEN_CANDIDATES:
        model, method, optimizer = object(), object(), object()
        graph = object()
        gradients = [object(), object()]
        objects.extend((model, method, optimizer, graph, *gradients))
        ledger.claim_candidate_objects(
            candidate,
            model=model,
            method=method,
            optimizer=optimizer,
            optimizer_state_entry_count=0,
        )
        ledger.claim_candidate_backward(
            candidate,
            autograd_graph=graph,
            gradient_buffers=gradients,
        )
    assert ledger.assert_complete() == {
        "candidate_count": 10,
        "fresh_model_count": 10,
        "fresh_method_count": 10,
        "fresh_optimizer_count": 10,
        "own_autograd_graph_count": 10,
        "own_backward_execution_count": 10,
        "gradient_reuse": False,
        "engineering_gate_passed": True,
        "scientific_gate_status": "unresolved",
        "stage2_authorized": False,
    }


def test_live_ledger_rejects_model_graph_and_gradient_buffer_reuse() -> None:
    first, second = FROZEN_CANDIDATES[:2]
    shared_model = object()

    model_reuse = IndependentCandidateExecutionLedger()
    model_reuse.claim_candidate_objects(
        first,
        model=shared_model,
        method=object(),
        optimizer=object(),
        optimizer_state_entry_count=0,
    )
    with pytest.raises(D0V2IndependentCandidateError, match="object reuse"):
        model_reuse.claim_candidate_objects(
            second,
            model=shared_model,
            method=object(),
            optimizer=object(),
            optimizer_state_entry_count=0,
        )

    graph_reuse = IndependentCandidateExecutionLedger()
    shared_graph = object()
    shared_gradient = object()
    first_objects = (object(), object(), object())
    second_objects = (object(), object(), object())
    graph_reuse.claim_candidate_objects(
        first,
        model=first_objects[0],
        method=first_objects[1],
        optimizer=first_objects[2],
        optimizer_state_entry_count=0,
    )
    graph_reuse.claim_candidate_backward(
        first, autograd_graph=shared_graph, gradient_buffers=[shared_gradient]
    )
    graph_reuse.claim_candidate_objects(
        second,
        model=second_objects[0],
        method=second_objects[1],
        optimizer=second_objects[2],
        optimizer_state_entry_count=0,
    )
    with pytest.raises(D0V2IndependentCandidateError, match="object reuse"):
        graph_reuse.claim_candidate_backward(
            second, autograd_graph=shared_graph, gradient_buffers=[object()]
        )

    gradient_reuse = IndependentCandidateExecutionLedger()
    first_objects = (object(), object(), object())
    second_objects = (object(), object(), object())
    first_graph, second_graph = object(), object()
    gradient_reuse.claim_candidate_objects(
        first,
        model=first_objects[0],
        method=first_objects[1],
        optimizer=first_objects[2],
        optimizer_state_entry_count=0,
    )
    gradient_reuse.claim_candidate_backward(
        first, autograd_graph=first_graph, gradient_buffers=[shared_gradient]
    )
    gradient_reuse.claim_candidate_objects(
        second,
        model=second_objects[0],
        method=second_objects[1],
        optimizer=second_objects[2],
        optimizer_state_entry_count=0,
    )
    with pytest.raises(D0V2IndependentCandidateError, match="object reuse"):
        gradient_reuse.claim_candidate_backward(
            second, autograd_graph=second_graph, gradient_buffers=[shared_gradient]
        )


def test_live_ledger_requires_empty_state_own_backward_and_complete_grid() -> None:
    candidate = FROZEN_CANDIDATES[0]
    ledger = IndependentCandidateExecutionLedger()
    with pytest.raises(D0V2IndependentCandidateError, match="empty state"):
        ledger.claim_candidate_objects(
            candidate,
            model=object(),
            method=object(),
            optimizer=object(),
            optimizer_state_entry_count=1,
        )

    model, method, optimizer = object(), object(), object()
    ledger.claim_candidate_objects(
        candidate,
        model=model,
        method=method,
        optimizer=optimizer,
        optimizer_state_entry_count=0,
    )
    with pytest.raises(D0V2IndependentCandidateError, match="at least one"):
        ledger.claim_candidate_backward(
            candidate, autograd_graph=object(), gradient_buffers=[]
        )
    with pytest.raises(D0V2IndependentCandidateError, match="incomplete"):
        ledger.assert_complete()
