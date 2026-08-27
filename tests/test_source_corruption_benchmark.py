from __future__ import annotations

import run_source_corruption_benchmark as benchmark


def test_canonical_json_equal_normalises_tuple_and_list_only() -> None:
    measured = {
        "fixed": {"iou": 0.75, "counts": (3, 1, 0)},
        "froc": ({"threshold": 0.5, "tp": 3},),
    }
    frozen_json = {
        "fixed": {"counts": [3, 1, 0], "iou": 0.75},
        "froc": [{"tp": 3, "threshold": 0.5}],
    }

    assert benchmark._canonical_json_equal(measured, frozen_json)


def test_canonical_json_equal_does_not_add_numeric_tolerance() -> None:
    assert not benchmark._canonical_json_equal(
        {"probability": (0.5,)}, {"probability": [0.5000000000000001]}
    )


def test_formal_condition_contract_is_exactly_thirteen() -> None:
    protocol, _ = benchmark._load_protocol(
        benchmark.DEFAULT_PROTOCOL, "IRSTD-1K"
    )

    conditions = benchmark._conditions(protocol)

    assert len(conditions) == 13
    assert conditions[0] == ("clean", 0)
    assert conditions[-1] == ("stripe_noise", 5)
