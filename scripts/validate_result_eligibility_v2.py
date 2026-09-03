#!/usr/bin/env python3
"""Inspect or publish the append-only artifact eligibility registry v2."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
METRICS_ROOT = PROJECT_ROOT / "metrics"
if str(METRICS_ROOT) not in sys.path:
    sys.path.insert(0, str(METRICS_ROOT))

import result_eligibility as v1
from result_eligibility_v2 import (
    EXPECTED_REGISTRY_RELATIVE,
    build_v2_registry,
    inspect_v2_config,
    registry_bytes,
)


DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "artifact_eligibility_registry_v2.yaml"


def expected_payload(config_path: Path) -> tuple[Path, bytes]:
    registry = build_v2_registry(
        config_path,
        project_root=PROJECT_ROOT,
        generator_path=Path(__file__),
    )
    return PROJECT_ROOT / EXPECTED_REGISTRY_RELATIVE, registry_bytes(registry)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("role", choices=("inspect", "materialize", "validate"))
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    config_path = args.config.absolute()
    if args.role == "inspect":
        result: dict[str, Any] = dict(inspect_v2_config(config_path, PROJECT_ROOT))
    else:
        output, payload = expected_payload(config_path)
        if args.role == "materialize":
            # v2 is immutable once published.  There is intentionally no
            # replace-existing option; any later extension must be v3.
            status = v1.materialize_registry(output, payload, replace_existing=False)
            result = {"role": "materialize", "status": status, "output": str(output)}
        else:
            v1.validate_materialized_registry(output, payload)
            result = {"role": "validate", "valid": True, "output": str(output)}
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
