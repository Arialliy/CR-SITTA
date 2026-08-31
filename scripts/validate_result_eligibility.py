#!/usr/bin/env python3
"""Inspect, materialize, or validate the external artifact eligibility registry."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping

PROJECT_ROOT = Path(__file__).resolve().parents[1]
METRICS_ROOT = PROJECT_ROOT / "metrics"
if str(METRICS_ROOT) not in sys.path:
    # Keep this metadata-only CLI independent from metrics/__init__.py, whose
    # evaluator exports intentionally import NumPy and image dependencies.
    sys.path.insert(0, str(METRICS_ROOT))

from result_eligibility import (
    EXPECTED_REGISTRY_RELATIVE,
    build_registry,
    compute_artifact_identity,
    load_config,
    materialize_registry,
    registry_bytes,
    validate_materialized_registry,
)


DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "artifact_eligibility_registry_v1.yaml"


def inspect(config_path: Path) -> dict[str, Any]:
    config = load_config(config_path, PROJECT_ROOT).value
    artifacts = config.get("artifacts")
    if not isinstance(artifacts, list):
        raise TypeError("config.artifacts must be a list")
    values = []
    for artifact in artifacts:
        if not isinstance(artifact, Mapping):
            raise TypeError("artifact entry must be a mapping")
        identity = compute_artifact_identity(
            PROJECT_ROOT, artifact, verify_expected=False
        )
        values.append(
            {
                "artifact_id": artifact.get("artifact_id"),
                "root": artifact.get("root"),
                "algorithm": identity.algorithm,
                "sha256": identity.sha256,
                "file_count": identity.file_count,
                "total_size_bytes": identity.total_size_bytes,
            }
        )
    return {"inspection_only": True, "artifacts": values}


def expected_payload(config_path: Path) -> tuple[Path, bytes]:
    registry = build_registry(
        config_path,
        project_root=PROJECT_ROOT,
        generator_path=Path(__file__),
    )
    # build_registry has already validated the output from the same immutable
    # config byte snapshot. Do not re-read mutable YAML to choose a write path.
    output = PROJECT_ROOT / EXPECTED_REGISTRY_RELATIVE
    return output, registry_bytes(registry)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("role", choices=("inspect", "materialize", "validate"))
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--replace-existing",
        action="store_true",
        help="Atomically replace a different registry after an external backup.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    config_path = args.config.absolute()
    if args.replace_existing and args.role != "materialize":
        raise SystemExit("--replace-existing is valid only for role=materialize")
    if args.role == "inspect":
        result = inspect(config_path)
    else:
        output, payload = expected_payload(config_path)
        if args.role == "materialize":
            status = materialize_registry(
                output,
                payload,
                replace_existing=args.replace_existing,
            )
            result = {
                "role": "materialize",
                "status": status,
                "output": str(output),
            }
        else:
            validate_materialized_registry(output, payload)
            result = {"role": "validate", "valid": True, "output": str(output)}
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
