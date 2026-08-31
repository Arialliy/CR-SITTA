#!/usr/bin/env python3
"""Materialize or read-only validate the TTA train-side Pilot v2 ID artifacts."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys


PROJECT_ROOT = Path(os.path.abspath(__file__)).parents[1]
DATAIO_ROOT = PROJECT_ROOT / "dataio"
if str(DATAIO_ROOT) not in sys.path:
    # Import the metadata-only module without executing dataio/__init__.py,
    # whose research-dataset exports intentionally load pixel dependencies.
    sys.path.insert(0, str(DATAIO_ROOT))

from train_side_pilot_protocol import (  # noqa: E402
    build_contract,
    materialize,
    validate_materialization,
)


DEFAULT_PROTOCOL = PROJECT_ROOT / "configs" / "tta_train_side_calibration_pilot_v2.yaml"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Read metadata only; perform no filesystem writes.",
    )
    return parser


def run(args: argparse.Namespace) -> dict[str, object]:
    context = build_contract(PROJECT_ROOT, args.protocol)
    if args.validate_only:
        return validate_materialization(context)
    return materialize(context)


def main() -> None:
    result = run(build_parser().parse_args())
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
