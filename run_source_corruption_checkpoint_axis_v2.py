#!/usr/bin/env python3
"""Run the role-aware Source corruption checkpoint axis v2."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from benchmark import checkpoint_axis as checkpoint_axis_api
from benchmark.source_corruption_axis_runner_v2 import (
    DATASETS,
    DEFAULT_SOURCE_PROTOCOL,
    run_source_corruption_axis,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=DATASETS)
    parser.add_argument(
        "--checkpoint-role", required=True, choices=("best_miou", "best_pd")
    )
    parser.add_argument(
        "--axis-config",
        type=Path,
        default=checkpoint_axis_api.DEFAULT_AXIS_CONFIG,
    )
    parser.add_argument("--source-protocol", type=Path, default=DEFAULT_SOURCE_PROTOCOL)
    parser.add_argument(
        "--clean-artifact",
        type=Path,
        default=None,
        help="Required best_miou parity clean artifact; best_pd uses the canonical clean root.",
    )
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--parity-reference",
        type=Path,
        default=None,
        help="Frozen v1 Source artifact; required only for best_miou parity runs.",
    )
    parser.add_argument(
        "--parity-receipt",
        type=Path,
        default=None,
        help=(
            "Global clean+Source+AdaBN best_miou parity receipt. If supplied, "
            "it must resolve to the one configured fixed path; best_pd otherwise "
            "uses that fixed path automatically."
        ),
    )
    return parser


def run(args: argparse.Namespace) -> dict[str, object]:
    config = checkpoint_axis_api.load_axis_config(args.axis_config)
    axis = checkpoint_axis_api.resolve_axis(
        config,
        dataset=args.dataset,
        role=args.checkpoint_role,
        artifact_kind="source",
        output_override=args.output_dir,
    )
    clean_artifact = args.clean_artifact
    if clean_artifact is None:
        if args.checkpoint_role == "best_miou":
            raise ValueError(
                "best_miou parity requires --clean-artifact from the clean-v2 parity run"
            )
        clean_artifact = checkpoint_axis_api.canonical_output_dir(
            config,
            artifact_kind="clean",
            role=args.checkpoint_role,
            dataset=args.dataset,
        )
    return run_source_corruption_axis(
        axis_config=config,
        axis=axis,
        clean_artifact=Path(clean_artifact),
        source_protocol_path=args.source_protocol,
        cache_dir=args.cache_dir,
        output_dir=args.output_dir,
        device_name=args.device,
        parity_reference=args.parity_reference,
        parity_receipt=args.parity_receipt,
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = run(args)
    print(f"Artifacts: {result['published_output_dir']}")
    print(
        f"{result['dataset']} / {result['checkpoint_role']}: "
        f"{result['condition_count']} conditions, "
        f"{result['evaluated_images_per_condition']} images/condition"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
