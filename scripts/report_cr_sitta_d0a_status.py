#!/usr/bin/env python3
"""Read-only status report for CR-SITTA D0-A full training.

This utility never imports torch and never opens checkpoint payloads.  A run is
considered running-like only when a live procfs command line contains an exact
``--output-dir`` argument matching the dataset directory.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Iterable, Mapping


DATASETS = ("IRSTD-1K", "NUAA-SIRST", "NUDT-SIRST")
DEFAULT_ROOT = Path("results/cr_sitta/d0a_supervised_lfhf_train_v2")
TOTAL_EPOCHS = 1000
MAX_TAIL_BYTES = 8 * 1024 * 1024
MAX_JSON_LINE_BYTES = 1024 * 1024


def _last_complete_json_object(path: Path) -> dict[str, Any] | None:
    """Return the last complete JSON object in a bounded file tail."""

    try:
        with path.open("rb") as handle:
            size = handle.seek(0, os.SEEK_END)
            handle.seek(max(0, size - MAX_TAIL_BYTES))
            payload = handle.read(MAX_TAIL_BYTES)
    except (FileNotFoundError, IsADirectoryError, OSError):
        return None

    lines = payload.splitlines()
    # If the read began in the middle of a line, do not consider that fragment.
    if size > MAX_TAIL_BYTES and lines:
        lines = lines[1:]
    for raw in reversed(lines):
        if not raw.strip() or len(raw) > MAX_JSON_LINE_BYTES:
            continue
        try:
            value = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            continue
        if isinstance(value, dict):
            return value
    return None


def _output_dirs_from_cmdline(raw: bytes) -> Iterable[str]:
    try:
        argv = [part.decode("utf-8") for part in raw.split(b"\0") if part]
    except UnicodeDecodeError:
        return ()
    found: list[str] = []
    for index, arg in enumerate(argv):
        if arg == "--output-dir" and index + 1 < len(argv):
            found.append(argv[index + 1])
        elif arg.startswith("--output-dir="):
            found.append(arg.split("=", 1)[1])
    return found


def _is_running_for_output_dir(output_dir: Path, proc_root: Path) -> bool:
    expected = os.path.abspath(os.fspath(output_dir))
    try:
        processes = list(proc_root.iterdir())
    except OSError:
        return False
    for process in processes:
        if not process.name.isdecimal():
            continue
        try:
            raw = (process / "cmdline").read_bytes()
        except (FileNotFoundError, PermissionError, ProcessLookupError, OSError):
            continue
        for candidate in _output_dirs_from_cmdline(raw):
            if os.path.abspath(candidate) == expected:
                return True
    return False


def _integer(record: Mapping[str, Any] | None, *keys: str) -> int | None:
    if record is None:
        return None
    for key in keys:
        value = record.get(key)
        if isinstance(value, int) and not isinstance(value, bool):
            return value
    return None


def dataset_status(output_dir: Path, proc_root: Path = Path("/proc")) -> dict[str, Any]:
    metrics = _last_complete_json_object(output_dir / "train_metrics.jsonl")
    last_checkpoint = output_dir / "last.pth.tar"
    final_checkpoint = output_dir / f"epoch_{TOTAL_EPOCHS}_train_only.pth.tar"
    summary = output_dir / "summary.json"
    last_exists = last_checkpoint.is_file()
    final_exists = final_checkpoint.is_file()
    summary_exists = summary.is_file()
    running = _is_running_for_output_dir(output_dir, proc_root)

    if final_exists and summary_exists:
        state = "complete"
    elif running:
        state = "running_like"
    elif not output_dir.exists():
        state = "queued"
    else:
        # Directory existence alone is deliberately not interpreted as running.
        state = "failed_or_stale"

    epoch = _integer(metrics, "epoch")
    global_step = _integer(metrics, "ending_optimizer_step", "global_optimizer_step")
    recent_loss = metrics.get("mean_combined_loss") if metrics is not None else None
    if not isinstance(recent_loss, (int, float)) or isinstance(recent_loss, bool):
        recent_loss = None
    return {
        "dataset": output_dir.name,
        "state": state,
        "epoch": epoch,
        "total_epochs": TOTAL_EPOCHS,
        "global_step": global_step,
        "recent_loss": recent_loss,
        "last_checkpoint_exists": last_exists,
        "final_checkpoint_exists": final_exists,
        "summary_exists": summary_exists,
        "process_match": running,
        "output_dir": os.path.abspath(os.fspath(output_dir)),
    }


def build_report(root: Path, proc_root: Path = Path("/proc")) -> dict[str, Any]:
    root = Path(os.path.abspath(os.fspath(root)))
    return {
        "root": os.fspath(root),
        "datasets": [dataset_status(root / name, proc_root) for name in DATASETS],
    }


def _render_text(report: Mapping[str, Any]) -> str:
    lines = [f"CR-SITTA D0-A status: {report['root']}"]
    for item in report["datasets"]:
        epoch = "?" if item["epoch"] is None else str(item["epoch"])
        step = "?" if item["global_step"] is None else str(item["global_step"])
        loss = "?" if item["recent_loss"] is None else f"{item['recent_loss']:.6f}"
        lines.append(
            f"{item['dataset']}: {item['state']} epoch={epoch}/{item['total_epochs']} "
            f"step={step} loss={loss} last={item['last_checkpoint_exists']} "
            f"final={item['final_checkpoint_exists']} summary={item['summary_exists']}"
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--json", action="store_true", dest="as_json")
    args = parser.parse_args()
    report = build_report(args.root)
    if args.as_json:
        print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    else:
        print(_render_text(report))


if __name__ == "__main__":
    main()
