from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = PROJECT_ROOT / "scripts" / "report_cr_sitta_d0a_status.py"
SPEC = importlib.util.spec_from_file_location("report_cr_sitta_d0a_status", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
STATUS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(STATUS)


def _proc(proc_root: Path, pid: int, argv: list[str]) -> None:
    directory = proc_root / str(pid)
    directory.mkdir(parents=True)
    (directory / "cmdline").write_bytes(b"\0".join(a.encode() for a in argv) + b"\0")


def test_states_tail_recovery_and_exact_process_match(tmp_path: Path) -> None:
    root = tmp_path / "runs"
    proc_root = tmp_path / "proc"
    proc_root.mkdir()

    running_dir = root / "IRSTD-1K"
    running_dir.mkdir(parents=True)
    (running_dir / "train_metrics.jsonl").write_text(
        '{"epoch": 8, "ending_optimizer_step": 88, "mean_combined_loss": 0.75}\n'
        '{"epoch": 9',
        encoding="utf-8",
    )
    (running_dir / "last.pth.tar").write_bytes(b"opaque checkpoint")
    _proc(proc_root, 101, ["python", "train.py", "--output-dir", str(running_dir)])

    stale_dir = root / "NUAA-SIRST"
    stale_dir.mkdir(parents=True)
    # A substring or unrelated positional argument must not count as a process match.
    _proc(proc_root, 102, ["python", "other.py", str(stale_dir)])

    report = STATUS.build_report(root, proc_root)
    items = {item["dataset"]: item for item in report["datasets"]}
    assert items["IRSTD-1K"]["state"] == "running_like"
    assert items["IRSTD-1K"]["epoch"] == 8
    assert items["IRSTD-1K"]["global_step"] == 88
    assert items["IRSTD-1K"]["recent_loss"] == 0.75
    assert items["IRSTD-1K"]["last_checkpoint_exists"] is True
    assert items["NUAA-SIRST"]["state"] == "failed_or_stale"
    assert items["NUDT-SIRST"]["state"] == "queued"


def test_complete_and_json_cli_with_injectable_root(tmp_path: Path) -> None:
    root = tmp_path / "runs"
    complete = root / "NUDT-SIRST"
    complete.mkdir(parents=True)
    (complete / "epoch_1000_train_only.pth.tar").write_bytes(b"not loaded")
    (complete / "summary.json").write_text("{}\n", encoding="utf-8")
    report = STATUS.build_report(root, tmp_path / "missing-proc")
    item = {entry["dataset"]: entry for entry in report["datasets"]}["NUDT-SIRST"]
    assert item["state"] == "complete"
    assert item["final_checkpoint_exists"] is True
    assert item["summary_exists"] is True

    completed = subprocess.run(
        [sys.executable, str(SCRIPT), "--root", str(root), "--json"],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(completed.stdout)
    assert payload["root"] == str(root.resolve())
    assert [entry["dataset"] for entry in payload["datasets"]] == list(STATUS.DATASETS)
