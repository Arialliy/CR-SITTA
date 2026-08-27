"""Print a compact status table for fixed-split NS-FPN retraining runs."""

from __future__ import annotations

import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = PROJECT_ROOT / "results" / "retraining_fixed_split"


def last_jsonl(path: Path) -> dict | None:
    if not path.is_file():
        return None
    last = None
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                last = json.loads(line)
    return last


def main() -> None:
    print("dataset\tepoch\tloss\tlatest_miou\tlatest_pd\tbest_files\tcomplete")
    for dataset in ("IRSTD-1K", "NUAA-SIRST", "NUDT-SIRST"):
        directory = OUTPUT_ROOT / dataset
        train = last_jsonl(directory / "train_metrics.jsonl") or {}
        test = last_jsonl(directory / "test_metrics.jsonl") or {}
        best_count = sum(
            (directory / name).is_file()
            for name in ("best_miou.pth.tar", "best_pd.pth.tar")
        )
        print(
            f"{dataset}\t{train.get('epoch', '-')}\t"
            f"{train.get('mean_loss', '-')}\t{test.get('miou', '-')}\t"
            f"{test.get('pd', '-')}\t{best_count}/2\t"
            f"{(directory / 'summary.json').is_file()}"
        )


if __name__ == "__main__":
    main()
