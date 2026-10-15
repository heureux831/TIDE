#!/usr/bin/env python3
"""Create review-friendly V3 metric and fixed-sample tables from raw JSONL."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("result_dir", type=Path)
    args = parser.parse_args()
    result_dir = args.result_dir
    curve = _read_jsonl(result_dir / "learning_curve.jsonl")
    audit = _read_jsonl(result_dir / "fixed_20_v2_audit.jsonl")
    evolution = _read_jsonl(result_dir / "fixed_20_evolution.jsonl")

    metric_fields = [
        "step",
        "train_required_fact_recall",
        "train_fact_joint",
        "train_text_loss",
        "dev_required_fact_recall",
        "dev_fact_joint",
        "dev_text_loss",
        "elapsed_seconds",
    ]
    with (result_dir / "checkpoint_metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=metric_fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(curve)

    by_example: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in evolution:
        by_example[row["example_id"]].append(row)
    lines = [
        "# V3 fixed-20 rewrite evolution",
        "",
        "Every checkpoint below is a free generation. `PASS` means deterministic FactJoint=1.",
        "",
    ]
    for baseline in audit:
        example_id = baseline["example_id"]
        lines.extend(
            [
                f"## {example_id}",
                "",
                f"- Task: {baseline['task']}",
                f"- Reference: {baseline['reference_rewrite']}",
                f"- Original context: {baseline['context']}",
                "",
                "| Checkpoint | FactJoint | Free-generation output |",
                "|---:|:---:|---|",
                f"| V2 best (300-step run) | FAIL | {baseline['output']} |",
            ]
        )
        for row in sorted(by_example[example_id], key=lambda item: int(item["step"])):
            status = "PASS" if row["fact_joint"] else "FAIL"
            lines.append(f"| {row['step']} | {status} | {row['output']} |")
        lines.append("")
    (result_dir / "fixed_20_evolution.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
