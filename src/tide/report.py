"""Aggregate saved evaluations without re-running models."""

from __future__ import annotations

import csv
import json
import random
from collections import defaultdict
from collections.abc import Callable
from pathlib import Path
from statistics import mean
from typing import Any

from tide.config import TideConfig
from tide.io import atomic_write_text, write_json


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _bootstrap(
    rows: list[dict[str, Any]], key: Callable[[dict[str, Any]], float], replicates: int, seed: int
) -> list[float]:
    by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_source[row["source_id"]].append(row)
    source_ids = sorted(by_source)
    rng = random.Random(seed)
    values = []
    for _ in range(replicates):
        sampled = [
            item for _source in range(len(source_ids)) for item in by_source[rng.choice(source_ids)]
        ]
        values.append(mean(key(item) for item in sampled))
    return sorted(values)


def _interval(values: list[float]) -> list[float]:
    return [values[int(0.025 * (len(values) - 1))], values[int(0.975 * (len(values) - 1))]]


def _paired_bootstrap(
    left: list[dict[str, Any]],
    right: list[dict[str, Any]],
    key: Callable[[dict[str, Any]], float],
    replicates: int,
    seed: int,
) -> tuple[float, list[float]]:
    left_by_key = {(row["source_id"], row["task_id"]): row for row in left}
    right_by_key = {(row["source_id"], row["task_id"]): row for row in right}
    if set(left_by_key) != set(right_by_key):
        raise ValueError("paired model results do not contain identical examples")
    source_ids = sorted({source_id for source_id, _task_id in left_by_key})

    def source_difference(source_id: str) -> float:
        task_keys = sorted(item for item in left_by_key if item[0] == source_id)
        return mean(key(left_by_key[item]) - key(right_by_key[item]) for item in task_keys)

    differences = {source_id: source_difference(source_id) for source_id in source_ids}
    rng = random.Random(seed)
    samples = sorted(
        mean(differences[rng.choice(source_ids)] for _ in source_ids) for _ in range(replicates)
    )
    return mean(differences.values()), _interval(samples)


def _task_switch_success(items: list[dict[str, Any]]) -> float:
    by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in items:
        by_source[item["source_id"]].append(item)
    return mean(
        float(len(source_rows) == 2 and all(row["fact_joint"] for row in source_rows))
        for source_rows in by_source.values()
    )


def build_report(config: TideConfig, root: Path) -> dict[str, Any]:
    evaluation_dir = root / config.experiment.output_dir / "evaluation"
    rows = _load_jsonl(evaluation_dir / "test.jsonl")
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["group"]].append(row)
    metrics: dict[str, Callable[[dict[str, Any]], float]] = {
        "task_acc": lambda row: float(row["task_answer"] == row["gold_answer"]),
        "any_leak_rate": lambda row: float(bool(row["protected_matches"])),
        "joint_success": lambda row: float(row["joint_success"]),
    }
    summary: dict[str, Any] = {}
    for group, items in grouped.items():
        summary[group] = {}
        for metric, function in metrics.items():
            values = _bootstrap(
                items,
                function,
                config.evaluation.bootstrap_replicates,
                config.evaluation.bootstrap_seed,
            )
            summary[group][metric] = {
                "value": mean(function(item) for item in items),
                "ci95": _interval(values),
            }
        summary[group].update(
            {
                "required_fact_recall": {
                    "value": sum(len(item["required_matches"]) for item in items)
                    / (2 * len(items)),
                    "ci95": None,
                },
                "direct_leak_rate": {
                    "value": sum(len(item["protected_matches"]) for item in items)
                    / (6 * len(items)),
                    "ci95": None,
                },
                "fact_joint": {
                    "value": mean(float(item["fact_joint"]) for item in items),
                    "ci95": None,
                },
                "parse_unknown_rate": {
                    "value": mean(float(item["parser_unknown"]) for item in items),
                    "ci95": None,
                },
                "task_switch_success": {
                    "value": _task_switch_success(items),
                    "ci95": None,
                },
            }
        )
        summary[group]["by_task"] = {
            task_id: {
                metric: mean(function(item) for item in items if item["task_id"] == task_id)
                for metric, function in metrics.items()
            }
            for task_id in ("A", "B")
        }
    paired: dict[str, Any] = {}
    m1 = "M1_FLAN_T5_LORA"
    m2 = "M2_FLAN_T5_MATRIX"
    if m1 in grouped and m2 in grouped:
        for metric, function in metrics.items():
            value, interval = _paired_bootstrap(
                grouped[m2],
                grouped[m1],
                function,
                config.evaluation.bootstrap_replicates,
                config.evaluation.bootstrap_seed,
            )
            paired[f"{m2}_minus_{m1}/{metric}"] = {"value": value, "ci95": interval}
    threshold_verdict: dict[str, Any] = {"evaluable": False}
    if all(group in summary for group in ("RAW", "ORACLE", m2)):
        baseline_ok = (
            summary["RAW"]["task_acc"]["value"] >= 0.95
            and summary["ORACLE"]["task_acc"]["value"] >= 0.95
        )
        m2_task = summary[m2]["task_acc"]["value"]
        m2_ok = (
            m2_task >= 0.90
            and summary["RAW"]["task_acc"]["value"] - m2_task <= 0.05
            and summary[m2]["required_fact_recall"]["value"] >= 0.95
            and summary[m2]["any_leak_rate"]["value"] <= 0.10
            and summary[m2]["joint_success"]["value"] >= 0.80
        )
        threshold_verdict = {
            "evaluable": baseline_ok,
            "raw_and_oracle_task_gate": baseline_ok,
            "m2_engineering_thresholds": m2_ok if baseline_ok else None,
            "matrix_benefit_over_m1_supported": None,
        }
        if m1 in summary:
            task_delta = m2_task - summary[m1]["task_acc"]["value"]
            leak_delta = (
                summary[m2]["any_leak_rate"]["value"] - summary[m1]["any_leak_rate"]["value"]
            )
            threshold_verdict["m2_minus_m1_task_acc"] = task_delta
            threshold_verdict["m2_minus_m1_any_leak_rate"] = leak_delta
            threshold_verdict["matrix_benefit_over_m1_supported"] = (
                task_delta > 0 and leak_delta <= 0
            ) or (task_delta >= 0 and leak_delta < 0)
    artifact = {
        "groups": summary,
        "paired_differences": paired,
        "threshold_verdict": threshold_verdict,
    }
    write_json(evaluation_dir / "bootstrap_summary.json", artifact)
    csv_path = evaluation_dir / "summary.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["group", "metric", "value", "ci95_low", "ci95_high"])
        for group, group_metrics in summary.items():
            for metric, result in group_metrics.items():
                if metric == "by_task":
                    continue
                interval = result["ci95"] or ["", ""]
                writer.writerow([group, metric, result["value"], *interval])
    failures = [row for row in rows if not row["joint_success"]]
    priority = sorted(
        failures,
        key=lambda row: (
            not row["parser_unknown"],
            not bool(row["protected_matches"]),
            not row["numeric_tamper"],
            row["source_id"],
            row["task_id"],
        ),
    )[:10]
    lines = [
        "# TIDE feasibility report",
        "",
        "This report separates code-path completion, threshold attainment, and M2-vs-M1 benefit.",
        "It does not claim differential privacy or real-world validation.",
        "",
        "## Test metrics",
        "",
        "| Group | TaskAcc | AnyLeakRate | JointSuccess |",
        "| --- | ---: | ---: | ---: |",
    ]
    for group in sorted(summary):
        lines.append(
            f"| {group} | {summary[group]['task_acc']['value']:.3f} | "
            f"{summary[group]['any_leak_rate']['value']:.3f} | "
            f"{summary[group]['joint_success']['value']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Prespecified decision gates",
            "",
            f"- RAW/ORACLE evaluator gate: {threshold_verdict.get('raw_and_oracle_task_gate')}",
            f"- M2 engineering thresholds: {threshold_verdict.get('m2_engineering_thresholds')}",
            f"- Additional M2 benefit over M1 supported: "
            f"{threshold_verdict.get('matrix_benefit_over_m1_supported')}",
        ]
    )
    lines.extend(["", "## Deterministically selected failures", ""])
    for row in priority:
        lines.append(
            f"- `{row['group']} / {row['example_id']}`: answer={row['task_answer']}, "
            f"protected={row['protected_matches']}, unknown={row['parser_unknown']}, "
            f"rewrite={row['rewrite']!r}"
        )
    atomic_write_text(root / config.experiment.output_dir / "report.md", "\n".join(lines) + "\n")
    return artifact
