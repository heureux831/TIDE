#!/usr/bin/env python3
"""Render the V3 train/dev fact-learning curve as PNG and vector PDF."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

TRAIN = "#0072B2"  # Okabe-Ito blue
DEV = "#D55E00"  # Okabe-Ito vermillion


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("curve", type=Path)
    parser.add_argument("output_stem", type=Path)
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.curve.read_text().splitlines() if line.strip()]
    if not rows:
        raise ValueError("learning curve is empty")

    steps = [row["step"] for row in rows]
    fig, axes = plt.subplots(1, 3, figsize=(11.2, 3.4), constrained_layout=True)
    panels = (
        ("required_fact_recall", "RequiredFactRecall", True),
        ("fact_joint", "FactJoint", True),
        ("text_loss", "Text loss", False),
    )
    for axis, (key, label, percentage) in zip(axes, panels, strict=True):
        axis.plot(
            steps,
            [row[f"train_{key}"] for row in rows],
            color=TRAIN,
            marker="o",
            markersize=3.5,
            linewidth=1.8,
            label="train (n=400)",
        )
        axis.plot(
            steps,
            [row[f"dev_{key}"] for row in rows],
            color=DEV,
            marker="s",
            markersize=3.5,
            linewidth=1.8,
            label="dev (n=100)",
        )
        axis.set_xlabel("Optimizer step")
        axis.set_ylabel(label)
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(axis="y", color="#D0D0D0", linewidth=0.6, alpha=0.6)
        if percentage:
            axis.set_ylim(-0.03, 1.03)
            axis.yaxis.set_major_formatter(PercentFormatter(1.0))
    axes[0].legend(frameon=False, loc="lower right")
    fig.suptitle("V3 long-budget M2: deterministic fact retention")
    args.output_stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output_stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(args.output_stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
