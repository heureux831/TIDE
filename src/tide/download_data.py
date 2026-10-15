"""Download small, auditable source-data samples for later TIDE extensions.

The controlled overnight feasibility experiment is generated locally and does
not train on these external samples. This command keeps provenance explicit.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.request import urlopen

from tide.env import load_project_env
from tide.io import sha256_file, write_json, write_jsonl

HEALTHBENCH_URL = (
    "https://openaipublic.blob.core.windows.net/simple-evals/healthbench/"
    "2025-05-07-06-14-12_oss_eval.jsonl"
)
WILDCHAT_REPO = "allenai/WildChat"


def download_healthbench(limit: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with urlopen(HEALTHBENCH_URL, timeout=120) as response:
        for raw_line in response:
            if raw_line.strip():
                rows.append(json.loads(raw_line))
            if len(rows) >= limit:
                break
    return rows, {"url": HEALTHBENCH_URL, "license": "MIT", "selection": "first_n"}


def download_wildchat(limit: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    from datasets import load_dataset

    stream = load_dataset(WILDCHAT_REPO, split="train", streaming=True)
    rows: list[dict[str, Any]] = []
    for row in stream:
        if row.get("toxic"):
            continue
        rows.append(dict(row))
        if len(rows) >= limit:
            break
    return rows, {
        "repository": WILDCHAT_REPO,
        "license": "ODC-BY",
        "selection": "first_n_non_toxic_streaming",
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=["healthbench", "wildchat"], required=True)
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--destination", type=Path, default=Path("data/downloads"))
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.limit <= 0:
        raise SystemExit("--limit must be positive")
    load_project_env(args.env_file)
    if args.source == "healthbench":
        rows, provenance = download_healthbench(args.limit)
    else:
        rows, provenance = download_wildchat(args.limit)
    output = args.destination / f"{args.source}_sample.jsonl"
    count = write_jsonl(output, rows)
    manifest = {
        "source": args.source,
        "record_count": count,
        "downloaded_at_utc": datetime.now(timezone.utc).isoformat(),
        "output": str(output),
        "sha256": sha256_file(output),
        "provenance": provenance,
        "use_in_controlled_overnight_experiment": False,
    }
    write_json(output.with_suffix(".manifest.json"), manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
