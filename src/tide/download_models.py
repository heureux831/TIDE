"""Reproducibly download the FLAN-T5 rewriter and Qwen task evaluator."""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path

from tide.env import load_project_env
from tide.io import write_json

MODEL_ENV = {
    "flan-t5": ("FLAN_T5_MODEL_ID", "FLAN_T5_REVISION", "google/flan-t5-base"),
    "qwen": ("QWEN_TASK_MODEL_ID", "QWEN_TASK_MODEL_REVISION", "Qwen/Qwen3-8B"),
}


def download_model(kind: str, destination: Path, dry_run: bool) -> dict[str, object]:
    from huggingface_hub import HfApi, snapshot_download

    id_env, revision_env, default_id = MODEL_ENV[kind]
    repo_id = os.getenv(id_env, default_id)
    revision = os.getenv(revision_env, "main")
    local_dir = destination / kind
    plan: dict[str, object] = {
        "kind": kind,
        "repo_id": repo_id,
        "requested_revision": revision,
        "local_dir": str(local_dir),
        "dry_run": dry_run,
    }
    if dry_run:
        return plan
    token = os.getenv("HF_TOKEN") or None
    info = HfApi(token=token).model_info(repo_id=repo_id, revision=revision)
    snapshot_download(
        repo_id=repo_id,
        revision=info.sha,
        local_dir=local_dir,
        token=token,
    )
    plan.update(
        {
            "resolved_revision": info.sha,
            "downloaded_at_utc": datetime.now(timezone.utc).isoformat(),
        }
    )
    write_json(local_dir / "tide_download_manifest.json", plan)
    return plan


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=["flan-t5", "qwen", "all"], default="all")
    parser.add_argument("--destination", type=Path, default=Path("models"))
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    load_project_env(args.env_file)
    kinds = list(MODEL_ENV) if args.model == "all" else [args.model]
    manifests = [download_model(kind, args.destination, args.dry_run) for kind in kinds]
    print(json.dumps(manifests, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
