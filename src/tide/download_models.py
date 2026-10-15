"""Reproducibly download the FLAN-T5 rewriter and Qwen task evaluator."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal, cast

from tide.env import load_project_env
from tide.io import sha256_file, write_json

MODEL_ENV = {
    "flan-t5": ("FLAN_T5_MODEL_ID", "FLAN_T5_REVISION", "google/flan-t5-base"),
    "qwen": ("QWEN_TASK_MODEL_ID", "QWEN_TASK_MODEL_REVISION", "Qwen/Qwen3-8B"),
}

Provider = Literal["huggingface", "hf-mirror", "modelscope"]
PROVIDERS: tuple[Provider, ...] = ("modelscope", "hf-mirror", "huggingface")
IGNORED_WEIGHT_FORMATS = ("*.bin", "*.h5", "*.msgpack")


def provider_order(requested: str) -> list[Provider]:
    if requested != "auto":
        if requested not in PROVIDERS:
            choices = ", ".join(PROVIDERS)
            raise ValueError(f"provider must be one of: {choices}")
        return [cast(Provider, requested)]
    configured = os.getenv("TIDE_MODEL_PROVIDER", "").strip()
    if configured:
        if configured not in PROVIDERS:
            choices = ", ".join(PROVIDERS)
            raise ValueError(f"TIDE_MODEL_PROVIDER must be one of: {choices}")
        preferred = cast(Provider, configured)
        return [preferred, *(provider for provider in PROVIDERS if provider != preferred)]
    return list(PROVIDERS)


def _artifact_inventory(local_dir: Path) -> list[dict[str, object]]:
    files = []
    for path in sorted(local_dir.rglob("*")):
        if not path.is_file() or path.name == "tide_download_manifest.json":
            continue
        if path.name.endswith(".incomplete"):
            continue
        files.append(
            {
                "path": str(path.relative_to(local_dir)),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return files


def _download_huggingface(
    repo_id: str, revision: str, local_dir: Path, endpoint: str | None
) -> str:
    from huggingface_hub import HfApi, snapshot_download

    token = os.getenv("HF_TOKEN") or None
    api = HfApi(endpoint=endpoint, token=token)
    info = api.model_info(repo_id=repo_id, revision=revision)
    if not info.sha:
        raise RuntimeError("Hugging Face did not return a resolved commit SHA")
    snapshot_download(
        repo_id=repo_id,
        revision=info.sha,
        local_dir=local_dir,
        token=token,
        endpoint=endpoint,
        ignore_patterns=list(IGNORED_WEIGHT_FORMATS),
    )
    return info.sha


def _download_modelscope(repo_id: str, revision: str, local_dir: Path) -> str:
    if importlib.util.find_spec("modelscope") is None:
        raise RuntimeError(
            "ModelScope is not installed; install it or choose --provider hf-mirror"
        )
    from modelscope import snapshot_download  # type: ignore[import-untyped]

    resolved_revision = "master" if revision == "main" else revision
    workers = int(os.getenv("MODELSCOPE_DOWNLOAD_MAX_WORKERS", "5"))
    os.environ.setdefault("MODELSCOPE_DOWNLOAD_PARALLEL_WORKERS", "4")
    os.environ.setdefault("MODELSCOPE_DOWNLOAD_PARALLEL_PART_SIZE_MB", "128")
    snapshot_download(
        model_id=repo_id,
        revision=resolved_revision,
        local_dir=str(local_dir),
        ignore_patterns=list(IGNORED_WEIGHT_FORMATS),
        max_workers=workers,
    )
    return resolved_revision


def _download_with_provider(
    provider: Provider, repo_id: str, revision: str, local_dir: Path
) -> str:
    if provider == "modelscope":
        return _download_modelscope(repo_id, revision, local_dir)
    endpoint = (
        os.getenv("HF_MIRROR_ENDPOINT", "https://hf-mirror.com")
        if provider == "hf-mirror"
        else None
    )
    return _download_huggingface(repo_id, revision, local_dir, endpoint)


def download_model(
    kind: str, destination: Path, dry_run: bool, requested_provider: str
) -> dict[str, object]:
    id_env, revision_env, default_id = MODEL_ENV[kind]
    repo_id = os.getenv(id_env, default_id)
    revision = os.getenv(revision_env, "main")
    local_dir = destination / kind
    providers = provider_order(requested_provider)
    plan: dict[str, object] = {
        "kind": kind,
        "repo_id": repo_id,
        "requested_revision": revision,
        "provider_order": providers,
        "local_dir": str(local_dir),
        "dry_run": dry_run,
    }
    if dry_run:
        return plan
    local_dir.mkdir(parents=True, exist_ok=True)
    failures: list[dict[str, str]] = []
    selected: Provider | None = None
    resolved_revision = ""
    for provider in providers:
        try:
            resolved_revision = _download_with_provider(
                provider, repo_id, revision, local_dir
            )
            selected = provider
            break
        except Exception as exc:
            failures.append({"provider": provider, "error": f"{type(exc).__name__}: {exc}"})
            if requested_provider != "auto":
                raise
    if selected is None:
        summary = "; ".join(f"{item['provider']}: {item['error']}" for item in failures)
        raise RuntimeError(f"all model download providers failed: {summary}")
    removed_partial_files = []
    for partial in sorted(local_dir.rglob("*.incomplete")):
        removed_partial_files.append(str(partial.relative_to(local_dir)))
        partial.unlink()
    plan.update(
        {
            "provider": selected,
            "resolved_revision": resolved_revision,
            "downloaded_at_utc": datetime.now(timezone.utc).isoformat(),
            "artifacts": _artifact_inventory(local_dir),
            "failed_provider_attempts": failures,
            "removed_partial_files": removed_partial_files,
        }
    )
    write_json(local_dir / "tide_download_manifest.json", plan)
    return plan


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=["flan-t5", "qwen", "all"], default="all")
    parser.add_argument("--destination", type=Path, default=Path("models"))
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    parser.add_argument(
        "--provider",
        choices=["auto", *PROVIDERS],
        default="auto",
        help="Download channel; auto tries ModelScope, hf-mirror, then Hugging Face",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    load_project_env(args.env_file)
    kinds = list(MODEL_ENV) if args.model == "all" else [args.model]
    manifests = [
        download_model(kind, args.destination, args.dry_run, args.provider) for kind in kinds
    ]
    print(json.dumps(manifests, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
