"""Command-line interface for the TIDE feasibility experiment."""

from __future__ import annotations

import argparse
import asyncio
import importlib.metadata
import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

from tide.config import TideConfig, load_config
from tide.data_build import build_and_write
from tide.env import env_status, load_project_env
from tide.io import write_json, write_jsonl
from tide.schemas import ExpandedRecord, SourceRecord
from tide.validate import dataset_validation_report


def _root_from_config(config_path: Path) -> Path:
    resolved = config_path.resolve()
    if resolved.parent.name != "configs":
        raise ValueError("config file must live in the project's configs directory")
    return resolved.parent.parent


def _load_sources(path: Path) -> list[SourceRecord]:
    with path.open(encoding="utf-8") as handle:
        return [SourceRecord.model_validate_json(line) for line in handle if line.strip()]


def _common(args: argparse.Namespace) -> tuple[TideConfig, Path]:
    root = _root_from_config(args.config)
    load_project_env(root / ".env")
    return load_config(args.config), root


def command_preflight(args: argparse.Namespace) -> int:
    config, root = _common(args)
    report: dict[str, Any] = {
        "project_root": str(root),
        "python": sys.version,
        "platform": platform.platform(),
        "environment_present": env_status("DEEPSEEK_API_KEY", "HF_TOKEN"),
        "configured_models": {
            "rewriter": os.getenv(config.model.backbone_env, config.model.backbone),
            "task_evaluator": os.getenv(
                config.evaluation.task_model_env, config.evaluation.task_model
            ),
        },
    }
    try:
        import torch

        report["torch"] = torch.__version__
        report["torch_cuda_runtime"] = torch.version.cuda
        report["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            device = torch.cuda.get_device_properties(0)
            report["gpu"] = {
                "name": device.name,
                "memory_bytes": device.total_memory,
                "capability": [device.major, device.minor],
            }
            if args.deep:
                if "RTX PRO 6000 Blackwell" not in device.name:
                    raise RuntimeError(f"expected RTX PRO 6000 Blackwell, found {device.name!r}")
                value = torch.randn(256, 256, device="cuda", dtype=torch.bfloat16)
                value.requires_grad_(True)
                loss_value = (value @ value).float().square().mean()
                loss_value.backward()  # type: ignore[no-untyped-call]
                if value.grad is None:
                    raise RuntimeError("BF16 smoke test did not produce gradients")
                report["bf16_forward_backward"] = bool(
                    torch.isfinite(loss_value) and torch.isfinite(value.grad).all()
                )
                from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

                local_model = root / "models/flan-t5"
                source = (
                    str(local_model)
                    if local_model.exists() and any(local_model.iterdir())
                    else os.getenv(config.model.backbone_env, config.model.backbone)
                )
                revision = os.getenv(config.model.revision_env, config.model.revision)
                tokenizer = AutoTokenizer.from_pretrained(source, revision=revision)
                smoke_model = AutoModelForSeq2SeqLM.from_pretrained(source, revision=revision).to(
                    "cuda"
                )
                smoke = tokenizer(
                    [
                        "Rewrite for the task. Task: retain age. "
                        "Context: The person is 34 years old."
                    ],
                    text_target=["The person is 34 years old."],
                    return_tensors="pt",
                )
                smoke = {key: tensor.to("cuda") for key, tensor in smoke.items()}
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    smoke_loss = smoke_model(**smoke).loss
                smoke_loss.backward()  # type: ignore[no-untyped-call]
                report["flan_t5_batch_forward_backward"] = bool(torch.isfinite(smoke_loss))
                del smoke_model, smoke, smoke_loss
                torch.cuda.empty_cache()
        if args.deep and not torch.cuda.is_available():
            raise RuntimeError("--deep preflight requires CUDA on the target server")
    except ImportError:
        report["torch"] = None
        report["cuda_available"] = False
    try:
        command_result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,memory.free,driver_version",
                "--format=csv",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        report["nvidia_smi"] = command_result.stdout.strip()
    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        report["nvidia_smi"] = None
        report["nvidia_smi_error"] = type(exc).__name__
    report["versions"] = {}
    for distribution in (
        "accelerate",
        "datasets",
        "huggingface-hub",
        "openai",
        "peft",
        "torch",
        "transformers",
    ):
        try:
            report["versions"][distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            report["versions"][distribution] = None
    disk = shutil.disk_usage(root)
    report["disk_free_bytes"] = disk.free
    try:
        import psutil

        report["host_memory_available_bytes"] = psutil.virtual_memory().available
    except ImportError:
        report["host_memory_available_bytes"] = None
    target = root / config.experiment.output_dir / "preflight.json"
    write_json(target, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


def command_build_data(args: argparse.Namespace) -> int:
    config, root = _common(args)
    report = build_and_write(config, root)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


def command_validate_data(args: argparse.Namespace) -> int:
    config, root = _common(args)
    sources_path = root / "data/raw/sources.jsonl"
    sources = _load_sources(sources_path)
    report = dataset_validation_report(sources)
    annotated: dict[str, list[ExpandedRecord]] = {}
    annotation_errors: list[str] = []
    for split in ("train", "dev", "test"):
        path = root / f"data/annotated/{split}.jsonl"
        if not path.exists():
            continue
        with path.open(encoding="utf-8") as handle:
            annotated[split] = [
                ExpandedRecord.model_validate_json(line) for line in handle if line.strip()
            ]
        expected = config.data.split_source_counts[split] * config.data.tasks_per_source
        if len(annotated[split]) != expected:
            annotation_errors.append(
                f"{split}: expected {expected} annotated records, found {len(annotated[split])}"
            )
        counts: dict[str, int] = {}
        for record in annotated[split]:
            counts[record.source_id] = counts.get(record.source_id, 0) + 1
            if record.split != split:
                annotation_errors.append(f"{record.example_id}: wrong split")
        if any(count != 2 for count in counts.values()):
            annotation_errors.append(f"{split}: sources do not each have exactly two tasks")
    report["annotated_split_counts"] = {split: len(records) for split, records in annotated.items()}
    report["annotation_errors"] = annotation_errors
    report["valid"] = bool(report["valid"] and not annotation_errors)
    report["source_frozen"] = bool(args.freeze and report["valid"])
    report["annotations_frozen"] = bool(args.freeze and report["valid"] and len(annotated) == 3)
    write_json(root / "data/raw/validation_report.json", report)
    if args.freeze and report["valid"]:
        write_jsonl(
            root / "data/frozen/sources.jsonl",
            (source.model_dump(mode="json") for source in sources),
        )
        if len(annotated) == 3:
            for split, records in annotated.items():
                write_jsonl(
                    root / f"data/frozen/{split}.jsonl",
                    (record.model_dump(mode="json") for record in records),
                )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["valid"] else 1


def command_annotate(args: argparse.Namespace) -> int:
    config, root = _common(args)
    from tide.annotate import DeepSeekAnnotator, expand_source

    sources = _load_sources(root / "data/raw/sources.jsonl")
    if args.smoke_sources:
        sources = [source for source in sources if source.split == "train"][: args.smoke_sources]
    if args.dry_run:
        plan = {
            "source_requests_without_retries": len(sources),
            "task_records": len(sources) * 2,
            "request_cap": config.annotation.max_api_requests,
            "cost_cap_usd": config.annotation.max_estimated_usd,
            "writes_external_state": False,
        }
        print(json.dumps(plan, indent=2, sort_keys=True))
        return 0
    annotator = DeepSeekAnnotator(config, root)
    pairs = asyncio.run(annotator.annotate_many(sources))
    by_split: dict[str, list[Any]] = {"train": [], "dev": [], "test": []}
    for source, response in pairs:
        by_split[source.split].extend(expand_source(source, response, annotator.model))
    for split, records in by_split.items():
        if records:
            write_jsonl(
                root / f"data/annotated/{split}.jsonl",
                (record.model_dump(mode="json") for record in records),
            )
    annotator.write_usage()
    return 0


def command_train(args: argparse.Namespace) -> int:
    config, root = _common(args)
    from tide.train import train_variant

    result = train_variant(config, root, args.variant, resume=args.resume)
    print(json.dumps(result, indent=2, sort_keys=True, default=str))
    return 0


def command_debug(args: argparse.Namespace) -> int:
    config, root = _common(args)
    from tide.train import train_variant

    debug_steps = args.steps or config.training.debug_max_optimizer_steps
    results = []
    for variant in ("M1_FLAN_T5_LORA", "M2_FLAN_T5_MATRIX"):
        result = train_variant(
            config,
            root,
            variant,
            resume=args.resume,
            debug_examples=args.examples,
            max_steps=debug_steps,
        )
        results.append(result)
    passed = all(result["best"] and result["best"][0] >= 0.9 for result in results)
    report = {"passed": passed, "required_fact_joint": 0.9, "results": results}
    write_json(root / config.experiment.output_dir / "debug/debug_report.json", report)
    print(json.dumps(report, indent=2, sort_keys=True, default=str))
    return 0 if passed else 1


def command_evaluate(args: argparse.Namespace) -> int:
    config, root = _common(args)
    from tide.evaluate import evaluate_groups

    result = evaluate_groups(config, root, args.split, args.groups)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def command_report(args: argparse.Namespace) -> int:
    config, root = _common(args)
    from tide.report import build_report

    result = build_report(config, root)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def command_diagnose_memory(args: argparse.Namespace) -> int:
    config, root = _common(args)
    from tide.diagnose import diagnose_memory

    result = diagnose_memory(config, root, args.split, args.variant)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def command_run(args: argparse.Namespace) -> int:
    config, root = _common(args)
    status_path = root / config.experiment.output_dir / "status.json"
    completed: list[str] = []

    def mark(state: str, current: str | None = None, error: str | None = None) -> None:
        write_json(
            status_path,
            {
                "status": state,
                "completed_stages": completed,
                "current_stage": current,
                "error": error,
                "resume_command": "tide run --stage all --resume",
            },
        )

    stages: list[tuple[str, Any]] = [
        ("preflight", lambda: command_preflight(argparse.Namespace(config=args.config, deep=True))),
        ("build-data", lambda: command_build_data(argparse.Namespace(config=args.config))),
        (
            "annotate-smoke",
            lambda: command_annotate(
                argparse.Namespace(
                    config=args.config, smoke_sources=10, resume=args.resume, dry_run=False
                )
            ),
        ),
        (
            "annotate-full",
            lambda: command_annotate(
                argparse.Namespace(
                    config=args.config, smoke_sources=None, resume=True, dry_run=False
                )
            ),
        ),
        (
            "validate-freeze",
            lambda: command_validate_data(argparse.Namespace(config=args.config, freeze=True)),
        ),
        (
            "debug",
            lambda: command_debug(
                argparse.Namespace(config=args.config, examples=32, steps=None, resume=args.resume)
            ),
        ),
    ]
    for variant in config.experiment.mandatory_variants:
        stages.append(
            (
                f"train-{variant}",
                lambda variant=variant: command_train(
                    argparse.Namespace(config=args.config, variant=variant, resume=args.resume)
                ),
            )
        )
    evaluation_groups = ["RAW", "EMPTY", "ORACLE", *config.experiment.mandatory_variants]
    stages.extend(
        [
            (
                "evaluate-dev",
                lambda: command_evaluate(
                    argparse.Namespace(config=args.config, split="dev", groups=evaluation_groups)
                ),
            ),
            (
                "diagnose-memory",
                lambda: command_diagnose_memory(
                    argparse.Namespace(
                        config=args.config,
                        split="dev",
                        variant="M2_FLAN_T5_MATRIX",
                    )
                ),
            ),
            (
                "evaluate-test",
                lambda: command_evaluate(
                    argparse.Namespace(config=args.config, split="test", groups=evaluation_groups)
                ),
            ),
        ]
    )
    if args.include_optional:
        for variant in config.experiment.optional_variants:
            stages.append(
                (
                    f"train-{variant}",
                    lambda variant=variant: command_train(
                        argparse.Namespace(config=args.config, variant=variant, resume=args.resume)
                    ),
                )
            )
            stages.append(
                (
                    f"evaluate-test-{variant}",
                    lambda variant=variant: command_evaluate(
                        argparse.Namespace(config=args.config, split="test", groups=[variant])
                    ),
                )
            )
    stages.append(("report", lambda: command_report(argparse.Namespace(config=args.config))))
    try:
        for name, action in stages:
            mark("running", current=name)
            exit_code = action()
            if exit_code:
                raise RuntimeError(f"stage {name} exited with status {exit_code}")
            completed.append(name)
        mark("completed")
    except Exception as exc:
        mark("failed", current=stages[len(completed)][0], error=f"{type(exc).__name__}: {exc}")
        raise
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="tide", description=__doc__)
    subcommands = parser.add_subparsers(dest="command", required=True)

    def configured(name: str, help_text: str) -> argparse.ArgumentParser:
        command = subcommands.add_parser(name, help=help_text)
        command.add_argument("--config", type=Path, default=Path("configs/overnight.yaml"))
        return command

    preflight = configured("preflight", "Record local or GPU-server environment details")
    preflight.add_argument(
        "--deep",
        action="store_true",
        help="Require CUDA and run a BF16 forward/backward operator smoke test",
    )
    preflight.set_defaults(handler=command_preflight)
    build = configured("build-data", "Generate the controlled synthetic source ledger")
    build.set_defaults(handler=command_build_data)
    validate = configured("validate-data", "Validate source isolation and optionally freeze it")
    validate.add_argument("--freeze", action="store_true")
    validate.set_defaults(handler=command_validate_data)
    annotate = configured("annotate", "Create and validate DeepSeek reference rewrites")
    annotate.add_argument("--smoke-sources", type=int)
    annotate.add_argument(
        "--resume", action="store_true", help="Reuse content-addressed valid cache entries"
    )
    annotate.add_argument("--dry-run", action="store_true")
    annotate.set_defaults(handler=command_annotate)
    train = configured("train", "Train one FLAN-T5 experiment variant on the GPU server")
    train.add_argument(
        "--variant",
        required=True,
        choices=["M1_FLAN_T5_LORA", "M2_FLAN_T5_MATRIX", "M3_FLAN_T5_MATRIX_AUX"],
    )
    train.add_argument("--resume", action="store_true")
    train.set_defaults(handler=command_train)
    debug = configured("debug", "Overfit M1 and M2 on a 32-example implementation gate")
    debug.add_argument("--examples", type=int, default=32)
    debug.add_argument(
        "--steps",
        type=int,
        help="Override training.debug_max_optimizer_steps from the selected config",
    )
    debug.add_argument("--resume", action="store_true")
    debug.set_defaults(handler=command_debug)
    evaluate = configured("evaluate", "Run independent Qwen task evaluation")
    evaluate.add_argument("--split", choices=["dev", "test"], required=True)
    evaluate.add_argument("--groups", nargs="+", required=True)
    evaluate.set_defaults(handler=command_evaluate)
    diagnose = configured(
        "diagnose-memory", "Compare normal, zeroed, and within-task shuffled matrix memory"
    )
    diagnose.add_argument("--split", choices=["dev", "test"], default="dev")
    diagnose.add_argument(
        "--variant",
        choices=["M2_FLAN_T5_MATRIX", "M3_FLAN_T5_MATRIX_AUX"],
        default="M2_FLAN_T5_MATRIX",
    )
    diagnose.set_defaults(handler=command_diagnose_memory)
    report = configured("report", "Aggregate saved test results and bootstrap intervals")
    report.set_defaults(handler=command_report)
    run = configured("run", "Run the mandatory feasibility pipeline with durable status")
    run.add_argument("--stage", choices=["all"], default="all")
    run.add_argument("--resume", action="store_true")
    run.add_argument("--include-optional", action="store_true")
    run.set_defaults(handler=command_run)
    return parser


def main(argv: list[str] | None = None) -> None:
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    args = build_parser().parse_args(argv)
    raise SystemExit(args.handler(args))


if __name__ == "__main__":
    main()
