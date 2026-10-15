"""V3 long-budget M2 fitting diagnostic, isolated from the V2 pipeline."""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup

from tide.config import TideConfig
from tide.io import write_json, write_jsonl
from tide.metrics import FactParse, parse_rewrite
from tide.models import assert_trainable_scope
from tide.rewrite import load_checkpoint_model
from tide.train import (
    RewriteCollator,
    RewriteDataset,
    Variant,
    _model_forward,
    _parameter_groups,
    _records_digest,
    build_rewriter,
    generate_batch,
    load_records,
    save_checkpoint,
)

VARIANT: Variant = "M2_FLAN_T5_MATRIX"
AUDIT_SIZE = 20
FIT_THRESHOLD = 0.95


def assert_v3_protocol(v2: TideConfig, v3: TideConfig) -> dict[str, Any]:
    """Fail closed if V3 changes anything outside the approved training protocol."""
    old = v2.model_dump(mode="json")
    new = v3.model_dump(mode="json")
    allowed = {
        "experiment.output_dir",
        "experiment.mandatory_variants",
        "experiment.optional_variants",
        "training.max_optimizer_steps",
        "training.eval_every_steps",
        "training.save_every_steps",
    }
    changes: dict[str, dict[str, Any]] = {}

    def compare(left: Any, right: Any, prefix: str = "") -> None:
        if isinstance(left, dict) and isinstance(right, dict):
            for key in sorted(set(left) | set(right)):
                path = f"{prefix}.{key}" if prefix else key
                compare(left.get(key), right.get(key), path)
        elif left != right:
            changes[prefix] = {"v2": left, "v3": right}

    compare(old, new)
    unexpected = sorted(set(changes) - allowed)
    if unexpected:
        raise ValueError(f"V3 has unapproved protocol changes: {unexpected}")
    expected = {
        "training.max_optimizer_steps": (300, 1500),
        "training.eval_every_steps": (50, 100),
        "training.save_every_steps": (50, 100),
    }
    for path, pair in expected.items():
        actual = changes.get(path)
        if actual != {"v2": pair[0], "v3": pair[1]}:
            raise ValueError(f"required V3 change missing at {path}: {actual}")
    effective_batch = v3.training.micro_batch_size * v3.training.gradient_accumulation_steps
    if effective_batch != 16:
        raise ValueError(f"V3 effective batch must be 16, got {effective_batch}")
    return {
        "protocol_changes": changes,
        "effective_batch_size": effective_batch,
        "scheduler_note": "fresh linear schedule over 1500 steps; not continued from V2",
    }


def _audit_flags(text: str, parsed: FactParse) -> dict[str, bool]:
    """Conservative triage; rows retain raw text for mandatory human inspection."""
    type_cue = bool(
        re.search(
            r"\b(?:age|aged|years? old|hours? per week|weekly|experience|training hours?)\b",
            text,
            re.I,
        )
    )
    exact_facts_present = len(parsed.required_matches) == 2
    metric_failed = not parsed.fact_joint
    parser_error = (
        exact_facts_present
        and metric_failed
        and not (parsed.protected_matches or parsed.numeric_tamper or parsed.premature_conclusion)
    )
    number_or_unit_error = metric_failed and type_cue and not exact_facts_present
    required_fact_deleted = metric_failed and not type_cue and not exact_facts_present
    return {
        "required_fact_deleted": required_fact_deleted,
        "number_or_unit_error": number_or_unit_error,
        "parser_error": parser_error,
    }


def _row(record: Any, output: str, parsed: FactParse, *, step: int | str) -> dict[str, Any]:
    return {
        "step": step,
        "example_id": record.example_id,
        "source_id": record.source_id,
        "task_id": record.task_id,
        "context": record.context,
        "task": record.task,
        "reference_rewrite": record.reference_rewrite,
        "output": output,
        "required_fact_ids": record.required_fact_ids,
        "required_matches": list(parsed.required_matches),
        "protected_matches": list(parsed.protected_matches),
        "numeric_tamper": parsed.numeric_tamper,
        "parser_unknown": parsed.parser_unknown,
        "premature_conclusion": parsed.premature_conclusion,
        "fact_joint": parsed.fact_joint,
        **_audit_flags(output, parsed),
    }


@torch.no_grad()
def audit_v2_failures(v2: TideConfig, v3: TideConfig, root: Path) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("V2 audit requires the target CUDA server")
    device = torch.device("cuda")
    checkpoint = root / v2.experiment.output_dir / VARIANT / "best.pt"
    model, tokenizer = load_checkpoint_model(v2, root, VARIANT, checkpoint, device)
    records = load_records(root / "data/annotated/train.jsonl")
    loader = DataLoader(
        RewriteDataset(records),
        batch_size=v2.training.micro_batch_size,
        shuffle=False,
        num_workers=v2.training.dataloader_num_workers,
        collate_fn=RewriteCollator(tokenizer, v2, VARIANT),
    )
    rows: list[dict[str, Any]] = []
    for raw_batch in loader:
        batch = raw_batch.to(device)
        outputs = tokenizer.batch_decode(
            generate_batch(model, batch, VARIANT, v2), skip_special_tokens=True
        )
        for record, output in zip(batch.records, outputs, strict=True):
            parsed = parse_rewrite(
                output, record.facts, record.required_fact_ids, record.protected_fact_ids
            )
            if not parsed.fact_joint:
                rows.append(_row(record, output, parsed, step="v2_best"))
                if len(rows) == AUDIT_SIZE:
                    break
        if len(rows) == AUDIT_SIZE:
            break
    if len(rows) != AUDIT_SIZE:
        raise RuntimeError(f"expected {AUDIT_SIZE} V2 failures, found {len(rows)}")
    output_dir = root / v3.experiment.output_dir
    write_jsonl(output_dir / "fixed_20_v2_audit.jsonl", rows)
    summary = {
        "checkpoint": str(checkpoint),
        "fixed_example_ids": [row["example_id"] for row in rows],
        "counts": {
            key: sum(bool(row[key]) for row in rows)
            for key in ("required_fact_deleted", "number_or_unit_error", "parser_error")
        },
        "parser_revision_required": any(bool(row["parser_error"]) for row in rows),
        "review_status": "machine_triage_complete; inspect raw rows before training",
    }
    write_json(output_dir / "fixed_20_v2_audit_summary.json", summary)
    del model
    torch.cuda.empty_cache()
    return summary


@dataclass(frozen=True)
class SplitMetrics:
    required_fact_recall: float
    fact_joint: float
    text_loss: float
    sample_rows: list[dict[str, Any]]


@torch.no_grad()
def evaluate_split(
    model: Any,
    tokenizer: Any,
    config: TideConfig,
    records: list[Any],
    device: torch.device,
    fixed_ids: set[str],
    step: int,
) -> SplitMetrics:
    loader = DataLoader(
        RewriteDataset(records),
        batch_size=config.training.micro_batch_size,
        shuffle=False,
        num_workers=config.training.dataloader_num_workers,
        collate_fn=RewriteCollator(tokenizer, config, VARIANT),
    )
    model.eval()
    required = joint = total = 0
    weighted_loss = 0.0
    sample_rows: list[dict[str, Any]] = []
    for raw_batch in loader:
        batch = raw_batch.to(device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            forward = _model_forward(model, batch, VARIANT, config)
        batch_size = len(batch.records)
        weighted_loss += float(forward.text_loss) * batch_size
        outputs = tokenizer.batch_decode(
            generate_batch(model, batch, VARIANT, config), skip_special_tokens=True
        )
        for record, output in zip(batch.records, outputs, strict=True):
            parsed = parse_rewrite(
                output, record.facts, record.required_fact_ids, record.protected_fact_ids
            )
            required += len(parsed.required_matches)
            joint += int(parsed.fact_joint)
            total += 1
            if record.example_id in fixed_ids:
                sample_rows.append(_row(record, output, parsed, step=step))
    model.train()
    return SplitMetrics(
        required_fact_recall=required / max(2 * total, 1),
        fact_joint=joint / max(total, 1),
        text_loss=weighted_loss / max(total, 1),
        sample_rows=sample_rows,
    )


def train_v3(v2: TideConfig, config: TideConfig, root: Path) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("V3 training requires the target CUDA server")
    protocol = assert_v3_protocol(v2, config)
    output_dir = root / config.experiment.output_dir
    audit_path = output_dir / "fixed_20_v2_audit_summary.json"
    if not audit_path.exists():
        raise RuntimeError("run the fixed-20 V2 audit before V3 training")
    audit = json.loads(audit_path.read_text())
    if audit["parser_revision_required"]:
        raise RuntimeError("audit flagged a parser error; revise and recompute V2 before training")
    fixed_ids = set(audit["fixed_example_ids"])
    existing = list((output_dir / VARIANT).glob("*.pt"))
    if existing:
        raise RuntimeError("V3 output already has checkpoints; refusing to mix or resume a run")

    train_records = load_records(root / "data/annotated/train.jsonl")
    dev_records = load_records(root / "data/annotated/dev.jsonl")
    if (len(train_records), len(dev_records)) != (400, 100):
        raise RuntimeError("V3 requires the frozen 400-train/100-dev split")
    model, tokenizer, targets = build_rewriter(config, root, VARIANT)
    device = torch.device("cuda")
    model.to(device)
    collator = RewriteCollator(tokenizer, config, VARIANT)
    optimizer = AdamW(_parameter_groups(model, config))
    scheduler = get_linear_schedule_with_warmup(
        optimizer, config.training.warmup_steps, config.training.max_optimizer_steps
    )
    run_dir = output_dir / VARIANT
    scope = assert_trainable_scope(model)
    write_json(run_dir / "trainable_parameters.json", {**scope, "lora_targets": targets})
    write_json(output_dir / "protocol.json", protocol)
    data_sha256 = _records_digest(train_records)
    optimizer.zero_grad(set_to_none=True)
    global_step = epoch = start_batch = 0
    micro_step = 0
    logs: list[dict[str, Any]] = []
    curve: list[dict[str, Any]] = []
    evolution: list[dict[str, Any]] = []
    best: tuple[float, float, int] | None = None
    best_step: int | None = None
    consecutive_fit = 0
    stop_reason = "max_steps"
    started = time.perf_counter()

    while global_step < config.training.max_optimizer_steps:
        generator = torch.Generator().manual_seed(config.experiment.seed + epoch)
        loader = DataLoader(
            RewriteDataset(train_records),
            batch_size=config.training.micro_batch_size,
            shuffle=True,
            generator=generator,
            num_workers=config.training.dataloader_num_workers,
            collate_fn=collator,
        )
        for batch_index, raw_batch in enumerate(loader):
            start_batch = batch_index + 1
            batch = raw_batch.to(device)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                forward = _model_forward(model, batch, VARIANT, config)
                loss = forward.loss / config.training.gradient_accumulation_steps
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite loss at micro step {micro_step}")
            loss.backward()
            micro_step += 1
            if micro_step % config.training.gradient_accumulation_steps:
                continue
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.training.max_grad_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1
            if global_step % config.training.log_every_steps == 0:
                logs.append(
                    {
                        "step": global_step,
                        "loss": float(forward.loss.detach()),
                        "lr_matrix": scheduler.get_last_lr()[0],
                        "elapsed_seconds": time.perf_counter() - started,
                    }
                )
                write_json(run_dir / "training_log.json", {"events": logs})
            if global_step % config.training.eval_every_steps == 0:
                train_metrics = evaluate_split(
                    model, tokenizer, config, train_records, device, fixed_ids, global_step
                )
                dev_metrics = evaluate_split(
                    model, tokenizer, config, dev_records, device, set(), global_step
                )
                metric = {
                    "train_required_fact_recall": train_metrics.required_fact_recall,
                    "train_fact_joint": train_metrics.fact_joint,
                    "train_text_loss": train_metrics.text_loss,
                    "dev_required_fact_recall": dev_metrics.required_fact_recall,
                    "dev_fact_joint": dev_metrics.fact_joint,
                    "dev_text_loss": dev_metrics.text_loss,
                }
                event = {
                    "step": global_step,
                    **metric,
                    "elapsed_seconds": time.perf_counter() - started,
                }
                curve.append(event)
                evolution.extend(train_metrics.sample_rows)
                write_jsonl(output_dir / "learning_curve.jsonl", curve)
                write_jsonl(output_dir / "fixed_20_evolution.jsonl", evolution)
                save_checkpoint(
                    run_dir / f"step_{global_step:04d}.pt",
                    model,
                    optimizer,
                    scheduler,
                    global_step,
                    epoch,
                    start_batch,
                    config,
                    VARIANT,
                    data_sha256,
                    metric,
                )
                candidate = (dev_metrics.fact_joint, -dev_metrics.text_loss, -global_step)
                if best is None or candidate > best:
                    best, best_step = candidate, global_step
                    save_checkpoint(
                        run_dir / "best.pt",
                        model,
                        optimizer,
                        scheduler,
                        global_step,
                        epoch,
                        start_batch,
                        config,
                        VARIANT,
                        data_sha256,
                        metric,
                    )
                consecutive_fit = (
                    consecutive_fit + 1 if train_metrics.fact_joint >= FIT_THRESHOLD else 0
                )
                print(json.dumps(event, sort_keys=True), flush=True)
                if consecutive_fit >= 2:
                    stop_reason = "train_fact_joint_ge_0.95_twice"
                    break
            if global_step >= config.training.max_optimizer_steps:
                break
        if stop_reason != "max_steps" or global_step >= config.training.max_optimizer_steps:
            break
        epoch += 1
        start_batch = 0

    elapsed = time.perf_counter() - started
    final_metric = curve[-1] if curve else {}
    save_checkpoint(
        run_dir / "final.pt",
        model,
        optimizer,
        scheduler,
        global_step,
        epoch,
        start_batch,
        config,
        VARIANT,
        data_sha256,
        {key: float(value) for key, value in final_metric.items() if key != "step"},
    )
    summary = {
        "variant": VARIANT,
        "global_step": global_step,
        "stop_reason": stop_reason,
        "best_step": best_step,
        "best_selection": {
            "primary": "dev_fact_joint_max",
            "tie_break": "dev_text_loss_min",
        },
        "elapsed_seconds": elapsed,
        "final_metrics": final_metric,
        "protocol": protocol,
    }
    write_json(output_dir / "summary.json", summary)
    return summary
