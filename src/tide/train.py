"""Training and checkpointing for the three named FLAN-T5 variants."""

from __future__ import annotations

import hashlib
import json
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np
import torch
from torch import Tensor, nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from tide.config import TideConfig
from tide.io import write_json
from tide.metrics import parse_rewrite
from tide.models import (
    FlanT5LoRARewriter,
    FlanT5TaskConditionedMatrixRewriter,
    TaskConditionedMatrixBottleneck,
    assert_trainable_scope,
    load_flan_t5_with_decoder_lora,
)
from tide.schemas import ExpandedRecord
from tide.tokenize import encode_rewrite_batch, fact_token_mask

Variant = Literal["M1_FLAN_T5_LORA", "M2_FLAN_T5_MATRIX", "M3_FLAN_T5_MATRIX_AUX"]


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_model_source(root: Path, config: TideConfig) -> tuple[str, str]:
    local = root / "models/flan-t5"
    if local.exists() and any(local.iterdir()):
        manifest = local / "tide_download_manifest.json"
        revision = "main"
        if manifest.exists():
            revision = json.loads(manifest.read_text())["resolved_revision"]
        return str(local), revision
    return os.getenv(config.model.backbone_env, config.model.backbone), os.getenv(
        config.model.revision_env, config.model.revision
    )


def load_records(path: Path) -> list[ExpandedRecord]:
    with path.open(encoding="utf-8") as handle:
        return [ExpandedRecord.model_validate_json(line) for line in handle if line.strip()]


class RewriteDataset(Dataset[ExpandedRecord]):
    def __init__(self, records: list[ExpandedRecord]) -> None:
        self.records = records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> ExpandedRecord:
        return self.records[index]


@dataclass
class RewriteBatch:
    tensors: dict[str, Tensor]
    records: list[ExpandedRecord]

    def to(self, device: torch.device) -> RewriteBatch:
        return RewriteBatch(
            {key: value.to(device) for key, value in self.tensors.items()}, self.records
        )


class RewriteCollator:
    def __init__(self, tokenizer: Any, config: TideConfig, variant: Variant) -> None:
        self.tokenizer = tokenizer
        self.config = config
        self.variant = variant

    def __call__(self, records: list[ExpandedRecord]) -> RewriteBatch:
        encoded = encode_rewrite_batch(
            self.tokenizer,
            [record.context for record in records],
            [record.task for record in records],
            [record.reference_rewrite for record in records],
            self.config.data.input_max_tokens,
            self.config.data.output_max_tokens,
            self.config.data.reject_truncation,
        )
        if (
            self.variant != "M1_FLAN_T5_LORA"
            and (encoded.context_mask.sum(dim=1) <= self.config.model.bottleneck_slots).any()
        ):
            raise ValueError("matrix examples must have more context tokens than bottleneck slots")
        tensors = {
            "input_ids": encoded.input_ids,
            "attention_mask": encoded.attention_mask,
            "context_mask": encoded.context_mask,
            "task_input_ids": encoded.task_input_ids,
            "task_attention_mask": encoded.task_attention_mask,
            "labels": cast(Tensor, encoded.labels),
        }
        if self.variant == "M3_FLAN_T5_MATRIX_AUX":
            required_masks = []
            protected_masks = []
            for record in records:
                by_id = {fact.fact_id: fact for fact in record.facts}
                required_masks.append(
                    [
                        (by_id[fact_id].char_start, by_id[fact_id].char_end)
                        for fact_id in record.required_fact_ids
                    ]
                )
                protected_masks.append(
                    [
                        (by_id[fact_id].char_start, by_id[fact_id].char_end)
                        for fact_id in record.protected_fact_ids
                    ]
                )
            first = fact_token_mask(
                encoded.offset_mapping,
                encoded.context_char_starts,
                [[spans[0]] for spans in required_masks],
            )
            second = fact_token_mask(
                encoded.offset_mapping,
                encoded.context_char_starts,
                [[spans[1]] for spans in required_masks],
            )
            tensors["required_fact_masks"] = torch.stack((first, second), dim=1)
            tensors["protected_fact_mask"] = fact_token_mask(
                encoded.offset_mapping, encoded.context_char_starts, protected_masks
            )
        return RewriteBatch(tensors, records)


def build_rewriter(
    config: TideConfig, root: Path, variant: Variant
) -> tuple[nn.Module, Any, list[str]]:
    seed_everything(config.experiment.seed)
    source, revision = resolve_model_source(root, config)
    tokenizer = AutoTokenizer.from_pretrained(source, revision=revision, use_fast=True)
    if not tokenizer.is_fast:
        raise RuntimeError("a fast tokenizer with offset mappings is required")
    decoder, targets = load_flan_t5_with_decoder_lora(
        source,
        revision,
        config.model.lora_rank,
        config.model.lora_alpha,
        config.model.lora_dropout,
    )
    if variant == "M1_FLAN_T5_LORA":
        model: nn.Module = FlanT5LoRARewriter(decoder)
    else:
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(config.experiment.matrix_init_seed)
            bottleneck = TaskConditionedMatrixBottleneck(
                hidden_size=decoder.get_base_model().config.d_model,
                slots=config.model.bottleneck_slots,
                heads=config.model.bottleneck_heads,
                ffn_dim=config.model.bottleneck_ffn_dim,
                gate_hidden_dim=config.model.gate_hidden_dim,
                gate_bias_init=config.model.gate_bias_init,
            )
        model = FlanT5TaskConditionedMatrixRewriter(decoder, bottleneck)
    assert_trainable_scope(model)
    return model, tokenizer, targets


def _parameter_groups(model: nn.Module, config: TideConfig) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, bool], list[nn.Parameter]] = {}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        family = "matrix" if "bottleneck." in name else "lora"
        no_decay = name.endswith(".bias") or "norm" in name.casefold()
        grouped.setdefault((family, no_decay), []).append(parameter)
    return [
        {
            "params": parameters,
            "lr": config.training.learning_rate_matrix
            if family == "matrix"
            else config.training.learning_rate_lora,
            "weight_decay": 0.0 if no_decay else config.training.weight_decay,
        }
        for (family, no_decay), parameters in grouped.items()
    ]


def _model_forward(
    model: nn.Module, batch: RewriteBatch, variant: Variant, config: TideConfig
) -> Any:
    values = batch.tensors
    if variant == "M1_FLAN_T5_LORA":
        return model(values["input_ids"], values["attention_mask"], values["labels"])
    return model(
        values["input_ids"],
        values["attention_mask"],
        values["context_mask"],
        values["task_input_ids"],
        values["task_attention_mask"],
        values["labels"],
        variant=variant,
        required_fact_masks=values.get("required_fact_masks"),
        protected_fact_mask=values.get("protected_fact_mask"),
        keep_weight=config.loss.m3_keep_weight,
        hide_weight=config.loss.m3_hide_weight,
    )


@torch.no_grad()
def generate_batch(
    model: nn.Module, batch: RewriteBatch, variant: Variant, config: TideConfig
) -> Tensor:
    values = batch.tensors
    kwargs = {
        "do_sample": config.generation.do_sample,
        "num_beams": config.generation.num_beams,
        "max_new_tokens": config.generation.max_new_tokens,
    }
    if variant == "M1_FLAN_T5_LORA":
        return model.generate(values["input_ids"], values["attention_mask"], **kwargs)
    return model.generate(
        values["input_ids"],
        values["attention_mask"],
        values["context_mask"],
        values["task_input_ids"],
        values["task_attention_mask"],
        **kwargs,
    )


@torch.no_grad()
def dev_fact_joint(
    model: nn.Module,
    loader: DataLoader[Any],
    tokenizer: Any,
    variant: Variant,
    config: TideConfig,
    device: torch.device,
) -> tuple[float, float]:
    model.eval()
    joint = 0
    total = 0
    losses: list[float] = []
    for raw_batch in loader:
        batch = raw_batch.to(device)
        with torch.autocast(
            device_type="cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"
        ):
            output = _model_forward(model, batch, variant, config)
        losses.append(float(output.text_loss))
        decoded = tokenizer.batch_decode(
            generate_batch(model, batch, variant, config), skip_special_tokens=True
        )
        for text, record in zip(decoded, batch.records, strict=True):
            joint += int(
                parse_rewrite(
                    text, record.facts, record.required_fact_ids, record.protected_fact_ids
                ).fact_joint
            )
            total += 1
    model.train()
    return joint / max(total, 1), sum(losses) / max(len(losses), 1)


def _rng_state() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: AdamW,
    scheduler: Any,
    step: int,
    epoch: int,
    batches_into_epoch: int,
    config: TideConfig,
    variant: Variant,
    data_sha256: str,
    metric: dict[str, float],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    trainable = {
        name: parameter.detach().cpu()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    torch.save(
        {
            "format_version": 1,
            "variant": variant,
            "trainable_state": trainable,
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "global_step": step,
            "epoch": epoch,
            "batches_into_epoch": batches_into_epoch,
            "rng_state": _rng_state(),
            "config": config.model_dump(mode="json"),
            "data_sha256": data_sha256,
            "selection_metric": metric,
        },
        path,
    )


def _records_digest(records: list[ExpandedRecord]) -> str:
    payload = "\n".join(record.model_dump_json() for record in records).encode()
    return hashlib.sha256(payload).hexdigest()


def _restore_rng(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if torch.cuda.is_available() and state["cuda"] is not None:
        torch.cuda.set_rng_state_all(state["cuda"])


def _load_resume_checkpoint(
    output_dir: Path,
    model: nn.Module,
    optimizer: AdamW,
    scheduler: Any,
    variant: Variant,
    data_sha256: str,
    device: torch.device,
) -> dict[str, Any] | None:
    candidates = sorted(output_dir.glob("step_*.pt"))
    if (output_dir / "final.pt").exists():
        candidates.append(output_dir / "final.pt")
    if not candidates:
        return None
    payloads = [torch.load(path, map_location=device, weights_only=False) for path in candidates]
    payload = max(payloads, key=lambda item: int(item["global_step"]))
    if payload["variant"] != variant or payload["data_sha256"] != data_sha256:
        raise RuntimeError("resume checkpoint variant or data hash does not match")
    named = dict(model.named_parameters())
    if set(payload["trainable_state"]) != {
        name for name, parameter in named.items() if parameter.requires_grad
    }:
        raise RuntimeError("resume checkpoint trainable parameter set does not match")
    with torch.no_grad():
        for name, value in payload["trainable_state"].items():
            named[name].copy_(value)
    optimizer.load_state_dict(payload["optimizer"])
    scheduler.load_state_dict(payload["scheduler"])
    _restore_rng(payload["rng_state"])
    return payload


def train_variant(
    config: TideConfig,
    root: Path,
    variant: Variant,
    *,
    resume: bool = False,
    debug_examples: int | None = None,
    max_steps: int | None = None,
) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError(
            "formal FLAN-T5 training requires a CUDA GPU; run this command on the target server"
        )
    train_path = root / "data/annotated/train.jsonl"
    dev_path = root / "data/annotated/dev.jsonl"
    train_records = load_records(train_path)
    dev_records = load_records(dev_path)
    if debug_examples is not None:
        train_records = train_records[:debug_examples]
        dev_records = train_records
    model, tokenizer, targets = build_rewriter(config, root, variant)
    device = torch.device("cuda")
    model.to(device)
    collator = RewriteCollator(tokenizer, config, variant)
    dev_loader = DataLoader(
        RewriteDataset(dev_records),
        batch_size=config.training.micro_batch_size,
        shuffle=False,
        num_workers=config.training.dataloader_num_workers,
        collate_fn=collator,
    )
    optimizer = AdamW(_parameter_groups(model, config))
    target_steps = max_steps or config.training.max_optimizer_steps
    scheduler = get_linear_schedule_with_warmup(
        optimizer, config.training.warmup_steps, target_steps
    )
    run_name = f"debug/{variant}" if debug_examples is not None else variant
    output_dir = root / config.experiment.output_dir / run_name
    scope = assert_trainable_scope(model)
    write_json(output_dir / "trainable_parameters.json", {**scope, "lora_targets": targets})
    optimizer.zero_grad(set_to_none=True)
    best: tuple[float, float, int] | None = None
    logs: list[dict[str, Any]] = []
    data_sha256 = _records_digest(train_records)
    global_step = 0
    epoch = 0
    start_batch = 0
    if resume:
        payload = _load_resume_checkpoint(
            output_dir, model, optimizer, scheduler, variant, data_sha256, device
        )
        if payload:
            global_step = int(payload["global_step"])
            epoch = int(payload["epoch"])
            start_batch = int(payload["batches_into_epoch"])
    if (output_dir / "best.pt").exists():
        best_payload = torch.load(output_dir / "best.pt", map_location="cpu", weights_only=False)
        metric = best_payload["selection_metric"]
        best = (
            float(metric["dev_fact_joint"]),
            -float(metric["dev_text_loss"]),
            -int(best_payload["global_step"]),
        )
    micro_step = global_step * config.training.gradient_accumulation_steps
    last_batch_index = start_batch
    while global_step < target_steps:
        generator = torch.Generator().manual_seed(config.experiment.seed + epoch)
        train_loader = DataLoader(
            RewriteDataset(train_records),
            batch_size=config.training.micro_batch_size,
            shuffle=True,
            generator=generator,
            num_workers=config.training.dataloader_num_workers,
            collate_fn=collator,
        )
        for batch_index, raw_batch in enumerate(train_loader):
            if batch_index < start_batch:
                continue
            last_batch_index = batch_index + 1
            batch = raw_batch.to(device)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                output = _model_forward(model, batch, variant, config)
                loss = output.loss / config.training.gradient_accumulation_steps
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
                        "loss": float(output.loss.detach()),
                        "max_memory_allocated": torch.cuda.max_memory_allocated(),
                        "max_memory_reserved": torch.cuda.max_memory_reserved(),
                    }
                )
                write_json(output_dir / "training_log.json", {"events": logs})
            should_evaluate = (
                global_step % config.training.eval_every_steps == 0 or global_step == target_steps
            )
            if should_evaluate:
                fact_joint, text_loss = dev_fact_joint(
                    model, dev_loader, tokenizer, variant, config, device
                )
                metric = {"dev_fact_joint": fact_joint, "dev_text_loss": text_loss}
                save_checkpoint(
                    output_dir / f"step_{global_step:04d}.pt",
                    model,
                    optimizer,
                    scheduler,
                    global_step,
                    epoch,
                    last_batch_index,
                    config,
                    variant,
                    data_sha256,
                    metric,
                )
                candidate = (fact_joint, -text_loss, -global_step)
                if best is None or candidate > best:
                    best = candidate
                    save_checkpoint(
                        output_dir / "best.pt",
                        model,
                        optimizer,
                        scheduler,
                        global_step,
                        epoch,
                        last_batch_index,
                        config,
                        variant,
                        data_sha256,
                        metric,
                    )
            if global_step >= target_steps:
                break
        if global_step < target_steps:
            epoch += 1
            start_batch = 0
    save_checkpoint(
        output_dir / "final.pt",
        model,
        optimizer,
        scheduler,
        global_step,
        epoch,
        last_batch_index,
        config,
        variant,
        data_sha256,
        {"final": 1.0},
    )
    return {
        "variant": variant,
        "global_step": global_step,
        "best": best,
        "output_dir": str(output_dir),
    }
