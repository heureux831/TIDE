"""Memory-dependence diagnostics for the selected FLAN-T5 matrix model."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any, cast

import torch
from torch.utils.data import DataLoader

from tide.config import TideConfig
from tide.evaluate import QwenTaskEvaluator
from tide.io import write_json, write_jsonl
from tide.metrics import parse_rewrite, parse_task_answer
from tide.models import FlanT5TaskConditionedMatrixRewriter
from tide.rewrite import load_checkpoint_model
from tide.train import RewriteCollator, RewriteDataset, Variant, load_records


def _within_task_permutation(records: list[Any], device: torch.device) -> torch.Tensor:
    indices = list(range(len(records)))
    by_task: dict[str, list[int]] = defaultdict(list)
    for index, record in enumerate(records):
        by_task[record.task_id].append(index)
    for group in by_task.values():
        if len(group) > 1:
            rotated = group[1:] + group[:1]
            for target, source in zip(group, rotated, strict=True):
                indices[target] = source
    return torch.tensor(indices, device=device)


@torch.no_grad()
def diagnose_memory(
    config: TideConfig,
    root: Path,
    split: str,
    variant: Variant = "M2_FLAN_T5_MATRIX",
) -> dict[str, Any]:
    if variant == "M1_FLAN_T5_LORA":
        raise ValueError("memory diagnosis applies only to matrix variants")
    if not torch.cuda.is_available():
        raise RuntimeError("memory diagnosis requires the target CUDA server")
    records = load_records(root / f"data/annotated/{split}.jsonl")
    device = torch.device("cuda")
    checkpoint = root / config.experiment.output_dir / variant / "best.pt"
    raw_model, tokenizer = load_checkpoint_model(config, root, variant, checkpoint, device)
    model = cast(FlanT5TaskConditionedMatrixRewriter, raw_model)
    collator = RewriteCollator(tokenizer, config, variant)
    loader = DataLoader(
        RewriteDataset(records),
        batch_size=config.training.micro_batch_size,
        shuffle=False,
        collate_fn=collator,
    )
    evaluator = QwenTaskEvaluator(root, config, device)
    rows: list[dict[str, Any]] = []
    generation_kwargs = {
        "do_sample": False,
        "num_beams": 1,
        "max_new_tokens": config.generation.max_new_tokens,
    }
    for raw_batch in loader:
        batch = raw_batch.to(device)
        tensors = batch.tensors
        memory, _attention = model.encode_memory(
            tensors["input_ids"],
            tensors["attention_mask"],
            tensors["context_mask"],
            tensors["task_input_ids"],
            tensors["task_attention_mask"],
        )
        permutation = _within_task_permutation(batch.records, device)
        memories = {
            "normal": memory,
            "zero": torch.zeros_like(memory),
            "shuffled": memory.index_select(0, permutation),
        }
        for condition, condition_memory in memories.items():
            token_ids = model.generate_from_memory(condition_memory, **generation_kwargs)
            rewrites = tokenizer.batch_decode(token_ids, skip_special_tokens=True)
            raw_answers, _lengths, _latencies = evaluator.answer(
                [record.task for record in batch.records], rewrites
            )
            for record, rewrite_text, raw_answer in zip(
                batch.records, rewrites, raw_answers, strict=True
            ):
                parsed = parse_rewrite(
                    rewrite_text,
                    record.facts,
                    record.required_fact_ids,
                    record.protected_fact_ids,
                )
                answer = parse_task_answer(raw_answer)
                rows.append(
                    {
                        "example_id": record.example_id,
                        "source_id": record.source_id,
                        "task_id": record.task_id,
                        "condition": condition,
                        "rewrite": rewrite_text,
                        "required_matches": list(parsed.required_matches),
                        "task_answer": answer,
                        "gold_answer": record.gold_answer,
                    }
                )
    summary: dict[str, Any] = {}
    by_condition: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_condition[row["condition"]].append(row)
    for condition, items in by_condition.items():
        summary[condition] = {
            "required_fact_recall": sum(len(item["required_matches"]) for item in items)
            / (2 * len(items)),
            "task_acc": sum(item["task_answer"] == item["gold_answer"] for item in items)
            / len(items),
        }
    normal = {row["example_id"]: row["rewrite"] for row in by_condition["normal"]}
    for condition in ("zero", "shuffled"):
        summary[condition]["rewrite_change_rate_vs_normal"] = sum(
            row["rewrite"] != normal[row["example_id"]] for row in by_condition[condition]
        ) / len(by_condition[condition])
    output = root / config.experiment.output_dir / "diagnostics" / f"{split}_{variant}.jsonl"
    write_jsonl(output, rows)
    write_json(output.with_suffix(".summary.json"), summary)
    return summary
