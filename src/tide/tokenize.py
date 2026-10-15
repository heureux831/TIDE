"""Tokenization utilities that make the context-only attention boundary explicit."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

INPUT_PREFIX = (
    "Rewrite the context for the task. Keep only facts needed for the task "
    "and do not answer it.\nTask: "
)
CONTEXT_SEPARATOR = "\nContext: "


@dataclass(frozen=True)
class EncodedRewriteBatch:
    input_ids: Any
    attention_mask: Any
    context_mask: Any
    task_input_ids: Any
    task_attention_mask: Any
    labels: Any | None
    offset_mapping: Any
    context_char_starts: list[int]


def format_rewriter_input(context: str, task: str) -> tuple[str, int]:
    prefix = f"{INPUT_PREFIX}{task}{CONTEXT_SEPARATOR}"
    return prefix + context, len(prefix)


def encode_rewrite_batch(
    tokenizer: Any,
    contexts: list[str],
    tasks: list[str],
    references: list[str] | None,
    input_max_tokens: int,
    output_max_tokens: int,
    reject_truncation: bool = True,
) -> EncodedRewriteBatch:
    if len(contexts) != len(tasks):
        raise ValueError("contexts and tasks must have equal lengths")
    formatted = [
        format_rewriter_input(context, task) for context, task in zip(contexts, tasks, strict=True)
    ]
    texts = [item[0] for item in formatted]
    starts = [item[1] for item in formatted]
    encoded = tokenizer(
        texts,
        padding=True,
        truncation=False,
        return_tensors="pt",
        return_offsets_mapping=True,
        add_special_tokens=True,
    )
    if encoded["input_ids"].shape[1] > input_max_tokens:
        if reject_truncation:
            raise ValueError("rewriter input exceeds input_max_tokens; truncation is forbidden")
        encoded = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=input_max_tokens,
            return_tensors="pt",
            return_offsets_mapping=True,
            add_special_tokens=True,
        )
    offsets = encoded.pop("offset_mapping")
    context_mask = encoded["attention_mask"].new_zeros(encoded["attention_mask"].shape)
    for row, (offset_row, start) in enumerate(zip(offsets.tolist(), starts, strict=True)):
        for column, (token_start, token_end) in enumerate(offset_row):
            if token_end > token_start and token_end > start:
                context_mask[row, column] = 1
    if (context_mask.sum(dim=1) == 0).any():
        raise ValueError("at least one example has no context tokens")
    task_encoded = tokenizer(
        [f"Task: {task}" for task in tasks],
        padding=True,
        truncation=False,
        return_tensors="pt",
        add_special_tokens=True,
    )
    labels = None
    if references is not None:
        target = tokenizer(
            text_target=references,
            padding=True,
            truncation=False,
            return_tensors="pt",
            add_special_tokens=True,
        )
        if target["input_ids"].shape[1] > output_max_tokens:
            raise ValueError("reference output exceeds output_max_tokens; truncation is forbidden")
        labels = target["input_ids"]
        labels[labels == tokenizer.pad_token_id] = -100
    return EncodedRewriteBatch(
        input_ids=encoded["input_ids"],
        attention_mask=encoded["attention_mask"],
        context_mask=context_mask,
        task_input_ids=task_encoded["input_ids"],
        task_attention_mask=task_encoded["attention_mask"],
        labels=labels,
        offset_mapping=offsets,
        context_char_starts=starts,
    )


def fact_token_mask(
    offset_mapping: Any,
    context_char_starts: list[int],
    fact_spans: list[list[tuple[int, int]]],
) -> Any:
    """Map context-relative character spans to input-token masks."""
    mask = offset_mapping.new_zeros(offset_mapping.shape[:2])
    for row, spans in enumerate(fact_spans):
        shifted = [
            (start + context_char_starts[row], end + context_char_starts[row])
            for start, end in spans
        ]
        for column, (token_start, token_end) in enumerate(offset_mapping[row].tolist()):
            if token_end <= token_start:
                continue
            if any(token_start < end and token_end > start for start, end in shifted):
                mask[row, column] = 1
    return mask
