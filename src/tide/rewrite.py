"""Checkpoint loading and the public rewrite(context, task) inference contract."""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import torch

from tide.config import TideConfig
from tide.models import FlanT5LoRARewriter, FlanT5TaskConditionedMatrixRewriter
from tide.tokenize import encode_rewrite_batch
from tide.train import Variant, build_rewriter


def load_checkpoint_model(
    config: TideConfig, root: Path, variant: Variant, checkpoint: Path, device: torch.device
) -> tuple[Any, Any]:
    model, tokenizer, _ = build_rewriter(config, root, variant)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if payload["variant"] != variant:
        raise ValueError(f"checkpoint variant {payload['variant']} does not match {variant}")
    named = dict(model.named_parameters())
    missing = sorted(set(payload["trainable_state"]) - set(named))
    if missing:
        raise RuntimeError(f"checkpoint contains unknown trainable parameters: {missing}")
    with torch.no_grad():
        for name, value in payload["trainable_state"].items():
            named[name].copy_(value)
    model.to(device).eval()
    return model, tokenizer


@torch.no_grad()
def rewrite(
    model: FlanT5LoRARewriter | FlanT5TaskConditionedMatrixRewriter,
    tokenizer: Any,
    config: TideConfig,
    variant: Variant,
    context: str,
    task: str,
) -> str:
    device = next(model.parameters()).device
    encoded = encode_rewrite_batch(
        tokenizer,
        [context],
        [task],
        None,
        config.data.input_max_tokens,
        config.data.output_max_tokens,
        config.data.reject_truncation,
    )
    tensors = {
        name: value.to(device)
        for name, value in {
            "input_ids": encoded.input_ids,
            "attention_mask": encoded.attention_mask,
            "context_mask": encoded.context_mask,
            "task_input_ids": encoded.task_input_ids,
            "task_attention_mask": encoded.task_attention_mask,
        }.items()
    }
    kwargs = {
        "do_sample": config.generation.do_sample,
        "num_beams": config.generation.num_beams,
        "max_new_tokens": config.generation.max_new_tokens,
    }
    if variant == "M1_FLAN_T5_LORA":
        tokens = cast(FlanT5LoRARewriter, model).generate(
            tensors["input_ids"], tensors["attention_mask"], **kwargs
        )
    else:
        tokens = cast(FlanT5TaskConditionedMatrixRewriter, model).generate(
            tensors["input_ids"],
            tensors["attention_mask"],
            tensors["context_mask"],
            tensors["task_input_ids"],
            tensors["task_attention_mask"],
            **kwargs,
        )
    return tokenizer.decode(tokens[0], skip_special_tokens=True).strip()
