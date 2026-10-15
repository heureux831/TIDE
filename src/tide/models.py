"""FLAN-T5 rewriters used by the controlled feasibility experiment."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import torch
from torch import Tensor, nn
from transformers.modeling_outputs import BaseModelOutput


@dataclass
class RewriteForwardOutput:
    loss: Tensor
    text_loss: Tensor
    attention_weights: Tensor | None
    memory: Tensor


class TaskConditionedMatrixBottleneck(nn.Module):
    """Compress context states into task-conditioned gated query slots."""

    def __init__(
        self,
        hidden_size: int = 768,
        slots: int = 64,
        heads: int = 8,
        ffn_dim: int = 1536,
        gate_hidden_dim: int = 256,
        gate_bias_init: float = 2.0,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.slots = slots
        self.query_slots = nn.Parameter(torch.empty(slots, hidden_size))
        self.task_projection = nn.Linear(hidden_size, hidden_size)
        self.cross_attention = nn.MultiheadAttention(
            hidden_size, heads, dropout=0.0, batch_first=True
        )
        self.attention_norm = nn.LayerNorm(hidden_size)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, ffn_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(ffn_dim, hidden_size),
        )
        self.ffn_norm = nn.LayerNorm(hidden_size)
        self.gate = nn.Sequential(
            nn.Linear(hidden_size * 2, gate_hidden_dim),
            nn.GELU(),
            nn.Linear(gate_hidden_dim, hidden_size),
        )
        nn.init.normal_(self.query_slots, mean=0.0, std=0.02)
        nn.init.constant_(self.gate[-1].bias, gate_bias_init)

    def forward(
        self, context_states: Tensor, context_mask: Tensor, task_vector: Tensor
    ) -> tuple[Tensor, Tensor]:
        if context_states.ndim != 3 or context_states.shape[-1] != self.hidden_size:
            raise ValueError("unexpected context state shape")
        if (context_mask.sum(dim=1) == 0).any():
            raise ValueError("each example needs at least one valid context token")
        query = self.query_slots.unsqueeze(0) + self.task_projection(task_vector).unsqueeze(1)
        attended, weights = self.cross_attention(
            query,
            context_states,
            context_states,
            key_padding_mask=~context_mask.bool(),
            need_weights=True,
            average_attn_weights=False,
        )
        content0 = self.attention_norm(query + attended)
        content = self.ffn_norm(content0 + self.ffn(content0))
        expanded_task = task_vector.unsqueeze(1).expand(-1, self.slots, -1)
        gate = torch.sigmoid(self.gate(torch.cat((content, expanded_task), dim=-1)))
        return gate * content, weights


def decoder_lora_targets(model: nn.Module) -> list[str]:
    targets = [
        name
        for name, module in model.named_modules()
        if name.startswith("decoder.")
        and (name.endswith(".q") or name.endswith(".v"))
        and isinstance(module, nn.Linear)
    ]
    if len(targets) != 48:
        raise RuntimeError(f"expected 48 FLAN-T5-base decoder q/v targets, found {len(targets)}")
    return targets


def load_flan_t5_with_decoder_lora(
    model_name_or_path: str,
    revision: str,
    rank: int,
    alpha: int,
    dropout: float,
) -> tuple[nn.Module, list[str]]:
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import AutoModelForSeq2SeqLM

    base = AutoModelForSeq2SeqLM.from_pretrained(model_name_or_path, revision=revision)
    if base.config.d_model != 768:
        raise RuntimeError(
            f"this experiment requires FLAN-T5-base d_model=768, got {base.config.d_model}"
        )
    base.requires_grad_(False)
    targets = decoder_lora_targets(base)
    lora = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        r=rank,
        lora_alpha=alpha,
        lora_dropout=dropout,
        bias="none",
        target_modules=targets,
    )
    model = get_peft_model(base, lora)
    for name, parameter in model.named_parameters():
        if "lora_" not in name:
            parameter.requires_grad_(False)
    return model, targets


class _FlanT5RewriterBase(nn.Module):
    def __init__(self, decoder_model: nn.Module) -> None:
        super().__init__()
        self.decoder_model = decoder_model

    @property
    def encoder(self) -> nn.Module:
        return self.decoder_model.get_encoder()

    def train(self, mode: bool = True) -> _FlanT5RewriterBase:
        super().train(mode)
        self.encoder.eval()
        return self

    def _encode(self, input_ids: Tensor, attention_mask: Tensor) -> Tensor:
        self.encoder.eval()
        with torch.no_grad():
            return self.encoder(
                input_ids=input_ids, attention_mask=attention_mask, return_dict=True
            ).last_hidden_state

    def _decode(self, memory: Tensor, labels: Tensor) -> Any:
        memory_mask = torch.ones(memory.shape[:2], device=memory.device, dtype=torch.long)
        return self.decoder_model(
            encoder_outputs=BaseModelOutput(last_hidden_state=memory),
            attention_mask=memory_mask,
            labels=labels,
            use_cache=False,
            return_dict=True,
        )

    @torch.no_grad()
    def generate_from_memory(self, memory: Tensor, **generation_kwargs: Any) -> Tensor:
        memory_mask = torch.ones(memory.shape[:2], device=memory.device, dtype=torch.long)
        return self.decoder_model.generate(
            encoder_outputs=BaseModelOutput(last_hidden_state=memory),
            attention_mask=memory_mask,
            **generation_kwargs,
        )


class FlanT5LoRARewriter(_FlanT5RewriterBase):
    """M1 baseline: decoder LoRA reads frozen FLAN-T5 encoder states."""

    def forward(
        self, input_ids: Tensor, attention_mask: Tensor, labels: Tensor
    ) -> RewriteForwardOutput:
        memory = self._encode(input_ids, attention_mask)
        output = self._decode(memory, labels)
        return RewriteForwardOutput(output.loss, output.loss, None, memory)

    @torch.no_grad()
    def generate(self, input_ids: Tensor, attention_mask: Tensor, **kwargs: Any) -> Tensor:
        return self.generate_from_memory(self._encode(input_ids, attention_mask), **kwargs)


class FlanT5TaskConditionedMatrixRewriter(_FlanT5RewriterBase):
    """M2/M3: decoder reads only task-conditioned matrix memory Z."""

    def __init__(
        self, decoder_model: nn.Module, bottleneck: TaskConditionedMatrixBottleneck
    ) -> None:
        super().__init__(decoder_model)
        self.bottleneck = bottleneck

    def encode_memory(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
        context_mask: Tensor,
        task_input_ids: Tensor,
        task_attention_mask: Tensor,
    ) -> tuple[Tensor, Tensor]:
        full_states = self._encode(input_ids, attention_mask)
        task_states = self._encode(task_input_ids, task_attention_mask)
        weights = task_attention_mask.unsqueeze(-1).to(task_states.dtype)
        task_vector = (task_states * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1)
        return self.bottleneck(full_states, context_mask, task_vector)

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
        context_mask: Tensor,
        task_input_ids: Tensor,
        task_attention_mask: Tensor,
        labels: Tensor,
        variant: Literal["M2_FLAN_T5_MATRIX", "M3_FLAN_T5_MATRIX_AUX"] = "M2_FLAN_T5_MATRIX",
        required_fact_masks: Tensor | None = None,
        protected_fact_mask: Tensor | None = None,
        keep_weight: float = 0.05,
        hide_weight: float = 0.05,
    ) -> RewriteForwardOutput:
        memory, attention = self.encode_memory(
            input_ids, attention_mask, context_mask, task_input_ids, task_attention_mask
        )
        output = self._decode(memory, labels)
        loss = output.loss
        if variant == "M3_FLAN_T5_MATRIX_AUX":
            if required_fact_masks is None or protected_fact_mask is None:
                raise ValueError("M3 requires fact token masks")
            probability = attention.mean(dim=(1, 2))
            masses = (probability.unsqueeze(1) * required_fact_masks.to(probability.dtype)).sum(
                dim=-1
            )
            keep_loss = -torch.log(masses + 1e-6).mean()
            hide_loss = (probability * protected_fact_mask.to(probability.dtype)).sum(dim=-1).mean()
            loss = loss + keep_weight * keep_loss + hide_weight * hide_loss
        return RewriteForwardOutput(loss, output.loss, attention, memory)

    @torch.no_grad()
    def generate(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
        context_mask: Tensor,
        task_input_ids: Tensor,
        task_attention_mask: Tensor,
        **kwargs: Any,
    ) -> Tensor:
        memory, _ = self.encode_memory(
            input_ids, attention_mask, context_mask, task_input_ids, task_attention_mask
        )
        return self.generate_from_memory(memory, **kwargs)


def assert_trainable_scope(model: nn.Module) -> dict[str, int | list[str]]:
    names = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
    bad = [name for name in names if "lora_" not in name and "bottleneck." not in name]
    if bad:
        raise RuntimeError(f"unexpected trainable parameters: {bad}")
    if any("encoder" in name and "lora_" in name for name in names):
        raise RuntimeError("encoder LoRA parameters must not be trainable")
    count = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    return {"trainable_parameter_count": count, "trainable_parameter_names": names}
