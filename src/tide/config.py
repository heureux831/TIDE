"""Strict experiment configuration models."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml  # type: ignore[import-untyped]
from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ExperimentConfig(StrictModel):
    seed: int
    matrix_init_seed: int
    output_dir: Path
    mandatory_variants: list[str]
    optional_variants: list[str]


class DataConfig(StrictModel):
    source_count: int = Field(gt=0)
    split_source_counts: dict[Literal["train", "dev", "test"], int]
    tasks_per_source: Literal[2]
    input_max_tokens: int = Field(gt=0)
    output_max_tokens: int = Field(gt=0)
    reject_truncation: bool

    @model_validator(mode="after")
    def counts_match(self) -> DataConfig:
        if sum(self.split_source_counts.values()) != self.source_count:
            raise ValueError("split_source_counts must sum to source_count")
        return self


class AnnotationConfig(StrictModel):
    model_env: str
    default_model: str
    concurrency: int = Field(gt=0)
    timeout_seconds: int = Field(gt=0)
    max_output_tokens: int = Field(gt=0)
    max_transport_attempts: int = Field(gt=0)
    max_semantic_repairs: int = Field(ge=0)
    max_api_requests: int = Field(gt=0)
    max_estimated_usd: float = Field(gt=0)


class ModelConfig(StrictModel):
    backbone_env: str
    backbone: str
    revision_env: str
    revision: str
    precision: Literal["bf16_autocast"]
    freeze_encoder: bool
    freeze_shared_embeddings: bool
    freeze_lm_head: bool
    lora_rank: int
    lora_alpha: int
    lora_dropout: float
    bottleneck_slots: int
    bottleneck_heads: int
    bottleneck_layers: Literal[1]
    bottleneck_ffn_dim: int
    gate_hidden_dim: int
    gate_bias_init: float


class TrainingConfig(StrictModel):
    micro_batch_size: int
    gradient_accumulation_steps: int
    max_optimizer_steps: int
    learning_rate_lora: float
    learning_rate_matrix: float
    optimizer: Literal["adamw"]
    weight_decay: float
    warmup_steps: int
    scheduler: Literal["linear"]
    max_grad_norm: float
    log_every_steps: int
    eval_every_steps: int
    save_every_steps: int
    gradient_checkpointing: bool
    dataloader_num_workers: int


class LossConfig(StrictModel):
    text_weight: float
    m3_keep_weight: float
    m3_hide_weight: float


class GenerationConfig(StrictModel):
    do_sample: bool
    num_beams: int
    max_new_tokens: int


class EvaluationConfig(StrictModel):
    task_model_env: str
    task_model: str
    task_revision_env: str
    task_revision: str
    task_dtype: Literal["bfloat16"]
    task_batch_size: int
    task_input_max_tokens: int
    task_max_new_tokens: int
    task_enable_thinking: bool
    bootstrap_replicates: int
    bootstrap_seed: int


class TideConfig(StrictModel):
    experiment: ExperimentConfig
    data: DataConfig
    annotation: AnnotationConfig
    model: ModelConfig
    training: TrainingConfig
    loss: LossConfig
    generation: GenerationConfig
    evaluation: EvaluationConfig


def load_config(path: Path) -> TideConfig:
    with path.open(encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    return TideConfig.model_validate(raw)
