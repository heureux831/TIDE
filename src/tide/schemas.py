"""Versioned, strict schemas shared by data, annotation, and evaluation."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

FACT_IDS = (
    "f_name",
    "f_address",
    "f_email",
    "f_employer",
    "f_age",
    "f_available",
    "f_experience",
    "f_training",
)


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Fact(StrictModel):
    fact_id: str
    field: str
    value: str | int
    surface: str
    char_start: int = Field(ge=0)
    char_end: int = Field(gt=0)
    aliases: list[str]


class SourceTask(StrictModel):
    task_id: Literal["A", "B"]
    task: str
    rule: dict[str, int]
    gold_answer: Literal["YES", "NO"]


class SourceRecord(StrictModel):
    schema_version: Literal["1.0"] = "1.0"
    source_id: str
    split: Literal["train", "dev", "test"]
    template_id: str
    context: str
    facts: list[Fact]
    tasks: list[SourceTask]

    @model_validator(mode="after")
    def validate_ledger(self) -> SourceRecord:
        ids = [fact.fact_id for fact in self.facts]
        if tuple(ids) != FACT_IDS:
            raise ValueError(f"fact ids must be exactly {FACT_IDS}")
        for fact in self.facts:
            if self.context[fact.char_start : fact.char_end] != fact.surface:
                raise ValueError(f"invalid span for {fact.fact_id}")
        if {task.task_id for task in self.tasks} != {"A", "B"}:
            raise ValueError("each source must contain tasks A and B")
        return self


class TaskAnnotation(StrictModel):
    task_id: Literal["A", "B"]
    required_fact_ids: list[str]
    protected_fact_ids: list[str]
    rewritten_context: str
    answer: Literal["YES", "NO", "UNKNOWN"]

    @model_validator(mode="after")
    def fact_partition(self) -> TaskAnnotation:
        required = set(self.required_fact_ids)
        protected = set(self.protected_fact_ids)
        if required & protected or required | protected != set(FACT_IDS):
            raise ValueError("required/protected facts must partition the ledger")
        return self


class AnnotationResponse(StrictModel):
    source_id: str
    annotations: list[TaskAnnotation]

    @model_validator(mode="after")
    def tasks_once(self) -> AnnotationResponse:
        if sorted(item.task_id for item in self.annotations) != ["A", "B"]:
            raise ValueError("annotations must contain task A and B exactly once")
        return self


class ExpandedRecord(StrictModel):
    schema_version: Literal["1.0"] = "1.0"
    source_id: str
    example_id: str
    split: Literal["train", "dev", "test"]
    template_id: str
    context: str
    facts: list[Fact]
    task_id: Literal["A", "B"]
    task: str
    rule: dict[str, int]
    required_fact_ids: list[str]
    protected_fact_ids: list[str]
    reference_rewrite: str
    gold_answer: Literal["YES", "NO"]
    annotation: dict[str, Any]
