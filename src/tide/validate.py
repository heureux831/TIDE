"""Deterministic validation for sources and DeepSeek annotations."""

from __future__ import annotations

import re
from collections import Counter
from typing import Any

from tide.schemas import FACT_IDS, AnnotationResponse, SourceRecord, TaskAnnotation

REQUIRED = {"A": ["f_age", "f_available"], "B": ["f_experience", "f_training"]}
CONCLUSION = re.compile(
    r"\b(yes|no|qualif(?:y|ies|ied)|eligible|ineligible|passes?|fails?)\b", re.I
)
SENTENCE = re.compile(r"[.!?]+(?:\s|$)")
NUMBER = re.compile(r"(?<![\w.])-?\d+(?:\.\d+)?(?![\w.])")


def protected_for(task_id: str) -> list[str]:
    return [fact_id for fact_id in FACT_IDS if fact_id not in REQUIRED[task_id]]


def validate_source(source: SourceRecord) -> list[str]:
    errors: list[str] = []
    words = len(source.context.split())
    if not 90 <= words <= 160:
        errors.append("context_word_count")
    if len(source.tasks) != 2:
        errors.append("task_count")
    if len({fact.surface for fact in source.facts}) != len(source.facts):
        errors.append("duplicate_fact_surface")
    for fact in source.facts:
        if source.context.count(fact.surface) != 1:
            errors.append(f"surface_occurrence:{fact.fact_id}")
    return errors


def _semantic_surface(fact_id: str, value: str | int) -> re.Pattern[str]:
    escaped = re.escape(str(value))
    patterns = {
        "f_age": rf"\b{escaped}\s+years?\s+old\b",
        "f_available": rf"\b{escaped}\s+hours?\s+per\s+week\b",
        "f_experience": rf"\b{escaped}\s+years?\s+(?:of\s+)?relevant\s+experience\b",
        "f_training": (
            rf"\b{escaped}\s+(?:completed\s+)?training\s+hours?\b|"
            rf"\bcompleted\s+{escaped}\s+training\s+hours?\b"
        ),
    }
    return re.compile(patterns[fact_id], re.I)


def annotation_error_codes(source: SourceRecord, annotation: TaskAnnotation) -> list[str]:
    errors: list[str] = []
    task = next(item for item in source.tasks if item.task_id == annotation.task_id)
    required = REQUIRED[annotation.task_id]
    protected = protected_for(annotation.task_id)
    if annotation.required_fact_ids != required:
        errors.append("required_fact_ids")
    if annotation.protected_fact_ids != protected:
        errors.append("protected_fact_ids")
    if annotation.answer != task.gold_answer:
        errors.append("answer")
    text = annotation.rewritten_context.strip()
    sentence_count = len(SENTENCE.findall(text))
    if not text or not 1 <= sentence_count <= 2:
        errors.append("sentence_count")
    by_id = {fact.fact_id: fact for fact in source.facts}
    for fact_id in required:
        if not _semantic_surface(fact_id, by_id[fact_id].value).search(text):
            errors.append(f"missing_required:{fact_id}")
    folded = text.casefold()
    for fact_id in protected:
        fact = by_id[fact_id]
        if fact_id in {"f_name", "f_address", "f_email", "f_employer"}:
            aliases = [fact.surface, *fact.aliases]
            if fact_id == "f_name":
                aliases.extend(str(fact.value).split())
            if any(alias.casefold() in folded for alias in aliases if len(alias.strip()) >= 3):
                errors.append(f"protected_disclosure:{fact_id}")
        elif _semantic_surface(fact_id, fact.value).search(text):
            errors.append(f"protected_disclosure:{fact_id}")
    if CONCLUSION.search(text):
        errors.append("eligibility_conclusion")
    allowed_numbers = {str(by_id[fact_id].value) for fact_id in required}
    if any(number not in allowed_numbers for number in NUMBER.findall(text)):
        errors.append("new_numeric_fact")
    return sorted(set(errors))


def validate_annotation_response(source: SourceRecord, response: AnnotationResponse) -> list[str]:
    if source.source_id != response.source_id:
        return ["source_id"]
    errors: list[str] = []
    for annotation in response.annotations:
        errors.extend(
            f"{annotation.task_id}:{code}" for code in annotation_error_codes(source, annotation)
        )
    return sorted(set(errors))


def dataset_validation_report(sources: list[SourceRecord]) -> dict[str, Any]:
    errors = {
        source.source_id: source_errors
        for source in sources
        if (source_errors := validate_source(source))
    }
    split_counts = Counter(source.split for source in sources)
    templates: dict[str, set[str]] = {split: set() for split in ("train", "dev", "test")}
    identities: dict[str, list[str]] = {key: [] for key in ("f_name", "f_address", "f_email")}
    for source in sources:
        templates[source.split].add(source.template_id)
        for fact in source.facts:
            if fact.fact_id in identities:
                identities[fact.fact_id].append(str(fact.value))
    template_overlap = {
        f"{left}/{right}": sorted(templates[left] & templates[right])
        for left, right in (("train", "dev"), ("train", "test"), ("dev", "test"))
    }
    duplicates = {key: len(values) - len(set(values)) for key, values in identities.items()}
    valid = not errors and not any(template_overlap.values()) and not any(duplicates.values())
    return {
        "valid": valid,
        "source_count": len(sources),
        "task_record_count": len(sources) * 2,
        "split_source_counts": dict(split_counts),
        "template_counts": {key: len(value) for key, value in templates.items()},
        "template_overlap": template_overlap,
        "identity_duplicate_counts": duplicates,
        "source_errors": errors,
    }
