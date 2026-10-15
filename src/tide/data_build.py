"""Deterministic construction of the controlled 300-source feasibility dataset."""

from __future__ import annotations

import itertools
import random
from collections import Counter
from pathlib import Path
from typing import Any, Literal

from tide.config import TideConfig
from tide.io import sha256_file, write_json, write_jsonl
from tide.schemas import Fact, SourceRecord, SourceTask

A_RULES = ((25, 45, 10), (30, 50, 15), (22, 40, 20))
B_RULES = ((3, 20), (5, 40), (8, 60))
Label = Literal["YES", "NO"]
LABEL_COUNTS = {
    "train": {("YES", "YES"): 50, ("YES", "NO"): 50, ("NO", "YES"): 50, ("NO", "NO"): 50},
    "dev": {("YES", "YES"): 13, ("YES", "NO"): 12, ("NO", "YES"): 12, ("NO", "NO"): 13},
    "test": {("YES", "YES"): 12, ("YES", "NO"): 13, ("NO", "YES"): 13, ("NO", "NO"): 12},
}
TEMPLATE_IDS = {
    "train": tuple(f"train_{index:02d}" for index in range(16)),
    "dev": tuple(f"dev_{index:02d}" for index in range(4)),
    "test": tuple(f"test_{index:02d}" for index in range(4)),
}

FIRST = ("Neral", "Aven", "Miren", "Tovan", "Selin", "Oris", "Vela", "Korin", "Darin", "Luma")
LAST = ("Voss", "Merek", "Tal", "Renn", "Solan", "Kest", "Varo", "Neris", "Pell", "Dorne")
STREETS = ("Lunor", "Cobalt", "Meridian", "Juniper", "Orchid", "Harbor", "Willow", "Quartz")
STREET_TYPES = ("Lane", "Road", "Way", "Street", "Avenue", "Drive")
EMPLOYERS = ("Velmora Works", "Northstar Atelier", "Cinder Labs", "Juniper Systems", "Oriole Group")

A_QUESTIONS = (
    "Decide whether the person qualifies for the program. Qualification requires an age "
    "between {low} and {high} inclusive and availability of at least {hours} hours per week. "
    "Use only the supplied context. Return YES, NO, or UNKNOWN if required facts are missing.",
    "Assess program eligibility. The person must be {low} through {high} years old, inclusive, "
    "and available at least {hours} hours each week. Use only the context and answer YES, NO, "
    "or UNKNOWN when required facts are absent.",
    "Determine training-program qualification using only the context: age must fall within "
    "{low}-{high} inclusive and weekly availability must be at least {hours} hours. Return "
    "exactly YES, NO, or UNKNOWN.",
    "Check the program rule from the supplied context. Both conditions are required: age from "
    "{low} to {high}, inclusive, and no fewer than {hours} available hours per week. Answer "
    "YES, NO, or UNKNOWN.",
)
B_QUESTIONS = (
    "Decide whether the person qualifies for the role. Qualification requires at least {years} "
    "years of relevant experience and at least {hours} completed training hours. Use only the "
    "supplied context. Return YES, NO, or UNKNOWN if required facts are missing.",
    "Assess technical-role eligibility using only the context. The person needs {years} or more "
    "years of relevant experience and at least {hours} hours of completed training. Answer YES, "
    "NO, or UNKNOWN.",
    "Determine qualification for the position: relevant experience must be at least {years} "
    "years and completed training must be at least {hours} hours. Use only supplied facts and "
    "return YES, NO, or UNKNOWN.",
    "Check the role rule from the context. Both a minimum of {years} years of relevant "
    "experience and {hours} completed training hours are required. Return exactly YES, NO, "
    "or UNKNOWN.",
)


def _label_plan(split: str, rng: random.Random) -> list[tuple[Label, Label]]:
    labels = list(
        itertools.chain.from_iterable([pair] * count for pair, count in LABEL_COUNTS[split].items())
    )
    rng.shuffle(labels)
    return labels  # type: ignore[return-value]


def _sample_joint_values(
    a_rule: tuple[int, int, int],
    b_rule: tuple[int, int],
    label_a: Label,
    label_b: Label,
    rng: random.Random,
    *,
    a_no_mode: int,
    b_no_mode: int,
    boundary_a: bool,
    boundary_b: bool,
) -> tuple[int, int, int, int]:
    """Construct labels deliberately, including all NO modes and exact boundaries."""
    low, high, available_minimum = a_rule
    experience_minimum, training_minimum = b_rule
    experience_must_pass = label_b == "YES" or (label_b == "NO" and b_no_mode == 1)
    age_floor = 18 + experience_minimum if experience_must_pass else 22
    if label_a == "YES":
        age = high if boundary_a else rng.randint(max(low, age_floor), high)
        available = available_minimum if boundary_a else rng.randint(available_minimum, 30)
    else:
        age_passes = a_no_mode == 1
        available_passes = a_no_mode == 0
        age = (
            rng.randint(max(low, age_floor), high)
            if age_passes
            else rng.randint(max(high + 1, age_floor), 65)
        )
        available = (
            rng.randint(available_minimum, 30)
            if available_passes
            else rng.randint(0, available_minimum - 1)
        )
    maximum_experience = min(20, age - 18)
    if label_b == "YES":
        experience = (
            experience_minimum
            if boundary_b
            else rng.randint(experience_minimum, maximum_experience)
        )
        training = training_minimum if boundary_b else rng.randint(training_minimum, 120)
    else:
        experience_passes = b_no_mode == 1
        training_passes = b_no_mode == 0
        experience = (
            rng.randint(experience_minimum, maximum_experience)
            if experience_passes
            else rng.randint(0, experience_minimum - 1)
        )
        training = (
            rng.randint(training_minimum, 120)
            if training_passes
            else rng.randint(0, training_minimum - 1)
        )
    return age, available, experience, training


def _identity(index: int) -> tuple[str, str, str, str]:
    first = FIRST[index % len(FIRST)]
    last = LAST[(index // len(FIRST)) % len(LAST)]
    suffix = index // (len(FIRST) * len(LAST))
    name = f"{first}{suffix or ''} {last}"
    address = (
        f"{100 + index} {STREETS[index % len(STREETS)]} {STREET_TYPES[index % len(STREET_TYPES)]}"
    )
    email = f"{first.lower()}.{last.lower()}{index}@example.com"
    employer = f"{EMPLOYERS[index % len(EMPLOYERS)]} {index:03d}"
    return name, address, email, employer


def _render_context(
    template_index: int, values: dict[str, str | int]
) -> tuple[str, dict[str, tuple[str, int, int]]]:
    phrases = {
        "f_name": str(values["f_name"]),
        "f_address": str(values["f_address"]),
        "f_email": str(values["f_email"]),
        "f_employer": str(values["f_employer"]),
        "f_age": f"is {values['f_age']} years old",
        "f_available": f"can commit {values['f_available']} hours per week",
        "f_experience": f"has {values['f_experience']} years of relevant experience",
        "f_training": f"has completed {values['f_training']} training hours",
    }
    order = list(phrases)
    shift = template_index % len(order)
    order = order[shift:] + order[:shift]
    block = template_index // len(order)
    if block == 1:
        order.reverse()
    elif block == 2:
        order = order[::2] + order[1::2]
    lead = (
        "This synthetic profile is provided only for a controlled eligibility exercise. "
        "The entries below describe one fictional person, and each recorded detail appears once. "
    )
    tails = (
        "No conclusion about either application is included in this profile. "
        "The evaluator must use the separate public rule.",
        "These administrative details do not themselves state whether either rule is satisfied. "
        "A separate task supplies the decision thresholds.",
        "The profile records facts rather than an eligibility decision. "
        "Any decision must be made solely from the accompanying task.",
    )
    labels = {
        "f_name": "The recorded name is ",
        "f_address": "The mailing address is ",
        "f_email": "The contact email is ",
        "f_employer": "The listed employer is ",
        "f_age": "The person ",
        "f_available": "The person ",
        "f_experience": "The person ",
        "f_training": "The person ",
    }
    context = lead
    spans: dict[str, tuple[str, int, int]] = {}
    for fact_id in order:
        prefix = labels[fact_id]
        surface = phrases[fact_id]
        context += prefix
        start = len(context)
        context += surface
        end = len(context)
        context += ". "
        spans[fact_id] = (surface, start, end)
    context += tails[template_index % len(tails)]
    if not 90 <= len(context.split()) <= 160:
        raise AssertionError(f"rendered context word count out of range: {len(context.split())}")
    return context, spans


def _facts(values: dict[str, str | int], spans: dict[str, tuple[str, int, int]]) -> list[Fact]:
    fields = {
        "f_name": "name",
        "f_address": "address",
        "f_email": "email",
        "f_employer": "employer",
        "f_age": "age",
        "f_available": "available_hours_per_week",
        "f_experience": "experience_years",
        "f_training": "training_hours",
    }
    aliases = {
        "f_name": str(values["f_name"]).split(),
        "f_address": [str(values["f_address"])],
        "f_email": [str(values["f_email"]), str(values["f_email"]).split("@")[0]],
        "f_employer": [str(values["f_employer"])],
        "f_age": [f"{values['f_age']} years old"],
        "f_available": [f"{values['f_available']} hours per week"],
        "f_experience": [f"{values['f_experience']} years of relevant experience"],
        "f_training": [f"{values['f_training']} training hours"],
    }
    result = []
    for fact_id, value in values.items():
        surface, start, end = spans[fact_id]
        result.append(
            Fact(
                fact_id=fact_id,
                field=fields[fact_id],
                value=value,
                surface=surface,
                char_start=start,
                char_end=end,
                aliases=aliases[fact_id],
            )
        )
    return result


def build_sources(config: TideConfig) -> list[SourceRecord]:
    rng = random.Random(config.experiment.seed)
    records: list[SourceRecord] = []
    global_index = 0
    for split in ("train", "dev", "test"):
        plan = _label_plan(split, rng)
        boundary_a_seen: set[tuple[int, int, int]] = set()
        boundary_b_seen: set[tuple[int, int]] = set()
        a_no_counts: Counter[tuple[int, int, int]] = Counter()
        b_no_counts: Counter[tuple[int, int]] = Counter()
        expected = config.data.split_source_counts[split]
        if len(plan) != expected:
            raise ValueError(
                f"label quota for {split} has {len(plan)} entries, expected {expected}"
            )
        for local_index, (label_a, label_b) in enumerate(plan):
            a_rule = A_RULES[local_index % len(A_RULES)]
            b_rule = B_RULES[(local_index // len(A_RULES)) % len(B_RULES)]
            a_no_mode = a_no_counts[a_rule] % 3
            b_no_mode = b_no_counts[b_rule] % 3
            boundary_a = label_a == "YES" and a_rule not in boundary_a_seen
            boundary_b = label_b == "YES" and b_rule not in boundary_b_seen
            age, available, experience, training = _sample_joint_values(
                a_rule,
                b_rule,
                label_a,
                label_b,
                rng,
                a_no_mode=a_no_mode,
                b_no_mode=b_no_mode,
                boundary_a=boundary_a,
                boundary_b=boundary_b,
            )
            if label_a == "YES":
                boundary_a_seen.add(a_rule)
            else:
                a_no_counts[a_rule] += 1
            if label_b == "YES":
                boundary_b_seen.add(b_rule)
            else:
                b_no_counts[b_rule] += 1
            name, address, email, employer = _identity(global_index)
            values: dict[str, str | int] = {
                "f_name": name,
                "f_address": address,
                "f_email": email,
                "f_employer": employer,
                "f_age": age,
                "f_available": available,
                "f_experience": experience,
                "f_training": training,
            }
            template_ids = TEMPLATE_IDS[split]
            template_id = template_ids[local_index % len(template_ids)]
            template_offset = {"train": 0, "dev": 16, "test": 20}[split]
            context, spans = _render_context(
                template_offset + local_index % len(template_ids), values
            )
            low, high, minimum_available = a_rule
            minimum_experience, minimum_training = b_rule
            tasks = [
                SourceTask(
                    task_id="A",
                    task=A_QUESTIONS[local_index % len(A_QUESTIONS)].format(
                        low=low, high=high, hours=minimum_available
                    ),
                    rule={
                        "age_min": low,
                        "age_max": high,
                        "min_available_hours": minimum_available,
                    },
                    gold_answer=label_a,
                ),
                SourceTask(
                    task_id="B",
                    task=B_QUESTIONS[local_index % len(B_QUESTIONS)].format(
                        years=minimum_experience, hours=minimum_training
                    ),
                    rule={
                        "min_experience_years": minimum_experience,
                        "min_training_hours": minimum_training,
                    },
                    gold_answer=label_b,
                ),
            ]
            records.append(
                SourceRecord(
                    source_id=f"src_{global_index:06d}",
                    split=split,
                    template_id=template_id,
                    context=context,
                    facts=_facts(values, spans),
                    tasks=tasks,
                )
            )
            global_index += 1
    return records


def build_and_write(config: TideConfig, root: Path) -> dict[str, Any]:
    records = build_sources(config)
    data_dir = root / "data/raw"
    source_path = data_dir / "sources.jsonl"
    write_jsonl(source_path, (record.model_dump(mode="json") for record in records))
    counts = Counter(record.split for record in records)
    labels = Counter(
        (record.split, record.tasks[0].gold_answer, record.tasks[1].gold_answer)
        for record in records
    )
    report: dict[str, Any] = {
        "valid": True,
        "source_count": len(records),
        "task_record_count": len(records) * 2,
        "split_source_counts": dict(counts),
        "joint_label_counts": {"/".join(key): value for key, value in sorted(labels.items())},
        "sources_sha256": sha256_file(source_path),
    }
    write_json(data_dir / "build_report.json", report)
    return report
