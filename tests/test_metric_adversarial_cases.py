from tide.metrics import parse_rewrite, parse_task_answer
from tide.schemas import Fact


def ledger() -> list[Fact]:
    values = {
        "f_name": ("Neral Voss", "name"),
        "f_address": ("471 Lunor Lane", "address"),
        "f_email": ("neral.voss17@example.com", "email"),
        "f_employer": ("Velmora Works", "employer"),
        "f_age": (34, "age"),
        "f_available": (12, "available_hours_per_week"),
        "f_experience": (6, "experience_years"),
        "f_training": (24, "training_hours"),
    }
    return [
        Fact(
            fact_id=fact_id,
            field=field,
            value=value,
            surface=str(value),
            char_start=index,
            char_end=index + 1,
            aliases=[str(value)],
        )
        for index, (fact_id, (value, field)) in enumerate(values.items())
    ]


def test_exact_required_facts_pass() -> None:
    parsed = parse_rewrite(
        "The person is 34 years old and can commit 12 hours per week.",
        ledger(),
        ["f_age", "f_available"],
        ["f_name", "f_address", "f_email", "f_employer", "f_experience", "f_training"],
    )
    assert parsed.fact_joint


def test_identity_alias_and_new_number_fail_conservatively() -> None:
    parsed = parse_rewrite(
        "Neral is 34 years old, can commit 12 hours per week, and scored 99 points.",
        ledger(),
        ["f_age", "f_available"],
        ["f_name", "f_address", "f_email", "f_employer", "f_experience", "f_training"],
    )
    assert "f_name" in parsed.protected_matches
    assert parsed.numeric_tamper
    assert not parsed.fact_joint


def test_task_answer_parser_accepts_only_complete_labels() -> None:
    assert parse_task_answer(" yes \n") == "YES"
    assert parse_task_answer("YES because...") == "INVALID"
