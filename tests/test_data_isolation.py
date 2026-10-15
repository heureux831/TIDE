from collections import Counter

from tide.config import load_config
from tide.data_build import LABEL_COUNTS, build_sources
from tide.validate import dataset_validation_report


def test_controlled_dataset_is_grouped_balanced_and_valid() -> None:
    config = load_config(__import__("pathlib").Path("configs/overnight.yaml"))
    sources = build_sources(config)
    assert len(sources) == 300
    assert len({source.source_id for source in sources}) == 300
    report = dataset_validation_report(sources)
    assert report["valid"], report
    counts = Counter(
        (source.split, source.tasks[0].gold_answer, source.tasks[1].gold_answer)
        for source in sources
    )
    for split, quotas in LABEL_COUNTS.items():
        for pair, expected in quotas.items():
            assert counts[(split, *pair)] == expected
    template_orders = {}
    for source in sources:
        order = tuple(
            fact.fact_id for fact in sorted(source.facts, key=lambda item: item.char_start)
        )
        previous = template_orders.setdefault(source.template_id, order)
        assert previous == order
    assert len(template_orders) == 24
    assert len(set(template_orders.values())) == 24

    for split in ("train", "dev", "test"):
        split_sources = [source for source in sources if source.split == split]
        a_failures = set()
        b_failures = set()
        a_boundary_rules = set()
        b_boundary_rules = set()
        for source in split_sources:
            values = {fact.fact_id: int(fact.value) for fact in source.facts[4:]}
            task_a, task_b = source.tasks
            age_ok = task_a.rule["age_min"] <= values["f_age"] <= task_a.rule["age_max"]
            available_ok = values["f_available"] >= task_a.rule["min_available_hours"]
            experience_ok = values["f_experience"] >= task_b.rule["min_experience_years"]
            training_ok = values["f_training"] >= task_b.rule["min_training_hours"]
            if task_a.gold_answer == "NO":
                a_failures.add((age_ok, available_ok))
            elif (
                values["f_age"] == task_a.rule["age_max"]
                and values["f_available"] == task_a.rule["min_available_hours"]
            ):
                a_boundary_rules.add(tuple(task_a.rule.values()))
            if task_b.gold_answer == "NO":
                b_failures.add((experience_ok, training_ok))
            elif (
                values["f_experience"] == task_b.rule["min_experience_years"]
                and values["f_training"] == task_b.rule["min_training_hours"]
            ):
                b_boundary_rules.add(tuple(task_b.rule.values()))
        assert a_failures == {(False, True), (True, False), (False, False)}
        assert b_failures == {(False, True), (True, False), (False, False)}
        assert len(a_boundary_rules) == 3
        assert len(b_boundary_rules) == 3


def test_dataset_generation_is_deterministic() -> None:
    from pathlib import Path

    config = load_config(Path("configs/overnight.yaml"))
    first = [record.model_dump_json() for record in build_sources(config)]
    second = [record.model_dump_json() for record in build_sources(config)]
    assert first == second


def test_model_input_whitelist_excludes_gold_and_ids() -> None:
    from tide.tokenize import format_rewriter_input

    text, _ = format_rewriter_input("The person is 34 years old.", "Apply the public rule.")
    assert "gold_answer" not in text
    assert "source_id" not in text
    assert "fact_id" not in text
