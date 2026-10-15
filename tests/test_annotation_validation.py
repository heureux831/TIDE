from pathlib import Path

from tide.config import load_config
from tide.data_build import build_sources
from tide.schemas import AnnotationResponse, TaskAnnotation
from tide.validate import REQUIRED, protected_for, validate_annotation_response


def test_reference_annotation_must_match_programmatic_partition_and_answer() -> None:
    source = build_sources(load_config(Path("configs/overnight.yaml")))[0]
    values = {fact.fact_id: fact.value for fact in source.facts}
    response = AnnotationResponse(
        source_id=source.source_id,
        annotations=[
            TaskAnnotation(
                task_id="A",
                required_fact_ids=REQUIRED["A"],
                protected_fact_ids=protected_for("A"),
                rewritten_context=(
                    f"The person is {values['f_age']} years old and is available for "
                    f"{values['f_available']} hours per week."
                ),
                answer=source.tasks[0].gold_answer,
            ),
            TaskAnnotation(
                task_id="B",
                required_fact_ids=REQUIRED["B"],
                protected_fact_ids=protected_for("B"),
                rewritten_context=(
                    f"The person has {values['f_experience']} years of relevant experience "
                    f"and has completed {values['f_training']} training hours."
                ),
                answer=source.tasks[1].gold_answer,
            ),
        ],
    )
    assert validate_annotation_response(source, response) == []

    response.annotations[0].rewritten_context += f" The name is {values['f_name']}."
    errors = validate_annotation_response(source, response)
    assert "A:protected_disclosure:f_name" in errors
