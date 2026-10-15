from pathlib import Path

from tide.config import load_config


def test_protocol_versions_keep_separate_debug_budgets_and_outputs() -> None:
    v1 = load_config(Path("configs/overnight.yaml"))
    v2 = load_config(Path("configs/overnight_v2.yaml"))

    assert v1.training.debug_max_optimizer_steps == 200
    assert v2.training.debug_max_optimizer_steps == 500
    assert v1.training.max_optimizer_steps == v2.training.max_optimizer_steps == 300
    assert v1.experiment.output_dir != v2.experiment.output_dir

    v1_payload = v1.model_dump()
    v2_payload = v2.model_dump()
    v1_payload["experiment"]["output_dir"] = v2_payload["experiment"]["output_dir"]
    v1_payload["training"]["debug_max_optimizer_steps"] = 500
    assert v1_payload == v2_payload
