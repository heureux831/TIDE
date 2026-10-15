from pathlib import Path

import pytest

from tide.config import load_config
from tide.metrics import FactParse
from tide.v3_long_m2 import FIT_THRESHOLD, _audit_flags, assert_v3_protocol

ROOT = Path(__file__).resolve().parents[1]


def test_v3_changes_only_approved_protocol_fields() -> None:
    v2 = load_config(ROOT / "configs/overnight_v2.yaml")
    v3 = load_config(ROOT / "configs/v3_long_m2.yaml")
    report = assert_v3_protocol(v2, v3)
    assert report["effective_batch_size"] == 16
    assert v3.training.max_optimizer_steps == 1500
    assert v3.training.learning_rate_matrix == pytest.approx(0.0003)
    assert v3.training.learning_rate_lora == pytest.approx(0.0002)
    assert pytest.approx(0.95) == FIT_THRESHOLD


def test_audit_triage_distinguishes_deletion_and_wrong_number() -> None:
    missing = FactParse((), (), False, False, False)
    assert _audit_flags("A short summary.", missing)["required_fact_deleted"]
    assert _audit_flags("The person is 99 years old.", missing)["number_or_unit_error"]
