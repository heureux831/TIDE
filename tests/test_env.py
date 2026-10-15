"""Environment-loading regression tests."""

import os
from pathlib import Path

import pytest

from tide.env import load_project_env


def test_blank_optional_endpoint_is_unset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text("HF_ENDPOINT=\n", encoding="utf-8")
    monkeypatch.delenv("HF_ENDPOINT", raising=False)

    load_project_env(env_file)

    assert "HF_ENDPOINT" not in os.environ


def test_explicit_endpoint_is_preserved(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text("HF_ENDPOINT=\n", encoding="utf-8")
    monkeypatch.setenv("HF_ENDPOINT", "https://hf-mirror.com")

    load_project_env(env_file)

    assert os.environ["HF_ENDPOINT"] == "https://hf-mirror.com"
