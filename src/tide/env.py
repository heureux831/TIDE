"""Environment loading without leaking secret values."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv


def load_project_env(path: Path | None = None) -> Path:
    """Load the project .env without overriding explicit process settings."""
    env_path = path or Path.cwd() / ".env"
    load_dotenv(env_path, override=False)
    return env_path


def required_env(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise RuntimeError(f"Required environment variable {name} is not set")
    return value


def env_status(*names: str) -> dict[str, bool]:
    """Return presence flags only; never return credential values."""
    return {name: bool(os.getenv(name, "").strip()) for name in names}
