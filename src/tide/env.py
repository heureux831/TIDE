"""Environment loading without leaking secret values."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

_OPTIONAL_URL_ENV_VARS = ("HF_ENDPOINT",)


def load_project_env(path: Path | None = None) -> Path:
    """Load the project .env without overriding explicit process settings."""
    env_path = path or Path.cwd() / ".env"
    load_dotenv(env_path, override=False)
    # Some clients interpret a present-but-empty URL as a literal endpoint.
    # Treat blank optional URLs as unset so their official defaults still work.
    for name in _OPTIONAL_URL_ENV_VARS:
        if not os.getenv(name, "").strip():
            os.environ.pop(name, None)
    return env_path


def required_env(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise RuntimeError(f"Required environment variable {name} is not set")
    return value


def env_status(*names: str) -> dict[str, bool]:
    """Return presence flags only; never return credential values."""
    return {name: bool(os.getenv(name, "").strip()) for name in names}
