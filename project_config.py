"""Shared, machine-independent paths and environment settings."""

from __future__ import annotations

import os
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent
DATA_ROOT = Path(
    os.environ.get("DATA_INTERDEPENDENCIES_DATA_ROOT", REPO_ROOT / "data")
).expanduser().resolve()


def require_env(name: str) -> str:
    """Return a required environment variable without exposing its value."""
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(
            f"{name} is not set. Configure it in your shell; do not store credentials "
            "in notebooks or committed files."
        )
    return value

