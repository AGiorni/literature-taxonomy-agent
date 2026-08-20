"""Load `.env` so credentials do not have to be exported by hand.

A real environment variable always wins over the file: exporting a key for one
run is the obvious way to override, and silently ignoring that would be a nasty
surprise.
"""

from __future__ import annotations

import os
from pathlib import Path

# Names this project reads. Used only to report *which* keys were found -
# never their values.
KNOWN_KEYS = (
    "ANTHROPIC_FOUNDRY_API_KEY",
    "ANTHROPIC_FOUNDRY_RESOURCE",
    "AZURE_OPENAI_API_KEY",
    "AZURE_OPENAI_ENDPOINT",
)

_loaded: Path | None = None


def _parse(text: str) -> dict[str, str]:
    """Minimal KEY=VALUE parser: comments, blank lines, optional `export`,
    and surrounding quotes."""
    values: dict[str, str] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        if key.startswith("export "):
            key = key[len("export ") :].strip()
        if not key:
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        values[key] = value
    return values


def load_env(path: Path | None = None) -> Path | None:
    """Read `.env` into os.environ without overriding what is already set.

    Returns the file that was loaded, or None. Searches the working directory
    and its parents so the command works from a subdirectory too.
    """
    global _loaded

    candidates = [path] if path else [
        parent / ".env" for parent in [Path.cwd(), *Path.cwd().parents]
    ]
    for candidate in candidates:
        if candidate is None or not candidate.is_file():
            continue
        try:
            text = candidate.read_text(encoding="utf-8")
        except OSError:
            continue
        for key, value in _parse(text).items():
            os.environ.setdefault(key, value)
        _loaded = candidate
        return candidate
    return None


def loaded_from() -> Path | None:
    return _loaded


def present_keys() -> list[str]:
    """Which known credential names are set. Names only - never values."""
    return [key for key in KNOWN_KEYS if os.environ.get(key)]
