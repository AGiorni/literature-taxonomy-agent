"""Content-addressed cache for expensive stages.

Keyed by the SHA-256 of the source file (plus a parser version), so editing the
parser invalidates every entry while re-running on an unchanged corpus costs
nothing.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

# Bump when a change to ingest/parse.py should invalidate cached parses.
PARSER_VERSION = 3


def corpus_key(*parts: str) -> str:
    """Stable hash over the inputs that would change an embedding matrix."""
    digest = hashlib.sha256()
    for part in parts:
        digest.update(part.encode("utf-8"))
        digest.update(b"\x00")
    return digest.hexdigest()[:32]


class ArrayCache:
    """Stores one numpy array per key. Used for the embedding matrix, which is
    both the slowest stage and perfectly deterministic given its inputs."""

    def __init__(self, root: Path, namespace: str) -> None:
        self.dir = Path(root) / namespace
        self.dir.mkdir(parents=True, exist_ok=True)

    def get(self, key: str):
        import numpy as np

        path = self.dir / f"{key}.npy"
        if not path.exists():
            return None
        try:
            return np.load(path)
        except (OSError, ValueError):
            return None

    def set(self, key: str, array) -> None:
        import numpy as np

        path = self.dir / f"{key}.npy"
        temp = path.with_suffix(".tmp.npy")
        np.save(temp, array)
        temp.replace(path)


class JsonCache:
    def __init__(self, root: Path, namespace: str, version: int | str = 1) -> None:
        self.dir = Path(root) / namespace / str(version)
        self.dir.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        # Shard by first two characters so no single directory holds 100k files.
        return self.dir / key[:2] / f"{key}.json"

    def get(self, key: str) -> dict[str, Any] | None:
        path = self._path(key)
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return None  # a corrupt entry is just a miss

    def set(self, key: str, value: dict[str, Any]) -> None:
        path = self._path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        temp = path.with_suffix(".tmp")
        temp.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")
        temp.replace(path)  # atomic, so an interrupted run leaves no partial entry
