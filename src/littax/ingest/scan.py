"""Walk the source directory and hash every PDF."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

_CHUNK = 1 << 20


@dataclass(slots=True)
class ScannedFile:
    path: Path
    relpath: str
    sha256: str
    size_bytes: int
    # sha256 of the first file seen with identical content, when this is a duplicate.
    duplicate_of: str | None = None


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        while chunk := fh.read(_CHUNK):
            digest.update(chunk)
    return digest.hexdigest()


def scan_directory(root: Path, glob: str = "**/*.pdf") -> list[ScannedFile]:
    """Return every matching file under `root`, sorted, with byte-duplicates flagged.

    Duplicates are kept in the list (so the report can show them) but marked, so
    downstream stages can skip them when building the taxonomy.
    """
    root = root.resolve()
    if not root.is_dir():
        raise NotADirectoryError(f"source root is not a directory: {root}")

    seen: dict[str, str] = {}
    out: list[ScannedFile] = []
    for path in sorted(root.glob(glob)):
        if not path.is_file():
            continue
        digest = sha256_file(path)
        relpath = path.relative_to(root).as_posix()
        first = seen.get(digest)
        if first is None:
            seen[digest] = relpath
        out.append(
            ScannedFile(
                path=path,
                relpath=relpath,
                sha256=digest,
                size_bytes=path.stat().st_size,
                duplicate_of=first,
            )
        )
    return out
