"""Execute the reorganisation plan.

This is the only module that writes outside `out/`, so it is deliberately
conservative: it verifies every source hash before touching anything, refuses
to overwrite, and defaults to a dry run.

`hardlink` is the mode worth reaching for on a corpus of any size: it produces
a fully browsable folder tree while the files still occupy their original bytes
once. `move` is the only mode that disturbs the source folder.
"""

from __future__ import annotations

import shutil
from collections import Counter
from pathlib import Path
from typing import Callable

from littax.ingest.scan import sha256_file
from littax.models import Taxonomy

MODES = ("dry-run", "hardlink", "copy", "move")


def apply_plan(
    taxonomy: Taxonomy,
    *,
    mode: str = "dry-run",
    min_confidence: float = 0.0,
    target_root: Path | None = None,
    log: Callable[[str], None] = print,
) -> dict[str, int]:
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, got {mode!r}")

    source_root = Path(taxonomy.source_root)
    root = Path(target_root) if target_root else Path(taxonomy.target_root)
    by_id = {document.id: document for document in taxonomy.documents}
    counts: Counter[str] = Counter()

    for instruction in taxonomy.move_plan:
        document = by_id.get(instruction.document_id)
        if document is None:
            counts["skipped (unknown document)"] += 1
            continue

        if instruction.confidence < min_confidence:
            counts["skipped (below --min-confidence)"] += 1
            continue

        source = source_root / (document.relpath or document.filename)
        if not source.exists():
            log(f"[yellow]missing source:[/] {document.filename}")
            counts["skipped (source missing)"] += 1
            continue

        # Re-hash rather than trust the plan: the corpus may have changed since
        # the analysis ran, and silently filing the wrong paper is worse than
        # refusing to file it.
        if sha256_file(source) != document.sha256:
            log(f"[yellow]changed since analysis:[/] {document.filename}")
            counts["skipped (content changed)"] += 1
            continue

        # Re-root the recorded target, which was written relative to the
        # taxonomy's own target_root.
        relative = Path(instruction.target)
        recorded_root = Path(taxonomy.target_root)
        if recorded_root.parts and relative.parts[: len(recorded_root.parts)] == recorded_root.parts:
            relative = Path(*relative.parts[len(recorded_root.parts) :])
        destination = root / relative

        if destination.exists():
            counts["skipped (already present)"] += 1
            continue

        if mode == "dry-run":
            log(f"  {relative}")
            counts["would place"] += 1
            continue

        destination.parent.mkdir(parents=True, exist_ok=True)
        try:
            if mode == "hardlink":
                try:
                    destination.hardlink_to(source)
                except OSError:
                    # Hard links fail across volumes and on some filesystems;
                    # a copy still produces a correct tree.
                    shutil.copy2(source, destination)
                    counts["copied (hardlink unavailable)"] += 1
                    continue
            elif mode == "copy":
                shutil.copy2(source, destination)
            else:
                shutil.move(str(source), str(destination))
        except OSError as exc:
            log(f"[red]failed:[/] {document.filename} - {exc}")
            counts["failed"] += 1
            continue
        counts[mode + "ed" if mode != "copy" else "copied"] += 1

    return dict(counts)
