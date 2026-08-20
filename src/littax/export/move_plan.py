"""Turn folder assignments into concrete file operations.

Nothing here touches the filesystem - it only produces the instruction list that
`littax apply` later executes.
"""

from __future__ import annotations

from pathlib import PurePosixPath

from littax.models import Band, Document, Folder, MoveInstruction


def build_move_plan(
    documents: list[Document],
    folders: dict[str, Folder],
    target_root: str,
    *,
    skip_duplicates: bool = True,
) -> list[MoveInstruction]:
    """Map each document to its destination path, resolving name collisions.

    Two different papers can legitimately share a filename, and the same paper
    can appear twice under different names; both would silently overwrite one
    another at apply time, so collisions get a numeric suffix here instead.
    """
    plan: list[MoveInstruction] = []
    used: set[str] = set()

    for document in documents:
        if document.assignment is None:
            continue
        if skip_duplicates and document.duplicate_of is not None:
            continue

        folder = folders.get(document.assignment.folder_id)
        folder_path = folder.path if folder else "_Unsorted"
        target = _unique_path(
            PurePosixPath(target_root) / folder_path / document.filename, used
        )
        used.add(str(target).lower())

        plan.append(
            MoveInstruction(
                source=document.source_path,
                target=str(target),
                document_id=document.id,
                confidence=document.assignment.confidence,
                band=document.assignment.band,
            )
        )
    return plan


def _unique_path(path: PurePosixPath, used: set[str]) -> PurePosixPath:
    if str(path).lower() not in used:
        return path
    stem, suffix = path.stem, path.suffix
    for n in range(2, 1000):
        candidate = path.with_name(f"{stem} ({n}){suffix}")
        if str(candidate).lower() not in used:
            return candidate
    raise RuntimeError(f"could not find a free filename for {path}")
