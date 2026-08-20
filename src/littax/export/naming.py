"""Folder names that are safe to actually create on disk.

Windows is the strict case and the one that matters here: it forbids a set of
characters, reserves a handful of device names regardless of extension, and
silently strips trailing dots and spaces.
"""

from __future__ import annotations

import re

_ILLEGAL = re.compile(r'[<>:"/\\|?*\x00-\x1f]')
_RESERVED = {
    "CON", "PRN", "AUX", "NUL",
    *(f"COM{i}" for i in range(1, 10)),
    *(f"LPT{i}" for i in range(1, 10)),
}
MAX_NAME_LENGTH = 70


def safe_folder_name(label: str, fallback: str = "topic") -> str:
    """Make `label` usable as a single path segment on Windows, macOS and Linux."""
    name = _ILLEGAL.sub(" ", label)
    name = re.sub(r"\s+", " ", name).strip(" .")

    if len(name) > MAX_NAME_LENGTH:
        # Cut on a word boundary so the name still reads.
        name = name[:MAX_NAME_LENGTH].rsplit(" ", 1)[0].strip(" .")

    if not name:
        return fallback
    if name.upper() in _RESERVED or name.split(".")[0].upper() in _RESERVED:
        return f"{name}_"
    return name


def placeholder_label(terms: list[str], node_id: str = "topic", max_terms: int = 3) -> str:
    """A readable stand-in name built from c-TF-IDF terms.

    Used until the Phase 3 labelling pass replaces it with a real topic name.
    Deliberately kept ugly-but-informative rather than pretty-but-vague, so it
    is obvious at a glance which folders have been named by the model.
    """
    if not terms:
        return node_id
    chosen: list[str] = []
    for term in terms:
        # Skip a term already contained in one we kept ("motor unit" after
        # "motor unit firing"), which otherwise wastes the name's budget.
        if any(term in kept or kept in term for kept in chosen):
            continue
        chosen.append(term)
        if len(chosen) == max_terms:
            break
    return ", ".join(chosen)
