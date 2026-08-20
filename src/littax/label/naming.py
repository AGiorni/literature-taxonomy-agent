"""Name every folder in the discovered tree, then review the tree as a whole."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from littax.cluster.tree import Node
from littax.config import LabelConfig
from littax.export.naming import safe_folder_name
from littax.label.client import FoundryLabeller
from littax.label.prompts import (
    COHERENCE_SYSTEM,
    NAMING_SYSTEM,
    VERIFY_SYSTEM,
    coherence_prompt,
    naming_prompt,
    verify_prompt,
)
from littax.models import CoherenceEdit

Report = Callable[[str, int, int, str], None]


@dataclass
class NodeLabel:
    label: str
    description: str
    distinguishing_note: str
    keywords: list[str]


def representative_titles(
    node: Node, X: np.ndarray, titles: list[str | None], limit: int
) -> list[str]:
    """Titles of the documents closest to the node's centroid.

    Sampling by proximity rather than at random matters: the model is being
    asked what the cluster is *about*, and its core members answer that far
    better than its fringe does.
    """
    indices = np.asarray(node.doc_indices)
    if node.centroid is None or not len(indices):
        return []
    similarity = X[indices] @ node.centroid
    order = np.argsort(similarity)[::-1][:limit]
    return [titles[indices[i]] or "(untitled)" for i in order]


def label_tree(
    labeller: FoundryLabeller,
    root: Node,
    X: np.ndarray,
    titles: list[str | None],
    terms_by_node: dict[str, list[str]],
    cfg: LabelConfig,
    report: Report | None = None,
) -> dict[str, NodeLabel]:
    """Name every node, parents before children.

    Level order is not an implementation detail: a child is named with its
    parent's final label in the prompt, so it can be made more specific than
    the folder it sits inside.
    """
    from littax.cluster.tree import iter_nodes

    nodes = list(iter_nodes(root))
    by_id = {node.id: node for node in nodes}
    children_of: dict[str | None, list[Node]] = {}
    for node in nodes:
        children_of.setdefault(node.parent_id, []).append(node)

    labels: dict[str, NodeLabel] = {}
    max_depth = max((n.level for n in nodes), default=1)
    done = 0

    for level in sorted({node.level for node in nodes}):
        cohort = [node for node in nodes if node.level == level]

        def describe(node: Node) -> dict[str, Any]:
            return {
                "id": node.id,
                "size": len(node.doc_indices),
                "distinctive_terms": terms_by_node.get(node.id, [])[:12],
            }

        def name_one(node: Node) -> tuple[str, NodeLabel]:
            siblings = [
                describe(other)
                for other in children_of.get(node.parent_id, [])
                if other.id != node.id
            ]
            parent = labels.get(node.parent_id or "")
            prompt = naming_prompt(
                node_id=node.id,
                parent_label=parent.label if parent else None,
                keywords=terms_by_node.get(node.id, []),
                titles=representative_titles(node, X, titles, cfg.representative_docs),
                size=len(node.doc_indices),
                siblings=siblings,
                level=node.level,
                max_depth=max_depth,
            )
            payload = labeller.ask_json(NAMING_SYSTEM, prompt, max_tokens=2000)
            return node.id, _coerce_label(payload, node.id)

        results = labeller.map_concurrent(cohort, name_one)
        for node_id, label in results:
            labels[node_id] = label
        done += len(cohort)
        if report:
            report("label", done, len(nodes), f"level {level}")

    _deduplicate_siblings(labels, children_of)
    return labels


def _coerce_label(payload: Any, node_id: str) -> NodeLabel:
    """Accept what the model returned, defensively."""
    if not isinstance(payload, dict):
        return NodeLabel(node_id, "", "", [])
    label = safe_folder_name(str(payload.get("label") or node_id), fallback=node_id)
    keywords = payload.get("keywords") or []
    if not isinstance(keywords, list):
        keywords = []
    return NodeLabel(
        label=label,
        description=str(payload.get("description") or ""),
        distinguishing_note=str(payload.get("distinguishing_note") or ""),
        keywords=[str(k) for k in keywords][:12],
    )


def _deduplicate_siblings(
    labels: dict[str, NodeLabel], children_of: dict[str | None, list[Node]]
) -> None:
    """Two sibling folders with the same name would collide on disk."""
    for siblings in children_of.values():
        seen: dict[str, int] = {}
        for node in siblings:
            label = labels.get(node.id)
            if label is None:
                continue
            key = label.label.casefold()
            if key in seen:
                seen[key] += 1
                label.label = f"{label.label} ({seen[key]})"
            else:
                seen[key] = 1


def review_coherence(
    labeller: FoundryLabeller,
    root: Node,
    labels: dict[str, NodeLabel],
    terms_by_node: dict[str, list[str]],
) -> list[CoherenceEdit]:
    """One pass over the whole tree, looking for faults only visible globally."""
    from littax.cluster.tree import iter_nodes

    tree = [
        {
            "id": node.id,
            "level": node.level,
            "parent_id": node.parent_id,
            "label": labels[node.id].label if node.id in labels else node.id,
            "description": labels[node.id].description if node.id in labels else "",
            "size": len(node.doc_indices),
            "distinctive_terms": terms_by_node.get(node.id, [])[:8],
            "children": [child.id for child in node.children],
        }
        for node in iter_nodes(root)
    ]

    payload = labeller.ask_json(
        COHERENCE_SYSTEM, coherence_prompt(tree), max_tokens=8000, effort="high"
    )
    raw_edits = payload.get("edits", []) if isinstance(payload, dict) else []

    valid_ids = {node["id"] for node in tree}
    edits: list[CoherenceEdit] = []
    for edit in raw_edits:
        if not isinstance(edit, dict):
            continue
        kind = edit.get("type")
        targets = edit.get("targets") or []
        if kind not in {"rename", "merge", "reparent"}:
            continue
        # A hallucinated folder id would corrupt the tree, so drop the edit.
        if not isinstance(targets, list) or not all(t in valid_ids for t in targets):
            continue
        edits.append(
            CoherenceEdit(
                type=kind,
                targets=[str(t) for t in targets],
                detail=str(edit.get("detail") or ""),
                reason=str(edit.get("reason") or ""),
            )
        )
    return edits


def apply_renames(labels: dict[str, NodeLabel], edits: list[CoherenceEdit]) -> None:
    """Apply the rename edits.

    Merge and reparent are recorded in the output for review but not applied:
    both restructure the tree, which would invalidate the centroids every
    confidence score was computed against. Renaming is safe because it changes
    only a name.
    """
    for edit in edits:
        if edit.type == "rename" and edit.targets and edit.detail:
            node_id = edit.targets[0]
            if node_id in labels:
                labels[node_id].label = safe_folder_name(edit.detail, fallback=node_id)


def verify_low_confidence(
    labeller: FoundryLabeller,
    candidates: list[dict[str, Any]],
    report: Report | None = None,
) -> dict[str, dict[str, Any]]:
    """Second-opinion pass over the documents the geometry was least sure about."""
    if not candidates:
        return {}

    def check(item: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        payload = labeller.ask_json(
            VERIFY_SYSTEM,
            verify_prompt(
                title=item["title"],
                abstract=item["abstract"],
                assigned=item["assigned"],
                candidates=item["candidates"],
            ),
            max_tokens=1000,
            effort="low",
        )
        if not isinstance(payload, dict):
            payload = {}
        return item["document_id"], {
            "correct": bool(payload.get("correct", True)),
            "best_folder_id": payload.get("best_folder_id"),
            "rationale": str(payload.get("rationale") or ""),
        }

    results = labeller.map_concurrent(
        candidates,
        check,
        on_done=(lambda n: report("verify", n, len(candidates), "")) if report else None,
    )
    return dict(results)
