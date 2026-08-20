"""Prompts for the labelling passes.

The naming prompt is deliberately *contrastive*: it always shows the sibling
clusters alongside the one being named. Naming clusters in isolation produces
sets of folders that are individually reasonable and collectively useless -
three siblings all called "Motor Cortex Excitability" because that is the
honest summary of each one on its own.
"""

from __future__ import annotations

import json
from typing import Any

NAMING_SYSTEM = """\
You name folders in a research literature taxonomy that was discovered by \
clustering paper embeddings. You are given one cluster plus its sibling \
clusters, and you name only the target cluster.

Rules:
- The label is a folder name: 2-5 words, title case, no punctuation beyond \
spaces and hyphens, never a filesystem-illegal character.
- The label must distinguish this cluster from its siblings. If siblings \
overlap conceptually, name what separates them, not what they share.
- Use the field's own terminology. These are domain experts; prefer \
"Corticospinal Excitability" over "Brain Stimulation Studies".
- Do not invent a topic the evidence does not support. If the cluster is \
genuinely mixed, say so in the description and pick the dominant theme.
- The description is 2-3 sentences describing what a paper in this folder is \
about, written for someone deciding whether to look inside.

Respond with a single JSON object, no prose, no code fences:
{"label": str, "description": str, "distinguishing_note": str, \
"keywords": [str, ...]}\
"""

COHERENCE_SYSTEM = """\
You review a complete research literature taxonomy for coherence. The tree was \
discovered by clustering, then each node was named independently, so typical \
faults are: sibling folders with near-identical meaning, labels at \
inconsistent levels of abstraction within one layer, a child that belongs \
under a different parent, and parent labels that do not cover their children.

Propose only high-confidence corrections. An unchanged taxonomy is a fine \
answer. Never propose deleting a folder or moving individual documents.

Respond with a single JSON object, no prose, no code fences:
{"edits": [{"type": "rename"|"merge"|"reparent", "targets": [folder_id, ...], \
"detail": str, "reason": str}, ...]}

- rename: targets is one folder id; detail is the new label.
- merge: targets is two or more sibling folder ids; detail is the label for \
the merged folder.
- reparent: targets is [child_id, new_parent_id]; detail is a short note.\
"""

VERIFY_SYSTEM = """\
You check whether a research paper was filed in the right folder. You are given \
the paper's title and abstract, the folder it was assigned to, and the other \
candidate folders it was closest to.

Answer honestly. Clustering put it where it is for a reason, but the reason \
may be superficial (shared methods, shared journal) rather than topical.

Respond with a single JSON object, no prose, no code fences:
{"correct": bool, "best_folder_id": str, "rationale": str}

`best_folder_id` is your choice among the candidates offered; repeat the \
assigned folder's id when it is already right. The rationale is one sentence.\
"""


def _cluster_block(
    label: str,
    keywords: list[str],
    titles: list[str],
    *,
    size: int,
) -> dict[str, Any]:
    return {
        "current_placeholder_label": label,
        "size": size,
        "distinctive_terms": keywords,
        "representative_titles": titles,
    }


def naming_prompt(
    *,
    node_id: str,
    parent_label: str | None,
    keywords: list[str],
    titles: list[str],
    size: int,
    siblings: list[dict[str, Any]],
    level: int,
    max_depth: int,
) -> str:
    payload = {
        "target_cluster": _cluster_block(node_id, keywords, titles, size=size),
        "parent_folder": parent_label,
        "sibling_clusters": siblings,
        "depth": {"level": level, "of": max_depth},
    }
    granularity = (
        "This is a top-level folder: name a broad research area."
        if level == 1
        else "This is a nested folder: name a specific sub-topic, more precise "
        "than its parent."
    )
    return (
        f"{granularity}\n\n"
        f"Name the target cluster.\n\n```json\n{json.dumps(payload, indent=2)}\n```"
    )


def coherence_prompt(tree: list[dict[str, Any]]) -> str:
    return (
        "Review this taxonomy and propose corrections.\n\n"
        f"```json\n{json.dumps(tree, indent=2)}\n```"
    )


def verify_prompt(
    *,
    title: str | None,
    abstract: str | None,
    assigned: dict[str, Any],
    candidates: list[dict[str, Any]],
) -> str:
    payload = {
        "paper": {"title": title, "abstract": abstract},
        "assigned_folder": assigned,
        "candidate_folders": candidates,
    }
    return (
        "Was this paper filed correctly?\n\n"
        f"```json\n{json.dumps(payload, indent=2)}\n```"
    )
