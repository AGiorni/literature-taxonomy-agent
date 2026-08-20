"""Recursive discovery of the topic hierarchy.

The recursion frame is deliberately shared by both algorithms: `_split` returns
a flat label array for one node's documents, or None if the node should stay a
leaf. Swapping agglomerative for HDBSCAN (Phase 2) only changes `_split`.

The number of children at each node is *discovered*, not configured: for every
admissible k we cut the dendrogram and keep the k with the best silhouette.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score

from littax.config import ClusterConfig


@dataclass
class Node:
    id: str
    level: int
    parent_id: str | None
    # Indices into the corpus-wide embedding matrix, for the whole subtree.
    doc_indices: list[int]
    children: list["Node"] = field(default_factory=list)
    centroid: np.ndarray | None = None
    # Per-document HDBSCAN membership strength, when the algorithm supplies it.
    membership: dict[int, float] = field(default_factory=dict)
    was_noise: set[int] = field(default_factory=set)

    @property
    def is_leaf(self) -> bool:
        return not self.children


def iter_nodes(root: Node, include_root: bool = False):
    """Depth-first walk over the tree."""
    stack = [root]
    while stack:
        node = stack.pop()
        if include_root or node is not root:
            yield node
        stack.extend(reversed(node.children))


def leaves(root: Node) -> list[Node]:
    return [n for n in iter_nodes(root, include_root=True) if n.is_leaf]


def _silhouette(X: np.ndarray, labels: np.ndarray) -> float:
    if len(set(labels)) < 2 or len(set(labels)) >= len(labels):
        return -1.0
    return float(silhouette_score(X, labels, metric="cosine"))


def _split_agglomerative(X: np.ndarray, cfg: ClusterConfig) -> np.ndarray | None:
    """Ward cut at the k that maximises silhouette.

    Ward needs Euclidean distance, but on L2-normalised rows Euclidean distance
    is a monotone function of cosine distance, so the tree it builds is the one
    we want. Silhouette is still scored with the cosine metric.

    The size floor is relaxed progressively rather than applied once. Ward
    routinely peels off one small outlier group at every k, and a single strict
    floor then rejects *all* of them - which is how a 145-document node ends up
    as an unsplit leaf. Falling back to a looser floor keeps the hierarchy from
    collapsing on exactly the nodes that most need splitting.
    """
    n = len(X)
    floors = [cfg.min_leaf_size]
    # Only relax for nodes so large that leaving them unsplit is the worse
    # outcome. Relaxing everywhere shatters mid-sized nodes into two-document
    # leaves, which are worthless as folders and score a spurious confidence of
    # 1.0 because a centroid over two points sits on top of both.
    if n >= 3 * cfg.min_split_size:
        relaxed = max(4, cfg.min_leaf_size // 2)
        if relaxed < cfg.min_leaf_size:
            floors.append(relaxed)

    for floor in floors:
        k_max = min(cfg.max_branching, n // floor)
        best: tuple[float, np.ndarray] | None = None
        for k in range(2, k_max + 1):
            labels = AgglomerativeClustering(n_clusters=k, linkage="ward").fit_predict(X)
            if np.bincount(labels, minlength=k).min() < floor:
                continue
            score = _silhouette(X, labels)
            if best is None or score > best[0]:
                best = (score, labels)
        if best is not None:
            return best[1]
    return None


def _adaptive_min_cluster_size(n: int, cfg: ClusterConfig) -> int:
    if cfg.min_cluster_size is not None:
        return cfg.min_cluster_size
    return max(cfg.min_leaf_size, n // 25)


def _split_hdbscan(X: np.ndarray, cfg: ClusterConfig) -> tuple[np.ndarray, np.ndarray] | None:
    """HDBSCAN on a UMAP-reduced view of *this node's* documents.

    Refitting the reducer inside each subtree is the important detail: reusing
    the corpus-wide projection destroys the local structure that makes
    sub-topics separable at all.
    """
    from sklearn.cluster import HDBSCAN

    from littax.cluster.reduce import reduce_dims

    n = len(X)
    if n < 2 * cfg.min_leaf_size:
        return None

    view = reduce_dims(X, cfg.reducer, cfg.reduce_components, cfg.random_state)
    min_cluster_size = _adaptive_min_cluster_size(n, cfg)
    model = HDBSCAN(
        min_cluster_size=min_cluster_size,
        # Defaulting min_samples to min_cluster_size makes HDBSCAN very
        # conservative: on a corpus this size it declares most of a node noise
        # and refuses to split at all. Halving it keeps the outlier detection
        # while letting real sub-topics through.
        min_samples=max(3, min_cluster_size // 2),
        metric="euclidean",
    )
    labels = model.fit_predict(view)
    probabilities = getattr(model, "probabilities_", np.ones(n))

    real = labels[labels >= 0]
    if len(set(real.tolist())) < 2:
        return None
    if (labels == -1).mean() > 0.5:
        return None
    return labels, probabilities


def _centroid(X: np.ndarray, indices: list[int]) -> np.ndarray:
    vector = X[indices].mean(axis=0)
    norm = np.linalg.norm(vector)
    return vector / norm if norm else vector


def build_tree(X: np.ndarray, cfg: ClusterConfig) -> Node:
    """Build the taxonomy over embedding matrix `X`. Returns a virtual root."""
    root = Node(id="root", level=0, parent_id=None, doc_indices=list(range(len(X))))
    root.centroid = _centroid(X, root.doc_indices)
    counter = {"n": 0}

    def recurse(node: Node) -> None:
        if node.level >= cfg.max_depth or len(node.doc_indices) < cfg.min_split_size:
            return

        indices = np.asarray(node.doc_indices)
        subset = X[indices]

        membership: dict[int, float] = {}
        noise: set[int] = set()
        labels: np.ndarray | None = None

        if cfg.algorithm == "hdbscan":
            outcome = _split_hdbscan(subset, cfg)
            if outcome is not None:
                labels, probabilities = outcome
                # Fold noise points into the nearest real cluster so every
                # document keeps a home; they stay flagged for the confidence
                # score.
                labels = _absorb_noise(subset, labels)
                for local, global_index in enumerate(indices):
                    membership[int(global_index)] = float(probabilities[local])
                noise = {int(indices[i]) for i in np.flatnonzero(probabilities == 0.0)}
            elif len(subset) < 2 * cfg.min_split_size:
                return  # small enough that "one coherent topic" is a fair answer

        if labels is None:
            # Either agglomerative is the configured algorithm, or HDBSCAN
            # declined to split a node too large to leave as a single folder.
            # A 69-document leaf is not a useful taxonomy node, so Ward gets
            # the last word rather than the hierarchy stalling.
            labels = _split_agglomerative(subset, cfg)
            if labels is None:
                return

        for label in sorted(set(labels.tolist())):
            member_indices = [int(indices[i]) for i in np.flatnonzero(labels == label)]
            counter["n"] += 1
            child = Node(
                id=f"L{node.level + 1}.{counter['n']:03d}",
                level=node.level + 1,
                parent_id=node.id,
                doc_indices=member_indices,
                membership={i: membership[i] for i in member_indices if i in membership},
                was_noise={i for i in member_indices if i in noise},
            )
            child.centroid = _centroid(X, member_indices)
            node.children.append(child)
            recurse(child)

    recurse(root)
    return root


def _absorb_noise(X: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Reassign HDBSCAN noise (-1) to the nearest cluster centroid."""
    noise = np.flatnonzero(labels == -1)
    if not len(noise):
        return labels
    real_labels = sorted({int(x) for x in labels if x >= 0})
    centroids = np.vstack(
        [_centroid(X, np.flatnonzero(labels == label).tolist()) for label in real_labels]
    )
    labels = labels.copy()
    similarity = X[noise] @ centroids.T
    labels[noise] = [real_labels[i] for i in similarity.argmax(axis=1)]
    return labels
