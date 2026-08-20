"""Assign documents to leaf folders and score how well they fit.

Every similarity here is computed in the *original* embedding space, never in
the reduced one: UMAP distances are not metrically meaningful, so a confidence
score derived from them would look precise and mean nothing.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from littax.config import ScoreConfig
from littax.cluster.tree import Node, leaves
from littax.models import UNSORTED_ID


@dataclass
class Calibration:
    """Corpus-derived scale for turning raw cosines into a 0-1 score.

    Absolute cosine values are not comparable across embedding backends: TF-IDF
    centroid similarities cluster around 0.4 where a sentence transformer's sit
    near 0.8, so a fixed threshold would call every TF-IDF assignment "low" and
    every transformer assignment "high" regardless of actual fit quality.

    Calibrating against the corpus instead makes confidence mean something
    stable: `chance` is how similar a document is to a cluster it does *not*
    belong to, and `ceiling` is what a well-placed document in this corpus
    actually achieves. Confidence is therefore relative - "how good is this fit
    compared to what is achievable here" - not an absolute probability.
    """

    chance: float
    ceiling: float
    margin_scale: float

    @property
    def span(self) -> float:
        return max(self.ceiling - self.chance, 1e-6)

    def scale_similarity(self, value: float) -> float:
        return float(np.clip((value - self.chance) / self.span, 0.0, 1.0))

    def scale_margin(self, value: float) -> float:
        return float(np.clip(value / self.margin_scale, 0.0, 1.0))

    def as_dict(self) -> dict[str, float]:
        return {
            "chance_similarity": round(self.chance, 4),
            "ceiling_similarity": round(self.ceiling, 4),
            "margin_scale": round(self.margin_scale, 4),
        }


def calibrate(similarity: np.ndarray, owner_column: np.ndarray) -> Calibration:
    """Derive the score scale from the corpus's own similarity distribution."""
    rows = np.arange(len(similarity))
    own = similarity[rows, owner_column]

    # With a single cluster there is no "other" to compare against, so the
    # calibration is undefined and the raw cosine is the honest fallback.
    if similarity.shape[1] < 2:
        return Calibration(chance=0.0, ceiling=1.0, margin_scale=0.25)

    others = similarity.copy()
    others[rows, owner_column] = np.nan
    chance = float(np.nanmedian(others))

    ceiling = float(np.percentile(own, 90))
    if ceiling <= chance:  # degenerate corpus - fall back to the raw cosine
        chance, ceiling = 0.0, 1.0

    margins = own - np.nanmax(others, axis=1)
    margin_scale = float(np.percentile(margins, 90))
    if not np.isfinite(margin_scale) or margin_scale <= 0:
        margin_scale = 0.25

    return Calibration(chance=chance, ceiling=ceiling, margin_scale=margin_scale)


@dataclass
class DocAssignment:
    folder_id: str
    similarity: float
    margin: float
    membership: float | None
    confidence: float
    band: str
    runner_up_id: str | None
    runner_up_similarity: float
    was_noise: bool


@dataclass
class AssignmentResult:
    assignments: dict[int, DocAssignment]
    # Documents that matched no cluster well enough to place.
    unsorted: list[int] = field(default_factory=list)
    near_duplicates: dict[int, list[int]] = field(default_factory=dict)
    calibration: Calibration | None = None


def _band(confidence: float, cfg: ScoreConfig) -> str:
    if confidence >= cfg.band_high:
        return "high"
    if confidence >= cfg.band_medium:
        return "medium"
    return "low"


def combine_confidence(
    similarity: float,
    margin: float,
    membership: float | None,
    cfg: ScoreConfig,
    calibration: Calibration,
) -> float:
    """Weighted blend of the fit signals, on 0-1.

    Components that the clustering algorithm cannot supply (agglomerative has no
    membership probability) are dropped and the remaining weights renormalised,
    so the score stays comparable across algorithms.
    """
    parts = [
        (cfg.w_similarity, calibration.scale_similarity(similarity)),
        (cfg.w_margin, calibration.scale_margin(margin)),
    ]
    if membership is not None:
        parts.append((cfg.w_membership, float(np.clip(membership, 0.0, 1.0))))

    total = sum(weight for weight, _ in parts)
    if total <= 0:
        return 0.0
    return sum(weight * value for weight, value in parts) / total


def assign_documents(X: np.ndarray, root: Node, cfg: ScoreConfig) -> AssignmentResult:
    leaf_nodes = leaves(root)
    if not leaf_nodes:
        return AssignmentResult(assignments={})

    leaf_ids = [node.id for node in leaf_nodes]
    centroids = np.vstack([node.centroid for node in leaf_nodes])

    # Which leaf each document actually belongs to, from the tree structure.
    owner: dict[int, int] = {}
    membership: dict[int, float] = {}
    noise: set[int] = set()
    for position, node in enumerate(leaf_nodes):
        for index in node.doc_indices:
            owner[index] = position
        membership.update(node.membership)
        noise |= node.was_noise

    similarity = X @ centroids.T  # rows are unit-norm, so this is cosine
    owner_column = np.array([owner[i] for i in range(len(X))])
    calibration = calibrate(similarity, owner_column)
    result = AssignmentResult(assignments={}, calibration=calibration)

    for index in range(len(X)):
        position = owner[index]
        own = float(similarity[index, position])

        # The runner-up is the best *other* leaf anywhere in the tree, which
        # makes it a genuinely useful "where else could this have gone".
        others = np.delete(similarity[index], position)
        if others.size:
            best_other = int(np.argmax(others))
            runner_position = best_other if best_other < position else best_other + 1
            runner_similarity = float(similarity[index, runner_position])
            runner_id = leaf_ids[runner_position]
        else:
            runner_similarity, runner_id = 0.0, None

        margin = own - runner_similarity
        strength = membership.get(index)
        confidence = combine_confidence(own, margin, strength, cfg, calibration)

        # Compared on the calibrated scale, so the threshold means the same
        # thing whichever embedding backend produced the vectors.
        if calibration.scale_similarity(own) < cfg.unsorted_threshold:
            folder_id = UNSORTED_ID
            result.unsorted.append(index)
        else:
            folder_id = leaf_ids[position]

        result.assignments[index] = DocAssignment(
            folder_id=folder_id,
            similarity=own,
            margin=margin,
            membership=strength,
            confidence=confidence,
            band=_band(confidence, cfg),
            runner_up_id=runner_id,
            runner_up_similarity=runner_similarity,
            was_noise=index in noise,
        )

    result.near_duplicates = find_near_duplicates(X, cfg.duplicate_threshold)
    return result


def find_near_duplicates(X: np.ndarray, threshold: float) -> dict[int, list[int]]:
    """Documents whose embeddings are almost identical - usually the same paper
    saved twice under different filenames."""
    similarity = X @ X.T
    np.fill_diagonal(similarity, -1.0)
    pairs = np.argwhere(similarity >= threshold)
    out: dict[int, list[int]] = {}
    for i, j in pairs:
        out.setdefault(int(i), []).append(int(j))
    return out


def confidence_stats(values: list[float]) -> dict[str, float]:
    if not values:
        return {"mean": 0.0, "median": 0.0, "p10": 0.0, "p90": 0.0}
    array = np.asarray(values, dtype=float)
    return {
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "p10": float(np.percentile(array, 10)),
        "p90": float(np.percentile(array, 90)),
    }
