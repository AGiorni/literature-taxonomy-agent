"""Clustering, confidence maths and naming, on synthetic data with known geometry."""

from __future__ import annotations

import numpy as np
import pytest

from littax.cluster.assign import Calibration, assign_documents, calibrate, combine_confidence
from littax.cluster.terms import is_junk_term, node_terms
from littax.cluster.tree import build_tree, iter_nodes, leaves
from littax.config import ClusterConfig, ScoreConfig, TermsConfig
from littax.export.naming import placeholder_label, safe_folder_name
from littax.export.move_plan import build_move_plan
from littax.models import Assignment, Document, Extraction, Folder


def blobs(n_groups: int = 4, per_group: int = 30, dim: int = 20, spread: float = 0.05):
    """Well-separated unit-norm clusters: the case clustering must get right."""
    rng = np.random.default_rng(0)
    centres = rng.normal(size=(n_groups, dim))
    centres /= np.linalg.norm(centres, axis=1, keepdims=True)
    rows = []
    for centre in centres:
        noise = rng.normal(scale=spread, size=(per_group, dim))
        block = centre + noise
        rows.append(block / np.linalg.norm(block, axis=1, keepdims=True))
    return np.vstack(rows).astype(np.float32)


class TestTree:
    def test_recovers_known_groups(self):
        X = blobs(n_groups=4, per_group=30)
        root = build_tree(X, ClusterConfig(max_depth=1, min_leaf_size=5, min_split_size=10))
        assert len(root.children) == 4
        # Each discovered cluster should be exactly one of the planted blobs.
        for child in root.children:
            assert {i // 30 for i in child.doc_indices} == {child.doc_indices[0] // 30}

    def test_respects_max_depth(self):
        X = blobs(n_groups=4, per_group=40)
        root = build_tree(X, ClusterConfig(max_depth=2, min_leaf_size=4, min_split_size=10))
        assert max(n.level for n in iter_nodes(root)) <= 2

    def test_small_node_stays_a_leaf(self):
        X = blobs(n_groups=2, per_group=5)
        root = build_tree(X, ClusterConfig(min_split_size=50))
        assert root.children == []

    def test_every_document_lands_in_exactly_one_leaf(self):
        X = blobs(n_groups=3, per_group=25)
        root = build_tree(X, ClusterConfig(max_depth=3, min_leaf_size=4, min_split_size=10))
        placed = [i for leaf in leaves(root) for i in leaf.doc_indices]
        assert sorted(placed) == list(range(len(X)))

    def test_does_not_produce_tiny_leaves(self):
        """The relaxed size floor must not shatter nodes into 2-document leaves."""
        X = blobs(n_groups=6, per_group=30)
        cfg = ClusterConfig(max_depth=3, min_leaf_size=8, min_split_size=25)
        root = build_tree(X, cfg)
        assert min(len(leaf.doc_indices) for leaf in leaves(root)) >= 4


class TestConfidence:
    def _calibration(self):
        return Calibration(chance=0.0, ceiling=1.0, margin_scale=0.25)

    def test_perfect_fit_scores_high(self):
        score = combine_confidence(1.0, 0.5, 1.0, ScoreConfig(), self._calibration())
        assert score == pytest.approx(1.0)

    def test_ambiguous_document_scores_low(self):
        """Sits between two clusters: decent similarity, no margin."""
        score = combine_confidence(0.5, 0.0, None, ScoreConfig(), self._calibration())
        assert score < 0.45

    def test_missing_membership_renormalises_weights(self):
        cfg = ScoreConfig()
        calibration = self._calibration()
        with_membership = combine_confidence(1.0, 0.5, 1.0, cfg, calibration)
        without = combine_confidence(1.0, 0.5, None, cfg, calibration)
        # Dropping a component must not depress the score of a perfect fit.
        assert with_membership == pytest.approx(without)

    def test_calibration_rescales_backend_specific_similarities(self):
        """A 0.45 cosine under TF-IDF should score like a 0.85 under a transformer."""
        tfidf = Calibration(chance=0.10, ceiling=0.50, margin_scale=0.2)
        transformer = Calibration(chance=0.50, ceiling=0.90, margin_scale=0.2)
        assert tfidf.scale_similarity(0.45) == pytest.approx(
            transformer.scale_similarity(0.85)
        )

    def test_calibration_handles_degenerate_corpus(self):
        similarity = np.full((5, 1), 0.4)
        calibration = calibrate(similarity, np.zeros(5, dtype=int))
        assert 0.0 <= calibration.scale_similarity(0.4) <= 1.0


class TestAssignment:
    def test_clear_membership_beats_ambiguous(self):
        X = blobs(n_groups=3, per_group=20, spread=0.02)
        root = build_tree(X, ClusterConfig(max_depth=1, min_leaf_size=5, min_split_size=10))
        result = assign_documents(X, root, ScoreConfig())
        assert len(result.assignments) == len(X)

        # A point placed exactly on a centroid must outscore the corpus median.
        confidences = [a.confidence for a in result.assignments.values()]
        assert max(confidences) > np.median(confidences)

    def test_runner_up_is_recorded(self):
        X = blobs(n_groups=3, per_group=20)
        root = build_tree(X, ClusterConfig(max_depth=1, min_leaf_size=5, min_split_size=10))
        result = assign_documents(X, root, ScoreConfig())
        for fit in result.assignments.values():
            assert fit.runner_up_id != fit.folder_id
            assert fit.similarity >= fit.runner_up_similarity

    def test_near_duplicates_detected(self):
        X = blobs(n_groups=2, per_group=10)
        X = np.vstack([X, X[0:1]])  # exact copy of row 0
        root = build_tree(X, ClusterConfig(max_depth=1, min_leaf_size=4, min_split_size=8))
        result = assign_documents(X, root, ScoreConfig())
        assert len(X) - 1 in result.near_duplicates.get(0, [])


class TestTerms:
    def test_surfaces_discriminative_not_common_terms(self):
        groups = {"a": [0, 1], "b": [2, 3]}
        texts = [
            "muscle motor cortex stimulation magnetic",
            "muscle motor cortex stimulation magnetic",
            "muscle motor shoulder impingement rotator",
            "muscle motor shoulder impingement rotator",
        ]
        terms = node_terms(texts, groups, TermsConfig(top_n=3, ngram_max=1))
        # "muscle"/"motor" appear in both groups, so must not distinguish them.
        assert "shoulder" in terms["b"] or "impingement" in terms["b"]
        assert "muscle" not in terms["a"]

    @pytest.mark.parametrize(
        "term", ["a1111111111", "xxxx", "2019", "10 1152", "ab", "rhythm123456"]
    )
    def test_rejects_junk(self, term):
        assert is_junk_term(term)

    @pytest.mark.parametrize("term", ["motor unit", "emg", "subacromial impingement", "tms"])
    def test_keeps_real_terms(self, term):
        assert not is_junk_term(term)


class TestNaming:
    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("Motor Unit: Firing/Rate", "Motor Unit Firing Rate"),
            ("trailing dots...", "trailing dots"),
            ("  spaced  out  ", "spaced out"),
            ("", "topic"),
            ('bad<>:"|?*chars', "bad chars"),
        ],
    )
    def test_sanitises(self, raw, expected):
        assert safe_folder_name(raw) == expected

    @pytest.mark.parametrize("reserved", ["CON", "PRN", "aux", "COM1", "NUL"])
    def test_windows_reserved_names_get_suffixed(self, reserved):
        assert safe_folder_name(reserved).endswith("_")

    def test_truncates_on_word_boundary(self):
        name = safe_folder_name(" ".join(["verylongword"] * 20))
        assert len(name) <= 70
        assert not name.endswith("verylongwor")

    def test_placeholder_drops_overlapping_terms(self):
        label = placeholder_label(["motor unit firing", "motor unit", "emg", "discharge"])
        assert label == "motor unit firing, emg, discharge"


class TestMovePlan:
    def _document(self, doc_id: str, filename: str, folder_id: str) -> Document:
        return Document(
            id=doc_id,
            source_path=f"raw/{filename}",
            filename=filename,
            sha256=doc_id * 8,
            size_bytes=1,
            extraction=Extraction(method="pymupdf"),
            assignment=Assignment(
                folder_id=folder_id,
                folder_path="Topic A",
                confidence=0.9,
                band="high",
                similarity_to_centroid=0.8,
                margin_over_next=0.2,
            ),
        )

    def _folders(self):
        return {"f1": Folder(id="f1", level=1, parent_id=None, path="Topic A", label="Topic A")}

    def test_builds_target_paths(self):
        plan = build_move_plan(
            [self._document("d1", "paper.pdf", "f1")], self._folders(), "organised"
        )
        assert plan[0].target == "organised/Topic A/paper.pdf"

    def test_resolves_filename_collisions(self):
        documents = [
            self._document("d1", "paper.pdf", "f1"),
            self._document("d2", "paper.pdf", "f1"),
        ]
        plan = build_move_plan(documents, self._folders(), "organised")
        assert plan[1].target == "organised/Topic A/paper (2).pdf"

    def test_skips_byte_duplicates(self):
        duplicate = self._document("d2", "copy.pdf", "f1")
        duplicate.duplicate_of = "paper.pdf"
        plan = build_move_plan(
            [self._document("d1", "paper.pdf", "f1"), duplicate], self._folders(), "organised"
        )
        assert len(plan) == 1

    def test_no_path_traversal_from_hostile_label(self):
        folders = {
            "f1": Folder(
                id="f1",
                level=1,
                parent_id=None,
                path=safe_folder_name("../../etc"),
                label="x",
            )
        }
        plan = build_move_plan([self._document("d1", "p.pdf", "f1")], folders, "organised")
        assert ".." not in plan[0].target
