"""Progress reporting.

The pipeline's longest stages used to report nothing until they finished, so a
healthy run looked hung: the bar sat at 0% for minutes. These tests pin the two
properties that fixed it - that encoding reports as it goes, and that every
stage says when it is done.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from littax.config import Config, EmbedConfig
from littax.embed.encoder import (
    MAX_CHUNK,
    MIN_CHUNK,
    PROGRESS_UPDATES,
    TfidfEncoder,
    progress_chunk,
)
from littax.pipeline import run

CORPUS_SIZES = [1, 12, 60, 376, 2_000, 10_000]


class TestProgressChunk:
    @pytest.mark.parametrize("n", CORPUS_SIZES)
    def test_stays_within_bounds(self, n):
        assert MIN_CHUNK <= progress_chunk(n) <= MAX_CHUNK

    @pytest.mark.parametrize("n", [n for n in CORPUS_SIZES if n >= MIN_CHUNK * 2])
    def test_updates_often_enough_to_look_alive(self, n):
        """A fixed chunk either updates twice on a small corpus or floods a
        large one; the point of sizing it is that neither happens."""
        updates = math.ceil(n / progress_chunk(n))
        assert 2 <= updates
        # Only a corpus past MAX_CHUNK * PROGRESS_UPDATES may exceed the target.
        if n <= MAX_CHUNK * PROGRESS_UPDATES:
            assert updates <= PROGRESS_UPDATES + 1

    def test_grows_with_the_corpus(self):
        sizes = [progress_chunk(n) for n in CORPUS_SIZES]
        assert sizes == sorted(sizes)


def sample_texts(n: int = 12) -> list[str]:
    """Two overlapping topics, so tf-idf has a vocabulary to work with."""
    topics = [
        "motor unit discharge rate firing recruitment threshold",
        "corticospinal excitability transcranial stimulation evoked potential",
    ]
    return [f"{topics[i % 2]} study {i // 2}" for i in range(n)]


# min_df/max_df defaults assume a real corpus; a dozen synthetic documents
# would leave the vocabulary empty.
TFIDF = EmbedConfig(backend="tfidf", min_df=1, max_df=1.0, svd_components=4)


class TestEncoderProgress:
    def test_reports_completion(self):
        """tfidf fits in one pass, but must still close the stage out."""
        seen: list[int] = []
        texts = sample_texts()
        TfidfEncoder(TFIDF).encode(texts, seen.append)
        assert seen and seen[-1] == len(texts)

    def test_is_optional(self):
        texts = sample_texts()
        vectors = TfidfEncoder(TFIDF).encode(texts)
        assert vectors.shape[0] == len(texts)


class TestPipelineStages:
    """Every stage must reach 100%, or the bar stalls part-way and stays there."""

    def test_each_stage_completes(self, tmp_path, monkeypatch):
        source = tmp_path / "corpus"
        source.mkdir()
        for i in range(6):
            # Distinct bytes: identical files are deduplicated, which would
            # leave one document and nothing to cluster.
            (source / f"paper{i}.pdf").write_bytes(f"%PDF-1.7 {i}".encode())

        # Real parsing needs real PDFs; the stage sequence is what matters here.
        import littax.pipeline as pipeline

        bodies = sample_texts(6)

        def fake_extract(path):
            from littax.ingest.parse import RawText

            body = bodies[int(path.stem[-1])] + " "
            return RawText(text=body * 40, pages=3, chars=900)

        monkeypatch.setattr(pipeline, "extract_pdf", fake_extract)

        cfg = Config(source_root=source, cache_dir=tmp_path / ".cache")
        cfg.embed = TFIDF.model_copy()
        cfg.cluster.reducer = "none"
        cfg.cluster.algorithm = "agglomerative"
        cfg.cluster.min_split_size = 2
        cfg.cluster.min_leaf_size = 1
        cfg.label.enabled = False

        frames: list[tuple[str, int, int]] = []
        run(cfg, lambda stage, done, total, detail: frames.append((stage, done, total)))

        by_stage: dict[str, list[tuple[int, int]]] = {}
        for stage, done, total in frames:
            by_stage.setdefault(stage, []).append((done, total))

        # "scan" is a single announcement before any work exists to measure.
        for stage, points in by_stage.items():
            if stage == "scan":
                continue
            done, total = points[-1]
            assert done == total, f"{stage} never reported completion: {points}"

