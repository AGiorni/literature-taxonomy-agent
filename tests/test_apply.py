"""The apply stage is the only code that writes outside out/, so it gets the
strictest tests: nothing is moved that should not be, and the source corpus is
never disturbed except in `move` mode."""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from pathlib import Path

import pytest

from littax.apply import apply_plan
from littax.models import (
    Assignment,
    Document,
    Extraction,
    Folder,
    MoveInstruction,
    Stats,
    Taxonomy,
)


def build_corpus(tmp_path: Path, names=("alpha.pdf", "beta.pdf")) -> tuple[Path, Taxonomy]:
    source = tmp_path / "raw"
    source.mkdir()
    documents, plan = [], []

    for n, name in enumerate(names):
        path = source / name
        path.write_bytes(f"pdf-content-{n}".encode())
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        documents.append(
            Document(
                id=f"doc_{n:04d}",
                source_path=f"raw/{name}",
                relpath=name,
                filename=name,
                sha256=digest,
                size_bytes=path.stat().st_size,
                title=name,
                extraction=Extraction(method="pymupdf"),
                assignment=Assignment(
                    folder_id="f1",
                    folder_path="Topic A",
                    confidence=0.9 if n == 0 else 0.3,
                    band="high" if n == 0 else "low",
                    similarity_to_centroid=0.8,
                    margin_over_next=0.2,
                ),
            )
        )
        plan.append(
            MoveInstruction(
                source=f"raw/{name}",
                target=f"organised/Topic A/{name}",
                document_id=f"doc_{n:04d}",
                confidence=0.9 if n == 0 else 0.3,
                band="high" if n == 0 else "low",
            )
        )

    taxonomy = Taxonomy(
        generated_at=datetime.now(timezone.utc),
        source_root=str(source),
        target_root="organised",
        config={},
        stats=Stats(
            n_documents=len(names), n_parsed=len(names), n_degraded=0, n_duplicates=0,
            n_folders=1, n_leaf_folders=1, n_unsorted=0, levels=1,
        ),
        folders=[Folder(id="f1", level=1, parent_id=None, path="Topic A", label="Topic A")],
        documents=documents,
        move_plan=plan,
    )
    return source, taxonomy


class TestDryRun:
    def test_writes_nothing(self, tmp_path):
        source, taxonomy = build_corpus(tmp_path)
        target = tmp_path / "organised"
        result = apply_plan(taxonomy, mode="dry-run", target_root=target, log=lambda _: None)
        assert result["would place"] == 2
        assert not target.exists()

    def test_is_the_default(self, tmp_path):
        source, taxonomy = build_corpus(tmp_path)
        target = tmp_path / "organised"
        apply_plan(taxonomy, target_root=target, log=lambda _: None)
        assert not target.exists()


class TestPlacement:
    @pytest.mark.parametrize("mode", ["hardlink", "copy"])
    def test_builds_tree_without_touching_source(self, tmp_path, mode):
        source, taxonomy = build_corpus(tmp_path)
        target = tmp_path / "organised"
        apply_plan(taxonomy, mode=mode, target_root=target, log=lambda _: None)

        assert (target / "Topic A" / "alpha.pdf").exists()
        assert (target / "Topic A" / "beta.pdf").exists()
        # The source corpus must be byte-identical afterwards.
        assert sorted(p.name for p in source.iterdir()) == ["alpha.pdf", "beta.pdf"]
        assert (source / "alpha.pdf").read_bytes() == b"pdf-content-0"

    def test_move_empties_the_source(self, tmp_path):
        source, taxonomy = build_corpus(tmp_path)
        target = tmp_path / "organised"
        apply_plan(taxonomy, mode="move", target_root=target, log=lambda _: None)
        assert list(source.iterdir()) == []
        assert (target / "Topic A" / "alpha.pdf").exists()

    def test_min_confidence_filters(self, tmp_path):
        source, taxonomy = build_corpus(tmp_path)
        target = tmp_path / "organised"
        result = apply_plan(
            taxonomy, mode="copy", min_confidence=0.5, target_root=target, log=lambda _: None
        )
        assert (target / "Topic A" / "alpha.pdf").exists()
        assert not (target / "Topic A" / "beta.pdf").exists()
        assert result["skipped (below --min-confidence)"] == 1

    def test_nested_source_paths(self, tmp_path):
        """A corpus with subdirectories must still be found - filename alone
        is not enough."""
        source, taxonomy = build_corpus(tmp_path, names=("alpha.pdf",))
        nested = source / "2024"
        nested.mkdir()
        (source / "alpha.pdf").rename(nested / "alpha.pdf")
        taxonomy.documents[0].relpath = "2024/alpha.pdf"

        target = tmp_path / "organised"
        result = apply_plan(taxonomy, mode="copy", target_root=target, log=lambda _: None)
        assert result.get("copied") == 1
        assert (target / "Topic A" / "alpha.pdf").exists()


class TestSafety:
    def test_refuses_when_content_changed(self, tmp_path):
        source, taxonomy = build_corpus(tmp_path)
        (source / "alpha.pdf").write_bytes(b"different content entirely")
        target = tmp_path / "organised"
        result = apply_plan(taxonomy, mode="copy", target_root=target, log=lambda _: None)
        assert result["skipped (content changed)"] == 1
        assert not (target / "Topic A" / "alpha.pdf").exists()

    def test_skips_missing_source(self, tmp_path):
        source, taxonomy = build_corpus(tmp_path)
        (source / "alpha.pdf").unlink()
        target = tmp_path / "organised"
        result = apply_plan(taxonomy, mode="copy", target_root=target, log=lambda _: None)
        assert result["skipped (source missing)"] == 1

    def test_never_overwrites(self, tmp_path):
        source, taxonomy = build_corpus(tmp_path)
        target = tmp_path / "organised"
        (target / "Topic A").mkdir(parents=True)
        (target / "Topic A" / "alpha.pdf").write_bytes(b"pre-existing")
        result = apply_plan(taxonomy, mode="copy", target_root=target, log=lambda _: None)
        assert result["skipped (already present)"] == 1
        assert (target / "Topic A" / "alpha.pdf").read_bytes() == b"pre-existing"

    def test_is_idempotent(self, tmp_path):
        source, taxonomy = build_corpus(tmp_path)
        target = tmp_path / "organised"
        apply_plan(taxonomy, mode="copy", target_root=target, log=lambda _: None)
        second = apply_plan(taxonomy, mode="copy", target_root=target, log=lambda _: None)
        assert second["skipped (already present)"] == 2

    def test_rejects_unknown_mode(self, tmp_path):
        _, taxonomy = build_corpus(tmp_path)
        with pytest.raises(ValueError, match="mode must be one of"):
            apply_plan(taxonomy, mode="delete-everything")
