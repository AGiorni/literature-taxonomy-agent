"""Server endpoints, exercised through FastAPI's TestClient (no live port)."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from littax.config import Config
from littax.models import (
    Assignment,
    Document,
    Extraction,
    Folder,
    Stats,
    Taxonomy,
)
from littax.server.app import create_app


@pytest.fixture
def workspace(tmp_path: Path):
    source = tmp_path / "raw"
    source.mkdir()
    (source / "paper.pdf").write_bytes(b"%PDF-1.7 fake")
    (source / "nested").mkdir()
    (source / "nested" / "deep.pdf").write_bytes(b"%PDF-1.7 nested")

    out = tmp_path / "out"
    out.mkdir()
    taxonomy = Taxonomy(
        generated_at=datetime.now(timezone.utc),
        source_root=str(source),
        target_root="organised",
        config={},
        stats=Stats(
            n_documents=2, n_parsed=2, n_degraded=0, n_duplicates=0,
            n_folders=1, n_leaf_folders=1, n_unsorted=0, levels=1,
        ),
        folders=[Folder(id="f1", level=1, parent_id=None, path="Topic A", label="Topic A")],
        documents=[
            Document(
                id="doc_0000", source_path="raw/paper.pdf", relpath="paper.pdf",
                filename="paper.pdf", sha256="a" * 64, size_bytes=13,
                extraction=Extraction(method="pymupdf"),
                assignment=Assignment(
                    folder_id="f1", folder_path="Topic A", confidence=0.8, band="high",
                    similarity_to_centroid=0.7, margin_over_next=0.2,
                ),
            ),
            Document(
                id="doc_0001", source_path="raw/nested/deep.pdf", relpath="nested/deep.pdf",
                filename="deep.pdf", sha256="b" * 64, size_bytes=15,
                extraction=Extraction(method="pymupdf"),
            ),
        ],
    )
    (out / "taxonomy.json").write_text(taxonomy.model_dump_json(), encoding="utf-8")

    cfg = Config(source_root=source, out_dir=out, cache_dir=tmp_path / ".cache")
    return cfg, tmp_path


@pytest.fixture
def client(workspace):
    cfg, _ = workspace
    return TestClient(create_app(cfg))


class TestPages:
    def test_index_serves_html(self, client):
        response = client.get("/")
        assert response.status_code == 200
        assert "littax" in response.text

    def test_config(self, client):
        body = client.get("/api/config").json()
        assert body["embed_backend"] == "bge"
        assert "raw" in body["source_root"]

    def test_state(self, client):
        body = client.get("/api/state").json()
        assert body["stats"]["n_documents"] == 2
        assert body["folders"][0]["label"] == "Topic A"


class TestStateMissing:
    def test_404_before_first_run(self, tmp_path):
        cfg = Config(source_root=tmp_path, out_dir=tmp_path / "empty")
        response = TestClient(create_app(cfg)).get("/api/state")
        assert response.status_code == 404


class TestPdf:
    def test_serves_a_document(self, client):
        response = client.get("/api/doc/doc_0000/pdf")
        assert response.status_code == 200
        assert response.headers["content-type"] == "application/pdf"

    def test_resolves_nested_relpath(self, client):
        """filename alone would not locate a document in a subdirectory."""
        assert client.get("/api/doc/doc_0001/pdf").status_code == 200

    def test_unknown_id_404s(self, client):
        assert client.get("/api/doc/nope/pdf").status_code == 404


class TestBrowse:
    def test_lists_directories(self, client, workspace):
        _, tmp_path = workspace
        body = client.get("/api/browse", params={"path": str(tmp_path / "raw")}).json()
        assert body["pdf_count"] == 1  # only the top level, not nested
        assert "nested" in [d["name"] for d in body["directories"]]

    def test_file_path_falls_back_to_its_parent(self, client, workspace):
        _, tmp_path = workspace
        body = client.get("/api/browse", params={"path": str(tmp_path / "raw" / "paper.pdf")}).json()
        assert body["path"].endswith("raw")


class TestRun:
    def test_rejects_a_non_directory(self, client, workspace):
        _, tmp_path = workspace
        response = client.post("/api/run", json={"source": str(tmp_path / "does-not-exist")})
        assert response.status_code == 400
