"""Server endpoints, exercised through FastAPI's TestClient (no live port)."""

from __future__ import annotations

import json
import threading
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

    out = source / "littax"
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

    # out_dir stays unset: the project lives beside the PDFs, at <source>/littax.
    cfg = Config(source_root=source, cache_dir=tmp_path / ".cache")
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
        cfg = Config(source_root=tmp_path / "empty")
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


SECRET = "sk-do-not-leak-0123456789"


class Captured(dict):
    """What the pipeline was asked to do, once the worker thread got there."""

    def __init__(self) -> None:
        super().__init__()
        self.called = threading.Event()

    def __getitem__(self, key):
        # /api/run returns as soon as the thread is started, so every read has
        # to wait for it rather than race it.
        assert self.called.wait(timeout=5), "the pipeline was never called"
        return super().__getitem__(key)


@pytest.fixture
def captured(monkeypatch):
    """Run /api/run without a pipeline, keeping what it was asked to do."""
    import littax.pipeline

    seen = Captured()

    def fake_run(cfg, progress=None, credentials=None):
        dict.__setitem__(seen, "cfg", cfg)
        dict.__setitem__(seen, "credentials", credentials)
        seen.called.set()
        raise RuntimeError("stop here - the arguments are what matter")

    monkeypatch.setattr(littax.pipeline, "run", fake_run)
    return seen


class TestProjectDetection:
    def test_browse_flags_analysed_folders(self, client, workspace):
        _, tmp_path = workspace
        body = client.get("/api/browse", params={"path": str(tmp_path)}).json()
        entry = next(d for d in body["directories"] if d["name"] == "raw")
        assert entry["has_taxonomy"] is True
        assert body["has_taxonomy"] is False  # tmp_path itself is not a project

    def test_open_switches_project(self, client, workspace):
        _, tmp_path = workspace
        assert client.post("/api/open", json={"source": str(tmp_path / "raw")}).status_code == 200
        assert client.get("/api/state").json()["stats"]["n_documents"] == 2

    def test_open_404s_on_a_plain_folder(self, client, workspace):
        _, tmp_path = workspace
        response = client.post("/api/open", json={"source": str(tmp_path / "raw" / "nested")})
        assert response.status_code == 404


class TestRunConfiguration:
    def test_standard_mode_leaves_labelling_off(self, client, workspace, captured):
        _, tmp_path = workspace
        client.post("/api/run", json={"source": str(tmp_path / "raw"), "ai_assisted": False})
        assert captured["cfg"].label.enabled is False
        assert captured["credentials"] is None

    def test_ai_mode_passes_credentials_beside_the_config(self, client, workspace, captured):
        _, tmp_path = workspace
        client.post(
            "/api/run",
            json={
                "source": str(tmp_path / "raw"),
                "ai_assisted": True,
                "llm": {
                    "provider": "azure-openai",
                    "model": "gpt-4o",
                    "deployment": "my-deployment",
                    "credentials": {
                        "api_key": SECRET,
                        "endpoint": "https://example.openai.azure.com/",
                    },
                },
            },
        )
        cfg = captured["cfg"]
        assert cfg.label.enabled is True
        assert cfg.label.provider == "azure-openai"
        assert cfg.label.deployment == "my-deployment"
        assert captured["credentials"].api_key == SECRET
        # The one that matters: Config is serialised into taxonomy.json.
        assert SECRET not in cfg.model_dump_json()

    def test_the_browser_cannot_change_the_analysis_path(self, client, workspace, captured):
        """embed/cluster are config-file territory, not UI territory."""
        cfg, tmp_path = workspace
        client.post(
            "/api/run",
            json={
                "source": str(tmp_path / "raw"),
                "ai_assisted": False,
                "embed": {"backend": "tfidf"},
                "cluster": {"reducer": "none"},
            },
        )
        assert captured["cfg"].embed == cfg.embed
        assert captured["cfg"].cluster == cfg.cluster

    def test_writes_beside_the_pdfs(self, client, workspace, captured):
        _, tmp_path = workspace
        client.post("/api/run", json={"source": str(tmp_path / "raw"), "ai_assisted": False})
        assert captured["cfg"].out == tmp_path / "raw" / "littax"


class TestMissingExtras:
    def test_run_400s_instead_of_failing_in_the_worker(self, client, workspace, monkeypatch):
        import littax.server.app as server

        monkeypatch.setattr(server, "missing_for", lambda cfg: ["uv sync --extra ml"])
        _, tmp_path = workspace
        response = client.post(
            "/api/run", json={"source": str(tmp_path / "raw"), "ai_assisted": False}
        )
        assert response.status_code == 400
        assert "uv sync --extra ml" in response.json()["detail"]


class TestProviders:
    def test_describes_every_provider_without_secrets(self, client):
        body = client.get("/api/providers").json()
        ids = {p["id"] for p in body["providers"]}
        assert ids == {"foundry", "azure-openai", "ollama"}
        azure = next(p for p in body["providers"] if p["id"] == "azure-openai")
        assert azure["needs_deployment"] is True
        assert {f["id"] for f in azure["fields"]} == {"api_key", "endpoint"}
        assert next(f for f in azure["fields"] if f["id"] == "api_key")["secret"] is True

    def test_ollama_asks_for_no_credentials(self, client):
        """The point of the local provider: nothing to sign up for."""
        body = client.get("/api/providers").json()
        ollama = next(p for p in body["providers"] if p["id"] == "ollama")
        assert ollama["fields"] == []
        assert ollama["local"] is True
        # No SDK and no key, so it is never gated behind an install or a login.
        assert ollama["sdk_installed"] is True
        assert ollama["env_ready"] is True
        assert ollama["base_url"]["default"].startswith("http://")

    def test_config_reports_key_names_only(self, client, monkeypatch):
        monkeypatch.setenv("AZURE_OPENAI_API_KEY", SECRET)
        monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://example.openai.azure.com/")
        body = client.get("/api/config").text
        assert "AZURE_OPENAI_API_KEY" in body
        assert SECRET not in body
