"""Preflight: what a config needs that this machine cannot provide."""

from __future__ import annotations

import pytest

from littax.capabilities import missing_for, probe_capabilities
from littax.config import Config


@pytest.fixture
def absent(monkeypatch):
    """Pretend a module is not installed, without uninstalling it."""

    def hide(*names: str) -> None:
        real = __import__("importlib.util", fromlist=["util"]).find_spec

        def find_spec(name, *args, **kwargs):
            return None if name in names else real(name, *args, **kwargs)

        monkeypatch.setattr("importlib.util.find_spec", find_spec)

    return hide


class TestProbe:
    def test_reports_every_known_module(self):
        caps = probe_capabilities()
        assert set(caps) >= {"sentence_transformers", "umap", "pymupdf"}
        assert all(isinstance(v, bool) for v in caps.values())


class TestMissingFor:
    def test_clean_when_everything_is_installed(self):
        assert missing_for(Config()) == []

    def test_names_the_ml_extra_for_bge(self, absent):
        absent("sentence_transformers")
        assert missing_for(Config()) == ["uv sync --extra ml"]

    def test_reads_the_config_not_the_defaults(self, absent):
        """A tfidf/none config does not need torch, whatever the default is."""
        absent("sentence_transformers", "umap")
        cfg = Config()
        cfg.embed.backend = "tfidf"
        cfg.cluster.reducer = "none"
        assert missing_for(cfg) == []

    def test_one_command_is_not_listed_twice(self, absent):
        absent("sentence_transformers", "umap")
        assert missing_for(Config()) == ["uv sync --extra ml"]

    def test_llm_sdk_only_when_labelling_is_on(self, absent):
        absent("openai")
        cfg = Config()
        cfg.label.provider = "azure-openai"

        cfg.label.enabled = False
        assert missing_for(cfg) == []

        cfg.label.enabled = True
        assert missing_for(cfg) == ["uv sync --extra llm"]
