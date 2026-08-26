"""Which optional dependency stacks are installed.

The heavy parts of the pipeline live behind extras, so a run configured for
`bge` embeddings on an install without `--extra ml` fails several minutes in,
inside a worker thread, as a bare ImportError. These probes let both the CLI
and the server ask the question *before* committing to a run.

Import attempts only: no network, no model download, cheap enough to call per
request.
"""

from __future__ import annotations

import importlib.util

from littax.config import Config

# The install command that provides each import, quoted back to the user.
EXTRA_FOR = {
    "sentence_transformers": "uv sync --extra ml",
    "umap": "uv sync --extra ml",
    "pymupdf": "uv sync",
    "anthropic": "uv sync --extra llm",
    "openai": "uv sync --extra llm",
}

# Which SDK each provider needs. "ollama" is deliberately absent: it talks HTTP
# over urllib, so a local model needs no extra at all - which is the whole point
# of offering it.
SDK_FOR_PROVIDER = {"foundry": "anthropic", "azure-openai": "openai"}


def _installed(module: str) -> bool:
    """Whether `module` can be imported, without importing it.

    find_spec keeps this cheap: importing torch to discover that torch exists
    would cost seconds and a few hundred MB of resident memory.
    """
    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ValueError):
        # A partially installed package can raise rather than return None.
        return False


def probe_capabilities() -> dict[str, bool]:
    """Which optional stacks are importable, by module name."""
    return {module: _installed(module) for module in EXTRA_FOR}


def missing_for(cfg: Config) -> list[str]:
    """Install commands this config needs and cannot satisfy.

    Driven by the config rather than by the shipped defaults, so it stays
    correct for someone who has set `tfidf` in their own config.toml.
    """
    needed: list[str] = []

    if not _installed("pymupdf"):
        needed.append(EXTRA_FOR["pymupdf"])
    if cfg.embed.backend != "tfidf" and not _installed("sentence_transformers"):
        needed.append(EXTRA_FOR["sentence_transformers"])
    if cfg.cluster.reducer == "umap" and not _installed("umap"):
        needed.append(EXTRA_FOR["umap"])
    if cfg.label.enabled:
        sdk = SDK_FOR_PROVIDER.get(cfg.label.provider)
        if sdk and not _installed(sdk):
            needed.append(EXTRA_FOR[sdk])

    # One command often fixes several rows; listing it twice reads as noise.
    return list(dict.fromkeys(needed))
