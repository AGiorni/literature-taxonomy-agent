"""Configuration. Loaded from a TOML file; secrets come from the environment."""

from __future__ import annotations

import os
import tomllib
from pathlib import Path
from typing import ClassVar, Literal
from urllib.parse import urlparse

from pydantic import BaseModel, Field

# A corpus folder becomes a "littax project" by gaining this subdirectory.
# Keeping the results beside the PDFs means a folder carries its own taxonomy:
# copy or move it and the analysis travels with it.
PROJECT_DIR = "littax"


def has_project(source: Path) -> bool:
    """Whether `source` has already been analysed. The single definition."""
    return (source / PROJECT_DIR / "taxonomy.json").is_file()


class IngestConfig(BaseModel):
    glob: str = "**/*.pdf"
    # A PDF yielding fewer than this many characters per page is treated as
    # scanned/image-only: flagged degraded, profiled from its filename alone.
    degraded_chars_per_page: int = 100
    abstract_max_chars: int = 3000
    # When no abstract is found, how many body words to use instead.
    profile_body_words: int = 400


class EmbedConfig(BaseModel):
    # bge wins clearly over tfidf on a real corpus - it recognises that
    # "corticospinal excitability" and "MEP amplitude" are the same subject,
    # which no bag-of-words model can. Needs `uv sync --extra ml`; set this to
    # "tfidf" for a zero-download run.
    backend: Literal["tfidf", "bge", "specter"] = "bge"
    # Overrides the backend's default HuggingFace model id. Ignored by "tfidf".
    model_name: str | None = None
    # "tfidf" backend only.
    svd_components: int = 256
    max_features: int = 50_000
    min_df: int = 2
    max_df: float = 0.5


class ClusterConfig(BaseModel):
    # hdbscan finds natural groups and flags outliers, falling back to Ward for
    # nodes it declines to split. Set both to "agglomerative"/"none" for a
    # deterministic, dependency-free run.
    algorithm: Literal["agglomerative", "hdbscan"] = "hdbscan"
    reducer: Literal["none", "pca", "umap"] = "umap"
    reduce_components: int = 10

    max_depth: int = 3
    # Nodes smaller than this are never split further.
    min_split_size: int = 25
    # A child cluster smaller than this is not allowed; constrains the branching search.
    min_leaf_size: int = 8
    # Upper bound on children per node. The actual count is chosen by silhouette.
    max_branching: int = 8
    # "hdbscan" only; None means adaptive (max(min_leaf_size, n // 25)).
    min_cluster_size: int | None = None
    random_state: int = 42


class ScoreConfig(BaseModel):
    """Weights for the combined confidence score.

    Components that are unavailable for a given algorithm (e.g. membership
    probability under agglomerative clustering) are dropped and the remaining
    weights renormalised, so the score stays on 0-1 either way.
    """

    w_similarity: float = 0.5
    w_margin: float = 0.3
    w_membership: float = 0.2

    band_high: float = 0.70
    band_medium: float = 0.45

    # Below this cosine similarity to every candidate centroid, a document goes
    # to _Unsorted rather than being forced into the least-bad folder.
    unsorted_threshold: float = 0.15
    # Cosine above which two documents are reported as near-duplicates.
    duplicate_threshold: float = 0.98


class TermsConfig(BaseModel):
    top_n: int = 25
    max_features: int = 20_000
    ngram_max: int = 2


class LabelConfig(BaseModel):
    """Phase 3. Disabled by default - folders get placeholder names from c-TF-IDF terms."""

    enabled: bool = False
    # "foundry"      -> Claude on Microsoft Foundry (Anthropic SDK)
    # "azure-openai" -> GPT models on Azure OpenAI (openai SDK)
    # "ollama"       -> a model running locally; free, no account, no SDK
    provider: Literal["foundry", "azure-openai", "ollama"] = "foundry"
    model: str = "claude-opus-5"
    # Azure OpenAI addresses a *deployment*, whose name is chosen at deploy time
    # and need not match the model id. Falls back to `model` when unset.
    deployment: str | None = None
    # Where a self-hosted provider listens. Unlike an api_key this is not a
    # secret, so it belongs in the config and is welcome in taxonomy.json as
    # provenance. "ollama" only; falls back to $OLLAMA_HOST then localhost.
    base_url: str | None = None
    api_version: str = "2024-10-21"
    max_workers: int = 6
    max_retries: int = 5
    representative_docs: int = 12
    coherence_pass: bool = True
    # Re-check documents whose band is "low" against their candidate folders.
    verify_low_confidence: bool = True


class Config(BaseModel):
    source_root: Path = Path("raw")
    # Unset means "beside the PDFs", at <source_root>/littax - so analysing a
    # second folder cannot overwrite the first. Read it through `.out`, never
    # directly; set it only to override that.
    out_dir: Path | None = None
    # Deliberately *not* per-project: the parsed and LLM caches are keyed by
    # file hash, so sharing one across corpora is a saving, not an oversight.
    cache_dir: Path = Path(".cache")
    # Root of the proposed reorganisation, used to build move_plan targets.
    target_root: Path = Path("organised")

    ingest: IngestConfig = Field(default_factory=IngestConfig)
    embed: EmbedConfig = Field(default_factory=EmbedConfig)
    cluster: ClusterConfig = Field(default_factory=ClusterConfig)
    score: ScoreConfig = Field(default_factory=ScoreConfig)
    terms: TermsConfig = Field(default_factory=TermsConfig)
    label: LabelConfig = Field(default_factory=LabelConfig)

    @property
    def out(self) -> Path:
        """Where taxonomy.json lives: per-corpus unless out_dir was set."""
        return self.out_dir if self.out_dir is not None else self.source_root / PROJECT_DIR

    @classmethod
    def load(cls, path: Path | None = None) -> "Config":
        """Load from TOML, falling back to config.toml then all-defaults."""
        if path is None:
            default = Path("config.toml")
            if not default.exists():
                return cls()
            path = default
        with open(path, "rb") as fh:
            return cls.model_validate(tomllib.load(fh))


class FoundryCredentials(BaseModel):
    api_key: str
    resource: str

    ENV_VARS: ClassVar[tuple[str, str]] = (
        "ANTHROPIC_FOUNDRY_API_KEY",
        "ANTHROPIC_FOUNDRY_RESOURCE",
    )

    @classmethod
    def from_env(cls) -> "FoundryCredentials | None":
        key, resource = (os.environ.get(name) for name in cls.ENV_VARS)
        if not key or not resource:
            return None
        return cls(api_key=key, resource=resource)


def azure_endpoint_problem(endpoint: str) -> str | None:
    """Explain why an endpoint is not usable, or None if it is fine.

    The SDK builds its own `/openai/deployments/...` path, so it needs the
    resource *base* URL. The Azure portal prominently shows a full operation
    URL instead (`.../openai/v1/responses`), and pasting that yields a 404
    reading "Resource not found" - which points at the deployment rather than
    the endpoint, and sends you looking in the wrong place.

    This validates rather than rewrites: quietly editing configuration means
    the value in the file is not the value in use, which is worse than an
    error that names the fix.
    """
    endpoint = endpoint.strip()
    if not endpoint:
        return "AZURE_OPENAI_ENDPOINT is empty."

    parsed = urlparse(endpoint)
    if not parsed.scheme or not parsed.netloc:
        return (
            f"AZURE_OPENAI_ENDPOINT must be a full URL such as "
            f"https://<resource>.openai.azure.com/ — got {endpoint!r}"
        )

    base = f"{parsed.scheme}://{parsed.netloc}/"
    if parsed.query:
        return (
            f"AZURE_OPENAI_ENDPOINT must not include a query string. "
            f"Set it to {base}"
        )
    if parsed.path.strip("/"):
        return (
            f"AZURE_OPENAI_ENDPOINT must be the resource base URL, not an "
            f"operation URL. Drop the '{parsed.path}' path and set it to {base}"
        )
    return None


class AzureOpenAICredentials(BaseModel):
    api_key: str
    endpoint: str

    ENV_VARS: ClassVar[tuple[str, str]] = (
        "AZURE_OPENAI_API_KEY",
        "AZURE_OPENAI_ENDPOINT",
    )

    @property
    def problem(self) -> str | None:
        return azure_endpoint_problem(self.endpoint)

    @classmethod
    def from_env(cls) -> "AzureOpenAICredentials | None":
        key, endpoint = (os.environ.get(name) for name in cls.ENV_VARS)
        if not key or not endpoint:
            return None
        return cls(api_key=key, endpoint=endpoint.strip())


def credentials_for(provider: str):
    """The credential class a provider needs, whether or not it is satisfied.

    None means the provider needs no credentials at all - a local model has
    nothing to authenticate against.
    """
    if provider == "ollama":
        return None
    return FoundryCredentials if provider == "foundry" else AzureOpenAICredentials
