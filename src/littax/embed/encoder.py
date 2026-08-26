"""Turn profile texts into L2-normalised vectors.

Backends are interchangeable so the rest of the pipeline never learns which one
produced the matrix. Everything downstream assumes unit-norm rows, which makes
cosine similarity a plain dot product.
"""

from __future__ import annotations

import math
from typing import Callable, Protocol

import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer

from littax.config import EmbedConfig

# Domain-generic academic filler. These dominate TF-IDF across any corpus of
# papers while saying nothing about topic, so they earn their place in the stop
# list alongside sklearn's English defaults.
_ACADEMIC_FILLER = """
    study studies result results method methods conclusion conclusions
    background objective objectives purpose aim aims introduction discussion
    significant significantly increase increased decrease decreased
    participant participants subject subjects patient patients group groups
    control controls data analysis analyses measure measured measurement
    effect effects research paper article review present presented
    used using use investigate investigated examine examined assess assessed
    demonstrate demonstrated suggest suggests observed found finding findings
    performed protocol trial trials session sessions test tested mean values
"""

# Publisher and front-matter boilerplate. Unlike the filler above these are not
# merely uninformative - they are actively misleading, because they appear in
# only a subset of papers and so score *highly* under c-TF-IDF, producing
# folder names like "preprint, biorxiv, pone".
_PUBLISHER_BOILERPLATE = """
    preprint biorxiv medrxiv arxiv plos pone pmc pubmed elsevier springer
    wiley frontiers sage taylor francis hindawi mdpi bmc lippincott
    copyright license licence creative commons rights reserved permission
    grant grants sponsor sponsors funding funded contract acknowledgement
    acknowledgements acknowledgments supplementary appendix figure figures
    table tables author authors manuscript journal editor editors issn isbn
    volume issue doi http https www com org pdf download downloaded
    correspondence email affiliation university department institute
    published publication received accepted revised submitted online
    et al ieee elsevier ltd inc reprints
"""

ACADEMIC_STOPWORDS = frozenset(
    (_ACADEMIC_FILLER + _PUBLISHER_BOILERPLATE).split()
)


# Roughly how many times a corpus should report progress while encoding. A
# fixed chunk size cannot serve both a 40-paper folder and a 4,000-paper one:
# it either updates twice or floods. Aiming for a number of updates instead
# keeps the bar moving at about the same rate whatever the corpus.
PROGRESS_UPDATES = 30
MIN_CHUNK, MAX_CHUNK = 8, 64


def progress_chunk(n_texts: int) -> int:
    """Texts to encode between progress reports."""
    return max(MIN_CHUNK, min(MAX_CHUNK, math.ceil(n_texts / PROGRESS_UPDATES)))

# Called with the number of texts finished so far.
EncodeProgress = Callable[[int], None]


class Encoder(Protocol):
    name: str

    def encode(self, texts: list[str], progress: EncodeProgress | None = None) -> np.ndarray: ...


class TfidfEncoder:
    """TF-IDF + LSA. No model download, no PyTorch - the zero-dependency baseline.

    Weaker than a sentence transformer at recognising that "corticospinal
    excitability" and "MEP amplitude" are the same subject, but good enough to
    prove the pipeline and fast enough to iterate on.
    """

    def __init__(self, cfg: EmbedConfig) -> None:
        self.cfg = cfg
        self.name = "tfidf"
        stop_words = sorted(
            set(TfidfVectorizer(stop_words="english").get_stop_words()) | ACADEMIC_STOPWORDS
        )
        self.vectoriser = TfidfVectorizer(
            stop_words=stop_words,
            max_features=cfg.max_features,
            min_df=cfg.min_df,
            max_df=cfg.max_df,
            ngram_range=(1, 2),
            sublinear_tf=True,
            strip_accents="unicode",
            lowercase=True,
        )

    def encode(self, texts: list[str], progress: EncodeProgress | None = None) -> np.ndarray:
        # One fit over the whole corpus; there is no partial state to report
        # against, and it is fast enough that there would be nothing to watch.
        counts = self.vectoriser.fit_transform(texts)
        n_components = min(self.cfg.svd_components, min(counts.shape) - 1)
        lsa = make_pipeline(
            TruncatedSVD(n_components=n_components, random_state=42),
            Normalizer(copy=False),
        )
        vectors = lsa.fit_transform(counts).astype(np.float32)
        if progress:
            progress(len(texts))
        return vectors


class SentenceTransformerEncoder:
    """Phase 2 backend. Requires the `ml` extra (`uv sync --extra ml`)."""

    DEFAULTS = {
        "bge": "BAAI/bge-base-en-v1.5",
        "specter": "sentence-transformers/allenai-specter",
    }

    def __init__(self, cfg: EmbedConfig) -> None:
        self.cfg = cfg
        self.name = cfg.model_name or self.DEFAULTS[cfg.backend]

    def encode(self, texts: list[str], progress: EncodeProgress | None = None) -> np.ndarray:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover - depends on optional extra
            raise RuntimeError(
                f"embed.backend='{self.cfg.backend}' needs sentence-transformers. "
                "Install it with:  uv sync --extra ml"
            ) from exc
        model = SentenceTransformer(self.name)

        # Encoded in chunks purely so the caller can report progress: this is
        # the slowest stage of a run, and encoding a whole corpus in one call
        # leaves the bar at zero for minutes, which reads as a hang. Each
        # vector depends only on its own text, so chunking changes nothing
        # about the result.
        done = 0
        chunks: list[np.ndarray] = []
        size = progress_chunk(len(texts))
        for start in range(0, len(texts), size):
            chunk = texts[start : start + size]
            chunks.append(
                model.encode(
                    chunk,
                    batch_size=16,
                    # We report progress ourselves; a tqdm bar per chunk would
                    # fight with the caller's own display.
                    show_progress_bar=False,
                    normalize_embeddings=True,
                    convert_to_numpy=True,
                )
            )
            done += len(chunk)
            if progress:
                progress(done)

        return np.vstack(chunks).astype(np.float32)


def build_encoder(cfg: EmbedConfig) -> Encoder:
    if cfg.backend == "tfidf":
        return TfidfEncoder(cfg)
    return SentenceTransformerEncoder(cfg)
