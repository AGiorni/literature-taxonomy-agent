"""c-TF-IDF: the terms that *distinguish* a cluster from its siblings.

Plain per-cluster term frequency just returns the vocabulary of the field
("muscle", "motor", "activity"). Treating each cluster as a single pseudo-
document and running TF-IDF across clusters instead surfaces the terms that
separate them, which is what both the viewer and the labelling prompt need.
"""

from __future__ import annotations

import re

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from littax.config import TermsConfig
from littax.embed.encoder import ACADEMIC_STOPWORDS


def _stop_words() -> list[str]:
    base = set(TfidfVectorizer(stop_words="english").get_stop_words())
    return sorted(base | ACADEMIC_STOPWORDS)


# PDF extraction leaves artefacts behind - figure ids, running heads, OCR
# rubbish like "a1111111111" from PLOS templates. These score highly under
# c-TF-IDF precisely because they are unique to one cluster, so they must be
# filtered out or they dominate the folder names.
_REPEATED_CHAR = re.compile(r"(.)\1{2,}")
_HAS_VOWEL = re.compile(r"[aeiouy]")


def is_junk_term(term: str) -> bool:
    if len(term) < 3:
        return True
    if _REPEATED_CHAR.search(term):
        return True
    digits = sum(c.isdigit() for c in term)
    if digits / len(term) > 0.3:
        return True
    # Every word of a real term should contain a vowel; acronyms are short
    # enough to survive the length check above.
    return any(len(word) > 4 and not _HAS_VOWEL.search(word) for word in term.split())


def _top_terms(row: np.ndarray, vocabulary: np.ndarray, top_n: int) -> list[str]:
    if row.size == 0:
        return []
    # Take a generous slice, then filter, so junk terms do not consume the budget.
    count = min(top_n * 4, row.size)
    order = np.argpartition(row, -count)[-count:]
    order = order[np.argsort(row[order])[::-1]]
    kept = [
        str(vocabulary[i]) for i in order if row[i] > 0 and not is_junk_term(str(vocabulary[i]))
    ]
    return kept[:top_n]


def node_terms(
    texts: list[str],
    groups: dict[str, list[int]],
    cfg: TermsConfig,
) -> dict[str, list[str]]:
    """Top discriminative terms per group. `groups` maps node id -> doc indices."""
    if not groups:
        return {}

    ids = list(groups)
    pseudo_docs = ["\n".join(texts[i] for i in groups[node_id]) for node_id in ids]

    vectoriser = TfidfVectorizer(
        stop_words=_stop_words(),
        max_features=cfg.max_features,
        ngram_range=(1, cfg.ngram_max),
        sublinear_tf=True,
        strip_accents="unicode",
        # A term present in every cluster cannot distinguish between them, but
        # with few clusters max_df must not be so tight it empties the matrix.
        max_df=0.8 if len(ids) > 4 else 1.0,
    )
    try:
        matrix = vectoriser.fit_transform(pseudo_docs)
    except ValueError:  # empty vocabulary
        return {node_id: [] for node_id in ids}

    vocabulary = vectoriser.get_feature_names_out()
    dense = matrix.toarray()
    return {
        node_id: _top_terms(dense[position], vocabulary, cfg.top_n)
        for position, node_id in enumerate(ids)
    }


def document_terms(texts: list[str], cfg: TermsConfig, top_n: int = 8) -> list[list[str]]:
    """Top TF-IDF terms for each individual document, for the detail panel."""
    vectoriser = TfidfVectorizer(
        stop_words=_stop_words(),
        max_features=cfg.max_features,
        ngram_range=(1, cfg.ngram_max),
        sublinear_tf=True,
        strip_accents="unicode",
        min_df=2,
        max_df=0.4,
    )
    try:
        matrix = vectoriser.fit_transform(texts)
    except ValueError:
        return [[] for _ in texts]

    vocabulary = vectoriser.get_feature_names_out()
    out: list[list[str]] = []
    for row in range(matrix.shape[0]):
        start, end = matrix.indptr[row], matrix.indptr[row + 1]
        columns, values = matrix.indices[start:end], matrix.data[start:end]
        if not len(columns):
            out.append([])
            continue
        order = np.argsort(values)[::-1][:top_n]
        out.append([str(vocabulary[columns[i]]) for i in order])
    return out
