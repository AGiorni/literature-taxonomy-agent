"""Dimensionality reduction, used both for clustering and for the 2D layout.

UMAP is the better choice for both, but it drags in numba, which is the most
fragile dependency in the stack. PCA keeps the pipeline running when UMAP
cannot be installed, at some cost in cluster separation.
"""

from __future__ import annotations

import numpy as np
from sklearn.decomposition import PCA


def reduce_dims(
    X: np.ndarray,
    method: str,
    n_components: int,
    random_state: int = 42,
    *,
    n_neighbors: int = 15,
    min_dist: float = 0.0,
) -> np.ndarray:
    """Project `X` to `n_components` dimensions. Returns `X` unchanged for 'none'."""
    if method == "none" or n_components >= X.shape[1]:
        return X

    # Neither backend can produce more components than it has samples. UMAP is
    # the strict one: its spectral initialisation solves for n_components
    # eigenvectors and raises outright once that reaches the sample count, so
    # small nodes need real headroom, not just `n_samples - 1`.
    n_components = min(n_components, X.shape[0] - 2, X.shape[1])
    if n_components < 2:
        return X

    if method == "umap":
        try:
            from umap import UMAP
        except ImportError as exc:  # pragma: no cover - depends on optional extra
            raise RuntimeError(
                "cluster.reducer='umap' needs umap-learn. Install it with "
                "`uv sync --extra ml`, or set cluster.reducer='pca'."
            ) from exc
        # n_neighbors must stay below the sample count or UMAP errors out.
        neighbours = max(2, min(n_neighbors, X.shape[0] - 1))
        return UMAP(
            n_components=n_components,
            n_neighbors=neighbours,
            min_dist=min_dist,
            metric="cosine",
            random_state=random_state,
        ).fit_transform(X)

    return PCA(n_components=n_components, random_state=random_state).fit_transform(X)


def layout_2d(X: np.ndarray, method: str, random_state: int = 42) -> np.ndarray:
    """A single corpus-wide 2D projection for the scatter plot.

    One layout serves every hierarchy level: the points do not move, only the
    colouring changes as you switch level. That makes the levels visually
    comparable, which per-level refits would not be.
    """
    method = "pca" if method == "none" else method
    coords = reduce_dims(X, method, 2, random_state)
    if coords.shape[1] < 2:  # degenerate corpus
        coords = np.pad(coords, ((0, 0), (0, 2 - coords.shape[1])))
    # Centre and scale to roughly unit range so the viewer needs no axis config.
    coords = coords - coords.mean(axis=0)
    scale = np.abs(coords).max() or 1.0
    return (coords / scale).astype(np.float32)
