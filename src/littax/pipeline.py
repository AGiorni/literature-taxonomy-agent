"""Stage orchestration: a folder of PDFs in, a Taxonomy out."""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

import numpy as np

from littax.cache import PARSER_VERSION, ArrayCache, JsonCache, corpus_key
from littax.cluster.assign import assign_documents, confidence_stats
from littax.cluster.reduce import layout_2d
from littax.cluster.terms import document_terms, node_terms
from littax.cluster.tree import Node, build_tree, iter_nodes, leaves
from littax.config import Config
from littax.embed.encoder import build_encoder
from littax.export.move_plan import build_move_plan
from littax.export.naming import placeholder_label, safe_folder_name
from littax.ingest import extract_pdf, parse_document, scan_directory
from littax.ingest.parse import ParsedDoc
from littax.models import (
    UNSORTED_ID,
    UNSORTED_LABEL,
    Assignment,
    CoherenceEdit,
    ConfidenceStats,
    Document,
    Extraction,
    Folder,
    RunnerUp,
    Stats,
    Taxonomy,
)

if TYPE_CHECKING:
    from littax.label.naming import NodeLabel

Progress = Callable[[str, int, int, str], None]


def _noop(stage: str, done: int, total: int, detail: str) -> None:  # pragma: no cover
    pass


def _parse_with_cache(
    files, cfg: Config, cache: JsonCache, progress: Progress
) -> list[tuple[object, ParsedDoc, Extraction]]:
    out = []
    for n, scanned in enumerate(files, start=1):
        progress("ingest", n, len(files), scanned.relpath)
        cached = cache.get(scanned.sha256)
        if cached is not None:
            parsed = ParsedDoc(**cached["parsed"])
            extraction = Extraction(**cached["extraction"])
        else:
            raw = extract_pdf(scanned.path)
            parsed = parse_document(
                raw,
                scanned.path.stem,
                degraded_chars_per_page=cfg.ingest.degraded_chars_per_page,
                abstract_max_chars=cfg.ingest.abstract_max_chars,
                profile_body_words=cfg.ingest.profile_body_words,
            )
            extraction = Extraction(
                method="pymupdf",
                pages=raw.pages,
                chars=raw.chars,
                degraded=parsed.degraded,
                abstract_found=parsed.abstract is not None,
                title_source=parsed.title_source,  # type: ignore[arg-type]
                error=raw.error,
            )
            # The body is the bulk of the payload and nothing downstream reads
            # it back, so it is dropped before caching.
            payload = asdict(parsed) | {"body": ""}
            cache.set(
                scanned.sha256,
                {"parsed": payload, "extraction": extraction.model_dump(mode="json")},
            )
        out.append((scanned, parsed, extraction))
    return out


def _folder_paths(root: Node, labels: dict[str, str]) -> dict[str, str]:
    """Build each node's full relative path from its ancestors' safe names."""
    paths: dict[str, str] = {"root": ""}
    for node in iter_nodes(root):
        parent = paths.get(node.parent_id or "root", "")
        segment = safe_folder_name(labels.get(node.id, node.id), fallback=node.id)
        paths[node.id] = f"{parent}/{segment}" if parent else segment
    return paths


def run(
    cfg: Config, progress: Progress | None = None, credentials: Any | None = None
) -> Taxonomy:
    """Discover the taxonomy.

    `credentials` overrides what the environment supplies for the labelling
    provider. It is a parameter and not a Config field because Config is
    serialised into taxonomy.json - an api_key there would be written to disk.
    """
    report = progress or _noop

    # ---- 1. scan -----------------------------------------------------------
    report("scan", 0, 1, str(cfg.source_root))
    files = scan_directory(cfg.source_root, cfg.ingest.glob)
    if not files:
        raise ValueError(f"no files matching {cfg.ingest.glob!r} under {cfg.source_root}")

    # ---- 2. ingest ---------------------------------------------------------
    cache = JsonCache(cfg.cache_dir, "parsed", PARSER_VERSION)
    parsed_rows = _parse_with_cache(files, cfg, cache, report)

    # Byte-identical duplicates are reported but excluded from the clustering,
    # so they cannot distort centroids or inflate folder counts.
    unique = [row for row in parsed_rows if row[0].duplicate_of is None]
    texts = [row[1].profile_text for row in unique]

    # ---- 3. embed ----------------------------------------------------------
    report("embed", 0, len(texts), cfg.embed.backend)
    encoder = build_encoder(cfg.embed)
    embeddings = ArrayCache(cfg.cache_dir, "embeddings")
    key = corpus_key(cfg.embed.backend, encoder.name, *texts)
    X = embeddings.get(key)
    if X is None:
        X = encoder.encode(texts)
        embeddings.set(key, X)
    report("embed", len(texts), len(texts), f"{X.shape[1]}d {encoder.name}")

    # ---- 4. cluster --------------------------------------------------------
    report("cluster", 0, 1, cfg.cluster.algorithm)
    root = build_tree(X, cfg.cluster)
    nodes = list(iter_nodes(root))
    leaf_nodes = leaves(root)
    report("cluster", 1, 1, f"{len(nodes)} folders, {len(leaf_nodes)} leaves")

    # ---- 5. score ----------------------------------------------------------
    report("score", 0, 1, "")
    assignment = assign_documents(X, root, cfg.score)

    # ---- 6. terms ----------------------------------------------------------
    report("terms", 0, 1, "")
    terms_by_node = node_terms(texts, {n.id: n.doc_indices for n in nodes}, cfg.terms)
    terms_by_doc = document_terms(texts, cfg.terms)

    # ---- 7. layout ---------------------------------------------------------
    report("layout", 0, 1, "")
    coords = layout_2d(X, cfg.cluster.reducer, cfg.cluster.random_state)

    # ---- 8. label ----------------------------------------------------------
    node_labels: dict[str, "NodeLabel"] | None = None
    coherence_edits: list[CoherenceEdit] = []
    labeller = None
    if cfg.label.enabled:
        from littax.label.client import build_labeller
        from littax.label.naming import apply_renames, label_tree, review_coherence

        report("label", 0, len(nodes), f"{cfg.label.provider}:{cfg.label.model}")
        labeller = build_labeller(cfg.label, cfg.cache_dir, credentials)
        titles = [row[1].title for row in unique]
        node_labels = label_tree(
            labeller, root, X, titles, terms_by_node, cfg.label, report
        )
        if cfg.label.coherence_pass:
            report("coherence", 0, 1, "reviewing whole tree")
            coherence_edits = review_coherence(labeller, root, node_labels, terms_by_node)
            apply_renames(node_labels, coherence_edits)

    # ---- 9. assemble -------------------------------------------------------
    if node_labels:
        labels = {n.id: node_labels[n.id].label for n in nodes}
    else:
        labels = {n.id: placeholder_label(terms_by_node.get(n.id, []), n.id) for n in nodes}
    paths = _folder_paths(root, labels)

    documents = _build_documents(
        parsed_rows, unique, assignment, terms_by_doc, coords, paths, cfg
    )
    folders = _build_folders(nodes, labels, paths, terms_by_node, documents, node_labels)
    folder_index = {f.id: f for f in folders}

    # ---- 10. verify the least certain assignments --------------------------
    if labeller is not None and cfg.label.verify_low_confidence:
        _verify_assignments(labeller, documents, folder_index, report)

    move_plan = build_move_plan(documents, folder_index, str(cfg.target_root))

    return Taxonomy(
        generated_at=datetime.now(timezone.utc),
        source_root=str(cfg.source_root.resolve()),
        target_root=str(cfg.target_root),
        config=cfg.model_dump(mode="json"),
        calibration=assignment.calibration.as_dict() if assignment.calibration else {},
        stats=_build_stats(parsed_rows, documents, folders, nodes, leaf_nodes),
        folders=folders,
        documents=documents,
        move_plan=move_plan,
        coherence_edits=coherence_edits,
    )


def _verify_assignments(labeller, documents, folder_index, report) -> None:
    """Ask Claude to second-guess the documents the geometry was least sure of."""
    from littax.label.naming import verify_low_confidence

    def summarise(folder_id: str | None) -> dict | None:
        folder = folder_index.get(folder_id or "")
        if folder is None:
            return None
        return {
            "id": folder.id,
            "path": folder.path,
            "label": folder.label,
            "description": folder.description,
        }

    candidates = []
    for document in documents:
        fit = document.assignment
        if fit is None or fit.band != "low" or document.duplicate_of is not None:
            continue
        assigned = summarise(fit.folder_id)
        if assigned is None:
            continue
        # Assigned plus runner-up: low confidence means the document was torn
        # between exactly these two, which is the case worth re-checking.
        options = [assigned]
        if fit.runner_up:
            alternative = summarise(fit.runner_up.folder_id)
            if alternative:
                options.append(alternative)
        if len(options) < 2:
            continue
        candidates.append(
            {
                "document_id": document.id,
                "title": document.title,
                "abstract": document.abstract_excerpt,
                "assigned": assigned,
                "candidates": options,
            }
        )

    if not candidates:
        return

    report("verify", 0, len(candidates), f"{len(candidates)} low-confidence documents")
    verdicts = verify_low_confidence(labeller, candidates, report)

    by_id = {d.id: d for d in documents}
    for document_id, verdict in verdicts.items():
        document = by_id.get(document_id)
        if document is None or document.assignment is None:
            continue
        # Record the second opinion; never silently move the document. The
        # geometry and the model disagreeing is exactly what the reviewer
        # wants to see, and hiding it behind an automatic reassignment would
        # destroy that signal.
        document.assignment.llm_verified = bool(verdict["correct"])
        document.assignment.rationale = verdict["rationale"]


def _build_documents(
    parsed_rows,
    unique,
    assignment,
    terms_by_doc: list[list[str]],
    coords: np.ndarray,
    paths: dict[str, str],
    cfg: Config,
) -> list[Document]:
    # Map a scanned file's position in `unique` back to its embedding row.
    row_of = {scanned.relpath: i for i, (scanned, _, _) in enumerate(unique)}
    relpath_of_row = {i: scanned.relpath for i, (scanned, _, _) in enumerate(unique)}

    documents: list[Document] = []
    for order, (scanned, parsed, extraction) in enumerate(parsed_rows):
        document = Document(
            id=f"doc_{order:04d}",
            source_path=f"{cfg.source_root.as_posix()}/{scanned.relpath}",
            relpath=scanned.relpath,
            filename=scanned.path.name,
            sha256=scanned.sha256,
            size_bytes=scanned.size_bytes,
            title=parsed.title,
            authors=parsed.authors,
            year=parsed.year,
            doi=parsed.doi,
            keywords=parsed.keywords,
            abstract_excerpt=(parsed.abstract or "")[:600] or None,
            extraction=extraction,
            duplicate_of=scanned.duplicate_of,
        )

        row = row_of.get(scanned.relpath)
        if row is not None:
            document.coords = {"global": [round(float(c), 5) for c in coords[row]]}
            document.top_terms = terms_by_doc[row]
            fit = assignment.assignments.get(row)
            if fit is not None:
                runner_up = None
                if fit.runner_up_id:
                    runner_up = RunnerUp(
                        folder_id=fit.runner_up_id,
                        folder_path=paths.get(fit.runner_up_id, ""),
                        similarity=round(fit.runner_up_similarity, 4),
                    )
                document.assignment = Assignment(
                    folder_id=fit.folder_id,
                    folder_path=paths.get(fit.folder_id, UNSORTED_LABEL),
                    confidence=round(fit.confidence, 4),
                    band=fit.band,  # type: ignore[arg-type]
                    similarity_to_centroid=round(fit.similarity, 4),
                    margin_over_next=round(fit.margin, 4),
                    membership_probability=fit.membership,
                    runner_up=runner_up,
                    was_noise=fit.was_noise,
                )
            document.near_duplicates = [
                relpath_of_row[other] for other in assignment.near_duplicates.get(row, [])
            ]
        documents.append(document)
    return documents


def _build_folders(
    nodes: list[Node],
    labels: dict[str, str],
    paths: dict[str, str],
    terms_by_node: dict[str, list[str]],
    documents: list[Document],
    node_labels: dict | None = None,
) -> list[Folder]:
    by_folder: dict[str, list[float]] = {}
    bands_by_folder: dict[str, Counter] = {}
    direct_counts: Counter[str] = Counter()
    for document in documents:
        if document.assignment is None:
            continue
        folder_id = document.assignment.folder_id
        by_folder.setdefault(folder_id, []).append(document.assignment.confidence)
        bands_by_folder.setdefault(folder_id, Counter())[document.assignment.band] += 1
        direct_counts[folder_id] += 1

    # Documents only ever land in leaves, so an internal node's statistics have
    # to be rolled up from its descendants - otherwise every parent folder
    # reports a confidence of zero.
    descendants: dict[str, list[str]] = {}
    for node in nodes:
        descendants[node.id] = [n.id for n in iter_nodes(node, include_root=True)]

    folders: list[Folder] = []
    for node in nodes:
        subtree = descendants[node.id]
        confidences = [c for folder_id in subtree for c in by_folder.get(folder_id, [])]
        bands: Counter = Counter()
        for folder_id in subtree:
            bands.update(bands_by_folder.get(folder_id, {}))
        stats = confidence_stats(confidences)
        named = (node_labels or {}).get(node.id)
        folders.append(
            Folder(
                id=node.id,
                level=node.level,
                parent_id=None if node.parent_id == "root" else node.parent_id,
                path=paths[node.id],
                label=labels[node.id],
                description=named.description if named else None,
                distinguishing_note=named.distinguishing_note if named else None,
                labelled_by_llm=named is not None,
                keywords=(named.keywords if named else terms_by_node.get(node.id, []))[:12],
                n_documents_direct=direct_counts.get(node.id, 0),
                n_documents_total=len(node.doc_indices),
                confidence_stats=ConfidenceStats(
                    **stats,
                    n_high=bands["high"],
                    n_medium=bands["medium"],
                    n_low=bands["low"],
                ),
                children=[child.id for child in node.children],
            )
        )

    unsorted = by_folder.get(UNSORTED_ID)
    if unsorted:
        stats = confidence_stats(unsorted)
        folders.append(
            Folder(
                id=UNSORTED_ID,
                level=1,
                parent_id=None,
                path=UNSORTED_LABEL,
                label=UNSORTED_LABEL,
                description=(
                    "Documents that matched no discovered topic closely enough to be "
                    "placed. Review these by hand."
                ),
                n_documents_direct=len(unsorted),
                n_documents_total=len(unsorted),
                confidence_stats=ConfidenceStats(**stats, n_low=len(unsorted)),
            )
        )
    return folders


def _build_stats(parsed_rows, documents, folders, nodes, leaf_nodes) -> Stats:
    total = len(parsed_rows)
    bands = Counter(d.assignment.band for d in documents if d.assignment)
    with_title = sum(1 for _, parsed, _ in parsed_rows if parsed.title)
    with_abstract = sum(1 for _, parsed, _ in parsed_rows if parsed.abstract)
    return Stats(
        n_documents=total,
        n_parsed=sum(1 for _, _, e in parsed_rows if not e.error),
        n_degraded=sum(1 for _, parsed, _ in parsed_rows if parsed.degraded),
        n_duplicates=sum(1 for s, _, _ in parsed_rows if s.duplicate_of is not None),
        n_folders=len(folders),
        n_leaf_folders=len(leaf_nodes),
        n_unsorted=sum(
            1 for d in documents if d.assignment and d.assignment.folder_id == UNSORTED_ID
        ),
        levels=max((n.level for n in nodes), default=0),
        band_counts=dict(bands),
        abstract_recall=round(with_abstract / total, 4) if total else 0.0,
        title_recall=round(with_title / total, 4) if total else 0.0,
    )


def write_taxonomy(taxonomy: Taxonomy, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "taxonomy.json"
    path.write_text(
        taxonomy.model_dump_json(indent=2, exclude_none=False), encoding="utf-8"
    )
    return path
