"""Pydantic schema for taxonomy.json - the primary deliverable."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field

SCHEMA_VERSION = "1.0"

Band = Literal["high", "medium", "low"]

# Folder id for documents that matched no cluster well enough to be placed.
UNSORTED_ID = "_unsorted"
UNSORTED_LABEL = "_Unsorted"


class Extraction(BaseModel):
    method: str
    pages: int = 0
    chars: int = 0
    # True when the PDF yielded too little text to be useful (likely a scan).
    degraded: bool = False
    abstract_found: bool = False
    title_source: Literal["pdf", "filename", "none"] = "none"
    error: str | None = None


class RunnerUp(BaseModel):
    folder_id: str
    folder_path: str
    similarity: float


class Assignment(BaseModel):
    folder_id: str
    folder_path: str
    confidence: float
    band: Band
    similarity_to_centroid: float
    margin_over_next: float
    membership_probability: float | None = None
    runner_up: RunnerUp | None = None
    was_noise: bool = False
    # Set by the Phase 3 low-confidence verification pass.
    llm_verified: bool | None = None
    rationale: str | None = None


class Document(BaseModel):
    id: str
    source_path: str
    # Path relative to source_root. `filename` alone is not enough to find the
    # file again once the corpus has subdirectories.
    relpath: str = ""
    filename: str
    sha256: str
    size_bytes: int

    title: str | None = None
    authors: list[str] = Field(default_factory=list)
    year: int | None = None
    doi: str | None = None
    keywords: list[str] = Field(default_factory=list)
    abstract_excerpt: str | None = None

    extraction: Extraction
    assignment: Assignment | None = None
    # 2D layout coordinates keyed by hierarchy level ("1", "2", "3").
    coords: dict[str, list[float]] = Field(default_factory=dict)
    top_terms: list[str] = Field(default_factory=list)
    # sha256 of another document with byte-identical content.
    duplicate_of: str | None = None
    near_duplicates: list[str] = Field(default_factory=list)


class ConfidenceStats(BaseModel):
    mean: float
    median: float
    p10: float
    p90: float
    n_high: int = 0
    n_medium: int = 0
    n_low: int = 0


class Folder(BaseModel):
    id: str
    level: int
    parent_id: str | None
    path: str
    label: str
    description: str | None = None
    distinguishing_note: str | None = None
    keywords: list[str] = Field(default_factory=list)
    n_documents_direct: int = 0
    n_documents_total: int = 0
    confidence_stats: ConfidenceStats | None = None
    centroid_2d: list[float] | None = None
    children: list[str] = Field(default_factory=list)
    # False until the Phase 3 labelling pass names it.
    labelled_by_llm: bool = False


class MoveInstruction(BaseModel):
    source: str
    target: str
    document_id: str
    confidence: float
    band: Band


class CoherenceEdit(BaseModel):
    type: Literal["rename", "merge", "reparent"]
    targets: list[str]
    detail: str
    reason: str


class Stats(BaseModel):
    n_documents: int
    n_parsed: int
    n_degraded: int
    n_duplicates: int
    n_folders: int
    n_leaf_folders: int
    n_unsorted: int
    levels: int
    band_counts: dict[str, int] = Field(default_factory=dict)
    abstract_recall: float = 0.0
    title_recall: float = 0.0


class Taxonomy(BaseModel):
    schema_version: str = SCHEMA_VERSION
    generated_at: datetime
    source_root: str
    target_root: str
    config: dict[str, Any]
    # Corpus-derived scale behind the confidence numbers; see cluster/assign.py.
    calibration: dict[str, float] = Field(default_factory=dict)
    stats: Stats
    folders: list[Folder]
    documents: list[Document]
    move_plan: list[MoveInstruction] = Field(default_factory=list)
    coherence_edits: list[CoherenceEdit] = Field(default_factory=list)

    def folder_by_id(self) -> dict[str, Folder]:
        return {f.id: f for f in self.folders}
