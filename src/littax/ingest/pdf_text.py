"""PyMuPDF text extraction.

Also returns the styled text spans of page 1, which the title heuristic needs:
a paper's title is almost always the largest non-trivial run of text in the
top portion of the first page.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path

import pymupdf

# Control chars and soft hyphens that PDF extraction leaves behind. Tabs and
# newlines are deliberately kept - line structure carries meaning for the
# abstract and title heuristics.
_JUNK_CODEPOINTS = (
    set(range(0x00, 0x09)) | {0x0B, 0x0C} | set(range(0x0E, 0x20)) | {0xAD, 0xFFFD}
)
_JUNK_TABLE = {cp: " " for cp in _JUNK_CODEPOINTS}
_LINE_BREAK_HYPHEN = re.compile(r"(\w)-\n(\w)")


def normalise(text: str) -> str:
    """NFKC-fold and de-noise extracted text.

    NFKC matters more than it looks here: PDF typesetting emits ligatures as
    single codepoints, so without it the corpus contains "firing" spelled with
    U+FB01, which tokenises differently from plain "firing" and quietly
    degrades both clustering and keyword extraction.
    """
    text = unicodedata.normalize("NFKC", text)
    text = text.translate(_JUNK_TABLE)
    # Rejoin words broken across a line by hyphenation.
    text = _LINE_BREAK_HYPHEN.sub(r"\1\2", text)
    return text


@dataclass(slots=True)
class Span:
    text: str
    size: float
    # Vertical position as a fraction of page height (0 = top).
    y_frac: float


@dataclass(slots=True)
class RawText:
    text: str = ""
    pages: int = 0
    chars: int = 0
    page1_spans: list[Span] = field(default_factory=list)
    error: str | None = None

    @property
    def chars_per_page(self) -> float:
        return self.chars / self.pages if self.pages else 0.0


def _page1_spans(page) -> list[Span]:
    height = page.rect.height or 1.0
    spans: list[Span] = []
    data = page.get_text("dict")
    for block in data.get("blocks", []):
        if block.get("type") != 0:  # 0 = text
            continue
        for line in block.get("lines", []):
            # Merge the spans of one line so a title split across style runs
            # (e.g. an italicised species name) stays a single candidate.
            raw_spans = line.get("spans", [])
            text = normalise("".join(s.get("text", "") for s in raw_spans)).strip()
            if not text:
                continue
            sizes = [s.get("size", 0.0) for s in raw_spans if s.get("text", "").strip()]
            if not sizes:
                continue
            y0 = line.get("bbox", (0, 0, 0, 0))[1]
            spans.append(Span(text=text, size=max(sizes), y_frac=y0 / height))
    return spans


def extract_pdf(path: Path) -> RawText:
    """Extract full text plus page-1 styling. Never raises; errors land on `.error`."""
    try:
        with pymupdf.open(path) as doc:
            pages = doc.page_count
            chunks: list[str] = []
            spans: list[Span] = []
            for i, page in enumerate(doc):
                chunks.append(page.get_text("text"))
                if i == 0:
                    spans = _page1_spans(page)
            text = normalise("\n".join(chunks))
            return RawText(text=text, pages=pages, chars=len(text), page1_spans=spans)
    except Exception as exc:  # corrupt/encrypted PDFs must not kill the run
        return RawText(error=f"{type(exc).__name__}: {exc}")
