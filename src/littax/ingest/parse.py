"""Turn raw PDF text into the structured fields the pipeline clusters on.

Everything here is offline heuristics - no bibliographic API lookups. The two
fields that matter most downstream are `title` and `abstract`: the profile text
built from them is what actually gets embedded.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from littax.ingest.pdf_text import RawText, Span

# --------------------------------------------------------------------------- #
# Filename parsing: this corpus is named "Author Year - Title.pdf"
# --------------------------------------------------------------------------- #

_FILENAME_RE = re.compile(
    r"^\s*(?P<author>.*?)\s*(?P<year>(?:19|20)\d{2})?\s*[-–—]\s*(?P<title>.+?)\s*$"
)
_YEAR_RE = re.compile(r"\b(19|20)\d{2}\b")


@dataclass(slots=True)
class FilenameHint:
    author: str | None = None
    year: int | None = None
    title: str | None = None


def _despacify(text: str) -> str:
    """Normalise separator characters used in place of spaces in filenames."""
    text = text.replace("_", " ")
    # Only treat hyphens as separators when the name is clearly hyphen-delimited
    # (e.g. "an-update-of-systematic-reviews"), not for real compounds.
    if text.count("-") >= 3:
        text = text.replace("-", " ")
    return re.sub(r"\s+", " ", text).strip()


def parse_filename(stem: str) -> FilenameHint:
    """Extract author/year/title from a `Author Year - Title` style filename."""
    hint = FilenameHint()
    match = _FILENAME_RE.match(stem)
    if match:
        author = (match.group("author") or "").strip(" ,.-")
        hint.author = author or None
        if match.group("year"):
            hint.year = int(match.group("year"))
        title = _despacify(match.group("title"))
        hint.title = title or None
    else:
        # No dash: the whole stem may still carry a year.
        cleaned = _despacify(stem)
        year = _YEAR_RE.search(cleaned)
        if year:
            hint.year = int(year.group(0))
        hint.title = cleaned or None
    if hint.year is None and hint.author:
        year = _YEAR_RE.search(hint.author)
        if year:
            hint.year = int(year.group(0))
    return hint


# --------------------------------------------------------------------------- #
# Title from page-1 typography
# --------------------------------------------------------------------------- #

_TITLE_NOISE = re.compile(
    r"^(journal of|vol\.|volume|downloaded from|received|accepted|copyright|"
    r"www\.|http|doi:|issn|page \d|\d+$)",
    re.IGNORECASE,
)


def _plausible_title(text: str) -> bool:
    if not (15 <= len(text) <= 300):
        return False
    if len(text.split()) < 3:
        return False
    if _TITLE_NOISE.match(text.strip()):
        return False
    alpha = sum(c.isalpha() or c.isspace() for c in text)
    return alpha / len(text) > 0.6


def title_from_spans(spans: list[Span]) -> str | None:
    """Largest text in the top half of page 1, merging consecutive equal-size lines."""
    candidates = [s for s in spans if s.y_frac < 0.5 and s.text.strip()]
    if not candidates:
        return None

    # Group runs of consecutive lines sharing a font size, so multi-line titles
    # come back whole.
    groups: list[tuple[float, list[str]]] = []
    for span in candidates:
        size = round(span.size, 1)
        if groups and abs(groups[-1][0] - size) < 0.15:
            groups[-1][1].append(span.text)
        else:
            groups.append((size, [span.text]))

    scored: list[tuple[float, int, str]] = []
    for order, (size, parts) in enumerate(groups):
        text = re.sub(r"\s+", " ", " ".join(parts)).strip()
        if _plausible_title(text):
            scored.append((size, -order, text))
    if not scored:
        return None
    scored.sort(reverse=True)  # biggest font, earliest on the page
    return scored[0][2]


def _tokens(text: str) -> set[str]:
    return {t for t in re.findall(r"[a-z]{4,}", text.lower())}


def _overlap(a: str, b: str) -> float:
    ta, tb = _tokens(a), _tokens(b)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / min(len(ta), len(tb))


def choose_title(pdf_title: str | None, hint_title: str | None) -> tuple[str | None, str]:
    """Reconcile the typography-derived title with the filename's.

    Returns (title, source). The PDF title is preferred when it is plausible,
    because filenames in the wild are abbreviated; the filename wins when the
    PDF gave us nothing usable.
    """
    if pdf_title and hint_title:
        if _overlap(pdf_title, hint_title) >= 0.35:
            return pdf_title, "pdf"
        # They disagree - trust the longer, more specific string.
        if len(pdf_title) >= len(hint_title):
            return pdf_title, "pdf"
        return hint_title, "filename"
    if pdf_title:
        return pdf_title, "pdf"
    if hint_title:
        return hint_title, "filename"
    return None, "none"


# --------------------------------------------------------------------------- #
# Abstract, keywords, DOI
# --------------------------------------------------------------------------- #

# Elsevier and friends letter-space their headings: "A B S T R A C T".
_ABSTRACT_HEAD = re.compile(
    r"^[ \t]*(?:a\s?b\s?s\s?t\s?r\s?a\s?c\s?t|summary)\b[ \t]*:?[ \t]*",
    re.IGNORECASE | re.MULTILINE,
)
_ABSTRACT_END = re.compile(
    r"^[ \t]*(?:"
    r"(?:\d+[\.\)]?\s*)?i\s?n\s?t\s?r\s?o\s?d\s?u\s?c\s?t\s?i\s?o\s?n"
    r"|k\s?e\s?y\s?[ ]?w\s?o\s?r\s?d\s?s"
    r"|©\s*\d{4}"
    r"|new\s+&\s+noteworthy"
    r")",
    re.IGNORECASE | re.MULTILINE,
)
_KEYWORDS_RE = re.compile(
    r"^[ \t]*k\s?e\s?y\s?[ ]?w\s?o\s?r\s?d\s?s?\b\s*:?\s*(?P<body>.+(?:\n(?!\s*\n).+)?)",
    re.IGNORECASE | re.MULTILINE,
)
_DOI_RE = re.compile(r"\b10\.\d{4,9}/[^\s\"'<>,;)\]]+", re.IGNORECASE)
_REFERENCES_RE = re.compile(
    r"^[ \t]*(?:\d+[\.\)]?\s*)?(?:references|bibliography|literature cited|reference list)"
    r"[ \t]*:?[ \t]*$",
    re.IGNORECASE | re.MULTILINE,
)


def strip_references(text: str) -> str:
    """Drop the reference list, which is pure noise for topic clustering.

    Only headings in the back half of the document count, so an in-body mention
    of "references" doesn't truncate the paper.
    """
    cutoff = None
    for match in _REFERENCES_RE.finditer(text):
        if match.start() > len(text) * 0.45:
            cutoff = match.start()
            break
    return text[:cutoff] if cutoff else text


def extract_abstract(text: str, max_chars: int = 3000) -> str | None:
    head = _ABSTRACT_HEAD.search(text[: max(len(text) // 2, 20_000)])
    if not head:
        return None
    body = text[head.end() : head.end() + max_chars * 2]
    end = _ABSTRACT_END.search(body)
    if end and end.start() > 100:
        body = body[: end.start()]
    body = re.sub(r"\s+", " ", body).strip()
    if len(body) < 100:
        return None
    return body[:max_chars]


def extract_keywords(text: str) -> list[str]:
    match = _KEYWORDS_RE.search(text[: max(len(text) // 2, 20_000)])
    if not match:
        return []
    body = re.sub(r"\s+", " ", match.group("body")).strip()
    parts = re.split(r"[;,·•]", body)
    out: list[str] = []
    for part in parts:
        word = part.strip(" .:-")
        if 2 <= len(word) <= 60 and not word.lower().startswith("introduction"):
            out.append(word)
    return out[:12]


def extract_doi(text: str) -> str | None:
    match = _DOI_RE.search(text)
    if not match:
        return None
    return match.group(0).rstrip(".,;)")


# --------------------------------------------------------------------------- #
# Unheaded abstracts
#
# Most journals in a typical corpus (J Physiol, J Appl Physiol, Frontiers, ...)
# print the abstract with no "Abstract" heading at all - it simply runs on after
# the affiliations. Heading-based extraction alone misses roughly a third of a
# real corpus, so when it fails we fall back to finding the lead prose block.
#
# For clustering purposes the distinction barely matters: what the embedding
# needs is topical prose from the front of the paper, not a bibliographically
# exact abstract.
# --------------------------------------------------------------------------- #

_INTRO_HEAD = re.compile(
    r"^[ \t]*(?:\d+[\.\)]?[ \t]*)?i\s?n\s?t\s?r\s?o\s?d\s?u\s?c\s?t\s?i\s?o\s?n\b",
    re.IGNORECASE | re.MULTILINE,
)
_LEAD_STOP = re.compile(
    r"^[ \t]*(?:k\s?e\s?y\s?[ ]?w\s?o\s?r\s?d\s?s?|abbreviations|highlights)\b",
    re.IGNORECASE | re.MULTILINE,
)
_BOILERPLATE = re.compile(
    r"(@|https?://|www\.|doi:|ISSN|©|\bReceived\b|\bAccepted\b|\bPublished\b"
    r"|\bCopyright\b|\bUniversit|\bDepartment\b|\bInstitute\b|\bHospital\b"
    r"|\bFacult|\bSchool of\b|\bLaborator|\bCent(?:er|re)\b|\bClinic\b"
    r"|\bCorrespondence\b|\bE-?mail\b|Downloaded from|All rights reserved"
    r"|Creative Commons|Conflicts? of Interest|Citation:|Edited by|Reviewed by"
    r"|Specialty section|\bFrontiers in\b|Author Contributions)",
    re.IGNORECASE,
)
_INITIALS = re.compile(r"\b[A-Z]\.")


def _demangle_citation(line: str) -> str:
    """Recover an abstract that runs on after a self-citation.

    J Appl Physiol and friends print "Author, A. Title. Journal 101: 1-7, 2006.
    doi:10.1152/...—The motor cortex..." on one line, with the abstract starting
    after the em dash.
    """
    if "—" in line and _BOILERPLATE.search(line):
        tail = line.rsplit("—", 1)[-1].strip()
        if len(tail) > 30:
            return tail
    return line


def _is_prose(line: str) -> bool:
    """True for a line of running body text, false for headings and metadata."""
    line = line.strip()
    if len(line) < 45:
        return False
    words = line.split()
    if len(words) < 7:
        return False
    if _BOILERPLATE.search(line):
        return False
    if sum(c.isdigit() for c in line) / len(line) > 0.15:
        return False
    if len(_INITIALS.findall(line)) >= 3:  # author list
        return False
    capitalised = sum(1 for w in words if w[:1].isupper())
    if capitalised / len(words) > 0.6:  # title-case heading
        return False
    return True


def extract_lead_prose(text: str, max_chars: int = 3000, min_chars: int = 250) -> str | None:
    intro = _INTRO_HEAD.search(text[:25_000])
    region = text[: intro.start()] if intro and intro.start() > 400 else text[:8_000]

    stop = _LEAD_STOP.search(region)
    if stop and stop.start() > 300:
        region = region[: stop.start()]

    kept = [
        stripped
        for line in region.split("\n")
        if _is_prose(stripped := _demangle_citation(line).strip())
    ]
    prose = re.sub(r"\s+", " ", " ".join(kept)).strip()
    return prose[:max_chars] if len(prose) >= min_chars else None


# --------------------------------------------------------------------------- #
# Assembly
# --------------------------------------------------------------------------- #


@dataclass(slots=True)
class ParsedDoc:
    title: str | None = None
    title_source: str = "none"
    authors: list[str] = field(default_factory=list)
    year: int | None = None
    doi: str | None = None
    keywords: list[str] = field(default_factory=list)
    abstract: str | None = None
    # "heading" = found under an explicit Abstract heading (high precision);
    # "lead-prose" = recovered from the unheaded lead block; "none" = neither.
    abstract_source: str = "none"
    body: str = ""
    profile_text: str = ""
    degraded: bool = False


def build_profile_text(
    title: str | None,
    keywords: list[str],
    abstract: str | None,
    body: str,
    body_words: int,
) -> str:
    """The string that actually gets embedded.

    Title is repeated twice: it is the single most topic-dense field, and under
    TF-IDF the repetition gives it roughly the weight it deserves relative to a
    much longer abstract.
    """
    parts: list[str] = []
    if title:
        parts += [title, title]
    if keywords:
        parts.append(". ".join(keywords))
    if abstract:
        parts.append(abstract)
    else:
        parts.append(" ".join(body.split()[:body_words]))
    return "\n".join(p for p in parts if p.strip())


def parse_document(
    raw: RawText,
    filename_stem: str,
    *,
    degraded_chars_per_page: int = 100,
    abstract_max_chars: int = 3000,
    profile_body_words: int = 400,
) -> ParsedDoc:
    hint = parse_filename(filename_stem)
    degraded = bool(raw.error) or raw.chars_per_page < degraded_chars_per_page

    if degraded:
        # Nothing trustworthy in the PDF: fall back to the filename alone.
        title = hint.title
        return ParsedDoc(
            title=title,
            title_source="filename" if title else "none",
            authors=[hint.author] if hint.author else [],
            year=hint.year,
            keywords=[],
            abstract=None,
            body="",
            profile_text=build_profile_text(title, [], None, "", profile_body_words),
            degraded=True,
        )

    body = strip_references(raw.text)
    pdf_title = title_from_spans(raw.page1_spans)
    title, title_source = choose_title(pdf_title, hint.title)

    abstract = extract_abstract(body, abstract_max_chars)
    abstract_source = "heading" if abstract else "none"
    if abstract is None:
        abstract = extract_lead_prose(body, abstract_max_chars)
        abstract_source = "lead-prose" if abstract else "none"

    keywords = extract_keywords(body)
    doi = extract_doi(raw.text)

    year = hint.year
    if year is None:
        copyright_year = re.search(r"©\s*(19|20)\d{2}", raw.text[:5000])
        if copyright_year:
            year = int(copyright_year.group(0)[-4:])

    return ParsedDoc(
        title=title,
        title_source=title_source,
        authors=[hint.author] if hint.author else [],
        year=year,
        doi=doi,
        keywords=keywords,
        abstract=abstract,
        abstract_source=abstract_source,
        body=body,
        profile_text=build_profile_text(title, keywords, abstract, body, profile_body_words),
        degraded=False,
    )
