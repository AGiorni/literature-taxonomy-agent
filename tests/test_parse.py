"""Ingest heuristics. Fixtures are drawn from real filenames and page layouts
in the corpus, including the awkward ones."""

from __future__ import annotations

import pytest

from littax.ingest.parse import (
    build_profile_text,
    choose_title,
    extract_abstract,
    extract_doi,
    extract_keywords,
    extract_lead_prose,
    parse_filename,
    strip_references,
    title_from_spans,
)
from littax.ingest.pdf_text import Span, normalise


class TestParseFilename:
    @pytest.mark.parametrize(
        "stem,author,year,title_contains",
        [
            ("Adkins 2006 - Motor training induces plasticity", "Adkins", 2006, "Motor training"),
            ("Allen 1995-twitch interpolation", "Allen", 1995, "twitch interpolation"),
            ("Wang - Chronic jaw pain attenuates", "Wang", None, "Chronic jaw pain"),
            ("Badr 2024 - Effect_of_Phasic_Experimental_Pain", "Badr", 2024, "Effect of Phasic"),
            (
                "pieters 2020 - an-update-of-systematic-reviews-examining",
                "pieters",
                2020,
                "an update of systematic reviews",
            ),
        ],
    )
    def test_extracts_parts(self, stem, author, year, title_contains):
        hint = parse_filename(stem)
        assert hint.author == author
        assert hint.year == year
        assert title_contains in hint.title

    def test_no_dash_still_finds_year(self):
        hint = parse_filename("Zandvoort thesis 2022")
        assert hint.year == 2022

    def test_hyphenated_compound_is_not_split(self):
        # Only 2 hyphens, so they are real compounds, not separators.
        hint = parse_filename("Smith 2020 - non-negative matrix factorisation")
        assert "non-negative" in hint.title


class TestTitle:
    def test_picks_largest_font_in_top_half(self):
        spans = [
            Span("Journal of Applied Physiology Vol 101", 8.0, 0.02),
            Span("Differential control of the scapulothoracic muscles in humans", 14.0, 0.15),
            Span("C. Alexander, R. Miley, S. Stynes", 9.0, 0.25),
        ]
        assert title_from_spans(spans) == (
            "Differential control of the scapulothoracic muscles in humans"
        )

    def test_merges_multi_line_title(self):
        spans = [
            Span("Differential control of the scapulothoracic", 14.0, 0.15),
            Span("muscles in humans and other primates", 14.0, 0.20),
        ]
        title = title_from_spans(spans)
        assert title.startswith("Differential control")
        assert title.endswith("primates")

    def test_ignores_boilerplate_and_lower_half(self):
        spans = [
            Span("Downloaded from journals.physiology.org at UCL", 16.0, 0.03),
            Span("A genuine paper title about motor cortex plasticity", 12.0, 0.18),
            Span("THIS HUGE TEXT IS BELOW THE FOLD AND MUST BE IGNORED", 30.0, 0.80),
        ]
        assert title_from_spans(spans) == "A genuine paper title about motor cortex plasticity"

    def test_returns_none_when_nothing_plausible(self):
        assert title_from_spans([Span("Vol. 3", 20.0, 0.1)]) is None


class TestChooseTitle:
    def test_prefers_pdf_when_they_agree(self):
        title, source = choose_title(
            "Differences in motor unit firing properties of the vastus lateralis",
            "Differences in MU firing properties of the VL during postural tasks",
        )
        assert source == "pdf"
        assert "vastus lateralis" in title

    def test_falls_back_to_filename(self):
        title, source = choose_title(None, "twitch interpolation")
        assert (title, source) == ("twitch interpolation", "filename")

    def test_none_when_neither(self):
        assert choose_title(None, None) == (None, "none")


class TestAbstract:
    def test_heading_form(self):
        text = "Title\nAbstract\n" + "We measured cortical excitability in humans. " * 6 + "\nIntroduction\nBody text"
        abstract = extract_abstract(text)
        assert abstract.startswith("We measured cortical excitability")
        assert "Introduction" not in abstract

    def test_letter_spaced_heading(self):
        text = "A B S T R A C T\n" + "Persistent inward currents sustain firing. " * 8
        assert extract_abstract(text).startswith("Persistent inward currents")

    def test_too_short_is_rejected(self):
        assert extract_abstract("Abstract\nShort.\nIntroduction") is None

    def test_lead_prose_recovers_unheaded_abstract(self):
        # J Physiol style: affiliations, then the abstract, then Introduction.
        text = (
            "J Physiol 580.3 (2007) pp 777-786\n"
            "Differential control of the scapulothoracic muscles\n"
            "C. Alexander, R. Miley, S. Stynes and P. J. Harrison\n"
            "1Department of Physiotherapy, Hammersmith Hospitals NHS Trust, London, UK\n"
            "The control of the scapulothoracic muscles trapezius and serratus anterior has been\n"
            "examined in normal human subjects using surface electromyography recordings taken\n"
            "bilaterally from the upper and lower portions of each muscle during arm elevation.\n"
            "Magnetic stimulation of the motor cortex was used to examine descending control and\n"
            "electrical stimulation of peripheral nerves was used to examine reflex responses.\n"
            "The average optimal site for evoking responses differed between the two muscles.\n"
            "Introduction\n"
            "The scapulothoracic muscles have long been\n"
        )
        prose = extract_lead_prose(text)
        assert prose.startswith("The control of the scapulothoracic")
        assert "Department of Physiotherapy" not in prose
        assert "Harrison" not in prose

    def test_lead_prose_recovers_after_citation_em_dash(self):
        long_tail = "The motor cortex and spinal cord alter structure in response to training. " * 4
        text = (
            "Adkins, DeAnna L. Motor training induces plasticity. J Appl Physiol 101: 1776, "
            f"2006. doi:10.1152/japplphysiol.00515.2006.—{long_tail}\n"
            "INTRODUCTION\n"
        )
        prose = extract_lead_prose(text)
        assert prose.startswith("The motor cortex and spinal cord")
        assert "doi" not in prose


class TestReferences:
    def test_strips_trailing_reference_list(self):
        body = "Body of the paper. " * 60
        text = body + "\nReferences\n" + "Smith J. et al. 2001. " * 40
        stripped = strip_references(text)
        assert "Smith J." not in stripped
        assert stripped.startswith("Body of the paper")

    def test_ignores_early_mention(self):
        text = "See references for details. " * 5 + ("Real body content here. " * 80)
        assert len(strip_references(text)) == len(text)


class TestMisc:
    def test_keywords(self):
        assert extract_keywords("Keywords: motor unit; recruitment, postural task\n") == [
            "motor unit",
            "recruitment",
            "postural task",
        ]

    def test_doi(self):
        assert extract_doi("see doi:10.1152/jn.00515.2006, published") == "10.1152/jn.00515.2006"

    def test_normalise_folds_ligatures(self):
        assert normalise("ﬁring speciﬁc") == "firing specific"

    def test_normalise_rejoins_hyphenated_linebreak(self):
        assert "corticospinal" in normalise("cortico-\nspinal tract")

    def test_profile_repeats_title(self):
        profile = build_profile_text("Motor unit firing", [], "An abstract sentence.", "", 400)
        assert profile.count("Motor unit firing") == 2

    def test_profile_falls_back_to_body(self):
        profile = build_profile_text("T", [], None, "one two three four five", 3)
        assert profile.endswith("one two three")
