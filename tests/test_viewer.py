"""Drive the viewer's JavaScript headlessly.

A syntax check cannot catch a `TypeError` inside `drawScatter`, and the viewer
is the part of this project least covered by the Python suite. `smoke.mjs`
stubs the DOM and Plotly, runs the real `app.js` against a generated
taxonomy, and exercises every interaction that could throw.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import pytest

from littax.models import (
    Assignment,
    ConfidenceStats,
    Document,
    Extraction,
    Folder,
    RunnerUp,
    Stats,
    Taxonomy,
)

ROOT = Path(__file__).resolve().parents[1]
HARNESS = ROOT / "tests" / "viewer" / "smoke.mjs"

pytestmark = pytest.mark.skipif(
    shutil.which("node") is None, reason="node is not installed"
)


def _folder(folder_id, level, parent, label, total, mean, children=()):
    return Folder(
        id=folder_id, level=level, parent_id=parent, path=label, label=label,
        description=f"{label} description",
        n_documents_direct=total if not children else 0,
        n_documents_total=total,
        confidence_stats=ConfidenceStats(
            mean=mean, median=mean, p10=mean - 0.1, p90=mean + 0.1,
        ),
        children=list(children),
    )


@pytest.fixture
def taxonomy_file(tmp_path: Path) -> Path:
    """A small taxonomy that still covers every shape the viewer must handle:
    nesting, a duplicate, an unassigned document, and all three bands."""
    folders = [
        _folder("L1.001", 1, None, "Motor Control", 6, 0.72, ["L2.001", "L2.002"]),
        _folder("L2.001", 2, "L1.001", "Motor Units", 3, 0.81),
        _folder("L2.002", 2, "L1.001", "Corticospinal", 3, 0.63),
        _folder("L1.002", 1, None, "Shoulder Pain", 2, 0.55),
    ]
    leaves = ["L2.001", "L2.001", "L2.001", "L2.002", "L2.002", "L2.002", "L1.002", "L1.002"]
    bands = ["high", "high", "medium", "medium", "low", "low", "high", "low"]

    documents = []
    for n, (leaf, band) in enumerate(zip(leaves, bands)):
        confidence = {"high": 0.85, "medium": 0.55, "low": 0.25}[band]
        documents.append(
            Document(
                id=f"doc_{n:04d}",
                source_path=f"raw/paper{n}.pdf", relpath=f"paper{n}.pdf",
                filename=f"paper{n}.pdf", sha256=f"{n:064d}", size_bytes=100,
                title=f"Paper {n} on motor unit discharge", authors=[f"Author{n}"],
                year=2000 + n, doi=f"10.1000/{n}",
                keywords=["motor unit"], abstract_excerpt="An abstract.",
                top_terms=["motor unit", "emg"],
                extraction=Extraction(method="pymupdf", pages=8, chars=1000),
                coords={"global": [n * 0.1 - 0.4, (n % 3) * 0.2 - 0.2]},
                assignment=Assignment(
                    folder_id=leaf,
                    folder_path=next(f.path for f in folders if f.id == leaf),
                    confidence=confidence, band=band,
                    similarity_to_centroid=0.7, margin_over_next=0.15,
                    membership_probability=0.9,
                    was_noise=(band == "low"),
                    llm_verified=False if band == "low" else None,
                    rationale="Torn between two topics." if band == "low" else None,
                    runner_up=RunnerUp(
                        folder_id="L2.002", folder_path="Corticospinal", similarity=0.55
                    ),
                ),
            )
        )

    # A byte-duplicate (no assignment) - the viewer must filter it out.
    documents.append(
        Document(
            id="doc_0100", source_path="raw/dup.pdf", relpath="dup.pdf",
            filename="dup.pdf", sha256="d" * 64, size_bytes=100,
            title="A duplicate", duplicate_of="paper0.pdf",
            extraction=Extraction(method="pymupdf"),
        )
    )

    taxonomy = Taxonomy(
        generated_at=datetime.now(timezone.utc),
        source_root="raw", target_root="organised", config={},
        calibration={"chance_similarity": 0.1, "ceiling_similarity": 0.6, "margin_scale": 0.3},
        stats=Stats(
            n_documents=9, n_parsed=9, n_degraded=0, n_duplicates=1,
            n_folders=4, n_leaf_folders=3, n_unsorted=0, levels=2,
            band_counts={"high": 3, "medium": 2, "low": 3},
            abstract_recall=0.98, title_recall=1.0,
        ),
        folders=folders,
        documents=documents,
    )
    path = tmp_path / "taxonomy.json"
    path.write_text(taxonomy.model_dump_json(), encoding="utf-8")
    return path


def test_viewer_survives_every_interaction(taxonomy_file: Path):
    result = subprocess.run(
        ["node", str(HARNESS), str(taxonomy_file)],
        cwd=ROOT, capture_output=True, text=True, timeout=180,
    )
    assert result.returncode == 0, (
        f"viewer smoke test failed\n--- stdout ---\n{result.stdout}\n"
        f"--- stderr ---\n{result.stderr}"
    )
    assert "viewer smoke test: OK" in result.stdout


class TestHiddenAttribute:
    """Regression guard.

    `hidden` is honoured only by the UA stylesheet's `display: none`, which any
    author-level `display` declaration outranks. Several panels set
    `display: grid`, so without a global override the progress overlay renders
    permanently on top of the page and blocks every click.
    """

    CSS = (ROOT / "src" / "littax" / "server" / "static" / "app.css").read_text(
        encoding="utf-8"
    )
    JS = (ROOT / "src" / "littax" / "server" / "static" / "app.js").read_text(
        encoding="utf-8"
    )
    HTML = (ROOT / "src" / "littax" / "server" / "static" / "index.html").read_text(
        encoding="utf-8"
    )

    def test_global_override_exists(self):
        rule = re.search(r"\[hidden\]\s*\{([^}]*)\}", self.CSS)
        assert rule, "app.css must define a global [hidden] rule"
        body = rule.group(1).replace(" ", "")
        assert "display:none!important" in body

    def test_every_toggled_panel_is_covered(self):
        """Any element the script shows/hides must start hidden in the markup,
        so nothing flashes before the first render."""
        toggled = set(re.findall(r'\$\("#([\w-]+)"\)\.hidden', self.JS))
        assert toggled, "expected the viewer to toggle some panels"
        for element_id in toggled:
            tag = re.search(rf'id="{element_id}"[^>]*>', self.HTML)
            assert tag, f"#{element_id} is toggled in JS but absent from the markup"

    def test_progress_overlay_starts_hidden(self):
        tag = re.search(r'<div id="progress"[^>]*>', self.HTML)
        assert tag and "hidden" in tag.group(0)


class TestScriptMarkupAgreement:
    """The script and the markup must describe the same page.

    `$("#missing")` returns null, and the TypeError kills the rest of whatever
    function it is in. When that function is one of the wiring calls at the
    bottom of app.js, every later one is skipped too - so a single stray id
    silently unwires the progress stream and leaves buttons doing nothing.
    That failure reads as "the app ignores me", which is expensive to diagnose
    from the outside; catching it here costs nothing.
    """

    JS = TestHiddenAttribute.JS
    HTML = TestHiddenAttribute.HTML

    def test_every_referenced_id_exists(self):
        referenced = set(re.findall(r'\$\("#([\w-]+)"\)', self.JS))
        defined = set(re.findall(r'id="([\w-]+)"', self.HTML))
        assert referenced, "expected the script to address elements by id"
        missing = sorted(referenced - defined)
        assert not missing, f"app.js addresses ids absent from index.html: {missing}"

    def test_a_broken_script_is_visible(self):
        """A dead page must say so, rather than just sitting there."""
        assert 'id="js-error"' in self.HTML
        # Registered ahead of the scripts it guards, so a parse error in
        # app.js still reaches it.
        handler = self.HTML.index('addEventListener("error"')
        assert handler < self.HTML.index('src="/static/app.js"')


def test_taxonomy_json_round_trips(taxonomy_file: Path):
    """The viewer reads raw JSON, so the schema must survive serialisation."""
    data = json.loads(taxonomy_file.read_text(encoding="utf-8"))
    restored = Taxonomy.model_validate(data)
    assert restored.stats.n_documents == 9
    assert restored.folder_by_id()["L2.001"].label == "Motor Units"
