# littax — literature taxonomy agent

Points at a folder of research PDFs, **discovers** a multi-level topic taxonomy
from the corpus itself, assigns every paper to a leaf folder with a confidence
score, and writes a reorganisation plan to `out/taxonomy.json`.

Your source folder is never modified. Reorganising is a separate, explicit step.

```bash
uv sync --extra ml --extra web      # + --extra llm for LLM folder names
uv run littax doctor raw            # check the environment first
uv run littax run raw               # discover the taxonomy
uv run littax serve raw             # inspect it at http://127.0.0.1:8000
uv run littax apply out/taxonomy.json --mode hardlink
```

## Starting from nothing

The only prerequisite is [uv](https://docs.astral.sh/uv/), which installs its
own Python — you do not need one already.

```powershell
# Windows
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```
```bash
# macOS / Linux
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Then, from the project folder:

```bash
uv sync --extra ml --extra web
uv run littax serve "C:/path/to/your/papers"
```

Open <http://127.0.0.1:8000>, click **Analyse a folder…**, pick your PDFs and
press Run. No API key, no account, nothing leaves your machine.

**First run downloads about 3 GB** (PyTorch plus one embedding model) and then
works offline forever. To skip that entirely, create `config.toml` with:

```toml
[embed]
backend = "tfidf"          # pure scikit-learn, no download
[cluster]
algorithm = "agglomerative"
reducer = "none"
```

and install with plain `uv sync --extra web`. The taxonomy is noticeably
coarser — bag-of-words cannot tell that "corticospinal excitability" and "MEP
amplitude" are the same subject — but it runs in about a minute on a few
hundred papers and proves the tool out before you commit to the big install.

Folder names are the only part that ever needs an LLM, and it is off by default.

## What it does

```
raw/*.pdf
  │
  ├─ INGEST    PyMuPDF text + page-1 typography → title, abstract, keywords,
  │            year, DOI. References stripped; ligatures NFKC-folded.
  ├─ EMBED     local sentence-transformer → one unit-norm vector per paper
  ├─ CLUSTER   recursive UMAP → HDBSCAN, refit inside every subtree
  ├─ SCORE     centroid similarity + margin over runner-up → confidence
  ├─ LABEL     Claude names and describes each folder  (optional)
  └─ EXPORT    taxonomy.json + move plan + 2D layout
```

Every stage is cached by content hash under `.cache/`, so re-runs only redo what
changed. Deleting `.cache/` costs time, nothing else.

## Reading the confidence score

Confidence is **relative to the corpus**, not an absolute probability. Raw cosine
values are not comparable across embedding models — TF-IDF centroid similarities
sit near 0.4 where a sentence transformer's sit near 0.8 — so the score is
calibrated against two corpus-derived quantities recorded in `taxonomy.json`:

- `chance_similarity` — how similar a paper is to a folder it does *not* belong to.
- `ceiling_similarity` — what a well-placed paper in this corpus actually achieves.

A confidence of 0.8 means "this fit is 80% of the way from chance to the best
fits seen in this corpus". Its components (`similarity_to_centroid`,
`margin_over_next`, `membership_probability`) are all exported alongside it, so
you can re-weight without re-running anything.

**The score works.** On a 376-paper motor-control corpus, every paper that the
clustering placed wrongly came back low-confidence, and every correct placement
came back high. Sorting the review queue by confidence surfaces the real errors
first.

## Configuration

Copy `config.example.toml` to `config.toml` and pass `-c config.toml`. The
defaults assume `--extra ml`; for a zero-download run set:

```toml
[embed]
backend = "tfidf"       # no PyTorch, no model download
[cluster]
algorithm = "agglomerative"
reducer = "none"
```

That path is meaningfully worse — bag-of-words cannot tell that "corticospinal
excitability" and "MEP amplitude" are the same subject — but it runs anywhere and
needs no downloads.

## Claude folder names (optional)

Without an LLM, folders are named from their discriminative c-TF-IDF terms
(`lbp, fascicles, trunk muscles`). With one, they get real names and
descriptions.

This project targets **Claude on Microsoft Foundry** (Azure). Set:

```bash
ANTHROPIC_FOUNDRY_API_KEY=...
ANTHROPIC_FOUNDRY_RESOURCE=...
```

then `enabled = true` under `[label]`. `littax doctor` verifies both before any
work starts.

Three passes run: each folder is named **contrastively** (its siblings are in the
prompt, so they cannot all come back as "Motor Cortex Excitability"); the whole
tree is then reviewed for coherence; and low-confidence papers get a second
opinion. Every response is cached by prompt hash, so an interrupted run resumes
for free.

Two Foundry-specific notes:

- **There is no Batch API on Foundry**, so calls run through a thread pool
  instead of the 50%-off asynchronous path. Cost for a 376-paper corpus is still
  well under $2 at Opus 5 rates.
- Only paper titles and c-TF-IDF terms are sent — roughly 20 titles per folder,
  plus a short abstract excerpt for the handful of low-confidence papers. Full
  PDF text never leaves the machine, and nothing is sent to any third-party
  bibliographic service.

## The viewer

`littax serve` opens a local page bound to `127.0.0.1`. It exposes a directory
browser and serves source PDFs, which is why it does not listen on a public
interface.

- **Taxonomy tree** — doc counts and a mean-confidence meter per folder.
  Selecting a folder filters every other panel.
- **Embedding space** — the UMAP projection, coloured by confidence, with folder
  names drawn at cluster centroids. Click a point for the paper.
- **Hierarchy** — sunburst sized by document count, shaded by mean confidence.
- **Confidence distribution** — the selection against the whole corpus.
- **Documents** — the review queue, lowest confidence first, with the folder each
  paper *almost* went to.

## Applying the plan

`littax apply` defaults to `--mode dry-run` and prints what it would do.

| mode | effect |
|---|---|
| `dry-run` | prints the tree, writes nothing (default) |
| `hardlink` | full browsable tree, no extra disk — the one to use |
| `copy` | duplicates the bytes |
| `move` | the only mode that disturbs `raw/` |

Every file is re-hashed against the analysis before being touched, existing
files are never overwritten, and re-running is idempotent. `--min-confidence 0.6`
files only what the model was sure about and leaves the rest for you.

## Development

```bash
uv run pytest
```

Tests cover parsing against real corpus filenames, confidence maths on synthetic
geometry with known answers, Windows path safety, the labelling layer against a
stub client (no network, no credentials), and every apply-stage refusal.
