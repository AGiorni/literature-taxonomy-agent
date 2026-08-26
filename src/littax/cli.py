"""Command line entry point."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Annotated, Optional

import typer
from rich.console import Console
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.table import Table
from rich.tree import Tree

from littax.capabilities import SDK_FOR_PROVIDER, probe_capabilities
from littax.config import PROJECT_DIR, Config, credentials_for
from littax.env import load_env, loaded_from, present_keys
from littax.models import Taxonomy

app = typer.Typer(add_completion=False, help=__doc__)
console = Console()


@app.callback()
def _bootstrap() -> None:
    """Runs before every command, so `.env` is honoured everywhere."""
    load_env()


def _record_ollama(cfg: Config, record) -> None:
    """Is the daemon up, is the model pulled, and how much context does it hold.

    Reported without --probe because none of it costs a token, and a local setup
    fails in ways the cloud providers cannot: daemon stopped, model never pulled.
    """
    from littax.label.client import OllamaLabeller

    labeller = OllamaLabeller(cfg.label, cfg.cache_dir)
    models = labeller.list_models()
    if not models:
        record(
            "ollama",
            False,
            f"not reachable at {labeller.base_url}. Install from https://ollama.com, "
            f"then: ollama pull {cfg.label.model}",
        )
        return

    record("ollama", True, f"{labeller.base_url} · {len(models)} model(s) installed")
    if cfg.label.model not in models:
        record(
            "model pulled",
            False,
            f"{cfg.label.model!r} not installed. Have: {', '.join(models)}. "
            f"Run: ollama pull {cfg.label.model}",
        )
    else:
        context = labeller.model_info()["context_length"]
        record("model pulled", True, f"{cfg.label.model} · {context} token context")


def _load_config(config_path: Optional[Path], source: Optional[Path]) -> Config:
    cfg = Config.load(config_path)
    if source is not None:
        cfg.source_root = source
    return cfg


@app.command()
def doctor(
    config: Annotated[Optional[Path], typer.Option("--config", "-c")] = None,
    source: Annotated[Optional[Path], typer.Argument()] = None,
    probe: Annotated[
        bool,
        typer.Option("--probe", help="Make one tiny real API call to confirm the model is deployed"),
    ] = False,
) -> None:
    """Check the environment before committing to a full run."""
    cfg = _load_config(config, source)
    table = Table("check", "status", "detail", show_lines=False)
    problems = 0

    def record(name: str, ok: bool, detail: str, fatal: bool = True) -> None:
        nonlocal problems
        if not ok and fatal:
            problems += 1
        mark = "[green]ok[/]" if ok else ("[red]fail[/]" if fatal else "[yellow]skip[/]")
        table.add_row(name, mark, detail)

    root = cfg.source_root
    if root.is_dir():
        pdfs = list(root.glob(cfg.ingest.glob))
        size_mb = sum(p.stat().st_size for p in pdfs) / 1e6
        record("source folder", bool(pdfs), f"{len(pdfs)} files, {size_mb:.0f} MB in {root}")
    else:
        record("source folder", False, f"not a directory: {root}")

    try:
        import pymupdf

        record("pymupdf", True, pymupdf.__doc__ or "installed")
    except ImportError as exc:
        record("pymupdf", False, str(exc))

    record("project", True, f"taxonomy.json in {cfg.out}", fatal=False)

    installed = probe_capabilities()

    if cfg.embed.backend == "tfidf":
        record("embedding backend", True, "tfidf (no download required)", fatal=False)
    elif installed["sentence_transformers"]:
        record("embedding backend", True, f"{cfg.embed.backend} via sentence-transformers")
    else:
        record(
            "embedding backend",
            False,
            f"backend '{cfg.embed.backend}' needs: uv sync --extra ml",
        )

    if cfg.cluster.reducer != "umap":
        record("umap", True, f"not needed (reducer={cfg.cluster.reducer})", fatal=False)
    elif installed["umap"]:
        record("umap", True, "installed")
    else:
        record("umap", False, "reducer='umap' needs: uv sync --extra ml")

    if cfg.label.enabled:
        from littax.label.client import (
            MODEL_RATES,
            PROVIDERS,
            supports_adaptive_thinking,
        )

        provider = cfg.label.provider
        if provider not in PROVIDERS:
            record("provider", False, f"unknown {provider!r}; expected one of "
                   + ", ".join(sorted(PROVIDERS)))
        else:
            target = cfg.label.deployment or cfg.label.model
            detail = f"{provider} → {target}"
            if cfg.label.deployment and cfg.label.deployment != cfg.label.model:
                detail += f" (deployment for {cfg.label.model})"
            record("provider", True, detail)

        model = cfg.label.model
        rates = MODEL_RATES.get(model)
        if provider == "ollama":
            # Not "pricing unknown": a local model costs nothing, and listing
            # the cloud price table here would only confuse.
            record("model", True, f"{model} · free, runs locally · json mode")
        elif rates:
            tuning = (
                "adaptive thinking"
                if supports_adaptive_thinking(model)
                else "json mode" if provider == "azure-openai" else "plain calls"
            )
            record("model", True, f"{model} · ~${rates[0]}/${rates[1]} per MTok · {tuning}")
        else:
            # Not fatal: an id this tool has not seen may still be valid.
            record(
                "model",
                True,
                f"{model} — unrecognised; pricing unknown. Known: "
                + ", ".join(sorted(MODEL_RATES)),
                fatal=False,
            )

        # Report the .env file by path and the keys by name only - never values.
        env_file = loaded_from()
        found = present_keys()
        record(
            ".env",
            True,
            f"{env_file} · found {', '.join(found)}" if env_file
            else "no .env found (using the shell environment)",
            fatal=False,
        )

        credential_class = credentials_for(provider)
        if credential_class is None:
            # A local model has nothing to authenticate against. `credentials`
            # stays truthy so the --probe branch below still runs.
            credentials = True
            record("credentials", True, "not needed - runs locally", fatal=False)
        else:
            credentials = credential_class.from_env()
            if credentials is None:
                missing = [n for n in credential_class.ENV_VARS if not os.environ.get(n)]
                hint = "set " + " and ".join(missing)
                if found:
                    # A common mix-up: the other provider's keys are present.
                    hint += f" (found instead: {', '.join(found)})"
                record("credentials", False, hint)
            else:
                problem = getattr(credentials, "problem", None)
                if problem:
                    record("credentials", False, problem)
                else:
                    record(
                        "credentials",
                        True,
                        getattr(credentials, "resource", None)
                        or getattr(credentials, "endpoint", ""),
                    )

        sdk = SDK_FOR_PROVIDER.get(provider)
        if sdk is None:
            record("sdk", True, f"none needed for {provider}", fatal=False)
        else:
            record(
                f"{sdk} sdk",
                installed.get(sdk, False),
                "installed" if installed.get(sdk) else "needs: uv sync --extra llm",
            )

        if provider == "ollama":
            _record_ollama(cfg, record)

        if probe and credentials:
            from littax.label.client import probe_labeller

            result = probe_labeller(cfg.label, cfg.cache_dir)

            if not result["ok"]:
                record("model deployed", False, result["reason"][:160])
            elif not result["matches"]:
                # A different deployment answering is worse than an outright
                # failure: the run would succeed and quietly bill at another
                # model's rate.
                record(
                    "model deployed",
                    False,
                    f"requested {result['requested']} but {result['served']} replied",
                )
            else:
                record("model deployed", True, f"{result['served']} answered")
        elif cfg.label.enabled:
            record(
                "model deployed",
                True,
                "not checked - re-run with --probe to make one tiny real call",
                fatal=False,
            )
    else:
        record("llm labelling", True, "disabled (label.enabled = false)", fatal=False)

    console.print(table)
    if problems:
        console.print(f"[red]{problems} problem(s) must be fixed before running.[/]")
        raise typer.Exit(code=1)
    console.print("[green]ready.[/]")


@app.command()
def run(
    source: Annotated[Optional[Path], typer.Argument(help="Folder of PDFs")] = None,
    config: Annotated[Optional[Path], typer.Option("--config", "-c")] = None,
    out: Annotated[Optional[Path], typer.Option("--out", "-o")] = None,
) -> None:
    """Discover the taxonomy and write <source>/littax/taxonomy.json."""
    from littax.pipeline import run as run_pipeline, write_taxonomy

    cfg = _load_config(config, source)
    if out is not None:
        cfg.out_dir = out

    with Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.fields[stage]:<8}[/]"),
        BarColumn(bar_width=30),
        TextColumn("{task.description}"),
        TimeElapsedColumn(),
        console=console,
    ) as bar:
        task = bar.add_task("starting", total=1, stage="init")

        def report(stage: str, done: int, total: int, detail: str) -> None:
            bar.update(
                task,
                completed=done,
                total=max(total, 1),
                stage=stage,
                description=detail[:60],
            )

        taxonomy = run_pipeline(cfg, report)

    path = write_taxonomy(taxonomy, cfg.out)
    _print_summary(taxonomy)
    console.print(f"\nwrote [bold]{path}[/]")


@app.command()
def serve(
    source: Annotated[Optional[Path], typer.Argument(help="Folder of PDFs")] = None,
    config: Annotated[Optional[Path], typer.Option("--config", "-c")] = None,
    out: Annotated[Optional[Path], typer.Option("--out", "-o")] = None,
    port: Annotated[int, typer.Option("--port", "-p")] = 8000,
) -> None:
    """Open the results viewer at http://localhost:8000."""
    try:
        from littax.server.app import serve as run_server
    except ImportError as exc:
        console.print(f"[red]the viewer needs FastAPI. Run: uv sync --extra web[/] ({exc})")
        raise typer.Exit(code=1) from exc

    cfg = _load_config(config, source)
    if out is not None:
        cfg.out_dir = out
    console.print(f"viewer on [bold]http://127.0.0.1:{port}[/]  (ctrl-c to stop)")
    run_server(cfg, port=port)


def _resolve_taxonomy(path: Optional[Path]) -> Path:
    """Find taxonomy.json when the user did not name one.

    Prefers the current project layout, <cwd>/littax/, and falls back to the
    legacy out/ so a working tree built before the move keeps working.
    """
    if path is not None:
        return path
    project = Path(PROJECT_DIR) / "taxonomy.json"
    legacy = Path("out") / "taxonomy.json"
    return legacy if not project.exists() and legacy.exists() else project


@app.command()
def apply(
    taxonomy_path: Annotated[Optional[Path], typer.Argument()] = None,
    mode: Annotated[str, typer.Option("--mode", help="dry-run|hardlink|copy|move")] = "dry-run",
    min_confidence: Annotated[float, typer.Option("--min-confidence")] = 0.0,
    target: Annotated[Optional[Path], typer.Option("--target")] = None,
) -> None:
    """Execute the reorganisation plan. Defaults to a dry run."""
    from littax.apply import apply_plan

    taxonomy_path = _resolve_taxonomy(taxonomy_path)
    taxonomy = Taxonomy.model_validate_json(taxonomy_path.read_text(encoding="utf-8"))
    result = apply_plan(
        taxonomy,
        mode=mode,
        min_confidence=min_confidence,
        target_root=target,
        log=console.print,
    )

    table = Table("outcome", "count", show_header=False, box=None)
    for name, count in result.items():
        table.add_row(name, str(count))
    console.print(table)
    if mode == "dry-run":
        console.print("\n[yellow]dry run - nothing was written.[/] Re-run with "
                      "[bold]--mode hardlink[/] to create the tree.")


@app.command("show")
def show(
    taxonomy_path: Annotated[Optional[Path], typer.Argument()] = None,
    max_depth: Annotated[int, typer.Option("--depth")] = 3,
) -> None:
    """Print the discovered hierarchy from an existing taxonomy.json."""
    taxonomy_path = _resolve_taxonomy(taxonomy_path)
    taxonomy = Taxonomy.model_validate_json(taxonomy_path.read_text(encoding="utf-8"))
    _print_summary(taxonomy)
    console.print()
    console.print(_render_tree(taxonomy, max_depth))


def _print_summary(taxonomy: Taxonomy) -> None:
    stats = taxonomy.stats
    table = Table("metric", "value", show_header=False, box=None)
    table.add_row("documents", str(stats.n_documents))
    table.add_row("duplicates skipped", str(stats.n_duplicates))
    table.add_row("degraded (likely scans)", str(stats.n_degraded))
    table.add_row("title recall", f"{stats.title_recall:.1%}")
    table.add_row("abstract recall", f"{stats.abstract_recall:.1%}")
    table.add_row("folders / leaves", f"{stats.n_folders} / {stats.n_leaf_folders}")
    table.add_row("levels", str(stats.levels))
    table.add_row("unsorted", str(stats.n_unsorted))
    bands = stats.band_counts
    table.add_row(
        "confidence",
        f"[green]high {bands.get('high', 0)}[/]  "
        f"[yellow]medium {bands.get('medium', 0)}[/]  "
        f"[red]low {bands.get('low', 0)}[/]",
    )
    console.print(table)


def _render_tree(taxonomy: Taxonomy, max_depth: int) -> Tree:
    folders = taxonomy.folder_by_id()
    children: dict[str | None, list] = {}
    for folder in taxonomy.folders:
        children.setdefault(folder.parent_id, []).append(folder)

    root = Tree(f"[bold]{taxonomy.source_root}[/]")

    def add(parent_widget, folder) -> None:
        stats = folder.confidence_stats
        mean = f"{stats.mean:.2f}" if stats else "-"
        colour = "green" if stats and stats.mean >= 0.7 else "yellow"
        node = parent_widget.add(
            f"[bold]{folder.label}[/]  "
            f"[dim]({folder.n_documents_total} docs, conf [{colour}]{mean}[/])[/]"
        )
        if folder.level >= max_depth:
            return
        for child in children.get(folder.id, []):
            add(node, child)

    for folder in sorted(
        children.get(None, []), key=lambda f: f.n_documents_total, reverse=True
    ):
        add(root, folder)
    return root


def main() -> None:  # pragma: no cover
    try:
        app()
    except KeyboardInterrupt:
        console.print("[yellow]interrupted[/]")
        sys.exit(130)


if __name__ == "__main__":  # pragma: no cover
    main()
