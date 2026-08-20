"""Local FastAPI server for the results viewer.

Binds to localhost only. It exposes a directory browser and serves source PDFs,
which is fine for a tool running on your own machine and would not be fine on a
network interface - hence the hard-coded default host.
"""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from littax.config import Config

STATIC = Path(__file__).parent / "static"


class RunState:
    """Progress of the currently running pipeline, shared with the SSE stream."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.status = "idle"  # idle | running | done | error
        self.stage = ""
        self.done = 0
        self.total = 1
        self.detail = ""
        self.error: str | None = None
        self.version = 0  # bumps whenever a run finishes, so the page reloads

    def update(self, stage: str, done: int, total: int, detail: str) -> None:
        with self.lock:
            self.stage, self.done, self.total, self.detail = stage, done, total, detail

    def snapshot(self) -> dict[str, Any]:
        with self.lock:
            return {
                "status": self.status,
                "stage": self.stage,
                "done": self.done,
                "total": self.total,
                "detail": self.detail,
                "error": self.error,
                "version": self.version,
            }


class RunRequest(BaseModel):
    source: str


def create_app(cfg: Config) -> FastAPI:
    app = FastAPI(title="littax", docs_url=None, redoc_url=None)
    state = RunState()

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        return (STATIC / "index.html").read_text(encoding="utf-8")

    @app.get("/api/config")
    def get_config() -> dict[str, Any]:
        return {
            "source_root": str(cfg.source_root),
            "out_dir": str(cfg.out_dir),
            "embed_backend": cfg.embed.backend,
            "algorithm": cfg.cluster.algorithm,
            "label_enabled": cfg.label.enabled,
        }

    @app.get("/api/state")
    def get_state() -> JSONResponse:
        path = cfg.out_dir / "taxonomy.json"
        if not path.exists():
            raise HTTPException(404, "no taxonomy.json yet - run the analysis first")
        # Served as raw text: it is already JSON, and re-validating a 376-document
        # payload through pydantic on every page load buys nothing.
        return JSONResponse(content=json.loads(path.read_text(encoding="utf-8")))

    @app.get("/api/browse")
    def browse(path: str = "") -> dict[str, Any]:
        """Directory picker. Browsers cannot hand JavaScript a real filesystem
        path, so the server has to do the walking."""
        target = Path(path).expanduser() if path else Path.home()
        try:
            target = target.resolve()
            if not target.is_dir():
                target = target.parent
            entries = sorted(
                (child for child in target.iterdir() if child.is_dir()),
                key=lambda c: c.name.lower(),
            )
            pdfs = len(list(target.glob("*.pdf")))
        except (OSError, PermissionError) as exc:
            raise HTTPException(400, f"cannot read {target}: {exc}") from exc

        return {
            "path": str(target),
            "parent": str(target.parent) if target.parent != target else None,
            "pdf_count": pdfs,
            "directories": [
                {"name": child.name, "path": str(child)} for child in entries[:500]
            ],
        }

    @app.post("/api/run")
    def start_run(request: RunRequest) -> dict[str, Any]:
        with state.lock:
            if state.status == "running":
                raise HTTPException(409, "a run is already in progress")
            state.status = "running"
            state.error = None
            state.stage, state.done, state.total, state.detail = "starting", 0, 1, ""

        source = Path(request.source).expanduser()
        if not source.is_dir():
            with state.lock:
                state.status = "error"
                state.error = f"not a directory: {source}"
            raise HTTPException(400, f"not a directory: {source}")

        def worker() -> None:
            from littax.pipeline import run as run_pipeline, write_taxonomy

            try:
                run_cfg = cfg.model_copy(deep=True)
                run_cfg.source_root = source
                taxonomy = run_pipeline(run_cfg, state.update)
                write_taxonomy(taxonomy, run_cfg.out_dir)
                with state.lock:
                    state.status = "done"
                    state.version += 1
            except Exception as exc:
                with state.lock:
                    state.status = "error"
                    state.error = f"{type(exc).__name__}: {exc}"

        threading.Thread(target=worker, daemon=True).start()
        return {"started": True}

    @app.get("/api/progress")
    def progress() -> StreamingResponse:
        def stream():
            last: str | None = None
            while True:
                payload = json.dumps(state.snapshot())
                if payload != last:
                    yield f"data: {payload}\n\n"
                    last = payload
                elif state.snapshot()["status"] != "running":
                    # Keep the connection alive but stop spamming identical frames.
                    yield ": keepalive\n\n"
                time.sleep(0.4)

        return StreamingResponse(
            stream(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @app.get("/api/doc/{document_id}/pdf")
    def open_pdf(document_id: str) -> FileResponse:
        """Serve a source PDF.

        The path comes from the stored taxonomy, never from the request, so a
        crafted id cannot reach outside the analysed corpus.
        """
        path = cfg.out_dir / "taxonomy.json"
        if not path.exists():
            raise HTTPException(404, "no taxonomy.json")
        data = json.loads(path.read_text(encoding="utf-8"))
        root = Path(data["source_root"]).resolve()

        for document in data["documents"]:
            if document["id"] == document_id:
                relative = document.get("relpath") or document["filename"]
                candidate = (root / relative).resolve()
                if root != candidate and root not in candidate.parents:
                    raise HTTPException(403, "outside the corpus root")
                if not candidate.exists():
                    raise HTTPException(404, f"file is gone: {candidate.name}")
                return FileResponse(candidate, media_type="application/pdf")
        raise HTTPException(404, f"unknown document {document_id}")

    app.mount("/static", StaticFiles(directory=STATIC), name="static")
    return app


def serve(cfg: Config, host: str = "127.0.0.1", port: int = 8000) -> None:
    import uvicorn

    uvicorn.run(create_app(cfg), host=host, port=port, log_level="warning")
