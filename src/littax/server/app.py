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

from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from littax.capabilities import SDK_FOR_PROVIDER, missing_for, probe_capabilities
from littax.config import (
    PROJECT_DIR,
    AzureOpenAICredentials,
    Config,
    FoundryCredentials,
    LabelConfig,
    has_project,
)
from littax.env import loaded_from, present_keys, write_keys

STATIC = Path(__file__).parent / "static"

Provider = Literal["foundry", "azure-openai"]

# What each provider needs from the user, so the setup form is described here
# rather than duplicated in JavaScript.
PROVIDER_FIELDS: dict[str, dict[str, Any]] = {
    "foundry": {
        "name": "Claude on Microsoft Foundry",
        "default_model": "claude-opus-5",
        "needs_deployment": False,
        "fields": [
            {
                "id": "resource",
                "label": "Foundry resource",
                "placeholder": "my-resource",
                "secret": False,
            },
            {"id": "api_key", "label": "API key", "placeholder": "", "secret": True},
        ],
    },
    "azure-openai": {
        "name": "GPT on Azure OpenAI",
        "default_model": "gpt-4o",
        "needs_deployment": True,
        "fields": [
            {
                "id": "endpoint",
                "label": "Endpoint",
                "placeholder": "https://my-resource.openai.azure.com/",
                "secret": False,
            },
            {"id": "api_key", "label": "API key", "placeholder": "", "secret": True},
        ],
    },
}


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


class Session:
    """Which project the viewer is showing, and the credentials it was given.

    Credentials live here and nowhere else: never on Config, which is
    serialised into taxonomy.json, and never written to disk unless the user
    asks for it explicitly.
    """

    def __init__(self, source: Path | None = None) -> None:
        self.lock = threading.Lock()
        self.source = source
        self.credentials: Any | None = None
        self.label: LabelConfig | None = None

    def taxonomy_path(self) -> Path | None:
        if self.source is None:
            return None
        return self.source / PROJECT_DIR / "taxonomy.json"


class Credentials(BaseModel):
    """One shape for both providers; each reads the fields it needs."""

    api_key: str = ""
    endpoint: str = ""  # azure-openai
    resource: str = ""  # foundry


class LlmSettings(BaseModel):
    provider: Provider = "foundry"
    model: str | None = None
    deployment: str | None = None
    api_version: str | None = None
    credentials: Credentials | None = None


class TestRequest(LlmSettings):
    pass


class OpenRequest(BaseModel):
    source: str


class RunRequest(BaseModel):
    source: str
    ai_assisted: bool = False
    llm: LlmSettings | None = None
    remember: bool = False


def _build_credentials(settings: LlmSettings) -> Any | None:
    """Turn form values into the provider's credential model.

    Dispatches on the literal provider rather than a fallback lookup, so an
    unexpected value cannot quietly be treated as Azure.
    """
    supplied = settings.credentials
    if supplied is None:
        return None
    if settings.provider == "foundry":
        if not (supplied.api_key and supplied.resource):
            return None
        return FoundryCredentials(api_key=supplied.api_key, resource=supplied.resource)
    if not (supplied.api_key and supplied.endpoint):
        return None
    return AzureOpenAICredentials(api_key=supplied.api_key, endpoint=supplied.endpoint)


def _env_values(credentials: Any) -> dict[str, str]:
    """Credentials as the environment variable names that hold them."""
    if isinstance(credentials, FoundryCredentials):
        return {
            "ANTHROPIC_FOUNDRY_API_KEY": credentials.api_key,
            "ANTHROPIC_FOUNDRY_RESOURCE": credentials.resource,
        }
    return {
        "AZURE_OPENAI_API_KEY": credentials.api_key,
        "AZURE_OPENAI_ENDPOINT": credentials.endpoint,
    }


def _label_config(base: LabelConfig, settings: LlmSettings) -> LabelConfig:
    """The run's label settings: the config file's, overridden by the form."""
    return base.model_copy(
        update={
            "enabled": True,
            "provider": settings.provider,
            "model": settings.model or PROVIDER_FIELDS[settings.provider]["default_model"],
            "deployment": settings.deployment or None,
            "api_version": settings.api_version or base.api_version,
        }
    )


def _scrub(message: str, secrets: list[str]) -> str:
    """Keep an api_key out of an error string.

    SDK errors occasionally quote the request that produced them, and this
    message is rendered straight into the page.
    """
    for secret in secrets:
        if len(secret) >= 8:
            message = message.replace(secret, "***")
    return message


def create_app(cfg: Config) -> FastAPI:
    app = FastAPI(title="littax", docs_url=None, redoc_url=None)
    state = RunState()
    # Open straight into a result when serve was pointed at an analysed folder.
    session = Session(cfg.source_root if has_project(cfg.source_root) else None)

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        return (STATIC / "index.html").read_text(encoding="utf-8")

    @app.get("/api/config")
    def get_config() -> dict[str, Any]:
        # `missing` is reported for a labelling run, so the mode step can block
        # before the user has invested any clicks. Standard mode re-checks
        # against its own config at /api/run.
        with_label = cfg.model_copy(deep=True)
        with_label.label = cfg.label.model_copy(update={"enabled": True})

        return {
            "source_root": str(cfg.source_root),
            "out_dir": str(cfg.out),
            "embed_backend": cfg.embed.backend,
            "algorithm": cfg.cluster.algorithm,
            "label_enabled": cfg.label.enabled,
            "has_project": session.source is not None,
            "project_source": str(session.source) if session.source else None,
            "missing": missing_for(cfg),
            "missing_for_llm": missing_for(with_label),
            # Names only, never values - so the form can pre-select a provider
            # whose credentials are already on this machine.
            "env_keys": present_keys(),
            "env_file": str(loaded_from() or (Path.cwd() / ".env")),
        }

    @app.get("/api/providers")
    def get_providers() -> dict[str, Any]:
        installed = probe_capabilities()
        present = set(present_keys())
        providers = []
        for key, spec in PROVIDER_FIELDS.items():
            sdk = SDK_FOR_PROVIDER[key]
            credential_class = (
                FoundryCredentials if key == "foundry" else AzureOpenAICredentials
            )
            providers.append(
                {
                    "id": key,
                    **spec,
                    "sdk_installed": installed.get(sdk, False),
                    "env_ready": all(n in present for n in credential_class.ENV_VARS),
                }
            )
        return {"providers": providers}

    @app.get("/api/state")
    def get_state() -> JSONResponse:
        path = session.taxonomy_path()
        if path is None or not path.exists():
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
            "has_taxonomy": has_project(target),
            "directories": [
                {
                    "name": child.name,
                    "path": str(child),
                    "has_taxonomy": has_project(child),
                }
                for child in entries[:500]
            ],
        }

    @app.post("/api/open")
    def open_project(request: OpenRequest) -> dict[str, Any]:
        """Show an already-analysed folder without re-running anything."""
        source = Path(request.source).expanduser()
        if not has_project(source):
            raise HTTPException(404, f"no {PROJECT_DIR}/taxonomy.json in {source}")
        with session.lock:
            session.source = source.resolve()
        return {"ok": True}

    @app.post("/api/llm/test")
    def test_llm(request: TestRequest) -> dict[str, Any]:
        """One tiny real call, to catch a bad key before a long run starts."""
        from littax.label.client import probe_labeller

        label_cfg = _label_config(cfg.label, request)
        missing = missing_for(cfg.model_copy(update={"label": label_cfg}))
        if missing:
            return {"ok": False, "reason": "; ".join(missing), "deployments": []}

        credentials = _build_credentials(request)
        try:
            result = probe_labeller(
                label_cfg, cfg.cache_dir, credentials, deployments=True
            )
        except Exception as exc:  # a transport failure is a result, not a 500
            result = {"ok": False, "reason": f"{type(exc).__name__}: {exc}", "deployments": []}

        secrets = [request.credentials.api_key] if request.credentials else []
        if result.get("reason"):
            result["reason"] = _scrub(str(result["reason"]), secrets)
        return result

    @app.post("/api/run")
    def start_run(request: RunRequest) -> dict[str, Any]:
        source = Path(request.source).expanduser()
        if not source.is_dir():
            raise HTTPException(400, f"not a directory: {source}")

        # Everything the run needs, decided before a thread exists. embed and
        # cluster are deliberately not overridable from the browser: the UI
        # chooses whether the LLM runs, not how documents are grouped.
        run_cfg = cfg.model_copy(deep=True)
        run_cfg.source_root = source
        run_cfg.out_dir = None  # per-project: <source>/littax
        credentials = None

        if request.ai_assisted:
            settings = request.llm or LlmSettings()
            run_cfg.label = _label_config(cfg.label, settings)
            credentials = _build_credentials(settings)
        else:
            run_cfg.label = cfg.label.model_copy(update={"enabled": False})

        missing = missing_for(run_cfg)
        if missing:
            # Fail here rather than several minutes in, inside the worker.
            raise HTTPException(400, "install first: " + "; ".join(missing))

        secrets = [
            c.api_key
            for c in [request.llm.credentials if request.llm else None]
            if c and c.api_key
        ]

        if request.remember and credentials is not None:
            write_keys(_env_values(credentials))

        with state.lock:
            if state.status == "running":
                raise HTTPException(409, "a run is already in progress")
            state.status = "running"
            state.error = None
            state.stage, state.done, state.total, state.detail = "starting", 0, 1, ""

        def worker() -> None:
            from littax.pipeline import run as run_pipeline, write_taxonomy

            try:
                taxonomy = run_pipeline(run_cfg, state.update, credentials)
                write_taxonomy(taxonomy, run_cfg.out)
                with session.lock:
                    session.source = source.resolve()
                    session.credentials = credentials
                    session.label = run_cfg.label
                with state.lock:
                    state.status = "done"
                    state.version += 1
            except Exception as exc:
                with state.lock:
                    state.status = "error"
                    state.error = _scrub(f"{type(exc).__name__}: {exc}", secrets)

        threading.Thread(target=worker, daemon=True).start()
        return {"started": True}

    @app.post("/api/dismiss")
    def dismiss() -> dict[str, Any]:
        """Clear a finished run's overlay.

        Without this the status stays "error" for the process lifetime and the
        page re-shows a stale failure on every load.
        """
        with state.lock:
            if state.status in ("done", "error"):
                state.status = "idle"
                state.error = None
        return {"ok": True}

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
        path = session.taxonomy_path()
        if path is None or not path.exists():
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
