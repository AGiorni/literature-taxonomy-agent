"""LLM providers for the labelling passes.

Three backends, one interface. Everything above this module only ever calls
`ask_json(system, prompt)` and `map_concurrent(...)`, so adding a provider does
not touch the naming, coherence or verification logic.

- `FoundryLabeller`     - Claude on Microsoft Foundry, via the Anthropic SDK.
- `AzureOpenAILabeller` - GPT models on Azure OpenAI, via the openai SDK.
- `OllamaLabeller`      - a model running locally. Free, offline, no account,
                          and no SDK: it speaks Ollama's HTTP API over urllib.

No provider has a Batch API available here, so throughput comes from a
modest thread pool. Every response is cached by prompt hash, which matters more
than it sounds: a labelling run that dies halfway through costs nothing to
resume, and switching models never returns a stale answer because the model id
is part of the cache key.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, TypeVar

from littax.cache import JsonCache
from littax.config import AzureOpenAICredentials, FoundryCredentials, LabelConfig

T = TypeVar("T")

_FENCE = re.compile(r"^\s*```(?:json)?\s*|\s*```\s*$", re.MULTILINE)


class LabellingUnavailable(RuntimeError):
    """Raised when labelling is requested but cannot run."""


# Characters Windows forbids in a path component. Model ids are not chosen with
# filesystems in mind, so anything used as a directory name has to be cleaned.
_UNSAFE_PATH_CHARS = re.compile(r'[<>:"/\\|?*\x00-\x1f]')


def cache_namespace(provider: str, model: str) -> str:
    """A path-safe directory name identifying one provider-and-model pair."""
    return _UNSAFE_PATH_CHARS.sub("-", f"{provider}-{model}").strip(" .") or provider


# --------------------------------------------------------------------------- #
# Model capabilities and pricing
# --------------------------------------------------------------------------- #

# Claude models that accept adaptive thinking and `output_config.effort`. Older
# ones reject both - Haiku 4.5 and Sonnet 4.5 error on `effort` - so they are
# called plainly.
_ADAPTIVE_THINKING_MODELS = (
    "claude-fable-5",
    "claude-mythos-5",
    "claude-opus-5",
    "claude-opus-4-6",
    "claude-opus-4-7",
    "claude-opus-4-8",
    "claude-sonnet-5",
    "claude-sonnet-4-6",
)

# OpenAI reasoning models: they take `reasoning_effort`, require
# `max_completion_tokens` instead of `max_tokens`, and reject `temperature`.
_OPENAI_REASONING_MODELS = ("o1", "o3", "o4", "gpt-5")

# Per-MTok (input, output). Claude on Foundry bills at standard Anthropic rates
# through the Microsoft Marketplace. Azure OpenAI rates vary by region and
# commitment tier, so those entries are indicative - check the Azure pricing
# page for your own resource before relying on an estimate.
MODEL_RATES = {
    # Claude
    "claude-opus-5": (5.0, 25.0),
    "claude-opus-4-8": (5.0, 25.0),
    "claude-fable-5": (10.0, 50.0),
    # Sonnet 5 has introductory pricing of $2/$10 through 2026-08-31; the
    # standard rate is used here so an estimate never understates the bill.
    "claude-sonnet-5": (3.0, 15.0),
    "claude-sonnet-4-6": (3.0, 15.0),
    "claude-haiku-4-5": (1.0, 5.0),
    # Azure OpenAI (indicative)
    "gpt-4o": (2.50, 10.00),
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4.1": (2.00, 8.00),
    "gpt-4.1-mini": (0.40, 1.60),
}


def supports_adaptive_thinking(model: str) -> bool:
    return model.startswith(_ADAPTIVE_THINKING_MODELS)


def is_openai_reasoning_model(model: str) -> bool:
    return model.startswith(_OPENAI_REASONING_MODELS)


def _is_unsupported_parameter_error(message: str) -> bool:
    """Does this 400 complain about a tuning parameter rather than our payload?

    Thinking and effort are beta on Foundry, so a resource without them refuses
    the request. That is recoverable - the same prompt works without them.

    The match deliberately requires a parameter name. Matching bare words like
    "unsupported" or "unknown" would also swallow "unknown model", turning a
    wrong-deployment error into a confusing silent retry.
    """
    lowered = message.lower()
    if "model" in lowered or "deployment" in lowered:
        return False
    return any(
        token in lowered
        for token in ("thinking", "output_config", "effort", "budget_tokens")
    )


def extract_json(text: str) -> Any:
    """Parse JSON out of a model response, tolerating prose and code fences."""
    cleaned = _FENCE.sub("", text).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    # Fall back to the outermost balanced object or array in the response.
    for opener, closer in (("{", "}"), ("[", "]")):
        start, end = cleaned.find(opener), cleaned.rfind(closer)
        if start != -1 and end > start:
            try:
                return json.loads(cleaned[start : end + 1])
            except json.JSONDecodeError:
                continue
    raise ValueError(f"no JSON found in response: {text[:200]!r}")


@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    calls: int = 0
    cache_hits: int = 0


# --------------------------------------------------------------------------- #
# Shared behaviour
# --------------------------------------------------------------------------- #


class BaseLabeller:
    """Caching, concurrency, retry policy and costing. Providers add transport."""

    provider = "base"

    def __init__(self, cfg: LabelConfig, cache_dir: Path) -> None:
        self.cfg = cfg
        self.usage = Usage()
        # Keyed by provider and model: switching either is a genuinely different
        # result, not a cache hit. The id becomes a directory name, and model
        # ids are not path-safe - an Ollama tag like "qwen3:8b" carries a colon,
        # which Windows rejects outright.
        self.cache = JsonCache(cache_dir, "llm", cache_namespace(self.provider, cfg.model))
        # Cleared if the endpoint refuses a tuning parameter.
        self.tuning_enabled = True
        self.degraded_reason: str | None = None

    @property
    def target(self) -> str:
        """What the request actually addresses (a deployment name on Azure)."""
        return self.cfg.deployment or self.cfg.model

    def ask_json(
        self,
        system: str,
        prompt: str,
        *,
        max_tokens: int = 4000,
        effort: str = "medium",
    ) -> Any:
        """One request that must come back as JSON. Cached by prompt hash."""
        key = hashlib.sha256(
            f"{self.provider}|{self.cfg.model}|{effort}|{system}|{prompt}".encode("utf-8")
        ).hexdigest()

        cached = self.cache.get(key)
        if cached is not None:
            self.usage.cache_hits += 1
            return cached["value"]

        text = self._call_with_retry(system, prompt, max_tokens, effort)
        value = extract_json(text)
        self.cache.set(key, {"value": value})
        return value

    def _call_with_retry(self, system: str, prompt: str, max_tokens: int, effort: str) -> str:
        last: Exception | None = None
        for attempt in range(self.cfg.max_retries):
            try:
                return self._call_once(system, prompt, max_tokens, effort)
            except LabellingUnavailable:
                raise  # never retry a configuration problem
            except _Recoverable as exc:
                last = exc.__cause__ or exc
            except _DroppedTuning:
                continue  # parameters removed; retry immediately

            # Full jitter: with several workers hitting the same rate limit,
            # synchronised retries would just collide again.
            time.sleep(random.uniform(0, min(2**attempt, 30)))

        raise RuntimeError(f"giving up after {self.cfg.max_retries} attempts: {last}")

    def _call_once(self, system: str, prompt: str, max_tokens: int, effort: str) -> str:
        raise NotImplementedError

    def probe(self) -> dict[str, Any]:
        """Smallest possible real call, to confirm the model is actually there.

        Reports the model the endpoint *served*, which is the only reliable way
        to know a request was not quietly satisfied by a different deployment.
        """
        raise NotImplementedError

    def map_concurrent(
        self,
        items: Iterable[T],
        fn: Callable[[T], Any],
        on_done: Callable[[int], None] | None = None,
    ) -> list[Any]:
        """Run `fn` over `items` on the thread pool, preserving input order."""
        items = list(items)
        results: list[Any] = [None] * len(items)
        with ThreadPoolExecutor(max_workers=self.cfg.max_workers) as pool:
            futures = {pool.submit(fn, item): i for i, item in enumerate(items)}
            for done, future in enumerate(as_completed(futures), start=1):
                results[futures[future]] = future.result()
                if on_done:
                    on_done(done)
        return results

    def estimated_cost_usd(self) -> float:
        input_rate, output_rate = MODEL_RATES.get(self.cfg.model, (5.0, 25.0))
        return (
            self.usage.input_tokens * input_rate + self.usage.output_tokens * output_rate
        ) / 1e6

    def _matches(self, served: Any) -> bool:
        """Foundry and Azure may report a dated snapshot id, so compare by prefix."""
        if not served:
            return False
        served = str(served)
        return served.startswith(self.cfg.model) or self.cfg.model.startswith(served)


class _Recoverable(Exception):
    """A transient failure worth backing off and retrying."""


class _DroppedTuning(Exception):
    """Tuning parameters were rejected and removed; retry at once."""


# --------------------------------------------------------------------------- #
# Claude on Microsoft Foundry
# --------------------------------------------------------------------------- #


class FoundryLabeller(BaseLabeller):
    provider = "foundry"

    def __init__(
        self,
        cfg: LabelConfig,
        cache_dir: Path,
        credentials: FoundryCredentials | None = None,
    ) -> None:
        super().__init__(cfg, cache_dir)

        credentials = credentials or FoundryCredentials.from_env()
        if credentials is None:
            raise LabellingUnavailable(
                "Claude labelling needs ANTHROPIC_FOUNDRY_API_KEY and "
                "ANTHROPIC_FOUNDRY_RESOURCE in the environment. See .env.example."
            )
        try:
            from anthropic import AnthropicFoundry
        except ImportError as exc:
            raise LabellingUnavailable(
                "the anthropic SDK is not installed. Run: uv sync --extra llm"
            ) from exc

        self.client = AnthropicFoundry(
            api_key=credentials.api_key, resource=credentials.resource
        )

    def _request_kwargs(self, system: str, prompt: str, max_tokens: int, effort: str) -> dict:
        kwargs: dict[str, Any] = {
            "model": self.target,
            "max_tokens": max_tokens,
            "system": system,
            "messages": [{"role": "user", "content": prompt}],
        }
        if self.tuning_enabled and supports_adaptive_thinking(self.cfg.model):
            kwargs["thinking"] = {"type": "adaptive"}
            kwargs["output_config"] = {"effort": effort}
        return kwargs

    def _call_once(self, system: str, prompt: str, max_tokens: int, effort: str) -> str:
        import anthropic

        try:
            response = self.client.messages.create(
                **self._request_kwargs(system, prompt, max_tokens, effort)
            )
        except anthropic.BadRequestError as exc:
            # Thinking and effort are beta on Foundry. If the resource has not
            # enabled them, drop both for the rest of the run rather than
            # failing the whole labelling pass over a tuning knob.
            if self.tuning_enabled and _is_unsupported_parameter_error(str(exc)):
                self.tuning_enabled = False
                self.degraded_reason = str(exc)[:200]
                raise _DroppedTuning from exc
            raise
        except anthropic.NotFoundError as exc:
            raise LabellingUnavailable(
                f"model {self.target!r} is not available on this Foundry resource. "
                f"Check the deployment in the Azure portal. ({exc})"
            ) from exc
        except anthropic.PermissionDeniedError as exc:
            raise LabellingUnavailable(f"Foundry rejected the credentials: {exc}") from exc
        except (
            anthropic.RateLimitError,
            anthropic.APIConnectionError,
            anthropic.InternalServerError,
        ) as exc:
            raise _Recoverable from exc
        except anthropic.APIStatusError as exc:
            if exc.status_code >= 500:
                raise _Recoverable from exc
            raise

        self.usage.calls += 1
        usage = getattr(response, "usage", None)
        if usage is not None:
            self.usage.input_tokens += getattr(usage, "input_tokens", 0) or 0
            self.usage.output_tokens += getattr(usage, "output_tokens", 0) or 0
        return "".join(block.text for block in response.content if block.type == "text")

    def probe(self) -> dict[str, Any]:
        import anthropic

        try:
            response = self.client.messages.create(
                model=self.target,
                max_tokens=16,
                messages=[{"role": "user", "content": "Reply with the word: ok"}],
            )
        except anthropic.NotFoundError as exc:
            return {"ok": False, "reason": f"not deployed on this resource ({exc})"}
        except anthropic.PermissionDeniedError as exc:
            return {"ok": False, "reason": f"credentials rejected ({exc})"}
        except anthropic.APIStatusError as exc:
            return {"ok": False, "reason": f"HTTP {exc.status_code}: {exc}"}
        except anthropic.APIConnectionError as exc:
            return {"ok": False, "reason": f"cannot reach the endpoint ({exc})"}

        served = getattr(response, "model", None) or "unreported"
        return {
            "ok": True,
            "requested": self.cfg.model,
            "served": served,
            "matches": self._matches(getattr(response, "model", None)),
        }


# --------------------------------------------------------------------------- #
# GPT on Azure OpenAI
# --------------------------------------------------------------------------- #


class AzureOpenAILabeller(BaseLabeller):
    provider = "azure-openai"

    def __init__(
        self,
        cfg: LabelConfig,
        cache_dir: Path,
        credentials: AzureOpenAICredentials | None = None,
    ) -> None:
        super().__init__(cfg, cache_dir)

        credentials = credentials or AzureOpenAICredentials.from_env()
        if credentials is None:
            raise LabellingUnavailable(
                "Azure OpenAI labelling needs AZURE_OPENAI_API_KEY and "
                "AZURE_OPENAI_ENDPOINT in the environment. See .env.example."
            )
        problem = credentials.problem
        if problem:
            raise LabellingUnavailable(problem)
        try:
            from openai import AzureOpenAI
        except ImportError as exc:
            raise LabellingUnavailable(
                "the openai SDK is not installed. Run: uv sync --extra llm"
            ) from exc

        self.credentials = credentials
        self.client = AzureOpenAI(
            api_key=credentials.api_key,
            azure_endpoint=credentials.endpoint,
            api_version=cfg.api_version,
        )

    def list_deployments(self) -> list[str]:
        """Deployment names on this resource, for a useful 404 message.

        Uses the data-plane listing directly: the openai SDK has no binding for
        it, and a stdlib request avoids depending on the SDK's HTTP internals.
        """
        import urllib.error
        import urllib.request

        url = (
            self.credentials.endpoint.rstrip("/")
            + "/openai/deployments?api-version=2023-03-15-preview"
        )
        request = urllib.request.Request(url, headers={"api-key": self.credentials.api_key})
        try:
            with urllib.request.urlopen(request, timeout=20) as response:
                payload = json.loads(response.read().decode("utf-8", "replace"))
        except (urllib.error.URLError, json.JSONDecodeError, OSError):
            return []
        return sorted(
            str(item.get("id"))
            for item in payload.get("data", [])
            if item.get("id")
        )

    def _request_kwargs(self, system: str, prompt: str, max_tokens: int, effort: str) -> dict:
        kwargs: dict[str, Any] = {
            # On Azure this is the *deployment* name, not the model id.
            "model": self.target,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
        }

        if is_openai_reasoning_model(self.cfg.model):
            # Reasoning models renamed the output cap and reject temperature.
            kwargs["max_completion_tokens"] = max_tokens
            if self.tuning_enabled:
                kwargs["reasoning_effort"] = "low" if effort == "low" else "medium"
        else:
            kwargs["max_tokens"] = max_tokens
            # Deterministic labels: the same corpus should name the same way twice.
            kwargs["temperature"] = 0

        if self.tuning_enabled:
            # JSON mode. The API requires the word "json" to appear in the
            # conversation, which every prompt in prompts.py satisfies.
            kwargs["response_format"] = {"type": "json_object"}
        return kwargs

    def _call_once(self, system: str, prompt: str, max_tokens: int, effort: str) -> str:
        import openai

        try:
            response = self.client.chat.completions.create(
                **self._request_kwargs(system, prompt, max_tokens, effort)
            )
        except openai.BadRequestError as exc:
            # JSON mode and reasoning_effort are not supported on every API
            # version or deployment. Drop them and fall back to the
            # prompt-and-parse path, which extract_json already handles.
            if self.tuning_enabled and _is_openai_tuning_error(str(exc)):
                self.tuning_enabled = False
                self.degraded_reason = str(exc)[:200]
                raise _DroppedTuning from exc
            raise
        except openai.NotFoundError as exc:
            raise LabellingUnavailable(
                f"deployment {self.target!r} does not exist on this Azure OpenAI "
                f"resource. The deployment name is chosen when you deploy and "
                f"need not match the model id - check the Azure portal. ({exc})"
            ) from exc
        except (openai.AuthenticationError, openai.PermissionDeniedError) as exc:
            raise LabellingUnavailable(f"Azure OpenAI rejected the credentials: {exc}") from exc
        except (
            openai.RateLimitError,
            openai.APIConnectionError,
            openai.InternalServerError,
        ) as exc:
            raise _Recoverable from exc
        except openai.APIStatusError as exc:
            if exc.status_code >= 500:
                raise _Recoverable from exc
            raise

        self.usage.calls += 1
        usage = getattr(response, "usage", None)
        if usage is not None:
            self.usage.input_tokens += getattr(usage, "prompt_tokens", 0) or 0
            self.usage.output_tokens += getattr(usage, "completion_tokens", 0) or 0

        choice = response.choices[0]
        # A response cut off mid-object cannot be parsed; say so plainly rather
        # than letting extract_json raise something opaque.
        if getattr(choice, "finish_reason", None) == "length":
            raise RuntimeError(
                f"response hit the {max_tokens}-token cap before finishing its JSON. "
                "Raise max_tokens or shorten the prompt."
            )
        return choice.message.content or ""

    def probe(self) -> dict[str, Any]:
        import openai

        try:
            response = self.client.chat.completions.create(
                model=self.target,
                max_tokens=16,
                messages=[{"role": "user", "content": "Reply with the word: ok"}],
            )
        except openai.NotFoundError as exc:
            # A 404 here is ambiguous - wrong deployment name, or an endpoint
            # that already carries an API path. Listing what the resource
            # actually has separates the two immediately.
            available = self.list_deployments()
            if available:
                reason = (
                    f"deployment {self.target!r} not found. This resource has: "
                    + ", ".join(available)
                    + " — set label.deployment to one of those."
                )
            else:
                reason = (
                    f"deployment {self.target!r} not found, and the deployment list "
                    f"could not be read from {self.credentials.endpoint} — check "
                    f"AZURE_OPENAI_ENDPOINT is the resource base URL. ({exc})"
                )
            return {"ok": False, "reason": reason}
        except (openai.AuthenticationError, openai.PermissionDeniedError) as exc:
            return {"ok": False, "reason": f"credentials rejected ({exc})"}
        except openai.APIStatusError as exc:
            return {"ok": False, "reason": f"HTTP {exc.status_code}: {exc}"}
        except openai.APIConnectionError as exc:
            return {"ok": False, "reason": f"cannot reach the endpoint ({exc})"}

        served = getattr(response, "model", None) or "unreported"
        return {
            "ok": True,
            "requested": self.cfg.model,
            "served": served,
            "matches": self._matches(getattr(response, "model", None)),
        }


def _is_openai_tuning_error(message: str) -> bool:
    """A 400 about JSON mode or reasoning effort, not about our payload."""
    lowered = message.lower()
    if "deployment" in lowered:
        return False
    return any(
        token in lowered
        for token in ("response_format", "json_object", "reasoning_effort", "temperature")
    )


# --------------------------------------------------------------------------- #
# A local model under Ollama
# --------------------------------------------------------------------------- #

DEFAULT_OLLAMA_URL = "http://localhost:11434"

# Ollama's own default context window. Both the naming and coherence prompts
# exceed it on a real corpus, which is why every request sizes num_ctx itself.
OLLAMA_DEFAULT_NUM_CTX = 4096

# Characters per token, deliberately pessimistic. Overestimating costs a little
# memory; underestimating costs a silently truncated prompt.
_CHARS_PER_TOKEN = 3.0


def _round_up_pow2(value: int, floor: int) -> int:
    size = floor
    while size < value:
        size *= 2
    return size


class OllamaLabeller(BaseLabeller):
    """A model served by a local Ollama daemon: free, offline, no account.

    Speaks Ollama's native /api/chat rather than its OpenAI-compatible /v1,
    because only the native API exposes `num_ctx` - and without that knob a
    prompt longer than the server's default window is silently truncated.
    Transport is urllib, so this provider adds no dependency at all: with
    Ollama the `llm` extra is not needed.
    """

    provider = "ollama"

    def __init__(
        self, cfg: LabelConfig, cache_dir: Path, credentials: Any = None
    ) -> None:
        super().__init__(cfg, cache_dir)
        # No credentials: a local daemon has nothing to authenticate against.
        # The host is configuration, and $OLLAMA_HOST is Ollama's own convention.
        self.base_url = (
            cfg.base_url or os.environ.get("OLLAMA_HOST") or DEFAULT_OLLAMA_URL
        ).rstrip("/")
        if "://" not in self.base_url:
            self.base_url = f"http://{self.base_url}"
        self._model_info: dict[str, Any] | None = None

    # -- transport ---------------------------------------------------------- #

    def _post(self, path: str, payload: dict[str, Any], timeout: int) -> dict[str, Any]:
        import urllib.error
        import urllib.request

        request = urllib.request.Request(
            f"{self.base_url}{path}",
            json.dumps(payload).encode("utf-8"),
            {"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return json.loads(response.read().decode("utf-8", "replace"))
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", "replace")
            if exc.code == 404:
                raise LabellingUnavailable(self._missing_model_message()) from exc
            if exc.code >= 500:
                raise _Recoverable(f"Ollama returned HTTP {exc.code}") from exc
            raise LabellingUnavailable(f"Ollama rejected the request: {body[:200]}") from exc
        except urllib.error.URLError as exc:
            # Unreachable is a setup problem, not a transient one: retrying for
            # a minute would only delay a message the user needs to act on.
            raise LabellingUnavailable(
                f"Ollama is not reachable at {self.base_url}. Start it, or install it "
                f"from https://ollama.com — then: ollama pull {self.cfg.model} ({exc.reason})"
            ) from exc
        except TimeoutError as exc:
            raise _Recoverable(f"Ollama timed out after {timeout}s") from exc

    def list_models(self) -> list[str]:
        """Models this daemon already has, for the picker and for a useful 404."""
        import urllib.error
        import urllib.request

        try:
            with urllib.request.urlopen(f"{self.base_url}/api/tags", timeout=10) as response:
                payload = json.loads(response.read().decode("utf-8", "replace"))
        except (urllib.error.URLError, json.JSONDecodeError, OSError, TimeoutError):
            return []
        return sorted(str(m.get("name")) for m in payload.get("models", []) if m.get("name"))

    def _missing_model_message(self) -> str:
        available = self.list_models()
        pull = f"Run: ollama pull {self.cfg.model}"
        if available:
            return (
                f"model {self.cfg.model!r} is not installed. This machine has: "
                + ", ".join(available)
                + f". {pull}"
            )
        return f"model {self.cfg.model!r} is not installed and none were found. {pull}"

    # -- model capabilities ------------------------------------------------- #

    def model_info(self) -> dict[str, Any]:
        """Context length and capabilities, read once and remembered."""
        if self._model_info is None:
            try:
                shown = self._post("/api/show", {"model": self.cfg.model}, timeout=30)
            except _Recoverable:
                shown = {}
            info = shown.get("model_info") or {}
            # The key is namespaced by family, e.g. "qwen3.context_length".
            context = next(
                (int(v) for k, v in info.items() if k.endswith("context_length") and v),
                OLLAMA_DEFAULT_NUM_CTX,
            )
            self._model_info = {
                "context_length": context,
                "capabilities": shown.get("capabilities") or [],
            }
        return self._model_info

    def _num_ctx(self, system: str, prompt: str, max_tokens: int) -> int:
        """A window this request certainly fits in.

        Ollama truncates an over-long prompt in silence - no error, and it drops
        the *start*, which is where the schema instruction lives. The result is
        confident, well-formed, wrong output. So the window is sized per request
        instead of trusting a default.
        """
        needed = int((len(system) + len(prompt)) / _CHARS_PER_TOKEN) + max_tokens
        limit = self.model_info()["context_length"]
        if needed > limit:
            raise LabellingUnavailable(
                f"this prompt needs about {needed} tokens but {self.cfg.model!r} holds "
                f"{limit}. Lower label.representative_docs, turn off "
                f"label.coherence_pass, or use a longer-context model."
            )
        return min(_round_up_pow2(needed, OLLAMA_DEFAULT_NUM_CTX), limit)

    # -- requests ----------------------------------------------------------- #

    def _request_kwargs(self, system: str, prompt: str, max_tokens: int, effort: str) -> dict:
        payload: dict[str, Any] = {
            "model": self.cfg.model,
            "stream": False,
            "format": "json",
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            "options": {
                "num_ctx": self._num_ctx(system, prompt, max_tokens),
                "num_predict": max_tokens,
                "temperature": 0,
            },
        }
        # Only sent to models that advertise it; the rest reject the parameter.
        # Reasoning lands in its own field, so this is a speed choice, not a
        # correctness one - and a local model is slow enough for that to matter.
        if self.tuning_enabled and "thinking" in self.model_info()["capabilities"]:
            payload["think"] = False
        return payload

    def _call_once(self, system: str, prompt: str, max_tokens: int, effort: str) -> str:
        payload = self._request_kwargs(system, prompt, max_tokens, effort)
        # Local inference is slow: a naming call on an 8B model can take minutes.
        response = self._post("/api/chat", payload, timeout=900)

        served_prompt = response.get("prompt_eval_count")
        expected = int((len(system) + len(prompt)) / (_CHARS_PER_TOKEN * 2))
        if served_prompt is not None and served_prompt < expected:
            # Belt and braces: the window was sized to fit, so a prompt that
            # still came back short means it was truncated anyway. Better to
            # fail than to label a folder from evidence the model never read.
            raise LabellingUnavailable(
                f"Ollama evaluated only {served_prompt} prompt tokens for a prompt of at "
                f"least {expected} — it was truncated. Reduce the prompt size via "
                f"label.representative_docs, or use a longer-context model."
            )

        usage_in = response.get("prompt_eval_count") or 0
        usage_out = response.get("eval_count") or 0
        self.usage.input_tokens += usage_in
        self.usage.output_tokens += usage_out
        self.usage.calls += 1
        return response.get("message", {}).get("content", "")

    def probe(self) -> dict[str, Any]:
        try:
            response = self._post(
                "/api/chat",
                {
                    "model": self.cfg.model,
                    "stream": False,
                    "messages": [{"role": "user", "content": "Reply with the word: ok"}],
                    "options": {"num_predict": 16},
                },
                timeout=120,
            )
        except LabellingUnavailable as exc:
            return {"ok": False, "reason": str(exc)}
        except _Recoverable as exc:
            return {"ok": False, "reason": str(exc)}

        served = response.get("model")
        return {
            "ok": True,
            "requested": self.cfg.model,
            "served": served or "unreported",
            "matches": self._matches(served),
        }

    def estimated_cost_usd(self) -> float:
        """Nothing. That is the entire point of running it locally."""
        return 0.0


# --------------------------------------------------------------------------- #

PROVIDERS = {
    "foundry": FoundryLabeller,
    "azure-openai": AzureOpenAILabeller,
    "ollama": OllamaLabeller,
}


def build_labeller(
    cfg: LabelConfig, cache_dir: Path, credentials: Any | None = None
) -> BaseLabeller:
    """Construct the provider's labeller.

    `credentials` lets a caller supply values the environment does not hold -
    the viewer collects them from a form. They stay a parameter rather than a
    config field on purpose: Config is serialised into taxonomy.json, so an
    api_key living there would be written to disk in cleartext.
    """
    try:
        provider = PROVIDERS[cfg.provider]
    except KeyError:
        raise LabellingUnavailable(
            f"unknown label.provider {cfg.provider!r}; expected one of "
            + ", ".join(sorted(PROVIDERS))
        ) from None
    return provider(cfg, cache_dir, credentials)


def probe_labeller(
    cfg: LabelConfig,
    cache_dir: Path,
    credentials: Any | None = None,
    *,
    deployments: bool = False,
) -> dict[str, Any]:
    """One tiny real call, reported as a result rather than an exception.

    Never raises: a configuration problem is something to show the user, not
    something every call site should have to catch. Shared by `littax doctor
    --probe` and the viewer's "Test connection" so the two cannot drift apart.

    `deployments` adds the provider's deployment names, for a UI that wants to
    offer them as a dropdown. Off by default - it is a second HTTP round trip,
    and the CLI already names them in the one case where it matters, a 404.
    """
    try:
        labeller = build_labeller(cfg, cache_dir, credentials)
        result = dict(labeller.probe())
    except LabellingUnavailable as exc:
        return {"ok": False, "reason": str(exc), "deployments": []}

    # Each provider names this after its own domain: Azure serves deployments,
    # Ollama serves installed models.
    lister = getattr(labeller, "list_deployments", None) or getattr(
        labeller, "list_models", None
    )
    result["deployments"] = lister() if deployments and lister else []
    return result
