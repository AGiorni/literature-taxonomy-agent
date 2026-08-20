"""Labelling layer. Exercised against a stub labeller so the suite needs no
Foundry credentials and makes no network calls."""

from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest

from littax.cluster.tree import Node
from littax.config import LabelConfig
from littax.label.client import (
    MODEL_RATES,
    AzureOpenAILabeller,
    FoundryLabeller,
    LabellingUnavailable,
    _is_openai_tuning_error,
    _is_unsupported_parameter_error,
    build_labeller,
    extract_json,
    supports_adaptive_thinking,
)
from littax.label.prompts import COHERENCE_SYSTEM, NAMING_SYSTEM, VERIFY_SYSTEM
from littax.label.naming import (
    NodeLabel,
    _coerce_label,
    _deduplicate_siblings,
    apply_renames,
    label_tree,
    representative_titles,
    review_coherence,
)
from littax.models import CoherenceEdit


class StubLabeller:
    """Stands in for FoundryLabeller: records prompts, returns canned JSON."""

    def __init__(self, responses=None):
        self.cfg = LabelConfig(max_workers=2)
        self.prompts: list[str] = []
        self.responses = responses or {}

    def ask_json(self, system, prompt, **kwargs):
        self.prompts.append(prompt)
        for marker, response in self.responses.items():
            if marker in prompt:
                return response
        if "Review this taxonomy" in prompt:
            return {"edits": []}
        return {
            "label": "Motor Unit Physiology",
            "description": "Papers about motor unit discharge.",
            "distinguishing_note": "Focuses on firing rate rather than force.",
            "keywords": ["motor unit", "firing rate"],
        }

    def map_concurrent(self, items, fn, on_done=None):
        return [fn(item) for item in items]


def make_tree() -> Node:
    root = Node(id="root", level=0, parent_id=None, doc_indices=list(range(12)))
    root.centroid = np.array([1.0, 0.0], dtype=np.float32)
    for n, name in enumerate(("L1.001", "L1.002"), start=0):
        child = Node(
            id=name,
            level=1,
            parent_id="root",
            doc_indices=list(range(n * 6, n * 6 + 6)),
        )
        child.centroid = np.array([1.0, 0.0], dtype=np.float32)
        root.children.append(child)
    return root


class TestModelCapabilities:
    @pytest.mark.parametrize(
        "model", ["claude-opus-5", "claude-sonnet-5", "claude-opus-4-8", "claude-fable-5"]
    )
    def test_modern_models_get_thinking(self, model):
        assert supports_adaptive_thinking(model)

    @pytest.mark.parametrize("model", ["claude-haiku-4-5", "claude-sonnet-4-5"])
    def test_older_models_do_not(self, model):
        """Haiku 4.5 errors on `effort` and rejects adaptive thinking, so the
        request must go out without either."""
        assert not supports_adaptive_thinking(model)

    def test_every_priced_model_is_classified(self):
        for model in MODEL_RATES:
            assert isinstance(supports_adaptive_thinking(model), bool)

    def test_cheaper_models_are_actually_cheaper(self):
        assert MODEL_RATES["claude-haiku-4-5"] < MODEL_RATES["claude-sonnet-5"]
        assert MODEL_RATES["claude-sonnet-5"] < MODEL_RATES["claude-opus-5"]


class FakeLabeller(FoundryLabeller):
    """Bypasses __init__ so request shaping can be tested without credentials."""

    def __init__(self, model: str, **overrides):
        self.cfg = LabelConfig(model=model, **overrides)
        self.tuning_enabled = True
        self.degraded_reason = None


class FakeAzureLabeller(AzureOpenAILabeller):
    def __init__(self, model: str, **overrides):
        self.cfg = LabelConfig(model=model, provider="azure-openai", **overrides)
        self.tuning_enabled = True
        self.degraded_reason = None


class TestProviderSelection:
    def test_factory_returns_the_configured_provider(self, tmp_path, monkeypatch):
        monkeypatch.setenv("AZURE_OPENAI_API_KEY", "k")
        monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://example.openai.azure.com/")
        labeller = build_labeller(
            LabelConfig(provider="azure-openai", model="gpt-4o-mini"), tmp_path
        )
        assert isinstance(labeller, AzureOpenAILabeller)

    def test_unknown_provider_is_rejected(self, tmp_path):
        cfg = LabelConfig.model_construct(provider="bedrock", model="x", max_workers=1)
        with pytest.raises(LabellingUnavailable, match="unknown label.provider"):
            build_labeller(cfg, tmp_path)

    def test_missing_credentials_name_the_right_variables(self, tmp_path, monkeypatch):
        for name in ("AZURE_OPENAI_API_KEY", "AZURE_OPENAI_ENDPOINT"):
            monkeypatch.delenv(name, raising=False)
        with pytest.raises(LabellingUnavailable, match="AZURE_OPENAI_API_KEY"):
            build_labeller(LabelConfig(provider="azure-openai"), tmp_path)

    def test_cache_is_namespaced_by_provider(self, tmp_path, monkeypatch):
        """Two providers must never serve each other's cached answers."""
        monkeypatch.setenv("AZURE_OPENAI_API_KEY", "k")
        monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://example.openai.azure.com/")
        azure = build_labeller(LabelConfig(provider="azure-openai", model="gpt-4o"), tmp_path)
        assert "azure-openai" in str(azure.cache.dir)


class TestAzureRequestShaping:
    def _kwargs(self, model, **overrides):
        return FakeAzureLabeller(model, **overrides)._request_kwargs(
            "sys", "prompt", 2000, "medium"
        )

    def test_system_and_user_are_separate_messages(self):
        kwargs = self._kwargs("gpt-4o-mini")
        assert kwargs["messages"] == [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "prompt"},
        ]

    def test_json_mode_and_deterministic_temperature(self):
        kwargs = self._kwargs("gpt-4o-mini")
        assert kwargs["response_format"] == {"type": "json_object"}
        assert kwargs["temperature"] == 0
        assert kwargs["max_tokens"] == 2000

    def test_addresses_the_deployment_not_the_model(self):
        """On Azure the request targets a deployment name, which need not match
        the model id."""
        kwargs = self._kwargs("gpt-4o-mini", deployment="my-naming-deployment")
        assert kwargs["model"] == "my-naming-deployment"

    def test_falls_back_to_model_when_no_deployment_set(self):
        assert self._kwargs("gpt-4o")["model"] == "gpt-4o"

    def test_reasoning_models_use_different_parameters(self):
        kwargs = self._kwargs("gpt-5")
        assert "max_tokens" not in kwargs
        assert kwargs["max_completion_tokens"] == 2000
        assert "temperature" not in kwargs  # rejected by reasoning models
        assert kwargs["reasoning_effort"] == "medium"

    def test_degrading_drops_json_mode(self):
        labeller = FakeAzureLabeller("gpt-4o-mini")
        assert "response_format" in labeller._request_kwargs("s", "p", 100, "low")
        labeller.tuning_enabled = False
        degraded = labeller._request_kwargs("s", "p", 100, "low")
        assert "response_format" not in degraded
        # The prompt still goes out intact - extract_json handles plain text.
        assert degraded["messages"][1]["content"] == "p"

    @pytest.mark.parametrize(
        "message,recoverable",
        [
            ("response_format is not supported with this API version", True),
            ("Unsupported parameter: reasoning_effort", True),
            ("temperature does not support 0 with this model", True),
            ("The API deployment for this resource does not exist", False),
            ("context_length_exceeded", False),
        ],
    )
    def test_only_tuning_errors_are_recoverable(self, message, recoverable):
        assert _is_openai_tuning_error(message) is recoverable


class TestJsonModePrecondition:
    """Azure's JSON mode rejects a request whose conversation never mentions
    'json'. Every system prompt must therefore contain the word."""

    @pytest.mark.parametrize(
        "system", [NAMING_SYSTEM, COHERENCE_SYSTEM, VERIFY_SYSTEM]
    )
    def test_system_prompts_mention_json(self, system):
        assert "json" in system.lower()


class TestRequestShaping:
    def _kwargs(self, model):
        return FakeLabeller(model)._request_kwargs("sys", "prompt", 2000, "medium")

    def test_modern_model_sends_thinking_and_effort(self):
        kwargs = self._kwargs("claude-sonnet-5")
        assert kwargs["thinking"] == {"type": "adaptive"}
        assert kwargs["output_config"] == {"effort": "medium"}

    def test_older_model_omits_both(self):
        kwargs = self._kwargs("claude-haiku-4-5")
        assert "thinking" not in kwargs
        assert "output_config" not in kwargs

    def test_core_fields_always_present(self):
        for model in ("claude-sonnet-5", "claude-haiku-4-5"):
            kwargs = self._kwargs(model)
            assert kwargs["model"] == model
            assert kwargs["max_tokens"] == 2000
            assert kwargs["messages"] == [{"role": "user", "content": "prompt"}]

    def test_degrading_drops_the_beta_parameters(self):
        """Foundry has thinking/effort in beta; a resource without them must
        not fail the whole run."""
        labeller = FakeLabeller("claude-sonnet-5")
        assert "thinking" in labeller._request_kwargs("s", "p", 100, "low")
        labeller.tuning_enabled = False
        degraded = labeller._request_kwargs("s", "p", 100, "low")
        assert "thinking" not in degraded and "output_config" not in degraded
        assert degraded["messages"] == [{"role": "user", "content": "p"}]

    @pytest.mark.parametrize(
        "message,recoverable",
        [
            ("thinking: unsupported parameter", True),
            ("output_config.effort is not supported on this resource", True),
            ("Unknown field: thinking", True),
            ("max_tokens must be greater than 0", False),
            ("credit balance is too low", False),
            # A wrong deployment must NOT look like a recoverable parameter
            # problem, or the real cause gets buried behind a silent retry.
            ("Unknown model: claude-sonnet-5", False),
            ("The model claude-sonnet-5 is not supported on this resource", False),
            ("DeploymentNotFound: no deployment for claude-sonnet-5", False),
        ],
    )
    def test_only_parameter_errors_are_recoverable(self, message, recoverable):
        assert _is_unsupported_parameter_error(message) is recoverable


class TestProbe:
    """`doctor --probe` must be able to tell "wrong model answered" apart from
    "model is missing" - a run that silently bills at another model's rate is
    worse than one that fails."""

    def _probe(self, requested, served):
        labeller = FakeLabeller(requested)
        labeller.client = SimpleNamespace(
            messages=SimpleNamespace(
                create=lambda **kw: SimpleNamespace(model=served, usage=None, content=[])
            )
        )
        return labeller.probe()

    def test_matching_model_passes(self):
        result = self._probe("claude-sonnet-5", "claude-sonnet-5")
        assert result["ok"] and result["matches"]

    def test_dated_snapshot_still_matches(self):
        result = self._probe("claude-sonnet-5", "claude-sonnet-5-20260514")
        assert result["matches"]

    def test_different_model_is_flagged(self):
        result = self._probe("claude-sonnet-5", "claude-opus-5")
        assert result["ok"] and not result["matches"]
        assert result["served"] == "claude-opus-5"

    def test_unreported_model_is_not_assumed_to_match(self):
        labeller = FakeLabeller("claude-sonnet-5")
        labeller.client = SimpleNamespace(
            messages=SimpleNamespace(create=lambda **kw: SimpleNamespace(usage=None, content=[]))
        )
        assert labeller.probe()["matches"] is False

    def test_probe_sends_no_beta_parameters(self):
        """The probe must isolate 'is the model there', so it cannot fail for
        an unrelated reason like thinking being disabled on the resource."""
        seen = {}
        labeller = FakeLabeller("claude-sonnet-5")
        labeller.client = SimpleNamespace(
            messages=SimpleNamespace(
                create=lambda **kw: (
                    seen.update(kw), SimpleNamespace(model="claude-sonnet-5", usage=None, content=[])
                )[1]
            )
        )
        labeller.probe()
        assert "thinking" not in seen and "output_config" not in seen
        assert seen["max_tokens"] == 16


class TestExtractJson:
    def test_plain(self):
        assert extract_json('{"a": 1}') == {"a": 1}

    def test_code_fenced(self):
        assert extract_json('```json\n{"a": 1}\n```') == {"a": 1}

    def test_with_surrounding_prose(self):
        assert extract_json('Here you go:\n{"a": 1}\nHope that helps.') == {"a": 1}

    def test_array(self):
        assert extract_json("```\n[1, 2]\n```") == [1, 2]

    def test_raises_when_absent(self):
        with pytest.raises(ValueError):
            extract_json("no json at all")


class TestCoerceLabel:
    def test_sanitises_illegal_label(self):
        label = _coerce_label({"label": "Motor/Unit: Physiology"}, "L1.001")
        assert label.label == "Motor Unit Physiology"

    def test_survives_wrong_shape(self):
        assert _coerce_label(["not", "a", "dict"], "L1.001").label == "L1.001"

    def test_survives_missing_fields(self):
        label = _coerce_label({"label": "Topic"}, "L1.001")
        assert (label.description, label.keywords) == ("", [])

    def test_non_list_keywords_ignored(self):
        assert _coerce_label({"label": "T", "keywords": "a,b"}, "x").keywords == []


class TestRepresentativeTitles:
    def test_returns_titles_nearest_the_centroid(self):
        X = np.array(
            [[1.0, 0.0], [0.9, 0.44], [0.0, 1.0]],
            dtype=np.float32,
        )
        node = Node(id="n", level=1, parent_id="root", doc_indices=[0, 1, 2])
        node.centroid = np.array([1.0, 0.0], dtype=np.float32)
        titles = ["closest", "middle", "furthest"]
        assert representative_titles(node, X, titles, 2) == ["closest", "middle"]

    def test_handles_missing_titles(self):
        X = np.array([[1.0, 0.0]], dtype=np.float32)
        node = Node(id="n", level=1, parent_id="root", doc_indices=[0])
        node.centroid = np.array([1.0, 0.0], dtype=np.float32)
        assert representative_titles(node, X, [None], 5) == ["(untitled)"]


class TestLabelTree:
    def test_names_every_node(self):
        root = make_tree()
        X = np.tile(np.array([[1.0, 0.0]], dtype=np.float32), (12, 1))
        labels = label_tree(
            StubLabeller(),
            root,
            X,
            ["title"] * 12,
            {"L1.001": ["a"], "L1.002": ["b"]},
            LabelConfig(max_workers=1),
        )
        assert set(labels) == {"L1.001", "L1.002"}

    def test_prompt_is_contrastive(self):
        """Each node must be named with its siblings visible, or the model
        cannot tell them apart."""
        stub = StubLabeller()
        root = make_tree()
        X = np.tile(np.array([[1.0, 0.0]], dtype=np.float32), (12, 1))
        label_tree(
            stub, root, X, ["t"] * 12, {"L1.001": ["a"], "L1.002": ["b"]}, LabelConfig()
        )
        first = json.loads(stub.prompts[0].split("```json")[1].split("```")[0])
        assert first["sibling_clusters"], "siblings missing from the naming prompt"
        assert first["sibling_clusters"][0]["id"] == "L1.002"

    def test_identical_sibling_names_are_disambiguated(self):
        """Two folders with the same name would collide on disk."""
        children = {
            "root": [
                Node(id="a", level=1, parent_id="root", doc_indices=[0]),
                Node(id="b", level=1, parent_id="root", doc_indices=[1]),
            ]
        }
        labels = {
            "a": NodeLabel("Motor Control", "", "", []),
            "b": NodeLabel("motor control", "", "", []),
        }
        _deduplicate_siblings(labels, children)
        assert labels["a"].label != labels["b"].label


class TestCoherence:
    def test_applies_renames(self):
        labels = {"L1.001": NodeLabel("Old Name", "", "", [])}
        apply_renames(
            labels,
            [CoherenceEdit(type="rename", targets=["L1.001"], detail="New Name", reason="")],
        )
        assert labels["L1.001"].label == "New Name"

    def test_ignores_merge_and_reparent(self):
        """Restructuring would invalidate the centroids every confidence score
        was computed against, so those edits are recorded, not applied."""
        labels = {"L1.001": NodeLabel("Kept", "", "", [])}
        apply_renames(
            labels,
            [CoherenceEdit(type="merge", targets=["L1.001"], detail="Merged", reason="")],
        )
        assert labels["L1.001"].label == "Kept"

    def test_drops_edits_naming_unknown_folders(self):
        stub = StubLabeller(
            responses={
                "Review this taxonomy": {
                    "edits": [
                        {"type": "rename", "targets": ["L9.999"], "detail": "X", "reason": ""},
                        {"type": "rename", "targets": ["L1.001"], "detail": "Y", "reason": ""},
                    ]
                }
            }
        )
        root = make_tree()
        edits = review_coherence(
            stub, root, {"L1.001": NodeLabel("A", "", "", [])}, {}
        )
        assert [e.targets for e in edits] == [["L1.001"]]

    def test_drops_unknown_edit_types(self):
        stub = StubLabeller(
            responses={
                "Review this taxonomy": {
                    "edits": [{"type": "delete", "targets": ["L1.001"], "detail": ""}]
                }
            }
        )
        assert review_coherence(stub, make_tree(), {}, {}) == []

    def test_survives_malformed_response(self):
        stub = StubLabeller(responses={"Review this taxonomy": ["not", "a", "dict"]})
        assert review_coherence(stub, make_tree(), {}, {}) == []
