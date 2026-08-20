"""`.env` loading. Credentials in a file must reach the SDK, and a real
environment variable must always win over the file."""

from __future__ import annotations

import os

import pytest

from littax.config import AzureOpenAICredentials, azure_endpoint_problem
from littax.env import _parse, load_env, present_keys


class TestParse:
    def test_plain_pairs(self):
        assert _parse("A=1\nB=2") == {"A": "1", "B": "2"}

    def test_ignores_comments_and_blanks(self):
        assert _parse("# note\n\nA=1\n   \n# A=2") == {"A": "1"}

    def test_strips_surrounding_quotes(self):
        parsed = _parse('A="https://x.openai.azure.com/"\nB=\'secret\'')
        assert parsed == {"A": "https://x.openai.azure.com/", "B": "secret"}

    def test_handles_export_prefix(self):
        assert _parse("export AZURE_OPENAI_API_KEY=abc") == {"AZURE_OPENAI_API_KEY": "abc"}

    def test_keeps_equals_inside_the_value(self):
        """Keys and tokens frequently contain '='."""
        assert _parse("K=ab=cd==") == {"K": "ab=cd=="}

    def test_tolerates_whitespace(self):
        assert _parse("  A = 1  ") == {"A": "1"}

    def test_skips_lines_without_equals(self):
        assert _parse("garbage\nA=1") == {"A": "1"}


class TestLoad:
    def test_loads_into_environment(self, tmp_path, monkeypatch):
        monkeypatch.delenv("AZURE_OPENAI_API_KEY", raising=False)
        env = tmp_path / ".env"
        env.write_text("AZURE_OPENAI_API_KEY=from-file\n", encoding="utf-8")
        load_env(env)
        assert os.environ["AZURE_OPENAI_API_KEY"] == "from-file"

    def test_real_environment_wins(self, tmp_path, monkeypatch):
        """Exporting a value for one run must override the file."""
        monkeypatch.setenv("AZURE_OPENAI_API_KEY", "from-shell")
        env = tmp_path / ".env"
        env.write_text("AZURE_OPENAI_API_KEY=from-file\n", encoding="utf-8")
        load_env(env)
        assert os.environ["AZURE_OPENAI_API_KEY"] == "from-shell"

    def test_missing_file_is_not_an_error(self, tmp_path):
        assert load_env(tmp_path / "nope.env") is None

    def test_finds_dotenv_in_a_parent_directory(self, tmp_path, monkeypatch):
        monkeypatch.delenv("AZURE_OPENAI_ENDPOINT", raising=False)
        (tmp_path / ".env").write_text("AZURE_OPENAI_ENDPOINT=https://x/\n", encoding="utf-8")
        nested = tmp_path / "a" / "b"
        nested.mkdir(parents=True)
        monkeypatch.chdir(nested)
        assert load_env() is not None
        assert os.environ["AZURE_OPENAI_ENDPOINT"] == "https://x/"


class TestAzureEndpointValidation:
    """The endpoint is validated, never rewritten: a config file whose value
    differs from the value in use is worse than an error naming the fix."""

    @pytest.mark.parametrize(
        "endpoint",
        [
            "https://andrea1us.services.ai.azure.com/",
            "https://andrea1us.services.ai.azure.com",
            "https://my-resource.openai.azure.com/",
        ],
    )
    def test_base_urls_are_accepted(self, endpoint):
        assert azure_endpoint_problem(endpoint) is None

    def test_operation_url_is_rejected_with_the_corrected_value(self):
        """The exact paste that caused a misleading 404."""
        problem = azure_endpoint_problem(
            "https://andrea1us.services.ai.azure.com/openai/v1/responses"
        )
        assert problem is not None
        assert "https://andrea1us.services.ai.azure.com/" in problem
        assert "/openai/v1/responses" in problem

    def test_deployment_url_is_rejected(self):
        problem = azure_endpoint_problem(
            "https://x.openai.azure.com/openai/deployments/gpt-4o/chat/completions"
        )
        assert problem and "base URL" in problem

    def test_query_string_is_rejected(self):
        problem = azure_endpoint_problem("https://x.openai.azure.com/?api-version=2024-10-21")
        assert problem and "query string" in problem

    @pytest.mark.parametrize("endpoint", ["", "   ", "x.openai.azure.com", "not a url"])
    def test_malformed_values_are_rejected(self, endpoint):
        assert azure_endpoint_problem(endpoint) is not None

    def test_credentials_surface_the_problem(self, monkeypatch):
        monkeypatch.setenv("AZURE_OPENAI_API_KEY", "k")
        monkeypatch.setenv(
            "AZURE_OPENAI_ENDPOINT",
            "https://andrea1us.services.ai.azure.com/openai/v1/responses",
        )
        credentials = AzureOpenAICredentials.from_env()
        # The value is preserved exactly as configured.
        assert credentials.endpoint.endswith("/openai/v1/responses")
        assert credentials.problem is not None

    def test_good_credentials_report_no_problem(self, monkeypatch):
        monkeypatch.setenv("AZURE_OPENAI_API_KEY", "k")
        monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://x.openai.azure.com/")
        assert AzureOpenAICredentials.from_env().problem is None


class TestPresentKeys:
    def test_reports_names_only(self, monkeypatch):
        monkeypatch.setenv("AZURE_OPENAI_API_KEY", "super-secret-value")
        monkeypatch.delenv("ANTHROPIC_FOUNDRY_API_KEY", raising=False)
        keys = present_keys()
        assert "AZURE_OPENAI_API_KEY" in keys
        assert "ANTHROPIC_FOUNDRY_API_KEY" not in keys
        # The value must never appear in anything we report.
        assert all("super-secret-value" not in key for key in keys)

    def test_ignores_empty_values(self, monkeypatch):
        monkeypatch.setenv("AZURE_OPENAI_API_KEY", "")
        assert "AZURE_OPENAI_API_KEY" not in present_keys()
