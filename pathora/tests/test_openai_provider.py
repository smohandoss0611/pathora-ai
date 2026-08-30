"""Tests for the OpenAI-compatible provider, against a stub server."""

from __future__ import annotations

import json

import httpx
import pytest

from pathora.config import Settings
from pathora.domain.models import ActivityAnalysis
from pathora.llm.base import StructuredOutputError
from pathora.llm.providers import OpenAICompatibleProvider, build_provider


def stub(handler):
    """Patch httpx.AsyncClient.post with a canned handler."""

    class _Response:
        def __init__(self, status_code: int, payload):
            self.status_code = status_code
            self._payload = payload
            self.text = json.dumps(payload) if not isinstance(payload, str) else payload

        def json(self):
            return self._payload

    async def _post(self, url, json=None, headers=None):  # noqa: A002
        return _Response(*handler(json, headers))

    return _post


def completion(content: str) -> tuple[int, dict]:
    return 200, {"choices": [{"message": {"content": content}}]}


class TestConfiguration:
    def test_local_backend_needs_no_key(self):
        provider = OpenAICompatibleProvider(
            Settings(llm_provider="openai", llm_base_url="http://localhost:11434/v1")
        )
        assert provider.base_url == "http://localhost:11434/v1"

    def test_hosted_backend_without_key_fails_loudly(self):
        with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
            OpenAICompatibleProvider(Settings(llm_provider="openai"))

    def test_factory_selects_it(self):
        provider = build_provider(
            Settings(llm_provider="openai", llm_base_url="http://localhost:11434/v1")
        )
        assert isinstance(provider, OpenAICompatibleProvider)

    def test_trailing_slash_is_normalized(self):
        provider = OpenAICompatibleProvider(
            Settings(llm_provider="openai", llm_base_url="http://localhost:11434/v1/")
        )
        assert not provider.base_url.endswith("/")

    def test_native_schema_only_for_openai(self):
        openai = OpenAICompatibleProvider(Settings(llm_provider="openai", openai_api_key="k"))
        ollama = OpenAICompatibleProvider(
            Settings(llm_provider="openai", llm_base_url="http://localhost:11434/v1")
        )
        assert openai._supports_native_schema()
        assert not ollama._supports_native_schema()


class TestRequests:
    @pytest.fixture
    def provider(self):
        return OpenAICompatibleProvider(
            Settings(llm_provider="openai", llm_base_url="http://localhost:11434/v1")
        )

    async def test_invoke_returns_content(self, provider, monkeypatch):
        monkeypatch.setattr(
            httpx.AsyncClient, "post", stub(lambda body, headers: completion("hello"))
        )
        assert await provider.invoke(task="profile", system="s", prompt="p") == "hello"

    async def test_structured_parses_valid_json(self, provider, monkeypatch):
        payload = ActivityAnalysis(themes=["technical"]).model_dump_json()
        monkeypatch.setattr(
            httpx.AsyncClient, "post", stub(lambda body, headers: completion(payload))
        )
        result = await provider.structured(
            task="activity", system="s", prompt="p", schema=ActivityAnalysis
        )
        assert result.themes == ["technical"]

    async def test_structured_tolerates_code_fences(self, provider, monkeypatch):
        fenced = "```json\n" + ActivityAnalysis().model_dump_json() + "\n```"
        monkeypatch.setattr(
            httpx.AsyncClient, "post", stub(lambda body, headers: completion(fenced))
        )
        assert await provider.structured(
            task="activity", system="s", prompt="p", schema=ActivityAnalysis
        )

    async def test_repair_loop_retries_then_gives_up(self, provider, monkeypatch):
        calls = []

        def handler(body, headers):
            calls.append(body)
            return completion("not json at all")

        monkeypatch.setattr(httpx.AsyncClient, "post", stub(handler))
        with pytest.raises(StructuredOutputError):
            await provider.structured(
                task="activity", system="s", prompt="p", schema=ActivityAnalysis
            )
        assert len(calls) == provider.settings.llm_max_retries + 1
        assert "rejected" in calls[-1]["messages"][-1]["content"]

    async def test_http_error_is_surfaced_with_context(self, provider, monkeypatch):
        monkeypatch.setattr(
            httpx.AsyncClient, "post", stub(lambda body, headers: (404, "model not found"))
        )
        with pytest.raises(RuntimeError, match="404"):
            await provider.invoke(task="profile", system="s", prompt="p")

    async def test_local_request_sends_no_auth_header(self, provider, monkeypatch):
        seen = {}

        def handler(body, headers):
            seen.update(headers)
            return completion("ok")

        monkeypatch.setattr(httpx.AsyncClient, "post", stub(handler))
        await provider.invoke(task="profile", system="s", prompt="p")
        assert "authorization" not in seen

    async def test_json_mode_requested_for_non_openai_hosts(self, provider, monkeypatch):
        seen = {}

        def handler(body, headers):
            seen.update(body)
            return completion(ActivityAnalysis().model_dump_json())

        monkeypatch.setattr(httpx.AsyncClient, "post", stub(handler))
        await provider.structured(task="activity", system="s", prompt="p", schema=ActivityAnalysis)
        assert seen["response_format"] == {"type": "json_object"}


class TestVendorNeutralKey:
    """Nebius, Groq, OpenRouter etc. issue their own key names."""

    def test_llm_api_key_is_accepted(self):
        provider = OpenAICompatibleProvider(
            Settings(
                llm_provider="openai",
                llm_base_url="https://api.studio.nebius.com/v1",
                llm_api_key="nebius-key",
            )
        )
        assert provider.settings.compatible_api_key == "nebius-key"

    def test_llm_api_key_takes_precedence_over_openai_key(self):
        settings = Settings(llm_api_key="vendor", openai_api_key="openai")
        assert settings.compatible_api_key == "vendor"

    def test_openai_key_still_works_alone(self):
        assert Settings(openai_api_key="openai").compatible_api_key == "openai"

    async def test_key_is_sent_as_bearer_token(self, monkeypatch):
        seen: dict = {}

        def handler(body, headers):
            seen.update(headers)
            return completion("ok")

        provider = OpenAICompatibleProvider(
            Settings(
                llm_provider="openai",
                llm_base_url="https://api.studio.nebius.com/v1",
                llm_api_key="nebius-key",
            )
        )
        monkeypatch.setattr(httpx.AsyncClient, "post", stub(handler))
        await provider.invoke(task="profile", system="s", prompt="p")
        assert seen["authorization"] == "Bearer nebius-key"

    def test_missing_key_on_hosted_endpoint_names_both_options(self):
        with pytest.raises(RuntimeError, match="LLM_API_KEY"):
            OpenAICompatibleProvider(
                Settings(llm_provider="openai", llm_base_url="https://api.studio.nebius.com/v1")
            )


class TestNamedVendors:
    """LLM_PROVIDER=nebius should not also require remembering a URL."""

    def test_nebius_resolves_its_endpoint(self):
        provider = build_provider(Settings(llm_provider="nebius", llm_api_key="k"))
        assert isinstance(provider, OpenAICompatibleProvider)
        assert provider.base_url == "https://api.studio.nebius.com/v1"

    @pytest.mark.parametrize(
        ("vendor", "expected"),
        [
            ("groq", "https://api.groq.com/openai/v1"),
            ("openrouter", "https://openrouter.ai/api/v1"),
            ("deepseek", "https://api.deepseek.com/v1"),
            ("together", "https://api.together.xyz/v1"),
            ("openai", "https://api.openai.com/v1"),
        ],
    )
    def test_each_named_vendor_resolves(self, vendor, expected):
        provider = build_provider(Settings(llm_provider=vendor, llm_api_key="k"))
        assert provider.base_url == expected

    @pytest.mark.parametrize("vendor", ["ollama", "lmstudio"])
    def test_local_vendors_need_no_key(self, vendor):
        provider = build_provider(Settings(llm_provider=vendor))
        assert "localhost" in provider.base_url

    def test_explicit_base_url_overrides_the_vendor_default(self):
        provider = build_provider(
            Settings(llm_provider="nebius", llm_api_key="k", llm_base_url="https://mirror/v1")
        )
        assert provider.base_url == "https://mirror/v1"

    def test_missing_key_names_the_vendor(self):
        with pytest.raises(RuntimeError, match="LLM_PROVIDER=nebius"):
            build_provider(Settings(llm_provider="nebius"))

    def test_unknown_provider_still_falls_back_to_fake(self):
        from pathora.llm.fake import FakeProvider

        assert isinstance(build_provider(Settings(llm_provider="nonsense")), FakeProvider)


class TestTransportResilience:
    """A ReadTimeout mid-analysis previously aborted the entire run."""

    @pytest.fixture
    def provider(self):
        return OpenAICompatibleProvider(
            Settings(
                llm_provider="nebius",
                llm_api_key="k",
                llm_max_retries=2,
                llm_timeout_seconds=1,
            )
        )

    async def test_timeout_is_retried_then_succeeds(self, provider, monkeypatch):
        calls = {"n": 0}

        async def _post(self, url, json=None, headers=None):  # noqa: A002
            calls["n"] += 1
            if calls["n"] == 1:
                raise httpx.ReadTimeout("too slow")

            class _Response:
                status_code = 200
                text = ""

                def json(self):
                    return {"choices": [{"message": {"content": "ok"}}]}

            return _Response()

        monkeypatch.setattr(httpx.AsyncClient, "post", _post)
        assert await provider.invoke(task="profile", system="s", prompt="p") == "ok"
        assert calls["n"] == 2

    async def test_persistent_timeout_raises_a_clear_error(self, provider, monkeypatch):
        async def _post(self, url, json=None, headers=None):  # noqa: A002
            raise httpx.ReadTimeout("too slow")

        monkeypatch.setattr(httpx.AsyncClient, "post", _post)
        with pytest.raises(RuntimeError, match="no response from"):
            await provider.invoke(task="profile", system="s", prompt="p")

    def test_default_timeout_suits_large_models(self):
        assert Settings().llm_timeout_seconds >= 120
