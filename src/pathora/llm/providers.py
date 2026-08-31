"""Concrete providers + factory.

One provider (Anthropic) is implemented fully, as instructed. The OpenAI shape is
sketched behind the same Protocol so swapping is a config change, not a rewrite.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, TypeVar

from pydantic import BaseModel

from pathora.config import Settings, get_settings
from pathora.llm.base import (
    LLMProvider,
    StructuredOutputError,
    extract_json,
    schema_instructions,
    validate,
)
from pathora.llm.fake import FakeProvider

log = logging.getLogger(__name__)
T = TypeVar("T", bound=BaseModel)


class AnthropicProvider:
    name = "anthropic"

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        try:
            from anthropic import AsyncAnthropic  # type: ignore
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "pip install 'pathora-ai[llm]' to use the Anthropic provider"
            ) from exc
        if not self.settings.anthropic_api_key:
            raise RuntimeError("ANTHROPIC_API_KEY is not set")
        self._client = AsyncAnthropic(
            api_key=self.settings.anthropic_api_key,
            timeout=self.settings.llm_timeout_seconds,
        )

    async def invoke(self, *, task: str, system: str, prompt: str, max_tokens: int = 2000) -> str:
        response = await self._client.messages.create(
            model=self.settings.model_for(task),
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": prompt}],
        )
        return "".join(block.text for block in response.content if block.type == "text")

    async def structured(
        self,
        *,
        task: str,
        system: str,
        prompt: str,
        schema: type[T],
        context: dict[str, Any] | None = None,
        max_tokens: int = 2000,
    ) -> T:
        instructions = f"{prompt}\n\n{schema_instructions(schema)}"
        last_error: Exception | None = None

        for attempt in range(self.settings.llm_max_retries + 1):
            try:
                text = await self.invoke(
                    task=task, system=system, prompt=instructions, max_tokens=max_tokens
                )
                return validate(schema, extract_json(text))
            except StructuredOutputError as exc:
                last_error = exc
                log.warning("structured output retry %s for task=%s: %s", attempt, task, exc)
                instructions = (
                    f"{prompt}\n\nYour previous response was rejected: {exc}\n\n"
                    f"{schema_instructions(schema)}"
                )
                await asyncio.sleep(0.2 * (attempt + 1))

        raise StructuredOutputError(f"task={task} failed after retries: {last_error}")


#: Base URLs for known OpenAI-compatible vendors, so LLM_PROVIDER=nebius works
#: without also remembering the URL. Nebius AI Studio was rebranded "Nebius
#: Token Factory" in 2026; the api.studio.nebius.com endpoint is what their
#: quickstart documents, and is overridable via LLM_BASE_URL if that changes.
COMPATIBLE_ENDPOINTS: dict[str, str] = {
    "nebius": "https://api.studio.nebius.com/v1",
    "groq": "https://api.groq.com/openai/v1",
    "openrouter": "https://openrouter.ai/api/v1",
    "deepseek": "https://api.deepseek.com/v1",
    "together": "https://api.together.xyz/v1",
    "ollama": "http://localhost:11434/v1",
    "lmstudio": "http://localhost:1234/v1",
    "openai": "https://api.openai.com/v1",
}


class OpenAICompatibleProvider:
    """Any endpoint speaking the OpenAI chat-completions protocol.

    One implementation covers OpenAI, Ollama, Groq, OpenRouter, DeepSeek,
    Together, vLLM and LM Studio — they differ only by base URL, model name and
    whether a key is required. Set LLM_BASE_URL to switch.

    Local backends (Ollama, LM Studio) need no key and cost nothing to run.
    """

    name = "openai"

    #: Endpoints that accept OpenAI's native JSON-schema enforcement. Others get
    #: prompt-level instructions plus the repair loop, same as Anthropic.
    NATIVE_SCHEMA_HOSTS = ("api.openai.com",)

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        # Explicit URL wins; otherwise fall back to the named vendor's endpoint.
        vendor = self.settings.llm_provider.lower()
        default = COMPATIBLE_ENDPOINTS.get(vendor, COMPATIBLE_ENDPOINTS["openai"])
        self.base_url = (self.settings.llm_base_url or default).rstrip("/")
        self.vendor = vendor
        self._local = any(host in self.base_url for host in ("localhost", "127.0.0.1", "0.0.0.0"))
        if not self.settings.compatible_api_key and not self._local:
            raise RuntimeError(
                f"No API key found for LLM_PROVIDER={vendor}. Set LLM_API_KEY "
                f"(vendor-neutral) or OPENAI_API_KEY. Local backends "
                f"(LLM_PROVIDER=ollama or lmstudio) need no key."
            )

    def _supports_native_schema(self) -> bool:
        return any(host in self.base_url for host in self.NATIVE_SCHEMA_HOSTS)

    async def _post(self, payload: dict[str, Any]) -> dict[str, Any]:
        import httpx

        headers = {"content-type": "application/json"}
        if key := self.settings.compatible_api_key:
            headers["authorization"] = f"Bearer {key}"

        last: Exception | None = None
        for attempt in range(self.settings.llm_max_retries + 1):
            try:
                async with httpx.AsyncClient(timeout=self.settings.llm_timeout_seconds) as client:
                    response = await client.post(
                        f"{self.base_url}/chat/completions", json=payload, headers=headers
                    )
            except (httpx.TimeoutException, httpx.TransportError) as exc:
                # Transport failures are transient and were previously fatal:
                # one slow response aborted the entire analysis.
                last = exc
                log.warning("transport error from %s (attempt %s): %s", self.base_url, attempt, exc)
                await asyncio.sleep(1.0 * (attempt + 1))
                continue

            if response.status_code >= 400:
                raise RuntimeError(
                    f"{response.status_code} from {self.base_url}: {response.text[:300]}"
                )
            return response.json()

        raise RuntimeError(f"no response from {self.base_url} after retries: {last}")

    async def invoke(self, *, task: str, system: str, prompt: str, max_tokens: int = 2000) -> str:
        body = await self._post(
            {
                "model": self.settings.model_for(task),
                "max_tokens": max_tokens,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt},
                ],
            }
        )
        return body["choices"][0]["message"]["content"] or ""

    async def structured(
        self,
        *,
        task: str,
        system: str,
        prompt: str,
        schema: type[T],
        context: dict[str, Any] | None = None,
        max_tokens: int = 2000,
    ) -> T:
        instructions = f"{prompt}\n\n{schema_instructions(schema)}"
        last_error: Exception | None = None

        for attempt in range(self.settings.llm_max_retries + 1):
            payload: dict[str, Any] = {
                "model": self.settings.model_for(task),
                "max_tokens": max_tokens,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": instructions},
                ],
            }
            # Server-side schema enforcement where available; elsewhere fall back
            # to plain JSON mode, then to the repair loop below.
            if self._supports_native_schema():
                payload["response_format"] = {
                    "type": "json_schema",
                    "json_schema": {
                        "name": schema.__name__,
                        "schema": schema.model_json_schema(),
                        "strict": False,
                    },
                }
            else:
                payload["response_format"] = {"type": "json_object"}

            try:
                body = await self._post(payload)
                text = body["choices"][0]["message"]["content"] or ""
                return validate(schema, extract_json(text))
            except StructuredOutputError as exc:
                last_error = exc
                log.warning("structured output retry %s for task=%s: %s", attempt, task, exc)
                instructions = (
                    f"{prompt}\n\nYour previous response was rejected: {exc}\n\n"
                    f"{schema_instructions(schema)}"
                )
                await asyncio.sleep(0.2 * (attempt + 1))

        raise StructuredOutputError(f"task={task} failed after retries: {last_error}")


def build_provider(settings: Settings | None = None) -> LLMProvider:
    settings = settings or get_settings()
    match settings.llm_provider.lower():
        case "anthropic":
            return AnthropicProvider(settings)
        case vendor if vendor in COMPATIBLE_ENDPOINTS:
            return OpenAICompatibleProvider(settings)
        case _:
            return FakeProvider()
