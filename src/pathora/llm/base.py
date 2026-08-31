"""LLM abstraction (Section 31).

Agents never import a vendor SDK. They call ``LLMProvider.structured(...)`` with
a Pydantic schema and get a validated model back, or a ``StructuredOutputError``.
"""

from __future__ import annotations

import json
import re
from typing import Any, Protocol, TypeVar, runtime_checkable

from pydantic import BaseModel, ValidationError

T = TypeVar("T", bound=BaseModel)

JSON_BLOCK = re.compile(r"```(?:json)?\s*(.*?)```", re.S)


class StructuredOutputError(RuntimeError):
    """Raised when a provider cannot produce schema-valid output."""


@runtime_checkable
class LLMProvider(Protocol):
    name: str

    async def invoke(
        self,
        *,
        task: str,
        system: str,
        prompt: str,
        max_tokens: int = 2000,
    ) -> str: ...

    async def structured(
        self,
        *,
        task: str,
        system: str,
        prompt: str,
        schema: type[T],
        context: dict[str, Any] | None = None,
        max_tokens: int = 2000,
    ) -> T: ...


def extract_json(text: str) -> Any:
    """Pull a JSON object out of a model response, tolerating code fences."""
    candidate = text.strip()
    if (block := JSON_BLOCK.search(candidate)) is not None:
        candidate = block.group(1).strip()
    else:
        start, end = candidate.find("{"), candidate.rfind("}")
        if start != -1 and end > start:
            candidate = candidate[start : end + 1]
    try:
        return json.loads(candidate)
    except json.JSONDecodeError as exc:
        raise StructuredOutputError(f"response was not valid JSON: {exc}") from exc


def validate[M: BaseModel](schema: type[M], payload: Any) -> M:
    try:
        return schema.model_validate(payload)
    except ValidationError as exc:
        raise StructuredOutputError(f"response did not match {schema.__name__}: {exc}") from exc


def schema_instructions(schema: type[BaseModel]) -> str:
    return (
        "Respond with a single JSON object and nothing else. No prose, no code "
        "fences, no explanation of your reasoning. It must validate against this "
        f"JSON schema:\n{json.dumps(schema.model_json_schema(), indent=2)}"
    )
