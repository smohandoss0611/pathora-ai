"""Offline provider used for tests, CI and degraded mode.

It honours the same interface as a real provider, so every node, route, retry
and interrupt in the graph is exercised without a network call.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, TypeVar

from pydantic import BaseModel

from pathora.domain.models import (
    ActivityAnalysis,
    AdmissionAssessment,
    CollegeCandidateList,
    CriticResult,
    NextActionList,
    ProfileAnalysis,
    Roadmap,
    STEMFitList,
)
from pathora.llm import heuristics
from pathora.llm.base import StructuredOutputError

T = TypeVar("T", bound=BaseModel)

HANDLERS: dict[type[BaseModel], Callable[[dict[str, Any]], BaseModel]] = {
    ProfileAnalysis: heuristics.profile_analysis,
    ActivityAnalysis: heuristics.activity_analysis,
    STEMFitList: heuristics.stem_fit,
    CollegeCandidateList: heuristics.college_discovery,
    AdmissionAssessment: heuristics.admission_assessment,
    CriticResult: heuristics.critic,
    NextActionList: heuristics.next_best_actions,
    Roadmap: heuristics.roadmap,
}


class FakeProvider:
    name = "fake"

    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    async def invoke(self, *, task: str, system: str, prompt: str, max_tokens: int = 2000) -> str:
        self.calls.append((task, "invoke"))
        return prompt[:max_tokens]

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
        self.calls.append((task, schema.__name__))
        handler = HANDLERS.get(schema)
        if handler is None:
            raise StructuredOutputError(f"FakeProvider has no handler for {schema.__name__}")
        result = handler(context or {})
        return schema.model_validate(result.model_dump())
