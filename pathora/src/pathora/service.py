"""Application-layer facade.

The API, the Streamlit UI and the tests all go through this module. Nothing in
here knows about HTTP or Streamlit, which is what keeps a later move to Next.js /
ECS from touching the agent layer.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any

from langgraph.types import Command

from pathora.config import Settings, get_settings
from pathora.domain.models import HumanResponse, WhatIfResult, WhatIfScenario
from pathora.graph.build import build_graph, default_config
from pathora.graph.checkpointer import build_checkpointer
from pathora.graph.nodes import Deps
from pathora.graph.state import PathoraState, initial_state
from pathora.graph.whatif import run_what_if
from pathora.llm.providers import build_provider
from pathora.rag.store import seeded_store


@dataclass
class RunResult:
    state: PathoraState
    interrupt: dict[str, Any] | None
    thread_id: str

    @property
    def awaiting_human(self) -> bool:
        return self.interrupt is not None


class PathoraService:
    """Owns the compiled graph, the checkpointer and the shared dependencies."""

    def __init__(self, deps: Deps, checkpointer: Any | None = None) -> None:
        self.deps = deps
        self.checkpointer = checkpointer or build_checkpointer(deps.settings)
        self.graph = build_graph(deps, checkpointer=self.checkpointer)

    @classmethod
    async def create(cls, settings: Settings | None = None) -> PathoraService:
        settings = settings or get_settings()
        store = await seeded_store(settings)
        return cls(Deps(provider=build_provider(settings), store=store, settings=settings))

    # -- workflow ----------------------------------------------------------- #
    async def start(
        self,
        *,
        thread_id: str,
        user_id: str,
        student_id: str,
        transcript_document: dict | None = None,
        student_input: dict | None = None,
    ) -> RunResult:
        state = initial_state(
            user_id=user_id,
            student_id=student_id,
            transcript_document=transcript_document,
            student_input=student_input,
        )
        return await self._run(self.graph.ainvoke(state, default_config(thread_id)), thread_id)

    async def resume(self, *, thread_id: str, response: HumanResponse) -> RunResult:
        return await self._run(
            self.graph.ainvoke(
                Command(resume=response.model_dump(mode="json")), default_config(thread_id)
            ),
            thread_id,
        )

    async def _run(self, coro, thread_id: str) -> RunResult:
        result = await coro
        snapshot = await self.graph.aget_state(default_config(thread_id))
        pending = None
        if snapshot.interrupts:
            pending = snapshot.interrupts[0].value
        state: PathoraState = result  # type: ignore[assignment]
        if pending is not None:
            state = dict(snapshot.values)  # type: ignore[assignment]
            state["pending_human_action"] = pending
            state["workflow_status"] = "awaiting_human"
        return RunResult(state=state, interrupt=pending, thread_id=thread_id)

    async def state(self, thread_id: str) -> PathoraState:
        snapshot = await self.graph.aget_state(default_config(thread_id))
        return dict(snapshot.values)  # type: ignore[return-value]

    # -- what-if ------------------------------------------------------------ #
    async def what_if(
        self, *, thread_id: str, scenario: WhatIfScenario
    ) -> tuple[PathoraState, WhatIfResult]:
        state = await self.state(thread_id)
        if not state.get("admission_results"):
            raise ValueError("run a full analysis before simulating scenarios")
        return await run_what_if(state, scenario, self.deps)


_singleton: PathoraService | None = None
_lock = asyncio.Lock()


async def get_service() -> PathoraService:
    global _singleton
    async with _lock:
        if _singleton is None:
            _singleton = await PathoraService.create()
    return _singleton
