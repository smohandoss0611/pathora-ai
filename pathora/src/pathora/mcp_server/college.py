"""College MCP server (Section 18).

MCP is a tool transport, not the orchestrator: LangGraph still decides what runs
next. Transcript parsing and GPA stay internal deterministic services because
exposing them over MCP would buy nothing.

The tool bodies live in ``TOOLS`` as plain async functions so they are unit
testable without a running MCP session; ``main()`` binds them to a FastMCP
server when the SDK is installed.
"""

from __future__ import annotations

import asyncio
from typing import Any

from pathora.agents.college_worker import college_research_worker
from pathora.config import get_settings
from pathora.domain.models import NOT_PUBLISHED, CollegeCandidate
from pathora.rag.store import VectorStore, seeded_store


async def research_college(
    university: str, target_major: str, deep: bool = False, *, store: VectorStore
) -> dict[str, Any]:
    """Full evidence-backed research pass for one university/major pair."""
    result = await college_research_worker(
        CollegeCandidate(university=university, target_major=target_major),
        store,
        deep=deep,
    )
    return result.model_dump(mode="json")


async def search_college_documents(
    query: str, university: str | None = None, top_k: int = 5, *, store: VectorStore
) -> list[dict[str, Any]]:
    """Hybrid search over indexed official college documents."""
    chunks = await store.query(query, university=university, top_k=top_k, max_depth=1)
    return [
        {
            "evidence_id": c.id,
            "university": c.university,
            "title": c.title,
            "snippet": c.text[:400],
            "source_url": c.source_url,
            "source_type": c.source_type,
            "published_at": c.published_at.isoformat() if c.published_at else None,
            "score": c.score,
        }
        for c in chunks
    ]


async def get_program_info(university: str, major: str, *, store: VectorStore) -> dict[str, Any]:
    """Program-level information for a specific major."""
    chunks = await store.query(
        f"{university} {major} program requirements", university=university, top_k=4, max_depth=1
    )
    facts: dict[str, Any] = {}
    for chunk in chunks:
        facts |= {k: v for k, v in chunk.facts.items() if k not in facts}
    return {
        "university": university,
        "major": major,
        "admission_structure": facts.get("admission_structure", NOT_PUBLISHED),
        "major_admit_rate": facts.get("major_admit_rate", NOT_PUBLISHED),
        "transfer_restrictions": facts.get("transfer_restrictions", NOT_PUBLISHED),
        "sources": [{"evidence_id": c.id, "source_url": c.source_url} for c in chunks],
    }


async def get_admission_policy(university: str, *, store: VectorStore) -> dict[str, Any]:
    """Testing policy, deadlines and published ranges for one university."""
    chunks = await store.query(
        f"{university} testing policy deadlines admission requirements",
        university=university,
        top_k=4,
        max_depth=1,
    )
    facts: dict[str, Any] = {}
    for chunk in chunks:
        facts |= {k: v for k, v in chunk.facts.items() if k not in facts}
    return {
        "university": university,
        "test_policy": facts.get("test_policy", NOT_PUBLISHED),
        "deadlines": facts.get("deadlines", NOT_PUBLISHED),
        "sat_range": facts.get("sat_range", NOT_PUBLISHED),
        "act_range": facts.get("act_range", NOT_PUBLISHED),
        "admit_rate": facts.get("admit_rate", NOT_PUBLISHED),
        "sources": [{"evidence_id": c.id, "source_url": c.source_url} for c in chunks],
    }


TOOLS = {
    "research_college": research_college,
    "search_college_documents": search_college_documents,
    "get_program_info": get_program_info,
    "get_admission_policy": get_admission_policy,
}


def main() -> None:  # pragma: no cover - requires the MCP SDK and a live session
    try:
        from mcp.server.fastmcp import FastMCP
    except ImportError as exc:
        raise SystemExit("pip install 'pathora-ai[mcp]' to run the MCP server") from exc

    settings = get_settings()
    store = asyncio.run(seeded_store(settings))
    server = FastMCP("pathora-college")

    @server.tool()
    async def research_college_tool(university: str, target_major: str, deep: bool = False) -> dict:
        return await research_college(university, target_major, deep, store=store)

    @server.tool()
    async def search_college_documents_tool(
        query: str, university: str | None = None, top_k: int = 5
    ) -> list[dict]:
        return await search_college_documents(query, university, top_k, store=store)

    @server.tool()
    async def get_program_info_tool(university: str, major: str) -> dict:
        return await get_program_info(university, major, store=store)

    @server.tool()
    async def get_admission_policy_tool(university: str) -> dict:
        return await get_admission_policy(university, store=store)

    server.run()


if __name__ == "__main__":  # pragma: no cover
    main()
