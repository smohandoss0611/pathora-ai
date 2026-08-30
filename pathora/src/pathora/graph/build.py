"""Graph assembly (Sections 5, 6, 7, 22, 23).

LangGraph is the authoritative orchestrator. There is no free-form agent chat and
no agent decides who speaks next: routing is either a static edge or an explicit
conditional function over typed state.
"""

from __future__ import annotations

from functools import partial
from typing import Any

from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

from pathora.config import get_settings
from pathora.domain.models import CriticResult
from pathora.graph import nodes
from pathora.graph.checkpointer import build_checkpointer
from pathora.graph.nodes import Deps
from pathora.graph.state import PathoraState

CANCELLED = "cancelled_by_user"


# --------------------------------------------------------------------------- #
# Conditional routers
# --------------------------------------------------------------------------- #
def after_verification(state: PathoraState) -> list[str] | str:
    if state.get("workflow_status") == CANCELLED:
        return "__end__"
    # Parallel branch: Profile Agent and Activity Agent run concurrently.
    return ["profile_agent", "activity_agent"]


def fan_out_research(state: PathoraState) -> list[Send] | str:
    """Dynamic fan-out: one worker instance per candidate college."""
    details = state.get("candidate_details", {})
    if not details:
        return "collect_research"
    return [
        Send("research_worker", {"candidate": candidate, "deep": False})
        for candidate in details.values()
    ]


def fan_out_targeted_research(state: PathoraState) -> list[Send] | str:
    """Re-research ONLY the colleges the Critic named, with a deeper sweep."""
    critic = CriticResult.model_validate(state["critic_results"])
    details = state.get("candidate_details", {})
    targets = [details[u] for u in critic.colleges_to_research if u in details]
    if not targets:
        return "collect_research"
    return [Send("research_worker", {"candidate": c, "deep": True}) for c in targets]


def route_critic(state: PathoraState) -> str:
    decision = CriticResult.model_validate(state["critic_results"]).decision
    return {
        "approve": "next_best_action",
        "research_more": "targeted_research",
        "human_review": "human_review",
    }[decision]


def after_human_review(state: PathoraState) -> str:
    if state.get("workflow_status") == CANCELLED:
        return "__end__"
    return "next_best_action"


# --------------------------------------------------------------------------- #
# Builder
# --------------------------------------------------------------------------- #
def build_graph(deps: Deps, *, checkpointer: Any | None = None):
    """Compile the Pathora workflow graph."""
    graph = StateGraph(PathoraState)

    bound: dict[str, Any] = {
        "parse_transcript": nodes.parse_transcript,
        "calculate_gpa": nodes.calculate_gpa,
        "verify_academic_profile": nodes.verify_academic_profile,
        "profile_agent": nodes.profile_agent,
        "activity_agent": nodes.activity_agent,
        "build_twin": nodes.build_twin,
        "stem_fit_agent": nodes.stem_fit_agent,
        "college_discovery": nodes.college_discovery,
        "research_worker": nodes.research_worker_node,
        "collect_research": nodes.collect_research,
        "admission_agent": nodes.admission_agent,
        "critic_agent": nodes.critic_agent,
        "targeted_research": nodes.targeted_research,
        "human_review": nodes.human_review,
        "next_best_action": nodes.next_best_action,
        "dynamic_roadmap": nodes.dynamic_roadmap,
    }
    for name, fn in bound.items():
        graph.add_node(name, partial(fn, deps=deps))

    graph.add_edge(START, "parse_transcript")
    graph.add_edge("parse_transcript", "calculate_gpa")
    graph.add_edge("calculate_gpa", "verify_academic_profile")

    graph.add_conditional_edges(
        "verify_academic_profile",
        after_verification,
        {"profile_agent": "profile_agent", "activity_agent": "activity_agent", "__end__": END},
    )
    graph.add_edge("profile_agent", "build_twin")
    graph.add_edge("activity_agent", "build_twin")

    graph.add_edge("build_twin", "stem_fit_agent")
    graph.add_edge("stem_fit_agent", "college_discovery")

    graph.add_conditional_edges(
        "college_discovery", fan_out_research, ["research_worker", "collect_research"]
    )
    graph.add_edge("research_worker", "collect_research")
    graph.add_edge("collect_research", "admission_agent")
    graph.add_edge("admission_agent", "critic_agent")

    graph.add_conditional_edges(
        "critic_agent",
        route_critic,
        {
            "next_best_action": "next_best_action",
            "targeted_research": "targeted_research",
            "human_review": "human_review",
        },
    )
    graph.add_conditional_edges(
        "targeted_research", fan_out_targeted_research, ["research_worker", "collect_research"]
    )
    graph.add_conditional_edges(
        "human_review", after_human_review, {"next_best_action": "next_best_action", "__end__": END}
    )

    graph.add_edge("next_best_action", "dynamic_roadmap")
    graph.add_edge("dynamic_roadmap", END)

    return graph.compile(checkpointer=checkpointer or build_checkpointer(deps.settings))


def default_config(thread_id: str) -> dict[str, Any]:
    settings = get_settings()
    return {
        "configurable": {"thread_id": thread_id},
        # Bounds the dynamic fan-out; workers also self-limit via a semaphore.
        "max_concurrency": settings.max_parallel_college_workers,
        # Hard stop: no infinite loops, ever.
        "recursion_limit": 60,
    }
