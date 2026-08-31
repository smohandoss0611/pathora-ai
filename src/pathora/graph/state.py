"""LangGraph state (Section 8).

Every value is JSON-serializable so the graph can be checkpointed and resumed.
Nodes read only the fields they need and write only the fields they own; the
merge reducers below make the parallel college fan-in safe.
"""

from __future__ import annotations

from typing import Annotated, Any, TypedDict


def merge_dicts(left: dict[str, Any] | None, right: dict[str, Any] | None) -> dict[str, Any]:
    """Reducer for concurrent writes from fanned-out college workers."""
    return {**(left or {}), **(right or {})}


def replace(_left: Any, right: Any) -> Any:
    return right


class PathoraState(TypedDict, total=False):
    user_id: str
    student_id: str

    transcript_document: dict | None
    student_input: dict
    extracted_academics: dict
    verified_academics: dict

    gpa_result: dict

    profile_analysis: dict
    activity_analysis: dict

    student_twin: dict

    stem_fit: list[dict]

    college_candidates: list[str]
    college_catalog: list[dict]
    candidate_details: dict[str, dict]
    college_research: Annotated[dict[str, dict], merge_dicts]

    admission_results: dict[str, dict]
    abstentions: dict[str, dict]
    gate_results: dict[str, dict]
    critic_results: dict[str, Any]
    evidence_passports: dict[str, dict]
    gap_analysis: dict[str, dict]

    research_retry_count: int
    critic_loop_count: int

    pending_human_action: dict | None
    human_responses: list[dict]

    next_actions: list[dict]
    roadmap: dict

    workflow_status: str
    warnings: list[str]


def initial_state(
    *,
    user_id: str,
    student_id: str,
    transcript_document: dict | None = None,
    student_input: dict | None = None,
) -> PathoraState:
    return PathoraState(
        user_id=user_id,
        student_id=student_id,
        transcript_document=transcript_document,
        student_input=student_input or {},
        extracted_academics={},
        verified_academics={},
        gpa_result={},
        profile_analysis={},
        activity_analysis={},
        student_twin={},
        stem_fit=[],
        college_candidates=[],
        college_catalog=[],
        candidate_details={},
        college_research={},
        admission_results={},
        abstentions={},
        gate_results={},
        critic_results={},
        evidence_passports={},
        gap_analysis={},
        research_retry_count=0,
        critic_loop_count=0,
        pending_human_action=None,
        human_responses=[],
        next_actions=[],
        roadmap={},
        workflow_status="started",
        warnings=[],
    )
