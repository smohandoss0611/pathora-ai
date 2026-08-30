"""Agent wrappers.

Each function builds a bounded, single-purpose prompt, sends it through the
LLMProvider Protocol, and returns a validated Pydantic model. No agent sees a
conversation history; each reads only the fields it needs.
"""

from __future__ import annotations

import json
from typing import Any

from pathora.domain.models import (
    ActivityAnalysis,
    AdmissionAssessment,
    AssessmentAbstention,
    CollegeCandidateList,
    CollegeResearchResult,
    CriticResult,
    NextActionList,
    ProfileAnalysis,
    Roadmap,
    STEMFit,
    STEMFitList,
    StudentDigitalTwin,
)
from pathora.llm.base import LLMProvider

GUARDRAILS = (
    "You must only restate facts present in the input. Never invent courses, "
    "hours, awards, statistics, technologies or responsibilities. If something "
    "is not in the input, say it is not available. Do not reveal your reasoning "
    "process; return conclusions only."
)

PROFILE_SYSTEM = (
    "You are an academic profile analyst for a college planning tool. You "
    "interpret an already-verified transcript. You must not recalculate or "
    "modify the GPA, which was computed deterministically upstream. " + GUARDRAILS
)

ACTIVITY_SYSTEM = (
    "You analyze a student's extracurricular record. Report only what is "
    "documented. Absence of evidence is a risk to name, not a gap to fill. " + GUARDRAILS
)

STEM_SYSTEM = (
    "You identify STEM disciplines that fit a student's verified academic "
    "record and stated interests. Never recommend a discipline because it is "
    "easier to be admitted to. " + GUARDRAILS
)

DISCOVERY_SYSTEM = (
    "You build an initial college candidate list from a student's digital twin, "
    "STEM fit ranking and stated preferences. Name real institutions and the "
    "specific major each would be applied to. Do NOT state admission rates, "
    "score ranges, deadlines or selectivity for any school you name — those are "
    "retrieved from official sources in a later step, and anything you assert "
    "about them will be discarded. Choose a spread of selectivity rather than "
    "only the best-known names.\n\n"
    "The target major for each college MUST come from the student's ranked STEM "
    "fit list, preferring the highest-ranked discipline the institution actually "
    "offers. Do not substitute a related-sounding major because it is easier to "
    "be admitted to, and do not default to a broad engineering discipline when a "
    "more specific one ranks higher. If you pair a college with anything other "
    "than the top-ranked available fit, say why in the reason field. " + GUARDRAILS
)

ADMISSION_SYSTEM = (
    "You classify admission fit using retrieved official evidence. Never output "
    "a numeric admission probability and never imply mathematical precision. "
    "Cite only evidence ids present in the research payload. If a university-wide "
    "admit rate is all that was published, say so explicitly rather than treating "
    "it as major-specific.\n\n"
    "Every strength and risk you list must be about THIS college. A bullet that "
    "would read identically for any institution — 'strong AP coursework', "
    "'demonstrated leadership' — belongs in the profile analysis, not here. Each "
    "strength must connect a specific fact about this student to a specific fact "
    "retrieved about this college: how their score sits against its published "
    "range, how their coursework maps onto the major it offers, what its "
    "admission structure means for them. If the retrieved evidence is too thin "
    "to say anything college-specific, list fewer strengths and say the evidence "
    "is thin. Do not pad the list with profile praise.\n\n"
    "The classification and confidence are computed deterministically and given "
    "to you. Return them EXACTLY as supplied. Your job is to explain why that "
    "label follows from the evidence, not to choose a different one. " + GUARDRAILS
)

CRITIC_SYSTEM = (
    "You validate another agent's admission assessments. You do not rewrite "
    "them. Flag unsupported claims, missing or stale evidence, contradictions, "
    "overconfidence, university-wide rates used as major-specific, and missing "
    "student information. Return only your decision and the issues. " + GUARDRAILS
)

ACTION_SYSTEM = (
    "You produce prioritized, practical next actions for a high-school student. "
    "Never suggest dishonest or misleading application strategies. " + GUARDRAILS
)

ROADMAP_SYSTEM = (
    "You lay prioritized actions onto a TODAY / THIS WEEK / THIS MONTH / "
    "UPCOMING roadmap. " + GUARDRAILS
)


def _today() -> str:
    """Models have no clock. Without this they call a completed junior year
    'planned or projected engagement', which is wrong and alarming to read."""
    from datetime import UTC, datetime

    now = datetime.now(UTC)
    # US academic years run Aug-May, so before August we are still in the year
    # that started the previous calendar year.
    start = now.year if now.month >= 8 else now.year - 1
    return (
        f"Today's date is {now:%Y-%m-%d}. The current academic year is "
        f"{start}-{start + 1}. Any academic year ending on or before {start} is "
        f"COMPLETED, not planned. Do not describe completed work as projected."
    )


def _dump(value: Any) -> str:
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")
    return json.dumps(value, indent=2, default=str)


async def run_profile_agent(
    provider: LLMProvider, twin: StudentDigitalTwin, grade_trend: str
) -> ProfileAnalysis:
    return await provider.structured(
        task="profile",
        system=PROFILE_SYSTEM,
        prompt=(
            f"{_today()}\n\n"
            "Verified academic profile:\n"
            f"{_dump(twin.academics)}\n\n"
            f"Deterministically computed grade trend: {grade_trend}\n\n"
            "Assess course rigor, math/science/CS preparation, strengths and risks."
        ),
        schema=ProfileAnalysis,
        context={"twin": twin, "grade_trend": grade_trend},
    )


async def run_activity_agent(provider: LLMProvider, twin: StudentDigitalTwin) -> ActivityAnalysis:
    return await provider.structured(
        task="activity",
        system=ACTIVITY_SYSTEM,
        prompt=(
            f"{_today()}\n\n"
            "Activities:\n"
            f"{_dump(twin.activities)}\n\nProjects:\n{_dump(twin.projects)}\n\n"
            f"Awards:\n{_dump(twin.awards)}\n\n"
            "Identify strengths, themes, leadership/technical/service evidence and risks."
        ),
        schema=ActivityAnalysis,
        context={"twin": twin},
    )


async def run_stem_fit_agent(
    provider: LLMProvider, twin: StudentDigitalTwin, *, limit: int = 6
) -> list[STEMFit]:
    result = await provider.structured(
        task="stem",
        system=STEM_SYSTEM,
        prompt=(
            f"Digital twin:\n{_dump(twin)}\n\n"
            f"Return {limit} STEM disciplines ranked by fit, each with supporting "
            "evidence drawn from the transcript, concerns and career paths."
        ),
        schema=STEMFitList,
        context={"twin": twin, "limit": limit},
    )
    return result.disciplines


async def run_college_discovery(
    provider: LLMProvider,
    twin: StudentDigitalTwin,
    stem_fit: list[STEMFit],
    catalog: list[dict[str, Any]],
    *,
    limit: int,
    mode: str = "catalog",
) -> CollegeCandidateList:
    if mode == "open":
        instruction = (
            f"Propose at most {limit} real universities this student should "
            f"consider, each paired with one target major drawn from the ranked "
            f"STEM fit list above. Use your own knowledge of institutions; ignore "
            f"any catalog. Include a mix of selectivity."
        )
        catalog_block = ""
    elif mode == "hybrid":
        instruction = (
            f"Select the best fits from the indexed institutions below first, then "
            f"propose additional real universities from your own knowledge until "
            f"you reach at most {limit} candidates."
        )
        catalog_block = f"Indexed institutions:\n{_dump(catalog)}\n\n"
    else:
        instruction = (
            f"Select at most {limit} candidates from the institutions below, "
            f"each paired with one target major."
        )
        catalog_block = f"Available institutions:\n{_dump(catalog)}\n\n"

    return await provider.structured(
        task="research",
        system=DISCOVERY_SYSTEM,
        prompt=(
            f"Digital twin preferences:\n{_dump(twin.preferences)}\n\n"
            f"Ranked STEM fit:\n{_dump([f.model_dump() for f in stem_fit])}\n\n"
            f"{catalog_block}{instruction}"
        ),
        schema=CollegeCandidateList,
        context={
            "twin": twin,
            "stem_fit": stem_fit,
            "catalog": catalog,
            "limit": limit,
            "mode": mode,
        },
    )


def _compact_twin(twin: StudentDigitalTwin) -> dict[str, Any]:
    """A summary, not the whole record.

    The admission prompt was shipping all 37 raw course rows to every college.
    That is tokens the model does not need — the profile analysis already
    interprets them — and it multiplies across every college in the list.
    """
    academics = twin.academics
    return {
        "gpa": academics.gpa.gpa,
        "graded_credits": academics.gpa.graded_credits,
        "class_rank": academics.class_rank,
        "graduation_year": academics.graduation_year,
        "ap_courses": academics.ap_courses,
        "ib_courses": academics.ib_courses,
        "math_progression": academics.math_progression,
        "science_progression": academics.science_progression,
        "cs_progression": academics.cs_progression,
        "senior_year_courses": academics.senior_year_courses,
        "testing": twin.testing.model_dump(mode="json"),
        "stem_interests": twin.stem_interests,
        "career_interests": twin.career_interests,
    }


async def run_admission_agent(
    provider: LLMProvider,
    twin: StudentDigitalTwin,
    research: CollegeResearchResult,
    profile_analysis: ProfileAnalysis,
    activity_analysis: ActivityAnalysis,
    baseline_description: str = "",
) -> AdmissionAssessment:
    return await provider.structured(
        task="admission",
        system=ADMISSION_SYSTEM,
        prompt=(
            f"{_today()}\n\n"
            f"Student summary:\n{_dump(_compact_twin(twin))}\n\n"
            f"Profile analysis:\n{_dump(profile_analysis)}\n\n"
            f"Activity analysis:\n{_dump(activity_analysis)}\n\n"
            f"Retrieved college research:\n{_dump(research)}\n\n"
            f"{baseline_description}\n\n"
            f"Explain the classification for {research.target_major} at "
            f"{research.university} from the evidence above. Ground each strength "
            f"and risk in this college's retrieved facts."
        ),
        schema=AdmissionAssessment,
        context={
            "twin": twin,
            "research": research,
            "profile_analysis": profile_analysis,
            "activity_analysis": activity_analysis,
        },
    )


async def run_critic_agent(
    provider: LLMProvider,
    assessments: dict[str, AdmissionAssessment],
    research: dict[str, CollegeResearchResult],
    *,
    abstentions: dict[str, AssessmentAbstention] | None = None,
    critic_loop_count: int,
    max_critic_loops: int,
    stale_after_days: int,
) -> CriticResult:
    return await provider.structured(
        task="critic",
        system=CRITIC_SYSTEM,
        prompt=(
            "Assessments to validate:\n"
            f"{_dump({k: v.model_dump() for k, v in assessments.items()})}\n\n"
            f"Underlying research:\n{_dump({k: v.model_dump() for k, v in research.items()})}\n\n"
            f"Colleges the evidence gate refused to assess: {sorted(abstentions or {})}\n\n"
            f"This is critic loop {critic_loop_count} of a maximum {max_critic_loops}. "
            "Decide: approve, research_more, or human_review."
        ),
        schema=CriticResult,
        context={
            "assessments": assessments,
            "research": research,
            "abstentions": abstentions or {},
            "critic_loop_count": critic_loop_count,
            "max_critic_loops": max_critic_loops,
            "stale_after_days": stale_after_days,
        },
    )


async def run_next_best_action(
    provider: LLMProvider,
    twin: StudentDigitalTwin,
    assessments: dict[str, AdmissionAssessment],
    activity_analysis: ActivityAnalysis,
) -> NextActionList:
    return await provider.structured(
        task="action",
        system=ACTION_SYSTEM,
        prompt=(
            f"Digital twin:\n{_dump(twin)}\n\n"
            f"Assessments:\n{_dump({k: v.model_dump() for k, v in assessments.items()})}\n\n"
            f"Activity analysis:\n{_dump(activity_analysis)}\n\n"
            "Return prioritized next actions."
        ),
        schema=NextActionList,
        context={"twin": twin, "assessments": assessments, "activity_analysis": activity_analysis},
    )


async def run_roadmap(
    provider: LLMProvider, actions: NextActionList, *, sections: list[str]
) -> Roadmap:
    return await provider.structured(
        task="roadmap",
        system=ROADMAP_SYSTEM,
        prompt=(
            f"Actions:\n{_dump(actions)}\n\nRegenerate only these roadmap sections: {sections}."
        ),
        schema=Roadmap,
        context={"actions": actions.actions, "sections": sections},
    )
