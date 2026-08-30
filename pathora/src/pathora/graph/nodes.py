"""Graph nodes.

Dependencies (LLM provider, vector store, settings) are injected once through
``Deps`` rather than imported inside nodes, which keeps every node unit-testable
and keeps vendor SDKs out of the agent layer.
"""

from __future__ import annotations

import asyncio
import base64
import logging
from dataclasses import dataclass, field
from typing import Any

from langgraph.types import interrupt

from pathora.agents.analysts import (
    run_activity_agent,
    run_admission_agent,
    run_college_discovery,
    run_critic_agent,
    run_next_best_action,
    run_profile_agent,
    run_roadmap,
    run_stem_fit_agent,
)
from pathora.agents.college_worker import college_research_worker
from pathora.config import Settings, get_settings
from pathora.domain.models import (
    Activity,
    ActivityAnalysis,
    AdmissionAssessment,
    AssessmentAbstention,
    Award,
    CollegeCandidate,
    CollegeResearchResult,
    CriticResult,
    ExtractedAcademics,
    HumanActionRequest,
    NextActionList,
    ProfileAnalysis,
    Project,
    STEMFit,
    StudentDigitalTwin,
    StudentPreferences,
    TestingProfile,
)
from pathora.graph.state import PathoraState
from pathora.llm.base import LLMProvider
from pathora.rag.store import VectorStore, load_seed_payload
from pathora.services.classifier import classify, describe
from pathora.services.evidence import analyze_gap, build_passport
from pathora.services.evidence_gate import evidence_gate, to_abstention
from pathora.services.gpa import calculate_unweighted_gpa, grade_trend
from pathora.services.transcript import parse_transcript_pdf, parse_transcript_text
from pathora.services.twin import build_digital_twin

log = logging.getLogger(__name__)


@dataclass
class Deps:
    provider: LLMProvider
    store: VectorStore
    settings: Settings = field(default_factory=get_settings)
    catalog: list[dict[str, Any]] | None = None

    def college_catalog(self) -> list[dict[str, Any]]:
        if self.catalog is None:
            self.catalog = load_seed_payload()["colleges"]
        return self.catalog


# --------------------------------------------------------------------------- #
# 1. Transcript
# --------------------------------------------------------------------------- #
async def parse_transcript(state: PathoraState, deps: Deps) -> dict:
    document = state.get("transcript_document") or {}
    if text := document.get("text"):
        extracted = parse_transcript_text(text)
    elif encoded := document.get("pdf_base64"):
        extracted = parse_transcript_pdf(base64.b64decode(encoded))
    else:
        return {
            "extracted_academics": ExtractedAcademics().model_dump(mode="json"),
            "workflow_status": "awaiting_transcript",
            "warnings": [*state.get("warnings", []), "No transcript document was supplied"],
        }

    return {
        "extracted_academics": extracted.model_dump(mode="json"),
        "workflow_status": "transcript_parsed",
    }


# --------------------------------------------------------------------------- #
# 2. GPA (deterministic)
# --------------------------------------------------------------------------- #
async def calculate_gpa(state: PathoraState, deps: Deps) -> dict:
    extracted = ExtractedAcademics.model_validate(state["extracted_academics"])
    result = calculate_unweighted_gpa(extracted.courses)
    return {"gpa_result": result.model_dump(mode="json"), "workflow_status": "gpa_calculated"}


# --------------------------------------------------------------------------- #
# 3. Human verification (interrupt)
# --------------------------------------------------------------------------- #
async def verify_academic_profile(state: PathoraState, deps: Deps) -> dict:
    extracted = ExtractedAcademics.model_validate(state["extracted_academics"])
    reported = extracted.reported_gpa
    computed = state["gpa_result"].get("gpa", 0.0)
    conflict = reported is not None and abs(reported - computed) > 0.2

    threshold = deps.settings.extraction_confidence_threshold
    if not extracted.needs_human_verification(threshold) and not conflict:
        return {
            "verified_academics": extracted.model_dump(mode="json"),
            "workflow_status": "academics_verified",
        }

    reasons = []
    if extracted.uncertain_fields:
        reasons.append(f"uncertain fields: {', '.join(extracted.uncertain_fields)}")
    if extracted.extraction_confidence < threshold:
        reasons.append(f"extraction confidence {extracted.extraction_confidence}")
    if conflict:
        reasons.append(f"transcript reports GPA {reported} but computed GPA is {computed}")

    request = HumanActionRequest(
        kind="resolve_conflict" if conflict else "verify_transcript",
        message="Please review the extracted academic record before we continue. "
        + "; ".join(reasons),
        payload={
            "extracted_academics": extracted.model_dump(mode="json"),
            "computed_gpa": computed,
            "reported_gpa": reported,
        },
    )

    response = interrupt(request.model_dump(mode="json")) or {}
    choice = response.get("choice", "confirm")
    edits = response.get("edits", {})

    if choice == "cancel":
        return {
            "workflow_status": "cancelled_by_user",
            "pending_human_action": None,
            "human_responses": [*state.get("human_responses", []), response],
        }

    verified = extracted
    warnings = list(state.get("warnings", []))
    if choice == "edit" and edits:
        verified = ExtractedAcademics.model_validate(extracted.model_dump() | edits)
    elif choice == "continue_with_uncertainty":
        warnings.append("Student continued with unverified transcript fields")

    updates = {
        "verified_academics": verified.model_dump(mode="json"),
        "pending_human_action": None,
        "human_responses": [*state.get("human_responses", []), response],
        "workflow_status": "academics_verified",
        "warnings": warnings,
    }
    if choice == "edit" and edits:
        updates["gpa_result"] = calculate_unweighted_gpa(verified.courses).model_dump(mode="json")
    return updates


# --------------------------------------------------------------------------- #
# 4/5. Profile + Activity agents (parallel)
# --------------------------------------------------------------------------- #
def _twin_from_state(state: PathoraState) -> StudentDigitalTwin:
    return StudentDigitalTwin.model_validate(state["student_twin"])


def _provisional_twin(state: PathoraState) -> StudentDigitalTwin:
    """Twin built from verified academics + student-supplied profile input."""
    verified = ExtractedAcademics.model_validate(state["verified_academics"])
    supplied = state.get("student_input", {}) or {}
    return build_digital_twin(
        student_id=state["student_id"],
        verified_academics=verified,
        testing=TestingProfile.model_validate(supplied.get("testing", {})),
        activities=[Activity.model_validate(a) for a in supplied.get("activities", [])],
        projects=[Project.model_validate(p) for p in supplied.get("projects", [])],
        awards=[Award.model_validate(a) for a in supplied.get("awards", [])],
        stem_interests=supplied.get("stem_interests", []),
        career_interests=supplied.get("career_interests", []),
        preferences=StudentPreferences.model_validate(supplied.get("preferences", {})),
    )


async def profile_agent(state: PathoraState, deps: Deps) -> dict:
    twin = _provisional_twin(state)
    analysis = await run_profile_agent(deps.provider, twin, grade_trend(twin.academics.courses))
    return {"profile_analysis": analysis.model_dump(mode="json")}


async def activity_agent(state: PathoraState, deps: Deps) -> dict:
    twin = _provisional_twin(state)
    analysis = await run_activity_agent(deps.provider, twin)
    return {"activity_analysis": analysis.model_dump(mode="json")}


# --------------------------------------------------------------------------- #
# 6. Digital twin (fan-in)
# --------------------------------------------------------------------------- #
async def build_twin(state: PathoraState, deps: Deps) -> dict:
    twin = _provisional_twin(state)
    profile = ProfileAnalysis.model_validate(state["profile_analysis"])
    activity = ActivityAnalysis.model_validate(state["activity_analysis"])
    twin.academic_strengths = [*profile.academic_strengths, *activity.strengths]
    twin.academic_risks = [*profile.academic_risks, *activity.risks]
    return {"student_twin": twin.model_dump(mode="json"), "workflow_status": "twin_built"}


# --------------------------------------------------------------------------- #
# 7. STEM fit
# --------------------------------------------------------------------------- #
async def stem_fit_agent(state: PathoraState, deps: Deps) -> dict:
    twin = _twin_from_state(state)
    fits = await run_stem_fit_agent(deps.provider, twin, limit=6)
    return {
        "stem_fit": [f.model_dump(mode="json") for f in fits],
        "workflow_status": "stem_fit_complete",
    }


# --------------------------------------------------------------------------- #
# 8. College discovery
# --------------------------------------------------------------------------- #
async def college_discovery(state: PathoraState, deps: Deps) -> dict:
    twin = _twin_from_state(state)
    fits = [STEMFit.model_validate(f) for f in state["stem_fit"]]
    catalog = state.get("college_catalog") or deps.college_catalog()

    result = await run_college_discovery(
        deps.provider,
        twin,
        fits,
        catalog,
        limit=deps.settings.max_colleges_per_analysis,
        mode=deps.settings.college_discovery_mode,
    )
    candidates = result.candidates[: deps.settings.max_colleges_per_analysis]

    # The prompt asks for majors drawn from the ranked STEM fits; verify it
    # rather than trusting it. A CS-heavy profile steered into a broad
    # engineering major is a real failure mode, not a stylistic quibble.
    ranked = [f.discipline for f in fits]
    warnings = list(state.get("warnings", []))
    off_list = [
        f"{c.university}: recommended {c.target_major}, which is not among the "
        f"student's ranked STEM fits ({', '.join(ranked[:3])}...)"
        for c in candidates
        if ranked and c.target_major not in ranked
    ]
    if off_list:
        warnings.extend(off_list)

    return {
        "college_candidates": [c.university for c in candidates],
        "candidate_details": {c.university: c.model_dump(mode="json") for c in candidates},
        "college_catalog": catalog,
        "warnings": warnings,
        "workflow_status": "colleges_discovered",
    }


# --------------------------------------------------------------------------- #
# 9. College research worker (Send target)
# --------------------------------------------------------------------------- #
async def research_worker_node(payload: dict, deps: Deps) -> dict:
    candidate = CollegeCandidate.model_validate(payload["candidate"])
    deep = bool(payload.get("deep"))
    result = await college_research_worker(candidate, deps.store, deep=deep, settings=deps.settings)
    return {"college_research": {candidate.university: result.model_dump(mode="json")}}


async def collect_research(state: PathoraState, deps: Deps) -> dict:
    """Fan-in barrier. Research dicts were merged by the state reducer."""
    return {"workflow_status": "research_complete"}


# --------------------------------------------------------------------------- #
# 10. Admission agent
# --------------------------------------------------------------------------- #
async def _assess_one(
    university: str,
    raw: dict,
    *,
    twin: StudentDigitalTwin,
    profile: ProfileAnalysis,
    activity: ActivityAnalysis,
    deps: Deps,
    profile_verified: bool,
) -> tuple[str, dict]:
    """Gate, classify and explain one college. Safe to run concurrently."""
    research = CollegeResearchResult.model_validate(raw)
    out: dict = {"warnings": []}

    # Code-level gate BEFORE generation. The model never sees the retrieved
    # passages unless this passes, so abstention is a decision rather than a
    # request the model may talk itself out of.
    gate = evidence_gate(research, settings=deps.settings)
    out["gate"] = gate.model_dump(mode="json")

    if not gate.passed:
        out["abstention"] = to_abstention(research, gate).model_dump(mode="json")
        out["passport"] = build_passport(
            research,
            AdmissionAssessment(
                university=university,
                recommended_major=research.target_major,
                classification="Reach",
                confidence="Low",
                missing_information=research.missing_information,
            ),
            profile_verified=profile_verified,
            stale_after_days=deps.settings.evidence_stale_after_days,
        ).model_dump(mode="json")
        return university, out

    # The label is computed here, not chosen by the model.
    baseline = classify(twin, research, profile)
    try:
        assessment = await run_admission_agent(
            deps.provider,
            twin,
            research,
            profile,
            activity,
            baseline_description=describe(baseline),
        )
    except Exception as exc:  # noqa: BLE001
        # The label is already deterministic; only the prose needs the model, so
        # a provider failure costs the explanation and nothing else.
        log.warning("admission agent failed for %s: %s", university, exc)
        out["warnings"].append(f"{university}: explanation unavailable ({type(exc).__name__})")
        assessment = AdmissionAssessment(
            university=university,
            recommended_major=research.target_major,
            classification=baseline.classification,
            confidence=baseline.confidence,
            strengths=baseline.signals,
            evidence_ids=[e.evidence_id for e in research.evidence],
            missing_information=research.missing_information,
            rationale_summary=(
                f"Classified {baseline.classification} from published evidence. "
                + " ".join(f"{sig}." for sig in baseline.signals)
            ),
        )

    if (
        assessment.classification != baseline.classification
        or assessment.confidence != baseline.confidence
    ):
        out["warnings"].append(
            f"{university}: model returned {assessment.classification}/"
            f"{assessment.confidence}; using computed "
            f"{baseline.classification}/{baseline.confidence}"
        )
        assessment = assessment.model_copy(
            update={
                "classification": baseline.classification,
                "confidence": baseline.confidence,
            }
        )

    out["assessment"] = assessment.model_dump(mode="json")
    out["passport"] = build_passport(
        research,
        assessment,
        profile_verified=profile_verified,
        stale_after_days=deps.settings.evidence_stale_after_days,
    ).model_dump(mode="json")
    out["gap"] = analyze_gap(twin, research, assessment).model_dump(mode="json")
    return university, out


async def admission_agent(state: PathoraState, deps: Deps) -> dict:
    twin = _twin_from_state(state)
    profile = ProfileAnalysis.model_validate(state["profile_analysis"])
    activity = ActivityAnalysis.model_validate(state["activity_analysis"])

    assessments = dict(state.get("admission_results", {}))
    passports = dict(state.get("evidence_passports", {}))
    gaps = dict(state.get("gap_analysis", {}))
    warnings = list(state.get("warnings", []))
    abstentions = dict(state.get("abstentions", {}))
    gate_results = dict(state.get("gate_results", {}))

    # Colleges are independent, so assess them concurrently. Running twelve
    # model calls end to end made the workflow feel broken; the same bound as
    # the research fan-out keeps provider rate limits in view.
    semaphore = asyncio.Semaphore(max(1, deps.settings.max_parallel_college_workers))
    profile_verified = bool(state.get("verified_academics"))

    async def guarded(university: str, raw: dict) -> tuple[str, dict]:
        async with semaphore:
            return await _assess_one(
                university,
                raw,
                twin=twin,
                profile=profile,
                activity=activity,
                deps=deps,
                profile_verified=profile_verified,
            )

    results = await asyncio.gather(
        *[guarded(u, raw) for u, raw in state.get("college_research", {}).items()],
        return_exceptions=True,
    )

    for outcome in results:
        if isinstance(outcome, BaseException):
            log.warning("assessment task failed: %s", outcome)
            warnings.append(f"An assessment task failed: {type(outcome).__name__}")
            continue
        university, out = outcome
        gate_results[university] = out["gate"]
        warnings.extend(out["warnings"])
        if "passport" in out:
            passports[university] = out["passport"]
        if "abstention" in out:
            abstentions[university] = out["abstention"]
            assessments.pop(university, None)
        else:
            abstentions.pop(university, None)
            assessments[university] = out["assessment"]
            gaps[university] = out["gap"]

    # Verify what the prompt asked for. Identical strength lists across
    # different institutions mean the agent restated the profile instead of
    # reasoning about each college.
    fingerprints: dict[tuple[str, ...], list[str]] = {}
    for university, raw in assessments.items():
        fingerprints.setdefault(tuple(sorted(raw.get("strengths", []))), []).append(university)
    for shared in fingerprints.values():
        if len(shared) > 1:
            warnings.append(
                "Identical reasoning was produced for "
                + ", ".join(sorted(shared))
                + " — these strengths are not college-specific."
            )

    return {
        "admission_results": assessments,
        "warnings": warnings,
        "abstentions": abstentions,
        "gate_results": gate_results,
        "evidence_passports": passports,
        "gap_analysis": gaps,
        "workflow_status": "assessed",
    }


async def critic_agent(state: PathoraState, deps: Deps) -> dict:
    assessments = {
        u: AdmissionAssessment.model_validate(a)
        for u, a in state.get("admission_results", {}).items()
    }
    research = {
        u: CollegeResearchResult.model_validate(r)
        for u, r in state.get("college_research", {}).items()
    }
    loop = state.get("critic_loop_count", 0)

    result = await run_critic_agent(
        deps.provider,
        assessments,
        research,
        abstentions={
            u: AssessmentAbstention.model_validate(a)
            for u, a in state.get("abstentions", {}).items()
        },
        critic_loop_count=loop,
        max_critic_loops=deps.settings.max_critic_loops,
        stale_after_days=deps.settings.evidence_stale_after_days,
    )

    # Retries are bounded twice over: by the critic loop counter and by the
    # research retry counter. Neither can be reset by an agent.
    if result.decision == "research_more" and (
        loop + 1 > deps.settings.max_critic_loops
        or state.get("research_retry_count", 0) >= deps.settings.max_research_retries
    ):
        result = CriticResult(
            decision="human_review",
            issues=[*result.issues, "Research retries exhausted; escalating to human review"],
            colleges_to_research=result.colleges_to_research,
            missing_information=result.missing_information,
        )

    return {
        "critic_results": result.model_dump(mode="json"),
        "critic_loop_count": loop + 1,
        "workflow_status": f"critic_{result.decision}",
    }


# --------------------------------------------------------------------------- #
# 12. Targeted re-research bookkeeping
# --------------------------------------------------------------------------- #
async def targeted_research(state: PathoraState, deps: Deps) -> dict:
    return {
        "research_retry_count": state.get("research_retry_count", 0) + 1,
        "workflow_status": "targeted_research",
    }


# --------------------------------------------------------------------------- #
# 13. Human review (interrupt)
# --------------------------------------------------------------------------- #
async def human_review(state: PathoraState, deps: Deps) -> dict:
    critic = CriticResult.model_validate(state["critic_results"])
    request = HumanActionRequest(
        kind="critic_human_review",
        message=(
            "We could not fully verify some admission information from official "
            "sources. Review the open issues and choose how to continue."
        ),
        payload={
            "issues": critic.issues,
            "missing_information": critic.missing_information,
            "colleges": critic.colleges_to_research,
        },
    )

    response = interrupt(request.model_dump(mode="json")) or {}
    choice = response.get("choice", "continue_with_uncertainty")
    warnings = list(state.get("warnings", []))

    if choice == "cancel":
        return {
            "workflow_status": "cancelled_by_user",
            "human_responses": [*state.get("human_responses", []), response],
        }

    warnings.append(
        "Assessments for "
        + ", ".join(critic.colleges_to_research or ["the current list"])
        + " carry unresolved evidence gaps accepted by the student"
    )
    return {
        "human_responses": [*state.get("human_responses", []), response],
        "warnings": warnings,
        "pending_human_action": None,
        "workflow_status": "human_reviewed",
    }


# --------------------------------------------------------------------------- #
# 14/15. Next best action + roadmap
# --------------------------------------------------------------------------- #
async def next_best_action(state: PathoraState, deps: Deps) -> dict:
    twin = _twin_from_state(state)
    assessments = {
        u: AdmissionAssessment.model_validate(a)
        for u, a in state.get("admission_results", {}).items()
    }
    activity = ActivityAnalysis.model_validate(state["activity_analysis"])
    result = await run_next_best_action(deps.provider, twin, assessments, activity)
    return {
        "next_actions": [a.model_dump(mode="json") for a in result.actions],
        "workflow_status": "actions_ready",
    }


async def dynamic_roadmap(
    state: PathoraState, deps: Deps, sections: list[str] | None = None
) -> dict:
    actions = NextActionList.model_validate({"actions": state.get("next_actions", [])})
    sections = sections or ["today", "this_week", "this_month", "upcoming"]
    result = await run_roadmap(deps.provider, actions, sections=sections)

    if existing := state.get("roadmap"):
        merged = dict(existing)
        for section in sections:
            merged[section] = result.model_dump(mode="json")[section]
        merged["sections_regenerated"] = sections
        merged["generated_at"] = result.generated_at.isoformat()
        return {"roadmap": merged, "workflow_status": "complete"}

    return {"roadmap": result.model_dump(mode="json"), "workflow_status": "complete"}
