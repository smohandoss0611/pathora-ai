"""Evidence gate: a code-level decision made BEFORE generation.

The system prompt already instructs the Admission Agent to ground every claim
and flag gaps. That instruction is a request, and it is evaluated by a model
that has already read the retrieved passages — at which point plausible-looking
text almost always produces an answer. Prompt-level abstention is therefore a
backup, not a control.

This module is the control. It inspects retrieval metadata only — source types,
counts, fact provenance, staleness — and decides whether an assessment may be
attempted at all. It never sees the student and never calls a model, so it is
deterministic, unit-testable, and cannot be talked out of its decision.

If the gate fails, no LLM call is made for that college and an abstention is
recorded with the specific checks that failed.
"""

from __future__ import annotations

from pathora.config import Settings, get_settings
from pathora.domain.models import (
    NOT_PUBLISHED,
    AssessmentAbstention,
    CollegeResearchResult,
    GateCheck,
    GateResult,
)

#: A classification is anchored on selectivity. Without at least one of these,
#: any Safety/Target/Reach label would be an invention.
ANCHOR_FACTS = ("admit_rate", "major_admit_rate")


def evidence_gate(
    research: CollegeResearchResult,
    *,
    settings: Settings | None = None,
    min_sources: int = 2,
    require_official: bool = True,
) -> GateResult:
    """Decide whether there is enough evidence to classify this college."""
    settings = settings or get_settings()
    checks: list[GateCheck] = []

    # 1. Something was retrieved at all.
    checks.append(
        GateCheck(
            name="evidence_retrieved",
            passed=bool(research.evidence),
            detail=f"{len(research.evidence)} evidence record(s) retrieved",
        )
    )

    # 3. At least one authoritative source, not just any page that mentions it.
    # Official institutional research counts: IPEDS is a mandatory federal
    # reporting dataset, which is a stronger source than most admissions pages.
    types = {e.source_type for e in research.evidence}
    authoritative = types & {
        "official_admissions",
        "common_data_set",
        "official_stem_program",
        "institutional_research",
    }
    checks.append(
        GateCheck(
            name="authoritative_source",
            passed=bool(authoritative) or not require_official,
            detail=f"authoritative source types present: {sorted(authoritative) or 'none'}",
        )
    )

    # 4. A selectivity anchor exists. Without it there is nothing to classify against.
    anchors = [f for f in ANCHOR_FACTS if getattr(research, f) != NOT_PUBLISHED]
    checks.append(
        GateCheck(
            name="selectivity_anchor",
            passed=bool(anchors),
            detail=f"selectivity facts available: {anchors or 'none'}",
        )
    )

    # 2. Independent corroboration: one page is not a corpus. Exempt a single
    # authoritative dataset that already carries a selectivity anchor —
    # corroboration exists to catch an unreliable lone page, and a mandatory
    # federal survey is not that. Demanding two URLs would make IPEDS, the most
    # complete admissions source available, permanently unusable.
    distinct = {e.source_url for e in research.evidence}
    corroborated = len(distinct) >= min_sources or (bool(authoritative) and bool(anchors))
    checks.append(
        GateCheck(
            name="independent_sources",
            passed=corroborated,
            detail=(
                f"{len(distinct)} distinct source URL(s); {min_sources} required unless a "
                f"single authoritative dataset supplies the selectivity anchor "
                f"(authoritative={sorted(authoritative) or 'none'}, anchors={anchors or 'none'})"
            ),
        )
    )

    # 5. Every stated fact traces to a retrieved document.
    stated = [
        f
        for f in (
            "admit_rate",
            "major_admit_rate",
            "sat_range",
            "act_range",
            "test_policy",
            "admission_structure",
        )
        if getattr(research, f) != NOT_PUBLISHED
    ]
    untraced = [f for f in stated if f not in research.fact_sources]
    checks.append(
        GateCheck(
            name="facts_traced",
            passed=not untraced,
            detail=f"untraced facts: {untraced or 'none'}",
        )
    )

    # 6. Evidence is not entirely stale. Annual surveys get a longer window
    # than live pages, because they are published on a reporting lag by design.
    annual = {"institutional_research", "common_data_set"}

    def _allowance(source_type: str) -> int:
        if source_type in annual:
            return settings.annual_survey_stale_after_days
        return settings.evidence_stale_after_days

    fresh = [e for e in research.evidence if e.age_days() <= _allowance(e.source_type)]
    checks.append(
        GateCheck(
            name="evidence_fresh",
            passed=bool(fresh) or not research.evidence,
            detail=(
                f"{len(fresh)} of {len(research.evidence)} record(s) within their "
                f"freshness window ({settings.evidence_stale_after_days}d for pages, "
                f"{settings.annual_survey_stale_after_days}d for annual surveys)"
            ),
        )
    )

    # 7. Research did not error out.
    checks.append(
        GateCheck(
            name="research_succeeded",
            passed=research.research_error is None,
            detail=research.research_error or "no research error",
        )
    )

    failed = [c for c in checks if not c.passed]
    return GateResult(
        university=research.university,
        passed=not failed,
        checks=checks,
        failed_checks=[c.name for c in failed],
        reason=(
            "sufficient evidence to assess"
            if not failed
            else "; ".join(f"{c.name}: {c.detail}" for c in failed)
        ),
        evidence_ids=[e.evidence_id for e in research.evidence],
    )


def to_abstention(research: CollegeResearchResult, gate: GateResult) -> AssessmentAbstention:
    """Turn a failed gate into a record the student can act on."""
    guidance = {
        "evidence_retrieved": "No official documents were found for this university.",
        "independent_sources": (
            "Only one source was found; a single page is not enough to classify."
        ),
        "authoritative_source": (
            "No official admissions, Common Data Set or program page was found."
        ),
        "selectivity_anchor": (
            "No published admit rate was found, so no fit label can be anchored."
        ),
        "facts_traced": "Some retrieved facts could not be traced to a source document.",
        "evidence_fresh": "All retrieved evidence is older than the freshness threshold.",
        "research_succeeded": "Research failed for this university.",
    }
    return AssessmentAbstention(
        university=research.university,
        recommended_major=research.target_major,
        reason=gate.reason,
        failed_checks=gate.failed_checks,
        what_would_help=[guidance[name] for name in gate.failed_checks if name in guidance],
        evidence_ids=gate.evidence_ids,
        missing_information=research.missing_information,
    )
