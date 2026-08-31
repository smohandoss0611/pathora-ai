"""What-If Lab (Section 27).

Selective re-execution: a scenario updates the Digital Twin and then reruns only
the nodes whose inputs actually changed. Transcript parsing and activity analysis
are never rerun for a test-score change; GPA is recomputed deterministically only
when a grade changes, and never re-derived from the PDF.
"""

from __future__ import annotations

import copy
from typing import Any, cast

from pathora.domain.models import (
    Activity,
    AdmissionAssessment,
    Course,
    Project,
    StudentDigitalTwin,
    WhatIfChange,
    WhatIfResult,
    WhatIfScenario,
)
from pathora.graph import nodes
from pathora.graph.nodes import Deps
from pathora.graph.state import PathoraState
from pathora.services.gpa import calculate_unweighted_gpa

ALWAYS_SKIPPED = ["parse_transcript", "verify_academic_profile", "profile_agent"]


def _apply_scenario(
    twin: StudentDigitalTwin, scenario: WhatIfScenario
) -> tuple[StudentDigitalTwin, set[str]]:
    """Return an updated twin plus the set of changed input domains."""
    updated = twin.model_copy(deep=True)
    changed: set[str] = set()

    if scenario.sat_total is not None:
        updated.testing.sat_total = scenario.sat_total
        changed.add("testing")
    if scenario.act_composite is not None:
        updated.testing.act_composite = scenario.act_composite
        changed.add("testing")

    if scenario.senior_grades:
        senior = set(updated.academics.senior_year_courses)
        courses: list[Course] = []
        for course in updated.academics.courses:
            if course.name in scenario.senior_grades and course.name in senior:
                course = course.model_copy(update={"grade": scenario.senior_grades[course.name]})
                changed.add("grades")
            courses.append(course)
        updated.academics.courses = courses
        if "grades" in changed:
            # Deterministic recompute only. The transcript is not re-parsed.
            updated.academics.gpa = calculate_unweighted_gpa(courses)

    if scenario.added_activity is not None:
        updated.activities = [*updated.activities, Activity.model_validate(scenario.added_activity)]
        changed.add("activities")
    if scenario.added_project is not None:
        updated.projects = [*updated.projects, Project.model_validate(scenario.added_project)]
        changed.add("activities")

    if scenario.major:
        updated.stem_interests = sorted({*updated.stem_interests, scenario.major})
        changed.add("major")
    if scenario.preferences is not None:
        updated.preferences = scenario.preferences
        changed.add("preferences")

    return updated, changed


def _reason(
    before: AdmissionAssessment,
    after: AdmissionAssessment,
    scenario: WhatIfScenario,
    state: PathoraState,
) -> str:
    if before.classification != after.classification:
        return (
            f"Classification moved from {before.classification} to {after.classification} "
            f"under this scenario."
        )
    gap = state.get("gap_analysis", {}).get(after.university, {})
    constraint = gap.get("primary_constraint", "")
    if scenario.sat_total or scenario.act_composite:
        return f"Test score was not the dominant constraint. {constraint}".strip()
    return f"No change under this scenario. {constraint}".strip()


async def run_what_if(
    state: PathoraState, scenario: WhatIfScenario, deps: Deps
) -> tuple[PathoraState, WhatIfResult]:
    before = {
        u: AdmissionAssessment.model_validate(a)
        for u, a in state.get("admission_results", {}).items()
    }

    # Mutable working copy. ``cast`` keeps the TypedDict contract at the
    # boundaries while allowing free-form updates in between.
    working: dict[str, Any] = copy.deepcopy(dict(state))
    new_state = cast(PathoraState, working)
    twin = StudentDigitalTwin.model_validate(state["student_twin"])
    updated_twin, changed = _apply_scenario(twin, scenario)
    working["student_twin"] = updated_twin.model_dump(mode="json")
    if "grades" in changed:
        working["gpa_result"] = updated_twin.academics.gpa.model_dump(mode="json")

    rerun: list[str] = []
    skipped = list(ALWAYS_SKIPPED)

    if "activities" in changed:
        working.update(await nodes.activity_agent(new_state, deps))
        rerun.append("activity_agent")
    else:
        skipped.append("activity_agent")

    if "grades" in changed:
        rerun.append("calculate_gpa (deterministic recompute)")
    else:
        skipped.append("calculate_gpa")

    if changed & {"major", "grades"}:
        working.update(await nodes.stem_fit_agent(new_state, deps))
        rerun.append("stem_fit_agent")
    else:
        skipped.append("stem_fit_agent")

    # A major change must rerun discovery too. Without it the candidate list
    # keeps its original target majors, and the simulation reports "Industrial
    # Engineering at Texas A&M" while the student asked about Data Science —
    # the label moves for a major they did not simulate.
    if changed & {"preferences", "major"}:
        working.update(await nodes.college_discovery(new_state, deps))

        # "Simulate Data Science" means apply for Data Science, not merely add
        # it as an interest. Discovery ranks by fit, so a CS-heavy record keeps
        # returning Computer Science and the simulation answers a question the
        # student did not ask. Force the major where the institution offers it.
        if scenario.major:
            catalog = {c["university"]: c for c in working.get("college_catalog", [])}
            forced: dict[str, dict] = {}
            for university, candidate in working.get("candidate_details", {}).items():
                offered = catalog.get(university, {}).get("majors")
                if offered is None or scenario.major in offered:
                    candidate = {**candidate, "target_major": scenario.major}
                forced[university] = candidate
            working["candidate_details"] = forced
        working["college_research"] = {}
        for candidate in new_state.get("candidate_details", {}).values():
            update = await nodes.research_worker_node({"candidate": candidate}, deps)
            working["college_research"].update(update["college_research"])
        rerun.extend(["college_discovery", "research_worker"])
    else:
        skipped.extend(["college_discovery", "research_worker"])

    working["admission_results"] = {}
    working.update(await nodes.admission_agent(new_state, deps))
    working["critic_loop_count"] = 0
    working.update(await nodes.critic_agent(new_state, deps))
    working.update(await nodes.next_best_action(new_state, deps))
    rerun.extend(["admission_agent", "critic_agent", "next_best_action"])

    sections = ["today", "this_week"]
    if changed & {"preferences", "major", "grades"}:
        sections = ["today", "this_week", "this_month", "upcoming"]
    working.update(await nodes.dynamic_roadmap(new_state, deps, sections=sections))
    rerun.append(f"dynamic_roadmap({','.join(sections)})")

    after = {
        u: AdmissionAssessment.model_validate(a)
        for u, a in new_state.get("admission_results", {}).items()
    }

    changes = [
        WhatIfChange(
            university=university,
            before=before[university].classification,
            after=assessment.classification,
            changed=before[university].classification != assessment.classification,
            reason=_reason(before[university], assessment, scenario, new_state),
        )
        for university, assessment in after.items()
        if university in before
    ]

    moved = [c.university for c in changes if c.changed]
    summary = (
        f"{len(moved)} of {len(changes)} classifications moved: {', '.join(moved)}."
        if moved
        else "No classification changed under this scenario."
    )

    return new_state, WhatIfResult(
        scenario=scenario,
        changes=sorted(changes, key=lambda c: c.university),
        nodes_rerun=rerun,
        nodes_skipped=sorted(set(skipped)),
        summary=summary,
    )
