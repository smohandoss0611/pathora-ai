"""Deterministic reference implementations behind the Fake LLM provider.

These exist so the entire graph is runnable, testable and reproducible with no
API key and no network. They are *rule-based*, they only ever restate facts
present in their inputs, and they never invent a statistic. When a real provider
is configured the same prompts are sent to a model instead.
"""

from __future__ import annotations

import re
from typing import Any

from pathora.domain.models import (
    NOT_PUBLISHED,
    ActivityAnalysis,
    AdmissionAssessment,
    CollegeCandidate,
    CollegeCandidateList,
    CollegeResearchResult,
    CriticResult,
    NextAction,
    NextActionList,
    ProfileAnalysis,
    Roadmap,
    RoadmapItem,
    STEMFit,
    STEMFitList,
    StudentDigitalTwin,
)

STEM_CATALOG: list[dict[str, Any]] = [
    {
        "discipline": "Computer Science",
        "signals": {"cs": 2.0, "math": 1.0},
        "careers": ["Software Engineer", "ML Engineer", "Security Engineer"],
    },
    {
        "discipline": "Data Science",
        "signals": {"cs": 1.2, "stats": 1.6, "math": 1.0},
        "careers": ["Data Scientist", "Analytics Engineer", "Quantitative Analyst"],
    },
    {
        "discipline": "Statistics",
        "signals": {"stats": 2.0, "math": 1.0},
        "careers": ["Statistician", "Biostatistician", "Actuary"],
    },
    {
        "discipline": "Applied Mathematics",
        "signals": {"math": 2.0, "stats": 0.6},
        "careers": ["Applied Mathematician", "Modeling Analyst"],
    },
    {
        "discipline": "Industrial Engineering",
        "signals": {"math": 1.2, "stats": 1.0, "science": 0.8},
        "careers": ["Process Engineer", "Supply Chain Analyst"],
    },
    {
        "discipline": "Systems Engineering",
        "signals": {"math": 1.0, "cs": 0.8, "science": 1.0},
        "careers": ["Systems Engineer", "Requirements Engineer"],
    },
    {
        "discipline": "Operations Research",
        "signals": {"math": 1.4, "stats": 1.2},
        "careers": ["Operations Research Analyst", "Optimization Engineer"],
    },
    {
        "discipline": "Computer Engineering",
        "signals": {"cs": 1.4, "physics": 1.4, "math": 1.0},
        "careers": ["Embedded Engineer", "Hardware Engineer"],
    },
    {
        "discipline": "Electrical Engineering",
        "signals": {"physics": 1.8, "math": 1.2},
        "careers": ["Electrical Engineer", "Power Systems Engineer"],
    },
    {
        "discipline": "Cybersecurity",
        "signals": {"cs": 1.6, "math": 0.6},
        "careers": ["Security Analyst", "Application Security Engineer"],
    },
    {
        "discipline": "Information Science",
        "signals": {"cs": 1.0, "stats": 0.8},
        "careers": ["Information Architect", "Product Analyst"],
    },
    {
        "discipline": "Computational Science",
        "signals": {"math": 1.4, "cs": 1.0, "science": 1.0},
        "careers": ["Computational Scientist", "Simulation Engineer"],
    },
]

FIT_ORDER = {"Excellent": 3, "Strong": 2, "Moderate": 1, "Weak": 0}
CLASSIFICATION_LADDER = ["Safety", "Likely", "Target", "Target-Reach", "Reach", "High Reach"]


# --------------------------------------------------------------------------- #
# Profile
# --------------------------------------------------------------------------- #
def _rigor(twin: StudentDigitalTwin) -> str:
    advanced = [c for c in twin.academics.courses if c.level in {"AP", "IB", "DualEnrollment"}]
    honors = [c for c in twin.academics.courses if c.level == "Honors"]
    score = len(advanced) + 0.5 * len(honors)
    if score >= 6:
        return "Excellent"
    if score >= 4:
        return "Strong"
    if score >= 2:
        return "Moderate"
    return "Weak"


def profile_analysis(context: dict[str, Any]) -> ProfileAnalysis:
    twin: StudentDigitalTwin = context["twin"]
    trend: str = context.get("grade_trend", "Unknown")
    academics = twin.academics

    def describe(label: str, sequence: list[str]) -> str:
        if not sequence:
            return f"No {label} coursework found on the verified transcript."
        return f"{label} sequence on file: {' -> '.join(sequence)}."

    strengths: list[str] = []
    risks: list[str] = []

    if any("calculus" in c.lower() for c in academics.math_progression):
        strengths.append("Reached calculus before graduation")
    else:
        risks.append("No calculus on the verified transcript, which most STEM majors expect")

    if academics.cs_progression:
        strengths.append(f"Computer science coursework completed ({len(academics.cs_progression)})")
    else:
        risks.append("No computer science coursework on file")

    if academics.ap_courses:
        strengths.append(f"{len(academics.ap_courses)} AP course(s) attempted")
    if trend == "Declining":
        risks.append("Grade trend is declining across academic years")
    if academics.gpa.gpa and academics.gpa.gpa < 3.0:
        risks.append("Unweighted GPA below 3.0 relative to selective STEM applicant pools")
    if not academics.class_rank:
        risks.append("Class rank not reported on the transcript")

    return ProfileAnalysis(
        course_rigor=_rigor(twin),  # type: ignore[arg-type]
        grade_trend=trend,  # type: ignore[arg-type]
        math_preparation=describe("Math", academics.math_progression),
        science_preparation=describe("Science", academics.science_progression),
        cs_preparation=describe("Computer science", academics.cs_progression),
        academic_strengths=strengths,
        academic_risks=risks,
    )


# --------------------------------------------------------------------------- #
# Activities
# --------------------------------------------------------------------------- #
LEADERSHIP_WORDS = (
    "president",
    "captain",
    "lead",
    "founder",
    "chair",
    "officer",
    "manager",
    "director",
    "head of",
    "organizer",
    "coordinator",
    # Eagle Scout is a leadership credential in its own right: the rank requires
    # planning and leading a service project. Keyword lists that only look for
    # job titles miss it entirely.
    "eagle scout",
    "scout",
    "drum major",
    "section leader",
    "team lead",
)
SERVICE_WORDS = (
    "volunteer",
    "service",
    "tutor",
    "mentor",
    "outreach",
    "food bank",
    "nonprofit",
    "shelter",
    "community",
    "charity",
)
TECHNICAL_WORDS = (
    "robotics",
    "code",
    "coding",
    "software",
    "data",
    "engineering",
    "hackathon",
    "programming",
    "app",
    "web",
    "machine learning",
    "analytics",
)
#: Internships and fellowships are neither leadership nor service, and were
#: previously invisible to the analysis.
PROFESSIONAL_WORDS = ("intern", "internship", "apprentice", "fellow", "co-op", "research assistant")


def activity_analysis(context: dict[str, Any]) -> ActivityAnalysis:
    twin: StudentDigitalTwin = context["twin"]
    leadership: list[str] = []
    service: list[str] = []
    technical: list[str] = []
    professional: list[str] = []
    themes: set[str] = set()
    risks: list[str] = []

    for activity in twin.activities:
        blob = " ".join(filter(None, [activity.name, activity.role, activity.description])).lower()
        label = f"{activity.name}" + (f" — {activity.role}" if activity.role else "")
        if any(w in blob for w in LEADERSHIP_WORDS):
            leadership.append(label)
            themes.add("leadership")
        if any(w in blob for w in SERVICE_WORDS):
            service.append(label)
            themes.add("service")
        if any(w in blob for w in TECHNICAL_WORDS):
            technical.append(label)
            themes.add("technical")
        if any(w in blob for w in PROFESSIONAL_WORDS):
            professional.append(label)
            themes.add("professional experience")
        if len(activity.years) >= 2:
            themes.add("sustained commitment")

    for project in twin.projects:
        technical.append(f"Project: {project.name}")
        themes.add("technical")

    strengths = []
    if leadership:
        strengths.append("Documented leadership roles")
    if technical:
        strengths.append("Documented technical work outside the classroom")
    if service:
        strengths.append("Documented service involvement")
    if professional:
        strengths.append(f"Documented internship or fellowship experience ({len(professional)})")
    if twin.awards:
        strengths.append(f"{len(twin.awards)} award(s) reported")

    if not twin.activities:
        risks.append("No activities were provided, so non-academic impact cannot be assessed")
    if all(len(a.years) < 2 for a in twin.activities) and twin.activities:
        risks.append("No activity shows multi-year sustained commitment")
    if not technical:
        risks.append("No technical activity or project evidence for a STEM application")

    return ActivityAnalysis(
        strengths=strengths,
        themes=sorted(themes),
        leadership_evidence=[*leadership, *professional],
        technical_evidence=technical,
        service_evidence=service,
        risks=risks,
    )


# --------------------------------------------------------------------------- #
# STEM fit
# --------------------------------------------------------------------------- #
def _signal_strengths(twin: StudentDigitalTwin) -> dict[str, float]:
    courses = twin.academics.courses
    names = [c.name.lower() for c in courses]
    return {
        "cs": float(sum(1 for c in courses if c.subject == "CS")),
        "math": float(sum(1 for c in courses if c.subject == "Math"))
        + (1.0 if any("calculus" in n for n in names) else 0.0),
        "stats": float(sum(1 for n in names if "statistic" in n)),
        "science": float(sum(1 for c in courses if c.subject == "Science")),
        "physics": float(sum(1 for n in names if "physics" in n)),
    }


def stem_fit(context: dict[str, Any]) -> STEMFitList:
    twin: StudentDigitalTwin = context["twin"]
    limit: int = context.get("limit", 6)
    signals = _signal_strengths(twin)
    interests = {i.lower() for i in twin.stem_interests}

    scored: list[tuple[float, STEMFit]] = []
    for entry in STEM_CATALOG:
        score = sum(signals.get(k, 0.0) * w for k, w in entry["signals"].items())
        if entry["discipline"].lower() in interests:
            score += 2.0

        evidence = [
            f"{key} signal from verified transcript: {signals.get(key, 0.0):g}"
            for key in entry["signals"]
            if signals.get(key, 0.0) > 0
        ]
        if entry["discipline"].lower() in interests:
            evidence.append("Student listed this as a stated interest")

        concerns = [
            f"No {key} coursework on the verified transcript"
            for key in entry["signals"]
            if signals.get(key, 0.0) == 0
        ]

        if score >= 6:
            fit = "Excellent"
        elif score >= 4:
            fit = "Strong"
        elif score >= 2:
            fit = "Moderate"
        else:
            fit = "Weak"

        scored.append(
            (
                score,
                STEMFit(
                    discipline=entry["discipline"],
                    fit=fit,  # type: ignore[arg-type]
                    supporting_evidence=evidence,
                    concerns=concerns,
                    career_paths=list(entry["careers"]),
                ),
            )
        )

    scored.sort(key=lambda pair: (-pair[0], pair[1].discipline))
    return STEMFitList(disciplines=[fit for _, fit in scored[:limit]])


# --------------------------------------------------------------------------- #
# College discovery
# --------------------------------------------------------------------------- #
def college_discovery(context: dict[str, Any]) -> CollegeCandidateList:
    twin: StudentDigitalTwin = context["twin"]
    catalog: list[dict[str, Any]] = context["catalog"]
    fits: list[STEMFit] = context["stem_fit"]
    limit: int = context["limit"]

    preferred_states = {s.strip().lower() for s in twin.preferences.locations if s.strip()}
    ranked_disciplines = [f.discipline for f in fits]

    scored: list[tuple[float, CollegeCandidate]] = []
    for college in catalog:
        offered = set(college.get("majors", []))
        match = next((d for d in ranked_disciplines if d in offered), None)
        if match is None:
            continue

        score = float(len(ranked_disciplines) - ranked_disciplines.index(match))
        reasons = [f"Offers {match}, a top STEM fit for this profile"]

        if preferred_states and college.get("state", "").lower() in preferred_states:
            score += 3
            reasons.append(f"Matches location preference ({college['state']})")
        if (
            twin.preferences.public_private != "NoPreference"
            and college.get("control") == twin.preferences.public_private
        ):
            score += 1
            reasons.append(f"{college['control']} institution as preferred")
        if (
            twin.preferences.school_size != "NoPreference"
            and college.get("size") == twin.preferences.school_size
        ):
            score += 1
            reasons.append(f"{college['size']} campus as preferred")

        scored.append(
            (
                score,
                CollegeCandidate(
                    university=college["university"],
                    target_major=match,
                    state=college.get("state"),
                    reason="; ".join(reasons),
                ),
            )
        )

    scored.sort(key=lambda pair: (-pair[0], pair[1].university))
    return CollegeCandidateList(candidates=[c for _, c in scored[:limit]])


# --------------------------------------------------------------------------- #
# Admission assessment
# --------------------------------------------------------------------------- #
def _selectivity_band(research: CollegeResearchResult) -> tuple[int, bool]:
    """Map a *published* admit rate to a ladder offset.

    Returns (offset, known). When nothing is published the offset is neutral and
    ``known`` is False, which forces a conservative floor downstream rather than
    an optimistic guess.
    """
    rate = research.major_admit_rate
    if rate == NOT_PUBLISHED:
        rate = research.admit_rate
    if rate == NOT_PUBLISHED:
        return 2, False
    digits = "".join(ch for ch in rate if ch.isdigit() or ch == ".")
    try:
        value = float(digits)
    except ValueError:
        return 2, False
    if value >= 70:
        return 0, True
    if value >= 50:
        return 1, True
    if value >= 30:
        return 2, True
    if value >= 15:
        return 3, True
    return 4, True


RANGE_RE = re.compile(r"(\d{2,4})\s*[-\u2013\u2014]\s*(\d{2,4})")


def parse_published_range(value: str) -> tuple[int, int] | None:
    """Pull a middle-50% band out of a published range string."""
    if value == NOT_PUBLISHED or not value:
        return None
    if (match := RANGE_RE.search(value)) is None:
        return None
    low, high = sorted((int(match.group(1)), int(match.group(2))))
    return low, high


def score_against_range(score: int, published: str, label: str) -> tuple[int, str]:
    """Compare a test score to THIS college's published band.

    A single global threshold (e.g. "1450 is good") makes every selective school
    react identically and leaves the What-If Lab reporting "nothing changed" for
    any plausible score change. Scoring relative to each college's own published
    middle 50% is both more defensible and actually informative.
    """
    band = parse_published_range(published)
    if band is None:
        # No published band: fall back to a coarse absolute benchmark and say so.
        if label == "SAT":
            high_mark, low_mark = 1450, 1250
        else:
            high_mark, low_mark = 32, 24
        if score >= high_mark:
            return 1, f"{label} {score} (no published range; general benchmark)"
        if score < low_mark:
            return -1, f"{label} {score} (no published range; general benchmark)"
        return 0, f"{label} {score} (no published range for this college)"

    low, high = band
    if score > high:
        return 1, f"{label} {score} is above the published {low}-{high} middle 50%"
    if score >= low:
        return 0, f"{label} {score} sits inside the published {low}-{high} middle 50%"
    return -1, f"{label} {score} is below the published {low}-{high} middle 50%"


def admission_assessment(context: dict[str, Any]) -> AdmissionAssessment:
    """Offline reference implementation.

    The label comes from the same deterministic engine the real provider path
    uses (`services.classifier`), so the two never disagree. Only the prose
    differs.
    """
    from pathora.services.classifier import classify

    twin: StudentDigitalTwin = context["twin"]
    research: CollegeResearchResult = context["research"]
    profile: ProfileAnalysis = context["profile_analysis"]

    baseline = classify(twin, research, profile)

    strengths = [s for s in baseline.signals if "below" not in s and "No " not in s]
    risks = [s for s in baseline.signals if "below" in s or s.startswith("No ")]

    missing = list(research.missing_information)
    if research.admission_structure == NOT_PUBLISHED:
        missing.append("Admission structure (direct-to-major vs. university-wide) not retrieved")
    if not baseline.major_level_known and research.admit_rate != NOT_PUBLISHED:
        missing.append(
            f"Major-specific admit rate for {research.target_major} not published; "
            "only a university-wide rate was retrieved"
        )
    if not twin.testing.sat_total and not twin.testing.act_composite:
        missing.append("No SAT or ACT score provided by the student")

    rationale = (
        f"Classified {baseline.classification} for {research.target_major} at "
        f"{research.university}. " + " ".join(f"{s}." for s in baseline.signals)
    )
    if baseline.caps_applied:
        rationale += " " + " ".join(f"Adjustment: {c}." for c in baseline.caps_applied)

    return AdmissionAssessment(
        university=research.university,
        recommended_major=research.target_major,
        classification=baseline.classification,
        confidence=baseline.confidence,
        strengths=strengths,
        risks=risks,
        evidence_ids=[e.evidence_id for e in research.evidence],
        missing_information=sorted(set(missing)),
        rationale_summary=rationale,
    )


# --------------------------------------------------------------------------- #
# Critic
# --------------------------------------------------------------------------- #
def critic(context: dict[str, Any]) -> CriticResult:
    assessments: dict[str, AdmissionAssessment] = context["assessments"]
    research: dict[str, CollegeResearchResult] = context["research"]
    abstentions: dict[str, Any] = context.get("abstentions", {})
    loop: int = context.get("critic_loop_count", 0)
    max_loops: int = context.get("max_critic_loops", 2)
    stale_after: int = context.get("stale_after_days", 365)

    issues: list[str] = []
    to_research: list[str] = []
    missing: list[str] = []

    for university, assessment in assessments.items():
        result = research.get(university)
        if result is None:
            issues.append(f"{university}: assessment produced without any research result")
            to_research.append(university)
            continue

        if not assessment.evidence_ids:
            issues.append(f"{university}: no evidence records support this classification")
            to_research.append(university)

        known_ids = {e.evidence_id for e in result.evidence}
        if unknown := [e for e in assessment.evidence_ids if e not in known_ids]:
            issues.append(f"{university}: cites evidence not present in research ({unknown})")
            to_research.append(university)

        if not any(e.source_type == "official_admissions" for e in result.evidence):
            issues.append(f"{university}: no official admissions source retrieved")
            to_research.append(university)

        if stale := [e.evidence_id for e in result.evidence if e.age_days() > stale_after]:
            issues.append(f"{university}: evidence older than {stale_after} days ({stale})")

        if (
            result.major_admit_rate == NOT_PUBLISHED
            and result.admit_rate != NOT_PUBLISHED
            and "university-wide" not in assessment.rationale_summary.lower()
            and not any("major-specific" in m.lower() for m in assessment.missing_information)
        ):
            issues.append(
                f"{university}: university-wide admit rate risks being read as major-specific"
            )
            to_research.append(university)

        if assessment.confidence == "High" and assessment.missing_information:
            issues.append(f"{university}: High confidence claimed despite open information gaps")

        if result.research_error:
            issues.append(f"{university}: research error — {result.research_error}")
            to_research.append(university)

        missing.extend(assessment.missing_information)

    # The gate already refused these; more retrieval is the only thing that can
    # change the outcome, and if retries are spent it is a human decision.
    for university, abstention in abstentions.items():
        issues.append(
            f"{university}: evidence gate refused assessment "
            f"({', '.join(getattr(abstention, 'failed_checks', []) or ['unknown'])})"
        )
        to_research.append(university)
        missing.extend(getattr(abstention, "missing_information", []) or [])

    to_research = sorted(set(to_research))

    if not issues:
        decision = "approve"
    elif to_research and loop < max_loops:
        decision = "research_more"
    elif to_research:
        decision = "human_review"
    else:
        # Issues exist but more retrieval will not fix them; approve with the
        # gaps surfaced rather than looping forever.
        decision = "approve"

    return CriticResult(
        decision=decision,  # type: ignore[arg-type]
        issues=issues,
        colleges_to_research=to_research if decision != "approve" else [],
        missing_information=sorted(set(missing)),
    )


# --------------------------------------------------------------------------- #
# Next best action + roadmap
# --------------------------------------------------------------------------- #
def next_best_actions(context: dict[str, Any]) -> NextActionList:
    twin: StudentDigitalTwin = context["twin"]
    assessments: dict[str, AdmissionAssessment] = context["assessments"]
    activity: ActivityAnalysis = context["activity_analysis"]

    actions: list[NextAction] = []
    reaches = [
        a.university for a in assessments.values() if a.classification in {"Reach", "High Reach"}
    ]
    safeties = [
        a.university for a in assessments.values() if a.classification in {"Safety", "Likely"}
    ]

    if not twin.testing.sat_total and not twin.testing.act_composite:
        actions.append(
            NextAction(
                title="Register for the next SAT or ACT date",
                priority="High",
                reason="No test score is on file; several list schools still consider scores.",
                related_colleges=sorted(assessments),
            )
        )

    gap_universities = sorted({u for u, a in assessments.items() if a.missing_information})
    if gap_universities:
        actions.append(
            NextAction(
                title="Confirm major-level admission rules directly with admissions offices",
                priority="High",
                reason="Retrieved sources did not publish the major-specific admission structure.",
                related_colleges=gap_universities,
            )
        )

    if not safeties:
        actions.append(
            NextAction(
                title="Add at least two likely/safety options you would be happy to attend",
                priority="High",
                reason="The current list has no Safety or Likely classification.",
                related_colleges=[],
            )
        )

    if any("technical" in r.lower() for r in activity.risks):
        actions.append(
            NextAction(
                title="Start or finish one substantial technical project",
                priority="Medium",
                reason="No technical project evidence was found in the verified profile.",
                related_colleges=reaches,
            )
        )

    if twin.academics.senior_year_courses:
        actions.append(
            NextAction(
                title="Protect senior-year grades in the most advanced courses",
                priority="Medium",
                reason="Senior coursework is the most recent academic signal reviewers will see.",
                related_colleges=reaches or sorted(assessments),
            )
        )

    actions.append(
        NextAction(
            title="Draft the STEM-specific essay paragraph for each application",
            priority="Low",
            reason=(
                "Programs asking for a major-specific statement need concrete, verifiable detail."
            ),
            related_colleges=sorted(assessments),
        )
    )

    order = {"High": 0, "Medium": 1, "Low": 2}
    actions.sort(key=lambda a: order[a.priority])
    return NextActionList(actions=actions)


def roadmap(context: dict[str, Any]) -> Roadmap:
    actions: list[NextAction] = context["actions"]
    sections: list[str] = context.get("sections", ["today", "this_week", "this_month", "upcoming"])

    def to_item(action: NextAction) -> RoadmapItem:
        return RoadmapItem(
            title=action.title,
            detail=action.reason,
            related_colleges=action.related_colleges,
        )

    high = [to_item(a) for a in actions if a.priority == "High"]
    medium = [to_item(a) for a in actions if a.priority == "Medium"]
    low = [to_item(a) for a in actions if a.priority == "Low"]

    result = Roadmap(sections_regenerated=sections)
    if "today" in sections:
        result.today = high[:1]
    if "this_week" in sections:
        result.this_week = high[1:]
    if "this_month" in sections:
        result.this_month = medium
    if "upcoming" in sections:
        result.upcoming = low
    return result
