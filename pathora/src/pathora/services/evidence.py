"""Evidence Passport (Section 25) and Admission Gap Analyzer (Section 26).

Both are deterministic. Evidence records are stored independently of generated
prose so a passport can be rebuilt and audited without re-running any agent.
The gap analyzer emits only High/Medium/Low impact — never invented weights.
"""

from __future__ import annotations

from pathora.config import get_settings
from pathora.domain.models import (
    NOT_PUBLISHED,
    AdmissionAssessment,
    CollegeResearchResult,
    EvidencePassport,
    GapAnalysis,
    GapFactor,
    Impact,
    StudentDigitalTwin,
)


def build_passport(
    research: CollegeResearchResult,
    assessment: AdmissionAssessment,
    *,
    profile_verified: bool,
    stale_after_days: int | None = None,
) -> EvidencePassport:
    stale_after = stale_after_days or get_settings().evidence_stale_after_days
    types = {e.source_type for e in research.evidence}

    has_admissions = "official_admissions" in types
    has_stem = "official_stem_program" in types
    has_cds = "common_data_set" in types
    annual = {"institutional_research", "common_data_set"}
    annual_after = get_settings().annual_survey_stale_after_days
    stale = [
        e.evidence_id
        for e in research.evidence
        if e.age_days() > (annual_after if e.source_type in annual else stale_after)
    ]

    signals = sum([has_admissions, has_stem, has_cds, profile_verified])
    if has_admissions and signals >= 3 and not assessment.missing_information:
        quality = "HIGH"
    elif signals >= 2:
        quality = "MEDIUM"
    else:
        quality = "LOW"

    missing = list(assessment.missing_information)
    if not has_admissions:
        missing.append("No official admissions source retrieved")
    if not has_cds:
        missing.append("No Common Data Set retrieved")
    if stale:
        missing.append(f"{len(stale)} evidence record(s) older than {stale_after} days")

    return EvidencePassport(
        university=research.university,
        quality=quality,  # type: ignore[arg-type]
        has_official_admissions=has_admissions,
        has_official_stem=has_stem,
        has_common_data_set=has_cds,
        has_verified_student_profile=profile_verified,
        stale_evidence_ids=stale,
        missing=sorted(set(missing)),
        evidence_ids=[e.evidence_id for e in research.evidence],
        retrieved_at=research.retrieved_at,
    )


def analyze_gap(
    twin: StudentDigitalTwin,
    research: CollegeResearchResult,
    assessment: AdmissionAssessment,
) -> GapAnalysis:
    selective = assessment.classification in {"Reach", "High Reach", "Target-Reach"}
    gpa = twin.academics.gpa.gpa
    factors: list[GapFactor] = []

    factors.append(
        GapFactor(
            factor="Academic record",
            impact="High" if (selective or gpa < 3.5) else "Medium",
            note=f"Unweighted GPA {gpa} across {twin.academics.gpa.graded_credits} graded credits.",
            controllable=bool(twin.academics.senior_year_courses),
        )
    )

    factors.append(
        GapFactor(
            factor="Class rank",
            impact="High" if twin.academics.class_rank is None and selective else "Medium",
            note=(
                f"Reported rank: {twin.academics.class_rank}"
                if twin.academics.class_rank
                else "Class rank was not on the transcript, so this cannot be assessed."
            ),
            controllable=False,
        )
    )

    test_impact: Impact
    test_policy = research.test_policy
    if test_policy != NOT_PUBLISHED and "blind" in test_policy.lower():
        test_impact = "Low"
        test_note = "This institution publishes a test-blind policy."
    elif not twin.testing.sat_total and not twin.testing.act_composite:
        test_impact = "High" if selective else "Medium"
        test_note = "No score on file and the published policy still considers scores."
    else:
        test_impact = "Medium"
        test_note = (
            f"Reported SAT {twin.testing.sat_total or 'n/a'} / ACT "
            f"{twin.testing.act_composite or 'n/a'} against published range "
            f"{research.sat_range}."
        )
    factors.append(
        GapFactor(factor="SAT / ACT", impact=test_impact, note=test_note, controllable=True)
    )

    advanced = len(twin.academics.ap_courses) + len(twin.academics.ib_courses)
    factors.append(
        GapFactor(
            factor="Course rigor",
            impact="Medium" if advanced < 4 else "Low",
            note=f"{advanced} AP/IB course(s) on the verified transcript.",
            controllable=bool(twin.academics.senior_year_courses),
        )
    )

    factors.append(
        GapFactor(
            factor="Activities",
            impact="Medium" if len(twin.activities) < 2 else "Low",
            note=(
                f"{len(twin.activities)} activity/activities and "
                f"{len(twin.projects)} project(s) on file."
            ),
            controllable=True,
        )
    )

    if (
        research.admission_structure != NOT_PUBLISHED
        and "capped" in research.admission_structure.lower()
    ):
        factors.append(
            GapFactor(
                factor="Major capacity",
                impact="High",
                note=research.admission_structure,
                controllable=False,
            )
        )

    high = [f for f in factors if f.impact == "High"]
    primary = high[0].factor if high else max(factors, key=lambda f: f.impact == "Medium").factor

    return GapAnalysis(
        university=research.university,
        factors=factors,
        primary_constraint=(
            f"{primary} is the dominant constraint for {research.target_major} at "
            f"{research.university}."
        ),
        already_strong=[f.factor for f in factors if f.impact == "Low"],
        still_controllable=[f.factor for f in factors if f.controllable and f.impact != "Low"],
    )
