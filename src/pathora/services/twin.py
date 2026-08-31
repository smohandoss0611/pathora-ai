"""Student Digital Twin construction (Section 13).

The twin is not an agent. It is the normalized, authoritative view of the
verified profile that every downstream node reads instead of re-parsing input.
"""

from __future__ import annotations

from datetime import UTC, datetime

from pathora.domain.models import (
    AcademicProfile,
    Activity,
    Award,
    ExtractedAcademics,
    Project,
    StudentDigitalTwin,
    StudentPreferences,
    TestingProfile,
)
from pathora.services.gpa import calculate_unweighted_gpa
from pathora.services.transcript import progression, senior_year_courses


def build_academic_profile(verified: ExtractedAcademics) -> AcademicProfile:
    courses = verified.courses
    return AcademicProfile(
        gpa=calculate_unweighted_gpa(courses),
        courses=courses,
        school_name=verified.school_name,
        graduation_year=verified.graduation_year,
        class_rank=verified.class_rank,
        ap_courses=[c.name for c in courses if c.level == "AP"],
        ib_courses=[c.name for c in courses if c.level == "IB"],
        honors_courses=[c.name for c in courses if c.level == "Honors"],
        math_progression=progression(courses, "Math"),
        science_progression=progression(courses, "Science"),
        cs_progression=progression(courses, "CS"),
        senior_year_courses=senior_year_courses(courses),
    )


def build_digital_twin(
    *,
    student_id: str,
    verified_academics: ExtractedAcademics,
    testing: TestingProfile | None = None,
    activities: list[Activity] | None = None,
    projects: list[Project] | None = None,
    awards: list[Award] | None = None,
    stem_interests: list[str] | None = None,
    career_interests: list[str] | None = None,
    preferences: StudentPreferences | None = None,
    academic_strengths: list[str] | None = None,
    academic_risks: list[str] | None = None,
) -> StudentDigitalTwin:
    return StudentDigitalTwin(
        student_id=student_id,
        academics=build_academic_profile(verified_academics),
        testing=testing or TestingProfile(),
        activities=activities or [],
        projects=projects or [],
        awards=awards or [],
        academic_strengths=academic_strengths or [],
        academic_risks=academic_risks or [],
        stem_interests=stem_interests or [],
        career_interests=career_interests or [],
        preferences=preferences or StudentPreferences(),
        updated_at=datetime.now(UTC),
    )


def apply_edits(twin: StudentDigitalTwin, edits: dict) -> StudentDigitalTwin:
    """Apply a shallow, validated patch to the twin (used by HITL and What-If)."""
    payload = twin.model_dump()
    for key, value in edits.items():
        if key not in payload:
            raise KeyError(f"unknown digital twin field: {key}")
        if isinstance(payload[key], dict) and isinstance(value, dict):
            payload[key] = payload[key] | value
        else:
            payload[key] = value
    payload["updated_at"] = datetime.now(UTC)
    return StudentDigitalTwin.model_validate(payload)
