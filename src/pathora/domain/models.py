"""Pydantic v2 domain models.

These are the contracts between graph nodes. Every agent returns one of these;
no node ever hands another node a raw conversation transcript.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

Fit = Literal["Excellent", "Strong", "Moderate", "Weak"]
Classification = Literal["Safety", "Likely", "Target", "Target-Reach", "Reach", "High Reach"]
Confidence = Literal["Low", "Moderate", "High"]
Impact = Literal["High", "Medium", "Low"]
Priority = Literal["High", "Medium", "Low"]
SourceType = Literal[
    "official_admissions",
    "official_stem_program",
    "common_data_set",
    "institutional_research",
    "other",
]

# Ranked by Section 17 grounding priority. Lower index == more authoritative.
SOURCE_PRIORITY: dict[str, int] = {
    "official_admissions": 0,
    "official_stem_program": 1,
    "common_data_set": 2,
    "institutional_research": 3,
    "other": 4,
}

NOT_PUBLISHED = "Not officially published"


def _utcnow() -> datetime:
    return datetime.now(UTC)


class Base(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)


# --------------------------------------------------------------------------- #
# Academics
# --------------------------------------------------------------------------- #
class Course(Base):
    name: str
    grade: str
    credits: float = 1.0
    academic_year: str | None = None
    level: Literal["Regular", "Honors", "AP", "IB", "DualEnrollment", "Unknown"] = "Regular"
    subject: Literal["Math", "Science", "CS", "English", "Social", "Language", "Other"] = "Other"
    pass_fail: bool = False
    confidence: float = 1.0

    @field_validator("credits")
    @classmethod
    def _credits_positive(cls, v: float) -> float:
        if v < 0:
            raise ValueError("credits must be >= 0")
        return v


class ExtractedAcademics(Base):
    courses: list[Course] = Field(default_factory=list)
    school_name: str | None = None
    graduation_year: int | None = None
    reported_gpa: float | None = None
    class_rank: str | None = None
    extraction_confidence: float = 0.0
    uncertain_fields: list[str] = Field(default_factory=list)
    raw_text_sha256: str | None = None

    def needs_human_verification(self, threshold: float = 0.75) -> bool:
        """Whether extraction is too uncertain to proceed without a human.

        ``threshold`` comes from EXTRACTION_CONFIDENCE_THRESHOLD. It is a method
        rather than a property precisely so the caller must supply the
        configured value instead of silently inheriting a hardcoded one.
        """
        return bool(self.uncertain_fields) or self.extraction_confidence < threshold


class GPAResult(Base):
    gpa: float
    graded_credits: float
    method: str = "standard_unweighted_4_scale"
    excluded_courses: list[str] = Field(default_factory=list)
    quality_points: float = 0.0


class TestingProfile(Base):
    sat_total: int | None = None
    sat_math: int | None = None
    sat_verbal: int | None = None
    act_composite: int | None = None
    ap_scores: dict[str, int] = Field(default_factory=dict)


class AcademicProfile(Base):
    gpa: GPAResult
    courses: list[Course] = Field(default_factory=list)
    school_name: str | None = None
    graduation_year: int | None = None
    class_rank: str | None = None
    ap_courses: list[str] = Field(default_factory=list)
    ib_courses: list[str] = Field(default_factory=list)
    honors_courses: list[str] = Field(default_factory=list)
    math_progression: list[str] = Field(default_factory=list)
    science_progression: list[str] = Field(default_factory=list)
    cs_progression: list[str] = Field(default_factory=list)
    senior_year_courses: list[str] = Field(default_factory=list)


class Activity(Base):
    name: str
    role: str | None = None
    years: list[str] = Field(default_factory=list)
    hours_per_week: float | None = None
    description: str | None = None


class Project(Base):
    name: str
    description: str | None = None
    technologies: list[str] = Field(default_factory=list)


class Award(Base):
    name: str
    level: str | None = None
    year: str | None = None


class StudentPreferences(Base):
    locations: list[str] = Field(default_factory=list)
    max_distance_miles: int | None = None
    setting: Literal["Urban", "Suburban", "Rural", "NoPreference"] = "NoPreference"
    school_size: Literal["Small", "Medium", "Large", "NoPreference"] = "NoPreference"
    public_private: Literal["Public", "Private", "NoPreference"] = "NoPreference"
    cost_sensitivity: Literal["Low", "Medium", "High"] = "Medium"
    must_have_majors: list[str] = Field(default_factory=list)


class StudentDigitalTwin(Base):
    """Normalized authoritative representation of the verified student profile.

    Not an agent. Downstream nodes read this instead of re-reading raw input.
    """

    student_id: str
    academics: AcademicProfile
    testing: TestingProfile = Field(default_factory=TestingProfile)
    activities: list[Activity] = Field(default_factory=list)
    projects: list[Project] = Field(default_factory=list)
    awards: list[Award] = Field(default_factory=list)
    academic_strengths: list[str] = Field(default_factory=list)
    academic_risks: list[str] = Field(default_factory=list)
    stem_interests: list[str] = Field(default_factory=list)
    career_interests: list[str] = Field(default_factory=list)
    preferences: StudentPreferences = Field(default_factory=StudentPreferences)
    updated_at: datetime = Field(default_factory=_utcnow)


# --------------------------------------------------------------------------- #
# Agent outputs
# --------------------------------------------------------------------------- #
class ProfileAnalysis(Base):
    course_rigor: Fit
    grade_trend: Literal["Improving", "Stable", "Declining", "Mixed", "Unknown"]
    math_preparation: str
    science_preparation: str
    cs_preparation: str
    academic_strengths: list[str] = Field(default_factory=list)
    academic_risks: list[str] = Field(default_factory=list)


class ActivityAnalysis(Base):
    strengths: list[str] = Field(default_factory=list)
    themes: list[str] = Field(default_factory=list)
    leadership_evidence: list[str] = Field(default_factory=list)
    technical_evidence: list[str] = Field(default_factory=list)
    service_evidence: list[str] = Field(default_factory=list)
    risks: list[str] = Field(default_factory=list)


class STEMFit(Base):
    discipline: str
    fit: Fit
    supporting_evidence: list[str] = Field(default_factory=list)
    concerns: list[str] = Field(default_factory=list)
    career_paths: list[str] = Field(default_factory=list)


class STEMFitList(Base):
    disciplines: list[STEMFit] = Field(default_factory=list)


class CollegeCandidate(Base):
    university: str
    target_major: str
    reason: str = ""
    state: str | None = None


class CollegeCandidateList(Base):
    candidates: list[CollegeCandidate] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Evidence
# --------------------------------------------------------------------------- #
class EvidenceRecord(Base):
    """Stored independently of any generated prose (Section 25)."""

    evidence_id: str
    university: str
    source_url: str
    source_type: SourceType = "other"
    title: str = ""
    snippet: str = ""
    published_at: datetime | None = None
    retrieved_at: datetime = Field(default_factory=_utcnow)

    def age_days(self, now: datetime | None = None) -> float:
        now = now or _utcnow()
        ref = self.published_at or self.retrieved_at
        if ref.tzinfo is None:
            ref = ref.replace(tzinfo=UTC)
        return (now - ref).total_seconds() / 86400.0


class CollegeResearchResult(Base):
    university: str
    target_major: str
    admit_rate: str = NOT_PUBLISHED
    major_admit_rate: str = NOT_PUBLISHED
    sat_range: str = NOT_PUBLISHED
    act_range: str = NOT_PUBLISHED
    test_policy: str = NOT_PUBLISHED
    admission_structure: str = NOT_PUBLISHED
    deadlines: str = NOT_PUBLISHED
    program_notes: list[str] = Field(default_factory=list)
    transfer_restrictions: str = NOT_PUBLISHED
    evidence: list[EvidenceRecord] = Field(default_factory=list)
    #: field name -> evidence_id the value was read from. Any fact without an
    #: entry here was not traced to a retrieved document.
    fact_sources: dict[str, str] = Field(default_factory=dict)
    missing_information: list[str] = Field(default_factory=list)
    research_error: str | None = None
    retrieved_at: datetime = Field(default_factory=_utcnow)


class EvidencePassport(Base):
    university: str
    quality: Literal["HIGH", "MEDIUM", "LOW"]
    has_official_admissions: bool = False
    has_official_stem: bool = False
    has_common_data_set: bool = False
    has_verified_student_profile: bool = False
    stale_evidence_ids: list[str] = Field(default_factory=list)
    missing: list[str] = Field(default_factory=list)
    evidence_ids: list[str] = Field(default_factory=list)
    retrieved_at: datetime = Field(default_factory=_utcnow)


# --------------------------------------------------------------------------- #
# Assessment
# --------------------------------------------------------------------------- #
class AdmissionAssessment(Base):
    university: str
    recommended_major: str
    classification: Classification
    confidence: Confidence
    strengths: list[str] = Field(default_factory=list)
    risks: list[str] = Field(default_factory=list)
    evidence_ids: list[str] = Field(default_factory=list)
    missing_information: list[str] = Field(default_factory=list)
    rationale_summary: str = ""


class GateCheck(Base):
    name: str
    passed: bool
    detail: str = ""


class GateResult(Base):
    """Outcome of the pre-generation evidence gate."""

    university: str
    passed: bool
    checks: list[GateCheck] = Field(default_factory=list)
    failed_checks: list[str] = Field(default_factory=list)
    reason: str = ""
    evidence_ids: list[str] = Field(default_factory=list)


class AssessmentAbstention(Base):
    """Recorded instead of an AdmissionAssessment when the gate refuses.

    Deliberately not a classification with Low confidence: there is no honest
    Safety/Target/Reach label to give, so none is produced.
    """

    university: str
    recommended_major: str
    reason: str
    failed_checks: list[str] = Field(default_factory=list)
    what_would_help: list[str] = Field(default_factory=list)
    evidence_ids: list[str] = Field(default_factory=list)
    missing_information: list[str] = Field(default_factory=list)


class CriticResult(Base):
    decision: Literal["approve", "research_more", "human_review"]
    issues: list[str] = Field(default_factory=list)
    colleges_to_research: list[str] = Field(default_factory=list)
    missing_information: list[str] = Field(default_factory=list)


class GapFactor(Base):
    factor: str
    impact: Impact
    note: str = ""
    controllable: bool = True


class GapAnalysis(Base):
    university: str
    factors: list[GapFactor] = Field(default_factory=list)
    primary_constraint: str = ""
    already_strong: list[str] = Field(default_factory=list)
    still_controllable: list[str] = Field(default_factory=list)


class NextAction(Base):
    title: str
    priority: Priority
    reason: str
    related_colleges: list[str] = Field(default_factory=list)


class NextActionList(Base):
    actions: list[NextAction] = Field(default_factory=list)


class RoadmapItem(Base):
    title: str
    detail: str = ""
    related_colleges: list[str] = Field(default_factory=list)


class Roadmap(Base):
    today: list[RoadmapItem] = Field(default_factory=list)
    this_week: list[RoadmapItem] = Field(default_factory=list)
    this_month: list[RoadmapItem] = Field(default_factory=list)
    upcoming: list[RoadmapItem] = Field(default_factory=list)
    generated_at: datetime = Field(default_factory=_utcnow)
    sections_regenerated: list[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# HITL + What-If
# --------------------------------------------------------------------------- #
HumanChoice = Literal["confirm", "edit", "continue_with_uncertainty", "cancel"]


def _default_choices() -> list[HumanChoice]:
    return ["confirm", "edit", "continue_with_uncertainty", "cancel"]


class HumanActionRequest(Base):
    kind: Literal[
        "verify_transcript",
        "resolve_conflict",
        "missing_information",
        "research_exhausted",
        "critic_human_review",
    ]
    message: str
    options: list[HumanChoice] = Field(default_factory=_default_choices)
    payload: dict = Field(default_factory=dict)


class HumanResponse(Base):
    choice: HumanChoice
    edits: dict = Field(default_factory=dict)
    note: str | None = None


class WhatIfScenario(Base):
    sat_total: int | None = None
    act_composite: int | None = None
    senior_grades: dict[str, str] = Field(default_factory=dict)
    major: str | None = None
    added_activity: Activity | None = None
    added_project: Project | None = None
    preferences: StudentPreferences | None = None


class WhatIfChange(Base):
    university: str
    before: Classification
    after: Classification
    changed: bool
    reason: str = ""


class WhatIfResult(Base):
    scenario: WhatIfScenario
    changes: list[WhatIfChange] = Field(default_factory=list)
    nodes_rerun: list[str] = Field(default_factory=list)
    nodes_skipped: list[str] = Field(default_factory=list)
    summary: str = ""
