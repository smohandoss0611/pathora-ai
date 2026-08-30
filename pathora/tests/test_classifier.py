"""Deterministic classification.

Regression origin: the Admission Agent classified Texas Tech as Target for a
student scoring above its published 75th percentile at a 72.7% admit rate. A
model reading prose and choosing a label produces results that do not follow
from the evidence it just cited.
"""

from __future__ import annotations

import pytest

from pathora.domain.models import (
    NOT_PUBLISHED,
    CollegeResearchResult,
    Course,
    EvidenceRecord,
    ExtractedAcademics,
    ProfileAnalysis,
)
from pathora.domain.models import TestingProfile as _Testing  # noqa: N813
from pathora.services.classifier import classify, parse_percent, parse_range
from pathora.services.twin import build_digital_twin


def student(*, sat: int | None = 1330, gpa: float = 3.62, rigor: str = "Excellent"):
    courses = [
        Course(name=f"AP Course {i}", grade="90", credits=1.0, level="AP", subject="Math")
        for i in range(20)
    ]
    twin = build_digital_twin(
        student_id="j",
        verified_academics=ExtractedAcademics(courses=courses),
        testing=_Testing(sat_total=sat),
    )
    twin.academics.gpa.gpa = gpa
    profile = ProfileAnalysis(
        course_rigor=rigor,  # type: ignore[arg-type]
        grade_trend="Mixed",
        math_preparation="",
        science_preparation="",
        cs_preparation="",
    )
    return twin, profile


def college(
    name="U", admit=NOT_PUBLISHED, sat=NOT_PUBLISHED, major=NOT_PUBLISHED, structure=NOT_PUBLISHED
):
    return CollegeResearchResult(
        university=name,
        target_major="Industrial Engineering",
        admit_rate=admit,
        major_admit_rate=major,
        sat_range=sat,
        admission_structure=structure,
        evidence=[
            EvidenceRecord(
                evidence_id="e1",
                university=name,
                source_url="https://x.example.edu",
                source_type="institutional_research",
            )
        ],
    )


class TestTheReportedRegression:
    def test_texas_tech_is_likely_not_target(self):
        """1330 SAT, above the published 1080-1280 band, 72.7% admit rate."""
        twin, profile = student()
        result = classify(twin, college("Texas Tech", "72.7%", "1080-1280"), profile)
        assert result.classification == "Likely"

    def test_it_is_not_safety_without_major_level_data(self):
        """Moderately selective + no major rate = Likely, not Safety."""
        twin, profile = student()
        result = classify(twin, college("Texas Tech", "72.7%", "1080-1280"), profile)
        assert result.classification != "Safety"
        assert any("Safety is not claimed" in c for c in result.caps_applied)


class TestCeilingsAndFloors:
    def test_very_selective_stays_a_reach_for_a_strong_student(self):
        twin, profile = student(sat=1560, gpa=4.0)
        result = classify(twin, college("Selective", "4%", "1460-1560"), profile)
        assert result.classification in {"Reach", "High Reach"}

    def test_below_the_published_band_is_never_likely(self):
        twin, profile = student(sat=1330)
        result = classify(twin, college("SMU", "63.3%", "1340-1500"), profile)
        assert result.classification == "Target"

    def test_broadly_accessible_school_can_be_safety(self):
        """Above the 75th at a 94% admit rate: refusing Safety is false caution."""
        twin, profile = student()
        result = classify(twin, college("UTRGV", "94.2%", "880-1090"), profile)
        assert result.classification == "Safety"

    def test_missing_admit_rate_caps_at_target(self):
        twin, profile = student()
        result = classify(twin, college("Unknown U"), profile)
        assert result.classification == "Target"
        assert not result.selectivity_known


class TestDowngradesRequireEvidence:
    def test_limited_access_major_downgrades_one_step(self):
        twin, profile = student()
        base = classify(twin, college("U", "36%", "1310-1460"), profile)
        restricted = classify(
            twin,
            college("U", "36%", "1310-1460", structure="Limited-access major: portfolio review"),
            profile,
        )
        from pathora.services.classifier import LADDER

        assert LADDER.index(restricted.classification) == LADDER.index(base.classification) + 1
        assert any("limited-access" in c for c in restricted.caps_applied)

    def test_missing_data_alone_does_not_downgrade(self):
        """Absence of information is not evidence of selectivity."""
        twin, profile = student()
        with_structure = classify(
            twin, college("U", "72.7%", "1080-1280", structure="Direct admission to major"), profile
        )
        without = classify(twin, college("U", "72.7%", "1080-1280"), profile)
        assert with_structure.classification == without.classification


class TestConfidence:
    def test_major_level_data_permits_high_confidence(self):
        twin, profile = student()
        research = college("U", "60%", "1100-1300", major="25% (Industrial Engineering)")
        research.evidence.append(
            EvidenceRecord(
                evidence_id="e2",
                university="U",
                source_url="https://y.example.edu",
                source_type="common_data_set",
            )
        )
        assert classify(twin, research, profile).confidence == "High"

    def test_no_published_rate_gives_low_confidence(self):
        twin, profile = student()
        assert classify(twin, college("U"), profile).confidence == "Low"


class TestParsing:
    @pytest.mark.parametrize(
        ("text", "expected"),
        [("1080-1280", (1080, 1280)), ("1280\u20131080", (1080, 1280)), (NOT_PUBLISHED, None)],
    )
    def test_range_parsing(self, text, expected):
        assert parse_range(text) == expected

    @pytest.mark.parametrize(
        ("text", "expected"),
        [("72.7%", 72.7), ("62.8% (university-wide)", 62.8), (NOT_PUBLISHED, None)],
    )
    def test_percent_parsing(self, text, expected):
        assert parse_percent(text) == expected


class TestDeterminism:
    def test_same_inputs_give_the_same_label_every_time(self):
        twin, profile = student()
        research = college("U", "72.7%", "1080-1280")
        labels = {classify(twin, research, profile).classification for _ in range(20)}
        assert len(labels) == 1

    def test_offline_and_online_paths_agree(self):
        """The fake provider must not disagree with the engine."""
        from pathora.llm.heuristics import admission_assessment

        twin, profile = student()
        research = college("Texas Tech", "72.7%", "1080-1280")
        baseline = classify(twin, research, profile)
        offline = admission_assessment(
            {"twin": twin, "research": research, "profile_analysis": profile}
        )
        assert offline.classification == baseline.classification
        assert offline.confidence == baseline.confidence


class TestCollegeLevelAdmission:
    """Large publics often admit to a COLLEGE, not a major.

    Where that is the published structure, the university-wide admit rate does
    not govern the major at all. Texas A&M's Entry to a Major (ETAM) is the
    case that prompted this: 62.8% overall says little about placing into
    Industrial Engineering.
    """

    ETAM = (
        "Applicants are admitted to the College of Engineering, then place into "
        "a major through Entry to a Major (ETAM) after the first year."
    )

    def test_etam_structure_downgrades(self):
        twin, profile = student()
        without = classify(twin, college("Texas A&M", "62.8%", "1170-1410"), profile)
        with_etam = classify(
            twin, college("Texas A&M", "62.8%", "1170-1410", structure=self.ETAM), profile
        )
        from pathora.services.classifier import LADDER

        assert without.classification == "Likely"
        assert with_etam.classification == "Target"
        assert LADDER.index(with_etam.classification) > LADDER.index(without.classification)

    def test_reason_is_recorded(self):
        twin, profile = student()
        result = classify(
            twin, college("Texas A&M", "62.8%", "1170-1410", structure=self.ETAM), profile
        )
        assert any("Entry to a Major" in c for c in result.caps_applied)

    def test_direct_admission_is_not_downgraded(self):
        """Where the university rate DOES govern the major, leave it alone."""
        twin, profile = student()
        direct = classify(
            twin,
            college("U", "72.7%", "1080-1280", structure="Direct admission to major"),
            profile,
        )
        plain = classify(twin, college("U", "72.7%", "1080-1280"), profile)
        assert direct.classification == plain.classification
