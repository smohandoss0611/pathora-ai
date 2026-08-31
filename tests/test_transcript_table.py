"""Tabular transcript parsing, against a real district PDF.

The original line-oriented parser was written and tested against a synthetic
fixture whose layout I chose. It extracted ZERO courses from the first real
transcript it saw, because PyMuPDF flattens tables to one cell per line. These
tests exist so that cannot regress.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from pathora.services.gpa import calculate_unweighted_gpa
from pathora.services.transcript import parse_transcript_pdf
from pathora.services.transcript_table import parse_row, rows_from_pdf
from pathora.services.twin import build_digital_twin

FIXTURE = Path(__file__).parent / "fixtures/real_tabular_transcript.pdf"

pytest.importorskip("pymupdf")


@pytest.fixture(scope="module")
def pdf_bytes() -> bytes:
    return FIXTURE.read_bytes()


@pytest.fixture(scope="module")
def extracted(pdf_bytes):
    return parse_transcript_pdf(pdf_bytes)


class TestRowReconstruction:
    def test_words_are_grouped_into_table_rows(self, pdf_bytes):
        rows = rows_from_pdf(pdf_bytes)
        assert any("ALG 1" in row and "1.0000" in row for row in rows)

    def test_plain_text_extraction_would_have_failed(self, pdf_bytes):
        """Documents why the coordinate reader exists."""
        from pathora.services.transcript import (
            extract_text_from_pdf,
            parse_transcript_text,
        )

        assert parse_transcript_text(extract_text_from_pdf(pdf_bytes)).courses == []


class TestRowParsing:
    def test_semester_and_final_grades(self):
        course = parse_row("03100500 - 1 ALG 1 89 95 92 1.0000", "2022-2023")
        assert course is not None
        assert course.name == "ALG 1"
        assert course.grade == "92"  # the final, not a semester
        assert course.credits == 1.0

    def test_course_number_is_not_eaten_as_a_grade(self):
        """Without a grade-column cap, 'ALG 1' collapses to 'ALG'."""
        assert parse_row("03100600 - 1 ALG 2 82 82 82 1.0000", None).name == "ALG 2"

    def test_missing_final_uses_last_semester(self):
        course = parse_row("A3330100 - 1 APUSGOVT 79 79 0.5000", None)
        assert course.grade == "79"
        assert course.credits == 0.5

    def test_no_grade_marker_is_skipped(self):
        course = parse_row("13011400 - 1 BUSIM1 NG 94 92 1.0000", None)
        assert course.grade == "92"
        assert not course.pass_fail

    def test_pass_only_course_is_pass_fail(self):
        course = parse_row("A3580120 - 1 APTACSAL P P 1.0000", None)
        assert course.pass_fail is True

    def test_in_progress_course_has_zero_credit_and_no_grade(self):
        course = parse_row("I3100500 - 1 IBMAASL 0.0000", "2026-2027")
        assert course.credits == 0.0
        assert course.grade == ""

    def test_header_rows_are_ignored(self):
        assert parse_row("Course Description Sem1 Sem2 Final Credit", None) is None

    def test_blank_row_is_ignored(self):
        assert parse_row("   ", None) is None


class TestSubjectAndLevel:
    @pytest.mark.parametrize(
        ("name", "subject"),
        [
            ("APSTATS", "Math"),
            ("APPRECAL", "Math"),
            ("IBMAASL", "Math"),
            ("IBPHYSHL", "Science"),
            ("APTACSAM", "CS"),
            ("APCSPRIN", "CS"),
            ("IBTACSSL", "CS"),
            ("IBLITHL", "English"),
            ("IBECO-HL", "Social"),
            ("APUSHIST", "Social"),
            ("IBSPANSL", "Language"),
        ],
    )
    def test_prefixed_abbreviations_are_classified(self, name, subject):
        """A leading \\b never matches inside AP*/IB* fused names."""
        course = parse_row(f"X1234567 - 1 {name} 90 90 90 1.0000", None)
        assert course.subject == subject

    @pytest.mark.parametrize(
        ("name", "level"),
        [("APSTATS", "AP"), ("IBPHYSHL", "IB"), ("GEOM", "Regular")],
    )
    def test_programme_prefix_sets_level(self, name, level):
        assert parse_row(f"X1234567 - 1 {name} 90 90 90 1.0000", None).level == level


class TestRealTranscript:
    def test_all_courses_extracted(self, extracted):
        assert len(extracted.courses) == 37

    def test_metadata(self, extracted):
        assert extracted.school_name == "Westwood High School"
        assert extracted.graduation_year == 2027

    def test_five_academic_years_detected(self, extracted):
        years = {c.academic_year for c in extracted.courses}
        assert years == {"2022-2023", "2023-2024", "2024-2025", "2025-2026", "2026-2027"}

    def test_gpa_computed_from_numeric_grades(self, extracted):
        result = calculate_unweighted_gpa(extracted.courses)
        assert 3.5 <= result.gpa <= 3.8
        assert result.graded_credits == 27.0

    def test_in_progress_senior_courses_excluded_from_gpa(self, extracted):
        result = calculate_unweighted_gpa(extracted.courses)
        assert any("IBMAASL" in e for e in result.excluded_courses)

    def test_missing_class_rank_triggers_verification(self, extracted):
        assert "class_rank" in extracted.uncertain_fields
        assert extracted.needs_human_verification(0.75)

    def test_digital_twin_progressions(self, extracted):
        twin = build_digital_twin(student_id="s", verified_academics=extracted)
        academics = twin.academics
        assert academics.math_progression[0] == "ALG 1"
        assert "APCSPRIN" in academics.cs_progression
        assert "IBPHYSHL" in academics.science_progression
        assert len(academics.ap_courses) == 8
        assert len(academics.ib_courses) == 12

    def test_senior_year_courses_are_the_in_progress_ones(self, extracted):
        twin = build_digital_twin(student_id="s", verified_academics=extracted)
        assert "IBMAASL" in twin.academics.senior_year_courses
        assert "ALG 1" not in twin.academics.senior_year_courses
