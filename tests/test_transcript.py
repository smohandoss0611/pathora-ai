from pathlib import Path

import pytest

from pathora.services.gpa import calculate_unweighted_gpa
from pathora.services.transcript import (
    parse_transcript_text,
    progression,
    senior_year_courses,
)

SAMPLE = (Path(__file__).parent.parent / "data/seed/sample_transcript.txt").read_text()


@pytest.fixture(scope="module")
def parsed():
    return parse_transcript_text(SAMPLE)


def test_extracts_all_course_rows(parsed):
    assert len(parsed.courses) == 18


def test_extracts_school_and_metadata(parsed):
    assert parsed.school_name == "Lakeview High School"
    assert parsed.graduation_year == 2027
    assert parsed.reported_gpa == 3.71
    assert parsed.class_rank == "24 of 412"


def test_assigns_academic_years(parsed):
    years = {c.academic_year for c in parsed.courses}
    assert years == {"2023-2024", "2024-2025", "2025-2026"}


def test_detects_ap_and_honors_levels(parsed):
    ap = [c.name for c in parsed.courses if c.level == "AP"]
    honors = [c.name for c in parsed.courses if c.level == "Honors"]
    assert "AP Calculus AB" in ap
    assert "AP Computer Science A" in ap
    assert "Chemistry Honors" in honors


def test_detects_subjects(parsed):
    by_name = {c.name: c.subject for c in parsed.courses}
    assert by_name["AP Calculus AB"] == "Math"
    assert by_name["AP Physics 1"] == "Science"
    assert by_name["AP Computer Science A"] == "CS"
    assert by_name["Spanish II"] == "Language"


def test_half_credit_courses_preserved(parsed):
    health = next(c for c in parsed.courses if c.name == "Health")
    assert health.credits == 0.5


def test_pass_fail_flagged(parsed):
    pe = next(c for c in parsed.courses if c.name.startswith("Physical Education"))
    assert pe.pass_fail is True


def test_confidence_is_reported(parsed):
    assert 0.0 < parsed.extraction_confidence <= 1.0
    assert all(0.0 <= c.confidence <= 1.0 for c in parsed.courses)


def test_hash_recorded_for_audit(parsed):
    assert parsed.raw_text_sha256 and len(parsed.raw_text_sha256) == 64


def test_gpa_from_parsed_transcript_is_close_to_reported(parsed):
    result = calculate_unweighted_gpa(parsed.courses)
    assert result.gpa == pytest.approx(parsed.reported_gpa, abs=0.15)
    assert any("pass/fail" in e for e in result.excluded_courses)


def test_math_progression_order(parsed):
    math = progression(parsed.courses, "Math")
    assert math[0] == "Algebra II"
    assert "AP Calculus AB" in math


def test_senior_year_courses(parsed):
    senior = senior_year_courses(parsed.courses)
    assert "AP Calculus AB" in senior
    assert "Algebra II" not in senior


def test_missing_metadata_triggers_verification():
    result = parse_transcript_text("Algebra I                         A     1.0\n")
    assert result.needs_human_verification() is True
    assert "class_rank" in result.uncertain_fields


def test_empty_document_is_uncertain():
    result = parse_transcript_text("")
    assert result.courses == []
    assert result.needs_human_verification() is True
    assert result.extraction_confidence == 0.0


def test_pdf_round_trip(tmp_path):
    pymupdf = pytest.importorskip("pymupdf")
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((40, 60), SAMPLE, fontname="cour", fontsize=8)
    pdf_path = tmp_path / "t.pdf"
    doc.save(pdf_path)
    doc.close()

    from pathora.services.transcript import parse_transcript_pdf

    parsed_pdf = parse_transcript_pdf(pdf_path.read_bytes())
    assert len(parsed_pdf.courses) >= 15
    assert parsed_pdf.school_name == "Lakeview High School"


def test_confidence_threshold_is_configurable():
    """EXTRACTION_CONFIDENCE_THRESHOLD must govern the decision, not just the message."""
    result = parse_transcript_text(SAMPLE)
    assert result.uncertain_fields == []
    assert result.needs_human_verification(threshold=0.0) is False
    assert result.needs_human_verification(threshold=1.0) is True
