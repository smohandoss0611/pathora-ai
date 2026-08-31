import pytest

from pathora.domain.models import Course
from pathora.services.gpa import (
    STANDARD_UNWEIGHTED_4_SCALE,
    calculate_unweighted_gpa,
    grade_trend,
    numeric_to_letter,
)


def c(name: str, grade: str, credits: float = 1.0, **kw) -> Course:
    return Course(name=name, grade=grade, credits=credits, **kw)


def test_empty_transcript_returns_zero_without_dividing_by_zero():
    r = calculate_unweighted_gpa([])
    assert r.gpa == 0.0
    assert r.graded_credits == 0.0


def test_all_a_grades():
    r = calculate_unweighted_gpa([c("Calc", "A"), c("Physics", "A")])
    assert r.gpa == 4.0
    assert r.graded_credits == 2.0


def test_mixed_letters_with_plus_minus():
    r = calculate_unweighted_gpa([c("A", "A"), c("B", "B+"), c("C", "B-"), c("D", "C")])
    # 4.0 + 3.3 + 2.7 + 2.0 = 12.0 / 4
    assert r.gpa == 3.0


def test_credit_weighting_changes_result():
    r = calculate_unweighted_gpa([c("Big", "A", credits=3), c("Small", "C", credits=1)])
    # (12 + 2) / 4 = 3.5
    assert r.gpa == 3.5
    assert r.graded_credits == 4.0


def test_half_credit_courses():
    r = calculate_unweighted_gpa([c("Health", "A", credits=0.5), c("Alg", "B", credits=1.0)])
    # (2.0 + 3.0) / 1.5
    assert r.gpa == pytest.approx(3.33, abs=0.005)
    assert r.graded_credits == 1.5


def test_pass_fail_flag_excluded():
    r = calculate_unweighted_gpa([c("PE", "P", pass_fail=True), c("Calc", "A")])
    assert r.gpa == 4.0
    assert r.graded_credits == 1.0
    assert any("pass/fail" in e for e in r.excluded_courses)


@pytest.mark.parametrize("mark", ["P", "NP", "W", "I", "AUD", "CR", "NC"])
def test_non_graded_marks_excluded(mark):
    r = calculate_unweighted_gpa([c("X", mark), c("Calc", "A")])
    assert r.gpa == 4.0
    assert r.graded_credits == 1.0


def test_unrecognized_grade_is_excluded_not_scored_zero():
    r = calculate_unweighted_gpa([c("Weird", "Z"), c("Calc", "A")])
    assert r.gpa == 4.0
    assert any("unrecognized" in e for e in r.excluded_courses)


def test_zero_credit_course_excluded():
    r = calculate_unweighted_gpa([c("Seminar", "A", credits=0), c("Calc", "B")])
    assert r.gpa == 3.0
    assert any("zero credit" in e for e in r.excluded_courses)


def test_whitespace_and_case_normalization():
    r = calculate_unweighted_gpa([c("X", " a- "), c("Y", "b+")])
    assert r.gpa == pytest.approx(3.5, abs=0.005)


def test_numeric_grades_supported():
    r = calculate_unweighted_gpa([c("X", "95"), c("Y", "85")])
    # 95 -> A (4.0), 85 -> B (3.0)
    assert r.gpa == 3.5


def test_numeric_grades_can_be_disabled():
    r = calculate_unweighted_gpa([c("X", "95")], allow_numeric_grades=False)
    assert r.graded_credits == 0.0
    assert any("unrecognized" in e for e in r.excluded_courses)


@pytest.mark.parametrize(
    ("value", "letter"),
    [(100, "A+"), (97, "A+"), (93, "A"), (90, "A-"), (60, "D-"), (59, "F"), (0, "F")],
)
def test_numeric_band_boundaries(value, letter):
    assert numeric_to_letter(value) == letter


def test_configurable_grade_scale():
    scale = dict(STANDARD_UNWEIGHTED_4_SCALE) | {"A": 5.0}
    r = calculate_unweighted_gpa([c("X", "A")], grade_scale=scale, method="custom")
    assert r.gpa == 5.0
    assert r.method == "custom"


def test_f_counts_as_zero_and_consumes_credits():
    r = calculate_unweighted_gpa([c("X", "A"), c("Y", "F")])
    assert r.gpa == 2.0
    assert r.graded_credits == 2.0


def test_quality_points_reported():
    r = calculate_unweighted_gpa([c("X", "A", credits=2)])
    assert r.quality_points == 8.0


def test_spec_example_shape():
    courses = [c(f"C{i}", "A") for i in range(20)] + [c(f"D{i}", "B") for i in range(5)]
    r = calculate_unweighted_gpa(courses)
    assert r.method == "standard_unweighted_4_scale"
    assert r.graded_credits == 25
    assert r.gpa == pytest.approx(3.8, abs=0.01)


def test_rounding_is_two_decimals_by_default():
    r = calculate_unweighted_gpa([c("X", "A"), c("Y", "B"), c("Z", "B")])
    assert r.gpa == 3.33


def test_result_is_deterministic_across_runs():
    courses = [c("X", "A-"), c("Y", "B+", credits=0.5), c("Z", "C")]
    assert calculate_unweighted_gpa(courses) == calculate_unweighted_gpa(courses)


class TestGradeTrend:
    def test_improving(self):
        courses = [
            c("X", "C", academic_year="2022-23"),
            c("Y", "B", academic_year="2023-24"),
            c("Z", "A", academic_year="2024-25"),
        ]
        assert grade_trend(courses) == "Improving"

    def test_declining(self):
        courses = [
            c("X", "A", academic_year="2022-23"),
            c("Y", "B", academic_year="2023-24"),
        ]
        assert grade_trend(courses) == "Declining"

    def test_stable(self):
        courses = [
            c("X", "A", academic_year="2022-23"),
            c("Y", "A", academic_year="2023-24"),
        ]
        assert grade_trend(courses) == "Stable"

    def test_unknown_without_years(self):
        assert grade_trend([c("X", "A")]) == "Unknown"

    def test_mixed(self):
        courses = [
            c("X", "C", academic_year="2021-22"),
            c("Y", "A", academic_year="2022-23"),
            c("Z", "C", academic_year="2023-24"),
        ]
        assert grade_trend(courses) == "Mixed"
