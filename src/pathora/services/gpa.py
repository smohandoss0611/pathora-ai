"""Deterministic GPA calculation.

Section 10: GPA is never computed by an LLM. This module is pure Python, has no
I/O, and is fully unit tested.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping

from pathora.domain.models import Course, GPAResult

STANDARD_UNWEIGHTED_4_SCALE: Mapping[str, float] = {
    "A+": 4.0,
    "A": 4.0,
    "A-": 3.7,
    "B+": 3.3,
    "B": 3.0,
    "B-": 2.7,
    "C+": 2.3,
    "C": 2.0,
    "C-": 1.7,
    "D+": 1.3,
    "D": 1.0,
    "D-": 0.7,
    "F": 0.0,
}

#: Grades that carry no quality points and are excluded from graded credits.
NON_GRADED = frozenset({"P", "PASS", "NP", "F/P", "CR", "NC", "W", "I", "INC", "AUD", "T", "TR"})

#: Optional numeric-percentage fallback (only used when enabled).
NUMERIC_BANDS: tuple[tuple[float, str], ...] = (
    (97, "A+"),
    (93, "A"),
    (90, "A-"),
    (87, "B+"),
    (83, "B"),
    (80, "B-"),
    (77, "C+"),
    (73, "C"),
    (70, "C-"),
    (67, "D+"),
    (63, "D"),
    (60, "D-"),
    (0, "F"),
)


def normalize_grade(raw: str) -> str:
    """Uppercase, strip whitespace and unicode minus signs."""
    return raw.strip().upper().replace("\u2212", "-").replace(" ", "")


def numeric_to_letter(value: float) -> str:
    for floor, letter in NUMERIC_BANDS:
        if value >= floor:
            return letter
    return "F"


def _resolve_grade(raw: str, allow_numeric: bool) -> str | None:
    """Return a normalized letter grade, or None if it is not gradeable."""
    grade = normalize_grade(raw)
    if not grade or grade in NON_GRADED:
        return None
    if allow_numeric:
        try:
            return numeric_to_letter(float(grade))
        except ValueError:
            pass
    return grade


def calculate_unweighted_gpa(
    courses: Iterable[Course],
    *,
    grade_scale: Mapping[str, float] | None = None,
    allow_numeric_grades: bool = True,
    method: str = "standard_unweighted_4_scale",
    round_to: int = 2,
) -> GPAResult:
    """Credit-weighted unweighted GPA on a 4.0 scale.

    - Credits are honored, including half-credit (0.5) courses.
    - ``pass_fail`` courses and non-graded marks (P/NP/W/I/AUD/...) are excluded.
    - Zero-credit courses contribute nothing but are not reported as excluded
      errors; they are listed so the student can see what was dropped.
    - Unknown grade symbols are excluded rather than silently scored 0.0.
    """
    scale = dict(grade_scale or STANDARD_UNWEIGHTED_4_SCALE)
    quality_points = 0.0
    graded_credits = 0.0
    excluded: list[str] = []

    for course in courses:
        if course.pass_fail:
            excluded.append(f"{course.name} (pass/fail)")
            continue

        grade = _resolve_grade(course.grade, allow_numeric_grades)
        if grade is None:
            excluded.append(f"{course.name} (non-graded: {course.grade.strip() or 'blank'})")
            continue

        if grade not in scale:
            excluded.append(f"{course.name} (unrecognized grade: {course.grade.strip()})")
            continue

        if course.credits <= 0:
            excluded.append(f"{course.name} (zero credit)")
            continue

        quality_points += scale[grade] * course.credits
        graded_credits += course.credits

    gpa = round(quality_points / graded_credits, round_to) if graded_credits else 0.0

    return GPAResult(
        gpa=gpa,
        graded_credits=round(graded_credits, 3),
        method=method,
        excluded_courses=excluded,
        quality_points=round(quality_points, 3),
    )


def grade_trend(courses: Iterable[Course]) -> str:
    """Deterministic year-over-year trend label used to ground the Profile Agent."""
    by_year: dict[str, list[float]] = {}
    for course in courses:
        if course.pass_fail or not course.academic_year:
            continue
        grade = _resolve_grade(course.grade, True)
        if grade in STANDARD_UNWEIGHTED_4_SCALE:
            by_year.setdefault(course.academic_year, []).append(
                STANDARD_UNWEIGHTED_4_SCALE[str(grade)]
            )

    if len(by_year) < 2:
        return "Unknown"

    averages = [sum(v) / len(v) for _, v in sorted(by_year.items())]
    deltas = [b - a for a, b in zip(averages, averages[1:], strict=False)]
    if all(d >= 0.05 for d in deltas):
        return "Improving"
    if all(d <= -0.05 for d in deltas):
        return "Declining"
    if all(abs(d) < 0.05 for d in deltas):
        return "Stable"
    return "Mixed"
