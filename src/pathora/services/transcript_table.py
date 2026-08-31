"""Parser for column-layout transcripts (the common real-world shape).

The original parser assumed one course per text line with whitespace-separated
columns. Real transcripts are tables, and PyMuPDF's plain-text extraction emits
them one *cell* per line:

    03100500 - 1
    ALG 1
    89
    95
    92
    1.0000

No line-oriented regex can recover a row from that. This module reconstructs
rows from word coordinates instead — words sharing a baseline are one row —
which is how the table actually reads on the page.

Handles the messiness that comes with real records: semester grades plus a
final, missing finals, ``P`` for pass, ``NG`` for no-grade, in-progress
senior courses carrying 0.0000 credit, and district course codes.
"""

from __future__ import annotations

import re
from typing import Any

from pathora.domain.models import Course

#: "2023-24 - Grade 09" or "2023-2024"
YEAR_ROW = re.compile(r"((?:19|20)\d{2})\s*[-/–]\s*((?:19|20)?\d{2})")
COURSE_CODE = re.compile(r"^[A-Z0-9]{6,9}\s*-\s*\d+\s+")
CREDIT = re.compile(r"^\d+\.\d{2,4}$")
NUMERIC_GRADE = re.compile(r"^\d{1,3}$")
LETTER_GRADE = re.compile(r"^(A\+|A-|A|B\+|B-|B|C\+|C-|C|D\+|D-|D|F)$", re.I)
NON_GRADED_MARK = {"P", "NG", "NC", "W", "I", "AUD", "CR", "INC", "--", "*"}

HEADER_WORDS = {
    "course",
    "description",
    "sem1",
    "sem2",
    "final",
    "credit",
    "building",
    "student",
    "district",
    "school",
    "transcript",
    "academic",
}

#: Abbreviated course names, as districts actually print them.
#: Matched as substrings, not whole words: districts print "APSTATS",
#: "IBPHYSHL", "APPRECAL" with the programme prefix fused to the subject, so a
#: leading \b would never match. CS is checked before Science so "TACS" is not
#: read as a science course.
SUBJECT_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"(TACS|CSPRIN|COMPSCI|PRINBMF|BUSIM|IED|ROBOT|PROG|CSA\b|CSP\b)", re.I), "CS"),
    (re.compile(r"(PRECAL|PRE-?CALC|CALC|ALG|GEOM|STAT|MAAS|MAA|MATH|TRIG)", re.I), "Math"),
    (re.compile(r"(BIO|CHEM|PHYS|ANAT|ENVSCI|SCIEN)", re.I), "Science"),
    (re.compile(r"(HUMGEO|HIST|GOVT|ECO|PSYCH|SOCST|GEOG|CIVIC)", re.I), "Social"),
    (re.compile(r"(LANLT|LITHL|LIT|ENG\d?|WRIT|ELA)", re.I), "English"),
    (re.compile(r"(SPAN|FREN|GERM|CHIN|LATIN|ASL)", re.I), "Language"),
]


def _detect_subject(name: str) -> str:
    for pattern, subject in SUBJECT_PATTERNS:
        if pattern.search(name):
            return subject
    return "Other"


def _detect_level(name: str, code: str) -> str:
    blob = f"{code} {name}".upper()
    if re.search(r"\bIB[A-Z]*\b|^I3", blob):
        return "IB"
    if re.search(r"\bAP[A-Z]*\b|^A3", blob):
        return "AP"
    if "HON" in blob or "ADV" in blob:
        return "Honors"
    if "DUAL" in blob or "DC" in blob:
        return "DualEnrollment"
    return "Regular"


def rows_from_pdf(pdf_bytes: bytes) -> list[str]:
    """Reconstruct visual table rows by grouping words on a shared baseline."""
    try:
        import pymupdf
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("PyMuPDF is required to parse PDF transcripts") from exc

    rows: list[str] = []
    with pymupdf.open(stream=pdf_bytes, filetype="pdf") as document:
        for page in document:
            buckets: dict[int, list[tuple[float, str]]] = {}
            for x0, y0, _x1, _y1, word, *_ in page.get_text("words"):
                # 3pt tolerance keeps sub/superscripts on the same row.
                buckets.setdefault(round(y0 / 3), []).append((x0, word))
            for key in sorted(buckets):
                rows.append(" ".join(word for _, word in sorted(buckets[key])))
    return rows


def parse_row(row: str, academic_year: str | None) -> Course | None:
    """Parse one reconstructed table row into a Course, or None if it isn't one."""
    stripped = row.strip()
    if not stripped:
        return None

    lowered = stripped.lower()
    if sum(word in lowered for word in HEADER_WORDS) >= 2:
        return None

    code = ""
    if (match := COURSE_CODE.match(stripped)) is not None:
        code = match.group(0).strip()
        stripped = stripped[match.end() :]

    tokens = stripped.split()
    if not tokens:
        return None

    # Trailing numeric with decimals is the credit value.
    credits = 0.0
    if CREDIT.match(tokens[-1]):
        credits = float(tokens.pop())
    elif not code:
        return None

    # Walk back over at most three grade columns (sem1, sem2, final). The cap
    # matters: without it "ALG 1 89 95 92" loses its course number to the grade
    # list and Algebra 1 becomes indistinguishable from Algebra 2.
    grades: list[str] = []
    while (
        tokens
        and len(grades) < 3
        and (
            NUMERIC_GRADE.match(tokens[-1])
            or LETTER_GRADE.match(tokens[-1])
            or tokens[-1].upper() in NON_GRADED_MARK
        )
    ):
        grades.insert(0, tokens.pop())

    name = " ".join(tokens).strip()
    if not name:
        return None

    graded = [g for g in grades if g.upper() not in NON_GRADED_MARK]
    pass_fail = bool(grades) and not graded

    # The rightmost real grade is the final (or the last semester if no final).
    grade = graded[-1] if graded else (grades[-1] if grades else "")

    confidence = 1.0
    if not grades:
        confidence -= 0.3  # in-progress course, no marks yet
    if len(name) <= 2:
        confidence -= 0.2
    if _detect_subject(name) == "Other":
        confidence -= 0.1

    return Course(
        name=name,
        grade=grade,
        credits=credits,
        academic_year=academic_year,
        level=_detect_level(name, code),  # type: ignore[arg-type]
        subject=_detect_subject(name),  # type: ignore[arg-type]
        pass_fail=pass_fail,
        confidence=round(max(confidence, 0.0), 3),
    )


def parse_rows(rows: list[str]) -> tuple[list[Course], dict[str, Any]]:
    """Parse reconstructed rows into courses plus document metadata."""
    courses: list[Course] = []
    current_year: str | None = None
    metadata: dict[str, Any] = {}

    for row in rows:
        if (match := YEAR_ROW.search(row)) is not None and not CREDIT.search(row):
            start, end = match.group(1), match.group(2)
            current_year = f"{start}-{end if len(end) == 4 else start[:2] + end}"
            continue

        if "high school" in row.lower() and "school_name" not in metadata:
            cleaned = re.sub(r"^(Building|High School)\s*:?\s*", "", row).strip()
            metadata["school_name"] = cleaned

        if course := parse_row(row, current_year):
            courses.append(course)

    years = sorted({c.academic_year for c in courses if c.academic_year})
    if years:
        # Final academic year end is the graduating year.
        metadata["graduation_year"] = int(years[-1].split("-")[-1])

    return courses, metadata
