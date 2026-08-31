"""Transcript ingestion (Section 9).

Deterministic, rule-based extraction. PDF text is pulled with PyMuPDF, then
parsed with regexes. Every course carries a per-field confidence; low overall
confidence or uncertain fields trigger human verification in the graph.

This is intentionally *not* an LLM node: an LLM re-reading a transcript on every
run is both non-reproducible and the single easiest place to hallucinate a
course that does not exist.
"""

from __future__ import annotations

import hashlib
import re
from typing import Literal

from pathora.domain.models import Course, ExtractedAcademics

Level = Literal["Regular", "Honors", "AP", "IB", "DualEnrollment", "Unknown"]
Subject = Literal["Math", "Science", "CS", "English", "Social", "Language", "Other"]

GRADE_RE = r"(A\+|A-|A|B\+|B-|B|C\+|C-|C|D\+|D-|D|F|P|NP|W|I|AUD|\d{1,3})"
COURSE_LINE = re.compile(
    rf"^\s*(?P<name>[A-Za-z][A-Za-z0-9 &/'.,\-()]+?)\s{{2,}}"
    rf"(?P<grade>{GRADE_RE})\s+"
    rf"(?P<credits>\d+(?:\.\d+)?)\s*$"
)
YEAR_HEADER = re.compile(r"(?P<year>(19|20)\d{2}\s*[-/–]\s*(19|20)?\d{2})")
GPA_RE = re.compile(r"(?:cumulative\s+)?gpa[^0-9]{0,12}(\d\.\d{1,3})", re.I)
RANK_RE = re.compile(r"rank[^0-9]{0,12}(\d+)\s*(?:of|/)\s*(\d+)", re.I)
GRAD_RE = re.compile(r"(?:graduation|class\s+of)[^0-9]{0,12}((?:19|20)\d{2})", re.I)
SCHOOL_RE = re.compile(
    r"^(?P<name>[A-Z][A-Za-z .'\-]*(High School|Academy|Preparatory|HS))\s*$", re.M
)

MATH_SEQUENCE = [
    "algebra i",
    "geometry",
    "algebra ii",
    "pre-calculus",
    "precalculus",
    "calculus",
    "ap calculus ab",
    "ap calculus bc",
    "multivariable calculus",
    "linear algebra",
    "statistics",
    "ap statistics",
]

SUBJECT_KEYWORDS: dict[str, Subject] = {
    "algebra": "Math",
    "geometry": "Math",
    "calculus": "Math",
    "precalc": "Math",
    "pre-calc": "Math",
    "statistic": "Math",
    "math": "Math",
    "trigonometry": "Math",
    "biology": "Science",
    "chemistry": "Science",
    "physics": "Science",
    "environmental": "Science",
    "anatomy": "Science",
    "science": "Science",
    "computer science": "CS",
    "programming": "CS",
    "software": "CS",
    "cybersecurity": "CS",
    "data structures": "CS",
    "robotics": "CS",
    "engineering": "CS",
    "english": "English",
    "literature": "English",
    "writing": "English",
    "history": "Social",
    "government": "Social",
    "economics": "Social",
    "psychology": "Social",
    "geography": "Social",
    "spanish": "Language",
    "french": "Language",
    "german": "Language",
    "chinese": "Language",
    "latin": "Language",
}


def _detect_level(name: str) -> Level:
    low = name.lower()
    if low.startswith("ap ") or " ap " in low or low.endswith(" ap"):
        return "AP"
    if low.startswith("ib ") or " ib " in low:
        return "IB"
    if "dual" in low or "dual enrollment" in low:
        return "DualEnrollment"
    if "honors" in low or low.endswith(" h") or "advanced" in low:
        return "Honors"
    return "Regular"


def _detect_subject(name: str) -> Subject:
    low = name.lower()
    # Longest keyword wins so "computer science" is not swallowed by "science".
    for keyword in sorted(SUBJECT_KEYWORDS, key=len, reverse=True):
        if keyword in low:
            return SUBJECT_KEYWORDS[keyword]
    return "Other"


def extract_text_from_pdf(pdf_bytes: bytes) -> str:
    """Extract raw text from a PDF. Raises RuntimeError if PyMuPDF is absent."""
    try:
        import pymupdf  # type: ignore
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError("PyMuPDF is required to parse PDF transcripts") from exc

    with pymupdf.open(stream=pdf_bytes, filetype="pdf") as doc:
        return "\n".join(page.get_text("text") for page in doc)


def parse_transcript_text(text: str) -> ExtractedAcademics:
    """Parse transcript text into structured academics with confidence scores."""
    courses: list[Course] = []
    current_year: str | None = None
    uncertain: list[str] = []

    for raw_line in text.splitlines():
        line = raw_line.rstrip()
        if not line.strip():
            continue

        year_match = YEAR_HEADER.search(line)
        if year_match and not COURSE_LINE.match(line):
            current_year = re.sub(r"\s*", "", year_match.group("year"))
            continue

        match = COURSE_LINE.match(line)
        if not match:
            continue

        name = match.group("name").strip(" .-")
        grade = match.group("grade").strip()
        credits = float(match.group("credits"))
        level = _detect_level(name)
        subject = _detect_subject(name)

        confidence = 1.0
        if subject == "Other":
            confidence -= 0.1
        if credits not in (0.5, 1.0, 2.0):
            confidence -= 0.1
        if grade.isdigit():
            confidence -= 0.05

        courses.append(
            Course(
                name=name,
                grade=grade,
                credits=credits,
                academic_year=current_year,
                level=level,
                subject=subject,
                pass_fail=grade.upper() in {"P", "NP"},
                confidence=round(max(confidence, 0.0), 3),
            )
        )

    reported_gpa = None
    if (m := GPA_RE.search(text)) is not None:
        reported_gpa = float(m.group(1))

    class_rank = None
    if (m := RANK_RE.search(text)) is not None:
        class_rank = f"{m.group(1)} of {m.group(2)}"
    else:
        uncertain.append("class_rank")

    graduation_year = int(m.group(1)) if (m := GRAD_RE.search(text)) else None
    if graduation_year is None:
        uncertain.append("graduation_year")

    school_name = None
    if (m := SCHOOL_RE.search(text)) is not None:
        school_name = m.group("name").strip()
    else:
        uncertain.append("school_name")

    if not courses:
        uncertain.append("courses")
    if any(c.academic_year is None for c in courses):
        uncertain.append("academic_year")

    course_confidence = sum(c.confidence for c in courses) / len(courses) if courses else 0.0
    field_penalty = 0.06 * len(uncertain)
    extraction_confidence = round(max(course_confidence - field_penalty, 0.0), 3)

    return ExtractedAcademics(
        courses=courses,
        school_name=school_name,
        graduation_year=graduation_year,
        reported_gpa=reported_gpa,
        class_rank=class_rank,
        extraction_confidence=extraction_confidence,
        uncertain_fields=sorted(set(uncertain)),
        raw_text_sha256=hashlib.sha256(text.encode()).hexdigest(),
    )


def parse_transcript_pdf(pdf_bytes: bytes) -> ExtractedAcademics:
    """Parse a PDF transcript, preferring the coordinate-based table reader.

    Real transcripts are tables. PyMuPDF's plain-text extraction flattens them
    to one cell per line, which no line-oriented regex can reassemble, so the
    table reader (which groups words by baseline) is tried first. The
    line-oriented parser remains the fallback for documents that really are
    laid out as text.
    """
    from pathora.services.transcript_table import parse_rows, rows_from_pdf

    text = extract_text_from_pdf(pdf_bytes)
    line_based = parse_transcript_text(text)

    try:
        courses, metadata = parse_rows(rows_from_pdf(pdf_bytes))
    except Exception:  # noqa: BLE001 - fall back rather than fail the upload
        return line_based

    if len(courses) <= len(line_based.courses):
        return line_based

    uncertain: list[str] = []
    if not metadata.get("school_name"):
        uncertain.append("school_name")
    if not metadata.get("graduation_year"):
        uncertain.append("graduation_year")

    reported_gpa = line_based.reported_gpa
    class_rank = line_based.class_rank
    if class_rank is None:
        uncertain.append("class_rank")
    if any(c.academic_year is None for c in courses):
        uncertain.append("academic_year")

    course_confidence = sum(c.confidence for c in courses) / len(courses)
    confidence = round(max(course_confidence - 0.06 * len(uncertain), 0.0), 3)

    return ExtractedAcademics(
        courses=courses,
        school_name=metadata.get("school_name"),
        graduation_year=metadata.get("graduation_year"),
        reported_gpa=reported_gpa,
        class_rank=class_rank,
        extraction_confidence=confidence,
        uncertain_fields=sorted(set(uncertain)),
        raw_text_sha256=hashlib.sha256(text.encode()).hexdigest(),
    )


def progression(courses: list[Course], subject: Subject) -> list[str]:
    """Ordered course progression for a subject, deterministically sorted."""
    selected = [c for c in courses if c.subject == subject]
    selected.sort(key=lambda c: (c.academic_year or "", _sequence_rank(c.name)))
    return [c.name for c in selected]


def _sequence_rank(name: str) -> int:
    low = name.lower()
    for idx, token in enumerate(MATH_SEQUENCE):
        if token in low:
            return idx
    return len(MATH_SEQUENCE)


def senior_year_courses(courses: list[Course]) -> list[str]:
    years = sorted({c.academic_year for c in courses if c.academic_year})
    if not years:
        return []
    return [c.name for c in courses if c.academic_year == years[-1]]
