"""Deterministic admission-fit classification.

The classification used to come out of the Admission Agent's prose. That is the
same error as prompt-level abstention: a model reading evidence and choosing a
label produces inconsistent results, and it produced one — Texas Tech scored
Target for a student above its published 75th percentile at a school admitting
72.7%.

So the label is computed here, in code, from four inputs:

    published selectivity + score position + GPA + course rigor
                              ↓
                        RULE ENGINE  ->  baseline classification
                              ↓
                             LLM     ->  explains why

The agent may still write the strengths, risks and rationale. It may not choose
the label.

Two asymmetries are deliberate:

- **Missing major-level data caps the ceiling, it does not lower the floor.**
  Without a published major admit rate the best available label is `Likely`,
  never `Safety`: you cannot call a program safe when you do not know how
  selective it is. But absence of data alone must not push a well-matched
  student down to `Target` — that is pessimism dressed as caution.
- **Downgrades require concrete retrieved evidence**, such as a capped or
  limited-access major, not the absence of information.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Literal

from pathora.domain.models import (
    NOT_PUBLISHED,
    Classification,
    CollegeResearchResult,
    Confidence,
    ProfileAnalysis,
    StudentDigitalTwin,
)

LADDER: list[Classification] = [
    "Safety",
    "Likely",
    "Target",
    "Target-Reach",
    "Reach",
    "High Reach",
]

RIGOR_POINTS = {"Excellent": 2, "Strong": 1, "Moderate": 0, "Weak": -1}

RANGE_RE = re.compile(r"(\d{2,4})\s*[-\u2013\u2014]\s*(\d{2,4})")

#: Retrieved wording that justifies a downgrade. Absence of data is not here.
RESTRICTION_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"capped major", re.I), "the major is capped"),
    (re.compile(r"limited[\s-]access", re.I), "the major is limited-access"),
    (re.compile(r"secondary admission", re.I), "the major requires secondary admission"),
    (
        re.compile(r"separate (?:review|committee|portfolio)", re.I),
        "the major is separately reviewed",
    ),
    (re.compile(r"competitive (?:entry|placement)", re.I), "major placement is competitive"),
    # Large publics commonly admit to a COLLEGE and place into the major later
    # (Texas A&M calls this Entry to a Major). Where that is the published
    # structure, the university-wide admit rate does not govern the major at
    # all, and treating it as a proxy overstates the student's position.
    (
        re.compile(r"entry to a major|\bETAM\b", re.I),
        "major placement runs through Entry to a Major",
    ),
    (
        re.compile(r"admitted to the college[^.]{0,80}(?:then|before|prior to)", re.I),
        "admission is to the college, with major placement decided later",
    ),
    (
        re.compile(r"place into (?:a |the )?major", re.I),
        "students place into the major after admission",
    ),
    (
        re.compile(r"university-? or college-level admission", re.I),
        "admission is to the university or college rather than directly to the major",
    ),
]


@dataclass
class Baseline:
    classification: Classification
    confidence: Confidence
    signals: list[str] = field(default_factory=list)
    caps_applied: list[str] = field(default_factory=list)
    selectivity_known: bool = True
    major_level_known: bool = False


def parse_range(value: str) -> tuple[int, int] | None:
    if not value or value == NOT_PUBLISHED:
        return None
    if (match := RANGE_RE.search(value)) is None:
        return None
    low, high = sorted((int(match.group(1)), int(match.group(2))))
    return low, high


def parse_percent(value: str) -> float | None:
    if not value or value == NOT_PUBLISHED:
        return None
    if (match := re.search(r"(\d{1,3}(?:\.\d+)?)\s*%", value)) is None:
        return None
    return float(match.group(1))


def selectivity_offset(rate: float | None) -> int:
    """Where a published admit rate sits on the ladder before student signal."""
    if rate is None:
        return 2
    if rate >= 70:
        return 0
    if rate >= 50:
        return 1
    if rate >= 30:
        return 2
    if rate >= 15:
        return 3
    return 4


def score_position(
    twin: StudentDigitalTwin, research: CollegeResearchResult
) -> tuple[int, str, bool]:
    """Compare the student's test score to THIS college's published band."""
    sat, act = twin.testing.sat_total, twin.testing.act_composite

    for score, published, label in (
        (sat, research.sat_range, "SAT"),
        (act, research.act_range, "ACT"),
    ):
        if not score:
            continue
        band = parse_range(published)
        if band is None:
            generic_high, generic_low = (1450, 1250) if label == "SAT" else (32, 24)
            if score >= generic_high:
                return 1, f"{label} {score} (no published range for this college)", False
            if score < generic_low:
                return -1, f"{label} {score} (no published range for this college)", False
            return 0, f"{label} {score} (no published range for this college)", False
        low, high = band
        if score > high:
            return 1, f"{label} {score} is above the published {low}-{high} range", True
        if score >= low:
            return 0, f"{label} {score} is inside the published {low}-{high} range", False
        return -1, f"{label} {score} is below the published {low}-{high} range", False

    return -1, "No standardized test score on file", False


def classify(
    twin: StudentDigitalTwin,
    research: CollegeResearchResult,
    profile: ProfileAnalysis,
) -> Baseline:
    """Compute the baseline label. Deterministic, reproducible, testable."""
    signals: list[str] = []
    caps: list[str] = []

    major_rate = parse_percent(research.major_admit_rate)
    overall_rate = parse_percent(research.admit_rate)
    rate = major_rate if major_rate is not None else overall_rate
    known = rate is not None
    offset = selectivity_offset(rate)

    if major_rate is not None:
        signals.append(f"Published major admit rate {research.major_admit_rate}")
    elif overall_rate is not None:
        signals.append(f"Published university-wide admit rate {research.admit_rate}")
    else:
        signals.append("No admit rate published in retrieved sources")

    strength = 0
    gpa = twin.academics.gpa.gpa
    if gpa >= 3.85:
        strength += 2
        signals.append(f"Unweighted GPA {gpa}")
    elif gpa >= 3.5:
        strength += 1
        signals.append(f"Unweighted GPA {gpa}")
    elif gpa < 3.0:
        strength -= 1
        signals.append(f"Unweighted GPA {gpa} is below typical selective-STEM pools")

    rigor_points = RIGOR_POINTS.get(profile.course_rigor, 0)
    strength += rigor_points
    signals.append(f"Course rigor {profile.course_rigor}")

    test_delta, test_note, above_75th = score_position(twin, research)
    strength += test_delta
    signals.append(test_note)

    if profile.grade_trend == "Improving":
        signals.append("Grade trend improving")
    elif profile.grade_trend == "Declining":
        strength -= 1
        signals.append("Grade trend declining")

    index = offset + (2 - strength)

    # Floors: a strong student cannot make a very selective program non-selective.
    if not known:
        index = max(index, 2)
        caps.append("no published admit rate, so nothing better than Target can be claimed")
    elif offset >= 4:
        index = max(index, 4)
    elif offset == 3:
        index = max(index, 3)

    # Below the published middle 50% is not a Likely anywhere, however
    # unselective the institution overall.
    band = parse_range(research.sat_range) or parse_range(research.act_range)
    if band is not None and test_delta < 0:
        index = max(index, 2)
        caps.append("score sits below the published middle 50%, so Likely is not claimed")

    # Ceiling: Safety normally requires knowing how selective the MAJOR is —
    # this caps the best case rather than pushing a matched student downward.
    # The exception is a broadly accessible institution where the student is
    # above the published 75th percentile: there the university-wide rate is
    # informative enough on its own, and refusing to say Safety would be
    # pessimism dressed as caution.
    if known and major_rate is None:
        broadly_accessible = overall_rate is not None and overall_rate >= 85 and above_75th
        if not broadly_accessible:
            index = max(index, 1)
            caps.append(
                "only a university-wide rate was published, so Safety is not claimed for this major"
            )

    # Downgrades need retrieved evidence of a restriction, not missing data.
    # Search the raw retrieved text as well as the normalised structure label:
    # the fact extractor collapses "admitted to the College of Engineering, then
    # place into a major through ETAM" to a generic category, which loses the
    # very signal that matters here.
    haystack = " ".join(
        [
            research.admission_structure if research.admission_structure != NOT_PUBLISHED else "",
            *research.program_notes,
            *(e.snippet for e in research.evidence),
        ]
    )
    if haystack.strip():
        for pattern, reason in RESTRICTION_PATTERNS:
            if pattern.search(haystack):
                index += 1
                caps.append(f"downgraded one step because {reason}")
                if research.admission_structure != NOT_PUBLISHED:
                    signals.append(f"Admission structure: {research.admission_structure}")
                break
    if research.transfer_restrictions != NOT_PUBLISHED:
        signals.append(f"Transfer restrictions: {research.transfer_restrictions}")

    index = max(0, min(len(LADDER) - 1, index))

    evidence_count = len(research.evidence)
    if major_rate is not None and evidence_count >= 2 and not research.missing_information:
        confidence: Confidence = "High"
    elif known and evidence_count >= 1:
        confidence = "Moderate"
    else:
        confidence = "Low"

    return Baseline(
        classification=LADDER[index],
        confidence=confidence,
        signals=signals,
        caps_applied=caps,
        selectivity_known=known,
        major_level_known=major_rate is not None,
    )


def describe(baseline: Baseline) -> str:
    """Render the baseline for the explaining agent."""
    lines = [
        f"BASELINE CLASSIFICATION (computed deterministically): {baseline.classification}",
        f"BASELINE CONFIDENCE: {baseline.confidence}",
        "Signals used:",
        *[f"  - {s}" for s in baseline.signals],
    ]
    if baseline.caps_applied:
        lines.append("Adjustments applied:")
        lines.extend(f"  - {c}" for c in baseline.caps_applied)
    lines.append(
        "You must EXPLAIN this classification, not change it. Return exactly this "
        "classification and confidence."
    )
    return "\n".join(lines)


Decision = Literal["explain"]
