"""Extract admission facts from retrieved document text.

The seeded demo corpus ships pre-structured ``facts`` on every chunk. Real
ingested pages do not — they are prose and tables. Without this module, pointing
the pipeline at real university websites returns "Not officially published" for
everything, because the worker has nowhere to read a fact from.

Every extractor here is a regex over retrieved text and returns the
``evidence_id`` and the literal matched span alongside the value. Nothing is
inferred, completed, or recalled from model memory: if the number is not in the
retrieved text, no value is produced. That is the enforcement point for Section
17, so it is deliberately dumber than an LLM and deliberately auditable.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from pathora.rag.store import RetrievedChunk


@dataclass(frozen=True)
class ExtractedFact:
    field: str
    value: str
    evidence_id: str
    matched_text: str


# --- admit rate -------------------------------------------------------------
ADMIT_RATE = [
    re.compile(r"(?:admit|admission|acceptance)\s+rate[^0-9%]{0,20}(\d{1,2}(?:\.\d)?)\s*%", re.I),
    re.compile(
        r"(?:admitted|accepted)\s+(\d{1,2}(?:\.\d)?)\s*%\s+of\s+(?:applicants|students)", re.I
    ),
    re.compile(r"(\d{1,2}(?:\.\d)?)\s*%\s+(?:admit|admission|acceptance)\s+rate", re.I),
]

# --- score ranges -----------------------------------------------------------
DASH = r"[-–—]"
SAT_RANGE = [
    re.compile(rf"SAT[^0-9]{{0,60}}(1[0-6]\d{{2}})\s*{DASH}\s*(1[0-6]\d{{2}})", re.I),
    re.compile(rf"(1[0-6]\d{{2}})\s*{DASH}\s*(1[0-6]\d{{2}})[^.]{{0,40}}SAT", re.I),
]
ACT_RANGE = [
    re.compile(rf"ACT[^0-9]{{0,60}}([1-3]\d)\s*{DASH}\s*([1-3]\d)", re.I),
    re.compile(rf"([1-3]\d)\s*{DASH}\s*([1-3]\d)[^.]{{0,40}}ACT\s+composite", re.I),
]

# --- policies ---------------------------------------------------------------
TEST_POLICY = [
    (re.compile(r"test[\s-]?blind", re.I), "Test blind"),
    (re.compile(r"test[\s-]?free", re.I), "Test free"),
    (re.compile(r"test[\s-]?optional", re.I), "Test optional"),
    (re.compile(r"(?:SAT|ACT|test)\s+scores?\s+(?:are\s+)?required", re.I), "Test required"),
    (re.compile(r"require[sd]?\s+(?:the\s+)?(?:SAT|ACT)", re.I), "Test required"),
]

ADMISSION_STRUCTURE = [
    (
        re.compile(r"direct\s+admi(?:t|ssion)\s+to\s+(?:the\s+)?major", re.I),
        "Direct admission to major",
    ),
    (re.compile(r"direct\s+admi(?:t|ssion)", re.I), "Direct admission"),
    (re.compile(r"capped\s+major", re.I), "Capped major"),
    (re.compile(r"limited[\s-]access\s+(?:major|program)", re.I), "Limited-access major"),
    (re.compile(r"secondary\s+admission", re.I), "Secondary admission to major"),
    (
        re.compile(r"admitted\s+to\s+the\s+(?:college|university)(?:\s+first)?", re.I),
        "University- or college-level admission",
    ),
]

#: Captured verbatim (no label): the restriction wording itself is the fact.
TRANSFER_RESTRICTION: list[re.Pattern[str]] = [
    re.compile(r"(?:change|transfer)\s+of\s+major[^.]{0,120}", re.I),
    re.compile(r"internal\s+transfer[^.]{0,120}", re.I),
]

MONTHS = "January|February|March|April|May|June|July|August|September|October|November|December"
DEADLINE = re.compile(
    rf"(?:early\s+(?:action|decision)|regular\s+decision|priority|application)\s+"
    rf"(?:deadline\s+)?(?:is\s+)?(?:by\s+)?({MONTHS})\s+(\d{{1,2}})",
    re.I,
)


def _search(patterns: list[re.Pattern[str]], text: str) -> re.Match[str] | None:
    for pattern in patterns:
        if (match := pattern.search(text)) is not None:
            return match
    return None


def extract_from_chunk(chunk: RetrievedChunk) -> list[ExtractedFact]:
    """Pull every admission fact present in one chunk's text."""
    text = f"{chunk.title}\n{chunk.text}"
    found: list[ExtractedFact] = []

    def add(field: str, value: str, match: re.Match[str]) -> None:
        found.append(
            ExtractedFact(
                field=field,
                value=value,
                evidence_id=chunk.id,
                matched_text=match.group(0).strip()[:200],
            )
        )

    if (m := _search(ADMIT_RATE, text)) is not None:
        add("admit_rate", f"{m.group(1)}%", m)

    if (m := _search(SAT_RANGE, text)) is not None:
        low, high = sorted((int(m.group(1)), int(m.group(2))))
        add("sat_range", f"{low}-{high} (as published)", m)

    if (m := _search(ACT_RANGE, text)) is not None:
        low, high = sorted((int(m.group(1)), int(m.group(2))))
        add("act_range", f"{low}-{high} (as published)", m)

    for pattern, label in TEST_POLICY:
        if (m := pattern.search(text)) is not None:
            add("test_policy", label, m)
            break

    for pattern, label in ADMISSION_STRUCTURE:
        if (m := pattern.search(text)) is not None:
            add("admission_structure", label, m)
            break

    for pattern in TRANSFER_RESTRICTION:
        if (m := pattern.search(text)) is not None:
            add("transfer_restrictions", m.group(0).strip()[:200], m)
            break

    if (m := DEADLINE.search(text)) is not None:
        add("deadlines", m.group(0).strip(), m)

    return found


def extract_facts(chunks: list[RetrievedChunk]) -> tuple[dict[str, str], dict[str, str]]:
    """Extract facts across chunks, most authoritative source winning.

    Returns ``(facts, provenance)`` where provenance maps each field to the
    evidence_id it came from, so the Evidence Passport and the Critic can both
    verify that a stated fact traces to a retrieved document.
    """
    from pathora.domain.models import SOURCE_PRIORITY

    facts: dict[str, str] = {}
    provenance: dict[str, str] = {}

    for chunk in sorted(chunks, key=lambda c: (SOURCE_PRIORITY.get(c.source_type, 9), c.id)):
        for fact in extract_from_chunk(chunk):
            if fact.field not in facts:
                facts[fact.field] = fact.value
                provenance[fact.field] = fact.evidence_id

    return facts, provenance
