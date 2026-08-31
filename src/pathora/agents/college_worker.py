"""One reusable college research worker (Section 16).

There is deliberately no PurdueAgent / VirginiaTechAgent. The graph fans this
single worker out over the candidate list with bounded concurrency.

Facts are copied from retrieved evidence only. Anything not present in retrieved
official material stays "Not officially published" — the worker never asks a
model to fill a gap, which is how Section 17 is actually enforced rather than
merely requested in a prompt.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from pathora.agents.fact_extractor import extract_facts
from pathora.config import Settings, get_settings
from pathora.domain.models import (
    NOT_PUBLISHED,
    SOURCE_PRIORITY,
    CollegeCandidate,
    CollegeResearchResult,
    EvidenceRecord,
)
from pathora.rag.store import RetrievedChunk, VectorStore

log = logging.getLogger(__name__)

#: university -> (expires_at, fetched) so repeated researches of the same
#: college inside a session do not re-hit the API.
_LOOKUP_CACHE: dict[str, tuple[float, bool]] = {}


def _needs_selectivity_anchor(chunks: Any) -> bool:
    """True when nothing retrieved can anchor a classification.

    A program page describing admission structure is valuable evidence, but it
    carries no admit rate. Without one, the evidence gate refuses — so the
    federal lookup must still run even though documents exist.
    """
    from pathora.agents.fact_extractor import extract_facts

    chunks = list(chunks)
    if not chunks:
        return True

    for chunk in chunks:
        if any(chunk.facts.get(field) for field in ANCHOR_FIELDS):
            return False

    extracted, _ = extract_facts(chunks)
    return not any(extracted.get(field) for field in ANCHOR_FIELDS)


async def live_lookup(
    university: str, store: VectorStore, settings: Settings
) -> list[RetrievedChunk]:
    """Fetch and index one college on demand. Returns the retrieved chunks."""
    import time

    cached = _LOOKUP_CACHE.get(university.lower())
    if cached and cached[0] > time.time() and not cached[1]:
        return []  # looked up recently and found nothing; do not retry

    try:
        from pathora.rag.ingest import ingest_records
        from pathora.rag.scorecard import lookup_by_name

        record = await lookup_by_name(university, settings=settings)
    except Exception as exc:  # noqa: BLE001 - a failed lookup must not fail research
        log.warning("live lookup failed for %s: %s", university, exc)
        _LOOKUP_CACHE[university.lower()] = (
            time.time() + settings.live_lookup_ttl_seconds,
            False,
        )
        return []

    _LOOKUP_CACHE[university.lower()] = (
        time.time() + settings.live_lookup_ttl_seconds,
        record is not None,
    )
    if record is None:
        log.info("live lookup found no federal record for %s", university)
        return []

    await ingest_records([record], store)
    log.info("live lookup indexed %s", university)
    return await store.query(
        f"{university} admission requirements admit rate",
        university=university,
        top_k=settings.rag_top_k,
        max_depth=1,
    )


FACT_FIELDS = (
    "admit_rate",
    "major_admit_rate",
    "sat_range",
    "act_range",
    "test_policy",
    "admission_structure",
    "deadlines",
    "transfer_restrictions",
)

REQUIRED_FOR_ASSESSMENT = ("admit_rate", "test_policy", "admission_structure")

#: Facts that can anchor a fit classification. Without one of these the gate
#: refuses regardless of how much other evidence exists.
ANCHOR_FIELDS = ("admit_rate", "major_admit_rate")


def _queries(college: CollegeCandidate, deep: bool) -> list[str]:
    base = [
        f"{college.university} first-year admission requirements admit rate",
        f"{college.university} {college.target_major} program admission structure",
    ]
    if deep:
        base += [
            f"{college.university} common data set admission section C",
            f"{college.university} {college.target_major} major admit rate transfer restrictions",
            f"{college.university} testing policy application deadlines",
        ]
    return base


def _merge_facts(chunks: list[RetrievedChunk]) -> tuple[dict[str, Any], dict[str, str]]:
    """Merge facts, most authoritative source wins. Returns (facts, provenance)."""
    facts: dict[str, Any] = {}
    provenance: dict[str, str] = {}
    ranked = sorted(chunks, key=lambda c: (SOURCE_PRIORITY.get(c.source_type, 9), c.id))
    for chunk in ranked:
        for key, value in chunk.facts.items():
            if key in FACT_FIELDS and value and key not in facts:
                facts[key] = value
                provenance[key] = chunk.id
    return facts, provenance


async def college_research_worker(
    college: CollegeCandidate,
    store: VectorStore,
    *,
    deep: bool = False,
    settings: Settings | None = None,
) -> CollegeResearchResult:
    settings = settings or get_settings()
    top_k = settings.rag_top_k * (2 if deep else 1)
    max_depth = 1 if deep else 0

    try:
        batches = await asyncio.gather(
            *[
                store.query(query, university=college.university, top_k=top_k, max_depth=max_depth)
                for query in _queries(college, deep)
            ]
        )
    except Exception as exc:  # pragma: no cover - defensive
        log.exception("research failed for %s", college.university)
        return CollegeResearchResult(
            university=college.university,
            target_major=college.target_major,
            research_error=str(exc),
            missing_information=[f"Research failed: {exc}"],
        )

    seen: dict[str, RetrievedChunk] = {}
    for batch in batches:
        for chunk in batch:
            existing = seen.get(chunk.id)
            if existing is None or chunk.score > existing.score:
                seen[chunk.id] = chunk

    # Fetch on demand when the corpus cannot anchor a classification. Two cases
    # matter, and only checking the first was a bug: indexing a program page
    # with no admit rate made `seen` non-empty, suppressed the federal lookup,
    # and turned a college that previously classified into an abstention.
    if settings.live_lookup_enabled and _needs_selectivity_anchor(seen.values()):
        for chunk in await live_lookup(college.university, store, settings):
            seen[chunk.id] = chunk

    chunks = sorted(seen.values(), key=lambda c: (-c.score, c.id))

    facts, provenance = _merge_facts(chunks)

    # Seeded corpora carry structured `facts`; real ingested pages do not.
    # Extract whatever the retrieved *text* actually states, without overriding
    # a fact that already came from a more authoritative structured source.
    extracted, extracted_provenance = extract_facts(chunks)
    for field, value in extracted.items():
        if field not in facts:
            facts[field] = value
            provenance[field] = extracted_provenance[field]
    evidence = [
        EvidenceRecord(
            evidence_id=chunk.id,
            university=chunk.university,
            source_url=chunk.source_url,
            source_type=chunk.source_type,
            title=chunk.title,
            snippet=chunk.text[:280],
            published_at=chunk.published_at,
            retrieved_at=chunk.retrieved_at,
        )
        for chunk in chunks
    ]

    missing = [
        f"{field.replace('_', ' ').title()} not found in retrieved official sources"
        for field in REQUIRED_FOR_ASSESSMENT
        if field not in facts
    ]
    if not evidence:
        missing.append("No official documents were retrieved for this university")

    return CollegeResearchResult(
        university=college.university,
        target_major=college.target_major,
        admit_rate=facts.get("admit_rate", NOT_PUBLISHED),
        major_admit_rate=facts.get("major_admit_rate", NOT_PUBLISHED),
        sat_range=facts.get("sat_range", NOT_PUBLISHED),
        act_range=facts.get("act_range", NOT_PUBLISHED),
        test_policy=facts.get("test_policy", NOT_PUBLISHED),
        admission_structure=facts.get("admission_structure", NOT_PUBLISHED),
        deadlines=facts.get("deadlines", NOT_PUBLISHED),
        transfer_restrictions=facts.get("transfer_restrictions", NOT_PUBLISHED),
        program_notes=[f"{c.title}: {c.text[:160]}" for c in chunks[:3]],
        evidence=evidence,
        fact_sources=provenance,
        missing_information=missing,
    )


async def research_many(
    colleges: list[CollegeCandidate],
    store: VectorStore,
    *,
    deep: bool = False,
    settings: Settings | None = None,
) -> dict[str, CollegeResearchResult]:
    """Dynamic fan-out with a bounded worker pool, then fan-in."""
    settings = settings or get_settings()
    semaphore = asyncio.Semaphore(max(1, settings.max_parallel_college_workers))

    async def run(college: CollegeCandidate) -> CollegeResearchResult:
        async with semaphore:
            return await college_research_worker(college, store, deep=deep, settings=settings)

    results = await asyncio.gather(*[run(c) for c in colleges], return_exceptions=True)

    fanned_in: dict[str, CollegeResearchResult] = {}
    for college, result in zip(colleges, results, strict=True):
        if isinstance(result, BaseException):
            fanned_in[college.university] = CollegeResearchResult(
                university=college.university,
                target_major=college.target_major,
                research_error=str(result),
                missing_information=[f"Research failed: {result}"],
            )
        else:
            fanned_in[college.university] = result
    return fanned_in
