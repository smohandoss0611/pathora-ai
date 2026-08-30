"""Metrics scored against a completed analysis.

Each returns (passed, detail). They check properties, never exact strings: an
eval that pins prose measures the prompt, not the system.
"""

from __future__ import annotations

import re
from typing import Any

LADDER = ["Safety", "Likely", "Target", "Target-Reach", "Reach", "High Reach"]

#: Language that claims precision the system cannot have.
FABRICATION = re.compile(
    r"\b\d{1,3}\s*%\s*(?:chance|probability|likelihood)|"
    r"\b(?:chance|probability|odds|likelihood)\s+of\s+(?:admission|acceptance|getting)",
    re.I,
)


def _assessments(state: dict[str, Any]) -> dict[str, dict]:
    return state.get("admission_results", {})


def min_colleges(state: dict[str, Any], threshold: int) -> tuple[bool, str]:
    total = len(_assessments(state)) + len(state.get("abstentions", {}))
    return total >= threshold, f"{total} colleges analysed, {threshold} required"


def max_abstention_rate(state: dict[str, Any], threshold: float) -> tuple[bool, str]:
    assessed = len(_assessments(state))
    abstained = len(state.get("abstentions", {}))
    total = assessed + abstained
    if not total:
        return False, "no colleges at all"
    rate = abstained / total
    return rate <= threshold, f"{abstained}/{total} abstained ({rate:.0%}), limit {threshold:.0%}"


def selectivity_spread(state: dict[str, Any], _: Any = None) -> tuple[bool, str]:
    """A list where everything is one label is not a college list."""
    labels = {a["classification"] for a in _assessments(state).values()}
    return len(labels) >= 2, f"labels produced: {sorted(labels) or 'none'}"


def grounded_evidence(state: dict[str, Any], _: Any = None) -> tuple[bool, str]:
    """Every cited evidence id must exist in that college's research."""
    research = state.get("college_research", {})
    orphans: list[str] = []
    for university, assessment in _assessments(state).items():
        known = {e["evidence_id"] for e in research.get(university, {}).get("evidence", [])}
        orphans += [f"{university}:{i}" for i in assessment["evidence_ids"] if i not in known]
    return not orphans, f"untraceable citations: {orphans or 'none'}"


def no_fabricated_probability(state: dict[str, Any], _: Any = None) -> tuple[bool, str]:
    hits: list[str] = []
    for university, assessment in _assessments(state).items():
        blob = " ".join(
            [assessment["rationale_summary"], *assessment["strengths"], *assessment["risks"]]
        )
        if match := FABRICATION.search(blob):
            hits.append(f"{university}: {match.group(0)!r}")
    return not hits, f"probability language: {hits or 'none'}"


def college_specific_reasoning(state: dict[str, Any], _: Any = None) -> tuple[bool, str]:
    """Identical strengths across colleges means profile boilerplate."""
    seen: dict[tuple[str, ...], list[str]] = {}
    for university, assessment in _assessments(state).items():
        seen.setdefault(tuple(sorted(assessment["strengths"])), []).append(university)
    duplicates = [group for group in seen.values() if len(group) > 1]
    return not duplicates, f"shared reasoning: {duplicates or 'none'}"


def names_missing_test_score(state: dict[str, Any], _: Any = None) -> tuple[bool, str]:
    for assessment in _assessments(state).values():
        blob = " ".join(assessment["missing_information"] + assessment["risks"]).lower()
        if "sat" in blob or "act" in blob or "test" in blob:
            return True, "missing test score is named"
    return False, "no assessment mentions the absent test score"


def names_major_rate_gap(state: dict[str, Any], _: Any = None) -> tuple[bool, str]:
    """A university-wide rate must not be passed off as major-specific."""
    research = state.get("college_research", {})
    unflagged: list[str] = []
    for university, assessment in _assessments(state).items():
        result = research.get(university, {})
        if result.get("major_admit_rate", "Not officially published") != "Not officially published":
            continue
        blob = " ".join(
            [assessment["rationale_summary"], *assessment["missing_information"]]
        ).lower()
        if "major" not in blob and "university-wide" not in blob:
            unflagged.append(university)
    return not unflagged, f"unflagged university-wide rates: {unflagged or 'none'}"


def no_unjustified_safety(state: dict[str, Any], _: Any = None) -> tuple[bool, str]:
    """Safety needs either major-level data or a broadly accessible institution."""
    research = state.get("college_research", {})
    bad: list[str] = []
    for university, assessment in _assessments(state).items():
        if assessment["classification"] != "Safety":
            continue
        result = research.get(university, {})
        has_major = result.get("major_admit_rate", "Not officially published") != (
            "Not officially published"
        )
        rate = result.get("admit_rate", "")
        digits = re.search(r"(\d{1,3}(?:\.\d+)?)\s*%", rate)
        accessible = bool(digits) and float(digits.group(1)) >= 85
        if not (has_major or accessible):
            bad.append(f"{university} ({rate or 'no rate'})")
    return not bad, f"unjustified Safety: {bad or 'none'}"


def all_abstain(state: dict[str, Any], _: Any = None) -> tuple[bool, str]:
    assessed = list(_assessments(state))
    return not assessed, f"classified despite no evidence: {assessed or 'none'}"


def abstention_gives_reasons(state: dict[str, Any], _: Any = None) -> tuple[bool, str]:
    silent = [
        university
        for university, record in state.get("abstentions", {}).items()
        if not record.get("what_would_help")
    ]
    return not silent, f"abstentions without a reason: {silent or 'none'}"


def average_ladder_position(state: dict[str, Any]) -> float | None:
    positions = [
        LADDER.index(a["classification"])
        for a in _assessments(state).values()
        if a["classification"] in LADDER
    ]
    return sum(positions) / len(positions) if positions else None


METRICS = {
    "min_colleges": min_colleges,
    "max_abstention_rate": max_abstention_rate,
    "selectivity_spread": selectivity_spread,
    "grounded_evidence": grounded_evidence,
    "no_fabricated_probability": no_fabricated_probability,
    "college_specific_reasoning": college_specific_reasoning,
    "names_missing_test_score": names_missing_test_score,
    "names_major_rate_gap": names_major_rate_gap,
    "no_unjustified_safety": no_unjustified_safety,
    "all_abstain": all_abstain,
    "abstention_gives_reasons": abstention_gives_reasons,
}
