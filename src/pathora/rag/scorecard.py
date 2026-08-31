"""College Scorecard client.

Lives in the package (not the script) because two callers need it: the bulk
ingestion CLI, and the on-demand lookup that runs when a college is researched
that nobody ingested in advance.

A note on "realtime": admissions figures are annual federal reporting. There is
no per-second truth to fetch, and a source claiming otherwise would be inventing
precision. What this makes live is *coverage* — a school named by discovery gets
looked up on demand instead of abstaining — not the recency of the numbers,
which remain whatever the Department of Education last published.
"""

from __future__ import annotations

import logging
import re
from datetime import UTC, datetime
from typing import Any

from pathora.config import Settings, get_settings

API = "https://api.data.gov/ed/collegescorecard/v1/schools"
SITE = "https://collegescorecard.ed.gov/"

FIELDS = ",".join(
    [
        "id",
        "school.name",
        "school.state",
        "school.ownership",
        "school.school_url",
        "latest.student.size",
        "latest.admissions.admission_rate.overall",
        "latest.admissions.sat_scores.25th_percentile.critical_reading",
        "latest.admissions.sat_scores.75th_percentile.critical_reading",
        "latest.admissions.sat_scores.25th_percentile.math",
        "latest.admissions.sat_scores.75th_percentile.math",
        "latest.admissions.act_scores.25th_percentile.cumulative",
        "latest.admissions.act_scores.75th_percentile.cumulative",
    ]
)

log = logging.getLogger(__name__)

OWNERSHIP = {1: "Public", 2: "Private", 3: "Private"}

DEFAULT_MAJORS = [
    "Computer Science",
    "Data Science",
    "Industrial Engineering",
    "Computer Engineering",
    "Electrical Engineering",
    "Statistics",
]


def get_field(row: dict[str, Any], key: str) -> Any:
    """Scorecard returns flat dotted keys; some clients nest them."""
    if key in row:
        return row[key]
    node: Any = row
    for part in key.split("."):
        if not isinstance(node, dict) or part not in node:
            return None
        node = node[part]
    return node


def size_band(size: int | None) -> str:
    if size is None:
        return "NoPreference"
    if size >= 15000:
        return "Large"
    if size >= 5000:
        return "Medium"
    return "Small"


def build_record(row: dict[str, Any], year_label: str) -> dict[str, Any] | None:
    name = get_field(row, "school.name")
    if not name:
        return None

    facts: dict[str, Any] = {}
    sentences = [f"College Scorecard (U.S. Department of Education) record for {name}."]

    rate = get_field(row, "latest.admissions.admission_rate.overall")
    # Scorecard reports 0 for "not reported", which must not read as "admits nobody".
    if isinstance(rate, int | float) and rate > 0:
        percent = round(rate * 100, 1)
        facts["admit_rate"] = f"{percent}% (university-wide)"
        sentences.append(
            f"Overall admission rate {percent}%. This is a university-wide figure and "
            f"is not the admit rate for any individual major."
        )

    read_lo = get_field(row, "latest.admissions.sat_scores.25th_percentile.critical_reading")
    read_hi = get_field(row, "latest.admissions.sat_scores.75th_percentile.critical_reading")
    math_lo = get_field(row, "latest.admissions.sat_scores.25th_percentile.math")
    math_hi = get_field(row, "latest.admissions.sat_scores.75th_percentile.math")
    if None not in (read_lo, read_hi, math_lo, math_hi):
        facts["sat_range"] = f"{read_lo + math_lo}-{read_hi + math_hi} (25th-75th percentile)"
        sentences.append(f"SAT {facts['sat_range']}.")

    act_lo = get_field(row, "latest.admissions.act_scores.25th_percentile.cumulative")
    act_hi = get_field(row, "latest.admissions.act_scores.75th_percentile.cumulative")
    if None not in (act_lo, act_hi):
        facts["act_range"] = f"{act_lo}-{act_hi} (25th-75th percentile)"
        sentences.append(f"ACT {facts['act_range']}.")

    if not facts:
        return None

    return {
        "id": f"scorecard-{get_field(row, 'id')}",
        "university": name,
        "title": f"College Scorecard — {name}",
        "text": " ".join(sentences),
        "source_url": get_field(row, "school.school_url") or SITE,
        "source_type": "institutional_research",
        "published_at": f"{year_label}-01-01T00:00:00+00:00",
        "depth": 0,
        "facts": facts,
    }


def catalog_entry(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "university": get_field(row, "school.name"),
        "state": get_field(row, "school.state") or "",
        "control": OWNERSHIP.get(get_field(row, "school.ownership"), "Public"),
        "size": size_band(get_field(row, "latest.student.size")),
        "majors": DEFAULT_MAJORS,
    }


async def query(
    params: dict[str, Any], *, settings: Settings | None = None, timeout: float | None = None
) -> list[dict[str, Any]]:
    """One Scorecard request. Raises with the response body on error."""
    import httpx

    settings = settings or get_settings()
    if not settings.scorecard_api_key:
        raise RuntimeError("SCORECARD_API_KEY is not set (free at https://api.data.gov/signup/)")

    async with httpx.AsyncClient(timeout=timeout or settings.live_lookup_timeout) as client:
        response = await client.get(
            API,
            params={
                "api_key": settings.scorecard_api_key,
                "fields": FIELDS,
                # A university with many campuses can fill a small page with
                # branches before the flagship appears.
                "per_page": 100,
                **params,
            },
        )
        if response.status_code == 403:
            raise RuntimeError("403 from Scorecard: check SCORECARD_API_KEY")
        if response.status_code >= 400:
            raise RuntimeError(f"{response.status_code} from Scorecard: {response.text[:200]}")
        return response.json().get("results", [])


#: Words that differ between how people name a university and how the federal
#: register does, and which must not block a match.
NOISE_WORDS = {"the", "university", "of", "at", "college", "main", "campus"}


def name_tokens(name: str) -> list[str]:
    cleaned = "".join(ch.lower() if ch.isalnum() else " " for ch in name).split()
    return [t for t in cleaned if t not in NOISE_WORDS]


def match_score(query_name: str, candidate: str) -> float:
    """Score a Scorecard institution name against what the user asked for.

    Federal records carry campus qualifiers people never type: Texas A&M is
    registered as "Texas A&M University-College Station", so an exact-string
    query returns nothing at all. Scoring candidates fixes that without
    accepting a different institution — Prairie View A&M and Texas A&M
    International must not match a request for Texas A&M.
    """
    wanted, found = name_tokens(query_name), name_tokens(candidate)
    if not wanted or not found:
        return 0.0
    if wanted == found:
        return 1.0
    # Score both directions. Extra tokens in the QUERY are usually the caller
    # qualifying a name ("St. Mary's University, Texas" for a school registered
    # without the state); extra tokens in the CANDIDATE mean a different campus.
    # Only the latter should be penalised.
    shared = set(wanted) & set(found)
    overlap = max(len(shared) / len(set(wanted)), len(shared) / len(set(found)))
    # Extra tokens are penalised so the flagship outranks a branch campus, but
    # only mildly: "college station" is a qualifier, not a different school.
    excess = len(set(found) - set(wanted))
    # Note: every Texas A&M campus scores identically here — Commerce,
    # Kingsville and College Station are all equally valid readings of "Texas
    # A&M University". No name-based rule can separate them, so the caller
    # breaks the tie on enrollment and records which institution it chose.
    return max(0.0, overlap - 0.08 * excess)


async def lookup_by_name(
    university: str, *, settings: Settings | None = None
) -> dict[str, Any] | None:
    """Find one institution by name and return an ingestible record."""
    settings = settings or get_settings()
    year = str(datetime.now(UTC).year)

    # Several passes, narrowing as they go. Scorecard returns HTTP 500 — not a
    # 4xx — for names containing punctuation such as "St. Mary's University,
    # Texas", so each attempt is isolated: one bad query must not abort the
    # remaining ones.
    attempts: list[str] = [university]
    depunctuated = re.sub(r"[^\w\s&-]", " ", university)
    depunctuated = re.sub(r"\s{2,}", " ", depunctuated).strip()
    if depunctuated and depunctuated.lower() != university.lower():
        attempts.append(depunctuated)
    core = " ".join(name_tokens(university))
    if core and core.lower() not in {a.lower() for a in attempts}:
        attempts.append(core)

    rows: list[dict[str, Any]] = []
    for attempt in attempts:
        try:
            rows = await query({"school.name": attempt}, settings=settings)
        except RuntimeError as exc:
            # A credential problem is a configuration error and must surface;
            # anything else is this one query failing, so try the next form.
            if "403" in str(exc) or "API_KEY" in str(exc):
                raise
            log.info("scorecard query %r failed (%s); trying a simpler form", attempt, exc)
            continue
        if rows:
            break

    if not rows:
        return None

    # Campus qualifiers score identically — "Texas A&M University-College
    # Station", "-Galveston" and "Texas A&M International University" all differ
    # from the query by exactly one token by exactly one token. Enrollment breaks the tie: when a
    # student names a university without qualifying it, they mean the flagship.
    ranked = sorted(
        (
            (
                match_score(university, str(get_field(r, "school.name"))),
                get_field(r, "latest.student.size") or 0,
                r,
            )
            for r in rows
        ),
        key=lambda triple: (triple[0], triple[1]),
        reverse=True,
    )
    best_score, best_size, best_row = ranked[0]

    # A weak lone match is worse than no match. "Texas A&M University-San
    # Antonio" scores 0.84 against a request for "Texas A&M University" — a
    # different institution with an 840-1070 SAT band, which is not a small
    # error to hand a student. Below the strict bar we abstain and say why,
    # rather than classify from the wrong campus.
    if best_score < settings.scorecard_match_threshold:
        log.warning(
            "no confident federal match for %r (best: %r at %.2f, threshold %.2f)",
            university,
            get_field(best_row, "school.name"),
            best_score,
            settings.scorecard_match_threshold,
        )
        return None

    tied = [r for r in ranked if abs(r[0] - best_score) < 0.01]
    if len(tied) > 1:
        log.info(
            "%s matched %d equally-named institutions; chose %r (enrollment %s). Others: %s",
            university,
            len(tied),
            get_field(best_row, "school.name"),
            best_size or "unreported",
            [get_field(r[2], "school.name") for r in tied[1:5]],
        )

    record = build_record(best_row, year)
    if record is not None:
        # Index under the name the caller used, since retrieval filters on it,
        # while keeping the official registration visible in the evidence.
        official = str(get_field(best_row, "school.name"))
        record["university"] = university
        if official.lower() != university.lower():
            record["title"] = f"College Scorecard — {official}"
            record["text"] = f"{record['text']} Registered federally as {official}."
    return record
