"""Ingest IPEDS admissions data — the fix for un-scrapeable admit rates.

Scraping admissions websites does not work for the numbers that matter. Those
pages are marketing; they say "test optional" and "we review holistically", not
"we admitted 26.4% of 68,000 applicants". Meanwhile some institutions (UT Austin
among them) reject automated requests outright, which is their access policy and
not something to work around.

IPEDS solves both problems. Every institution receiving federal Title IV funding
is *required* to report admissions counts and test-score ranges annually, so the
data is authoritative, complete, consistent across institutions, and free to
download in bulk. No scraping, no blocks, no per-school URL hunting.

Usage — download the two survey files first (they are ZIP archives of CSVs):

    https://nces.ed.gov/ipeds/datacenter/DataFiles.aspx
      * ADM   "Admissions and Test Scores"      -> adm2023.csv
      * HD    "Institutional Characteristics"   -> hd2023.csv

    python scripts/ingest_ipeds.py --adm adm2023.csv --hd hd2023.csv \\
        --only "Texas A&M University,The University of Texas at Austin"

Admit rate is computed as ADMSSN / APPLCN, exactly the definition institutions
report against. Nothing is inferred: a school missing either count produces no
admit rate rather than an estimate.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

IPEDS_SOURCE = "https://nces.ed.gov/ipeds/datacenter/DataFiles.aspx"

#: ADMCON7 — institutional policy on admission test scores.
TEST_POLICY = {
    "1": "Required for all or some first-time applicants",
    "2": "Recommended",
    "3": "Neither required nor recommended (test optional)",
    "5": "Considered but not required",
}

CONTROL = {"1": "Public", "2": "Private", "3": "Private"}


def _int(value: str | None) -> int | None:
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        return None


def _read_csv(path: Path) -> list[dict[str, str]]:
    # IPEDS files ship as latin-1 with a BOM on some years.
    with path.open(encoding="latin-1", newline="") as handle:
        return list(csv.DictReader(handle))


def _range(low: str | None, high: str | None, label: str) -> str | None:
    lo, hi = _int(low), _int(high)
    if lo is None or hi is None:
        return None
    return f"{lo}-{hi} ({label} 25th-75th percentile, IPEDS)"


def build_record(row: dict[str, str], institution: dict[str, str], year: str) -> dict[str, Any]:
    """Turn one IPEDS admissions row into an ingestible document."""
    name = institution.get("INSTNM", "").strip()
    applied, admitted = _int(row.get("APPLCN")), _int(row.get("ADMSSN"))

    facts: dict[str, Any] = {}
    sentences = [f"IPEDS Admissions and Test Scores, survey year {year}, for {name}."]

    if applied and admitted:
        rate = round(admitted / applied * 100, 1)
        facts["admit_rate"] = f"{rate}%"
        sentences.append(f"Admit rate {rate}% ({admitted:,} admitted of {applied:,} applicants).")

    sat = _range(row.get("SATMT25"), row.get("SATMT75"), "SAT Math")
    sat_verbal = _range(row.get("SATVR25"), row.get("SATVR75"), "SAT EBRW")
    act = _range(row.get("ACTCM25"), row.get("ACTCM75"), "ACT Composite")

    # IPEDS reports SAT sections separately; the combined middle-50 band is the
    # sum of the section bands, which is how institutions publish it.
    lo_m, hi_m = _int(row.get("SATMT25")), _int(row.get("SATMT75"))
    lo_v, hi_v = _int(row.get("SATVR25")), _int(row.get("SATVR75"))
    if None not in (lo_m, hi_m, lo_v, hi_v):
        facts["sat_range"] = f"{lo_m + lo_v}-{hi_m + hi_v} (IPEDS section 25th-75th, combined)"
        sentences.append(f"SAT {facts['sat_range']}.")
    if act:
        facts["act_range"] = act
        sentences.append(f"ACT {act}.")
    for extra in (sat, sat_verbal):
        if extra:
            sentences.append(extra + ".")

    if policy := TEST_POLICY.get(str(row.get("ADMCON7", "")).strip()):
        facts["test_policy"] = policy
        sentences.append(f"Test score policy: {policy}.")

    return {
        "id": f"ipeds-{row.get('UNITID')}-{year}",
        "university": name,
        "title": f"IPEDS Admissions and Test Scores {year} — {name}",
        "text": " ".join(sentences),
        "source_url": IPEDS_SOURCE,
        "source_type": "institutional_research",
        "published_at": f"{year}-01-01T00:00:00+00:00",
        "depth": 0,
        "facts": facts,
    }


def catalog_entry(institution: dict[str, str], majors: list[str]) -> dict[str, Any]:
    return {
        "university": institution.get("INSTNM", "").strip(),
        "state": institution.get("STABBR", "").strip(),
        "control": CONTROL.get(str(institution.get("CONTROL", "")).strip(), "Public"),
        "size": "Large",
        "majors": majors,
    }


DEFAULT_MAJORS = [
    "Computer Science",
    "Data Science",
    "Industrial Engineering",
    "Computer Engineering",
    "Electrical Engineering",
    "Statistics",
]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adm", type=Path, required=True, help="IPEDS ADM csv")
    parser.add_argument("--hd", type=Path, required=True, help="IPEDS HD (directory) csv")
    parser.add_argument("--year", default="2023")
    parser.add_argument(
        "--only",
        default="",
        help="comma-separated institution names (substring match, case-insensitive)",
    )
    parser.add_argument("--state", default="", help="restrict to a state abbreviation, e.g. TX")
    parser.add_argument("--out", type=Path, default=ROOT / "data/seed/ipeds.colleges.json")
    args = parser.parse_args()

    for path in (args.adm, args.hd):
        if not path.exists():
            print(f"missing file: {path}\nDownload both from {IPEDS_SOURCE}")
            return 1

    directory = {row["UNITID"]: row for row in _read_csv(args.hd)}
    wanted = [n.strip().lower() for n in args.only.split(",") if n.strip()]

    documents, colleges = [], []
    for row in _read_csv(args.adm):
        institution = directory.get(row.get("UNITID", ""))
        if institution is None:
            continue

        name = institution.get("INSTNM", "").strip()
        if wanted and not any(w in name.lower() for w in wanted):
            continue
        if args.state and institution.get("STABBR", "").strip().upper() != args.state.upper():
            continue

        record = build_record(row, institution, args.year)
        if not record["facts"]:
            print(f"skip (no reported figures): {name}")
            continue

        documents.append(record)
        colleges.append(catalog_entry(institution, DEFAULT_MAJORS))
        print(f"OK   {name:52s} {record['facts'].get('admit_rate', 'no admit rate')}")

    if not documents:
        print("\nNothing matched. Check --only spelling against INSTNM in the HD file.")
        return 1

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(
            {
                "_source": f"IPEDS survey year {args.year}, {IPEDS_SOURCE}",
                "_retrieved_at": datetime.now(UTC).isoformat(),
                "colleges": colleges,
                "documents": documents,
            },
            indent=2,
        )
        + "\n"
    )
    print(f"\nWrote {args.out}: {len(colleges)} colleges, {len(documents)} documents.")
    print("Restart the app to load it.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
