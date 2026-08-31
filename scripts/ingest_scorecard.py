"""Ingest from the U.S. Dept. of Education College Scorecard API.

Preferred over manual IPEDS CSV downloads for Pathora: same federal source data,
but a queryable JSON API with per-institution filters, so refreshing a college
list is one command instead of a download-and-join.

    export SCORECARD_API_KEY=...        # free at https://api.data.gov/signup/
    python scripts/ingest_scorecard.py --state TX --min-size 5000
    python scripts/ingest_scorecard.py --names "Texas A&M,Purdue,Virginia Tech"

Two limitations worth stating plainly, because they shape what Pathora can
honestly claim afterwards:

1. `admission_rate.overall` is **university-wide**. It is not the admit rate for
   Computer Science, and at most large publics the major-level rate is far
   lower. The Critic flags any assessment that lets one stand in for the other,
   so expect Low/Moderate confidence from Scorecard data alone.
2. The `latest.*` fields are federal reporting on a lag, typically one to two
   cycles behind. That is why annual surveys have their own freshness window.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pathora.rag.scorecard import (  # noqa: E402
    API,
    FIELDS,
    SITE,
    build_record,
    catalog_entry,
    size_band,
)

__all__ = ["build_record", "catalog_entry", "size_band"]


async def fetch_page(client: Any, params: dict[str, Any]) -> dict[str, Any]:
    response = await client.get(API, params=params)
    if response.status_code == 403:
        raise RuntimeError("403 from Scorecard: check SCORECARD_API_KEY")
    if response.status_code >= 400:
        raise RuntimeError(f"{response.status_code} from Scorecard: {response.text[:200]}")
    return response.json()


async def collect(args: argparse.Namespace, key: str) -> list[dict[str, Any]]:
    import httpx

    base: dict[str, Any] = {
        "api_key": key,
        "fields": FIELDS,
        "per_page": 100,
        "latest.student.size__range": f"{args.min_size}..",
        "school.degrees_awarded.predominant__range": "3..4",
    }
    if args.state:
        base["school.state"] = args.state

    rows: list[dict[str, Any]] = []
    async with httpx.AsyncClient(timeout=args.timeout) as client:
        if args.names:
            for name in [n.strip() for n in args.names.split(",") if n.strip()]:
                payload = await fetch_page(client, {**base, "school.name": name})
                rows.extend(payload.get("results", []))
                print(f"  {name:40s} {len(payload.get('results', []))} match(es)")
        else:
            page = 0
            while True:
                payload = await fetch_page(client, {**base, "page": page})
                results = payload.get("results", [])
                rows.extend(results)
                total = payload.get("metadata", {}).get("total", len(rows))
                print(f"  page {page}: {len(results)} rows ({len(rows)}/{total})")
                page += 1
                if not results or len(rows) >= total or page > args.max_pages:
                    break
    return rows


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state", default="", help="two-letter state code, e.g. TX")
    parser.add_argument("--names", default="", help="comma-separated institution names")
    parser.add_argument("--min-size", type=int, default=2000)
    parser.add_argument("--max-pages", type=int, default=5)
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--year", default=str(datetime.now(UTC).year))
    parser.add_argument("--out", type=Path, default=ROOT / "data/seed/scorecard.colleges.json")
    args = parser.parse_args()

    key = os.environ.get("SCORECARD_API_KEY", "")
    if not key:
        print("SCORECARD_API_KEY is not set. Get a free key at https://api.data.gov/signup/")
        return 1

    try:
        rows = await collect(args, key)
    except Exception as exc:  # noqa: BLE001
        print(f"failed: {exc}")
        return 1

    documents, colleges, skipped = [], [], 0
    for row in rows:
        record = build_record(row, args.year)
        if record is None:
            skipped += 1
            continue
        documents.append(record)
        colleges.append(catalog_entry(row))

    if not documents:
        print("\nNo institutions with reported admissions figures matched.")
        return 1

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(
            {
                "_source": f"College Scorecard API, retrieved {args.year}. {SITE}",
                "_retrieved_at": datetime.now(UTC).isoformat(),
                "colleges": colleges,
                "documents": documents,
            },
            indent=2,
        )
        + "\n"
    )
    print(f"\nWrote {args.out}: {len(colleges)} colleges ({skipped} had no reported figures).")
    print("Restart the app to load it.")
    print(
        "\nNote: admission_rate.overall is university-wide. The Critic will flag any "
        "assessment that treats it as major-specific, so expect Low/Moderate confidence "
        "until program-level sources are added."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
