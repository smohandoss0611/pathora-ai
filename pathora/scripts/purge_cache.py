"""Remove cached federal records so they can be re-fetched.

    python scripts/purge_cache.py                    # all live-lookup records
    python scripts/purge_cache.py --university "Texas A&M University"

A record fetched before a matching bug was fixed stays in the corpus and keeps
being used: it satisfies the selectivity anchor, so the live lookup never runs
again and the bad data never gets replaced. That is how a wrong-campus match
survives a code fix.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

SEED_DIR = ROOT / "data/seed"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--university", default="", help="only this institution")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    removed = 0
    for path in sorted(SEED_DIR.glob("*.colleges.json")):
        payload = json.loads(path.read_text())
        documents = payload.get("documents", [])

        def keep(doc: dict) -> bool:
            """Keep everything except the federal records we are purging."""
            is_federal = str(doc.get("id", "")).startswith("scorecard-")
            targeted = not args.university or doc.get("university") == args.university
            return not (is_federal and targeted)

        kept = [d for d in documents if keep(d)]
        dropped = len(documents) - len(kept)
        if not dropped:
            continue

        for doc in documents:
            if doc not in kept:
                print(f"  {path.name}: {doc.get('university')} -> {doc.get('title')}")
        removed += dropped

        if not args.dry_run:
            payload["documents"] = kept
            if args.university:
                payload["colleges"] = [
                    c for c in payload.get("colleges", []) if c["university"] != args.university
                ]
            path.write_text(json.dumps(payload, indent=2) + "\n")

    if not removed:
        print("No cached federal records found.")
        return 0

    verb = "Would remove" if args.dry_run else "Removed"
    print(f"\n{verb} {removed} record(s). Restart the app; they will be re-fetched.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
