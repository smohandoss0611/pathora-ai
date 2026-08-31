"""Print exactly which corpus files the running app will load.

    python scripts/diagnose_corpus.py

If a university you ingested is not in this list, the app cannot show it —
the problem is the corpus file, not the UI.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pathora.rag.store import SEED_DIR, corpus_files, load_seed_payload  # noqa: E402


def main() -> None:
    print(f"seed dir : {SEED_DIR}")
    print(f"exists   : {SEED_DIR.exists()}\n")

    print("files on disk:")
    for file in sorted(SEED_DIR.glob("*.json")):
        print(f"   {file.name:30s} {file.stat().st_size:>9,} bytes")

    picked = [f.name for f in corpus_files()]
    print(f"\nloader picks up: {picked}")
    if "real.colleges.json" not in picked:
        print("   -> real.colleges.json is NOT being loaded.")
        print("      Run: python scripts/ingest_real_colleges.py   (without --dry-run)")

    payload = load_seed_payload()
    print(f"\nmerged: {len(payload['colleges'])} colleges, {len(payload['documents'])} documents")
    for college in payload["colleges"]:
        print(f"   - {college['university']}")


if __name__ == "__main__":
    main()
