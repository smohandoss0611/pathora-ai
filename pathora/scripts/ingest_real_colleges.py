"""Ingest real university documents from official sources.

    python scripts/ingest_real_colleges.py --dry-run   # show what it would fetch
    python scripts/ingest_real_colleges.py             # fetch, chunk, index

Every URL below was verified to exist as an official institutional page. None of
the *values* on those pages are reproduced here — this script fetches them at
run time so `retrieved_at` is honest and the figures are whatever the university
publishes today, not whatever was true when this file was written.

Source priority follows Section 17: official admissions > official STEM program
> Common Data Set > institutional research.

Two caveats you will hit immediately:

1. **Virginia Tech does not publish its CDS publicly.** Its Analytics &
   Institutional Effectiveness office distributes CDS files by email request
   (aiesupport@vt.edu). There is no URL to ingest. Pathora will correctly report
   "Not officially published" for anything only the CDS would have answered.
   That is the honest result, not a bug — fix it by requesting the file and
   adding it via `--file`.

2. **Major-level admit rates are rarely published anywhere.** Universities
   publish an institution-wide rate. A CS-specific admit rate usually does not
   exist as an official figure, which is exactly the confusion the Critic is
   built to flag.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pathora.rag.ingest import ingest_records  # noqa: E402
from pathora.rag.store import build_store  # noqa: E402

# (university, title, url, source_type, depth)
SOURCES: list[tuple[str, str, str, str, int]] = [
    # --- Texas A&M University -----------------------------------------------
    (
        "Texas A&M University",
        "Texas A&M Undergraduate Catalog: Admission (PDF)",
        "https://catalog.tamu.edu/undergraduate/general-information/admission/admission.pdf",
        "official_admissions",
        1,
    ),
    (
        "Texas A&M University",
        "Texas A&M Freshman Admissions",
        "https://admissions.tamu.edu/apply/freshman/index.html",
        "official_admissions",
        0,
    ),
    (
        "Texas A&M University",
        "Texas A&M College of Engineering Admissions",
        "https://catalog.tamu.edu/undergraduate/general-information/admission/",
        "official_stem_program",
        0,
    ),
    (
        "Texas A&M University",
        "Texas A&M Engineering: Entry to a Major (ETAM)",
        "https://engineering.tamu.edu/academics/entry-to-a-major.html",
        "official_stem_program",
        0,
    ),
    # --- The University of Texas at Dallas ----------------------------------
    (
        "The University of Texas at Dallas",
        "UT Dallas Common Data Set",
        "https://oisds.utdallas.edu/common-data-set",
        "common_data_set",
        1,
    ),
    (
        "The University of Texas at Dallas",
        "UT Dallas Freshman Admission Requirements",
        "https://www.utdallas.edu/admissions/",
        "official_admissions",
        0,
    ),
    (
        "The University of Texas at Dallas",
        "UT Dallas Erik Jonsson School of Engineering and Computer Science",
        "https://engineering.utdallas.edu/",
        "official_stem_program",
        0,
    ),
    # --- Purdue University ---------------------------------------------------
    (
        "Purdue University",
        "Purdue Common Data Set",
        "https://www.purdue.edu/idata/products-services/common-data-set/",
        "common_data_set",
        1,
    ),
    (
        "Purdue University",
        "Purdue Undergraduate Admissions",
        "https://www.admissions.purdue.edu/apply/index.php",
        "official_admissions",
        0,
    ),
    (
        "Purdue University",
        "Purdue College of Engineering First-Year Engineering",
        "https://engineering.purdue.edu/Engr/Academics/Undergraduate",
        "official_stem_program",
        0,
    ),
    # --- The University of Texas at Austin ----------------------------------
    (
        "The University of Texas at Austin",
        "UT Austin Common Data Set",
        "https://reports.utexas.edu/common-data-set",
        "common_data_set",
        1,
    ),
    (
        "The University of Texas at Austin",
        "UT Austin Freshman Admission",
        "https://admissions.utexas.edu/apply/freshman-admission/",
        "official_admissions",
        0,
    ),
    (
        "The University of Texas at Austin",
        "Cockrell School of Engineering Undergraduate Admissions",
        "https://cockrell.utexas.edu/admissions/undergraduate/",
        "official_stem_program",
        0,
    ),
    (
        "The University of Texas at Austin",
        "UT Austin Engineering Admission and Registration (internal transfer rules)",
        "https://catalog.utexas.edu/undergraduate/engineering/admission-and-registration/",
        "official_stem_program",
        1,
    ),
    # --- Texas Tech University ----------------------------------------------
    (
        "Texas Tech University",
        "Texas Tech Common Data Sets",
        "https://www.depts.ttu.edu/irim/CommonDataSets/",
        "common_data_set",
        1,
    ),
    (
        "Texas Tech University",
        "Texas Tech Undergraduate Admissions",
        "https://www.depts.ttu.edu/admissions/apply/freshman/",
        "official_admissions",
        0,
    ),
    # --- University of North Texas ------------------------------------------
    (
        "University of North Texas",
        "UNT Common Data Set",
        "https://institutionalresearch.unt.edu/common-data-set.html",
        "common_data_set",
        1,
    ),
    (
        "University of North Texas",
        "UNT Freshman Admissions",
        "https://admissions.unt.edu/freshman",
        "official_admissions",
        0,
    ),
    # --- University of Houston ----------------------------------------------
    (
        "University of Houston",
        "UH Undergraduate Admissions",
        "https://uh.edu/admissions/apply/freshman/",
        "official_admissions",
        0,
    ),
    # --- The University of Texas at Arlington --------------------------------
    (
        "The University of Texas at Arlington",
        "UTA Freshman Admission",
        "https://www.uta.edu/admissions/apply/undergraduate/freshman",
        "official_admissions",
        0,
    ),
    # --- Prairie View A&M University -----------------------------------------
    (
        "Prairie View A&M University",
        "PVAMU Undergraduate Admissions",
        "https://www.pvamu.edu/admissions/",
        "official_admissions",
        0,
    ),
    # --- Virginia Tech -------------------------------------------------------
    # NOTE: no public CDS. aie.vt.edu/analytics-and-ai/common-data-set.html
    # states files are available by request to aiesupport@vt.edu.
    (
        "Virginia Tech",
        "Virginia Tech Undergraduate Admissions",
        "https://www.admissions.vt.edu/apply.html",
        "official_admissions",
        0,
    ),
    (
        "Virginia Tech",
        "Virginia Tech College of Engineering Undergraduate",
        "https://eng.vt.edu/academics/undergraduate-students.html",
        "official_stem_program",
        0,
    ),
]

CATALOG = [
    {
        "university": "Texas A&M University",
        "state": "TX",
        "control": "Public",
        "size": "Large",
        "majors": [
            "Computer Science",
            "Industrial Engineering",
            "Computer Engineering",
            "Statistics",
        ],
    },
    {
        "university": "The University of Texas at Dallas",
        "state": "TX",
        "control": "Public",
        "size": "Large",
        "majors": [
            "Computer Science",
            "Data Science",
            "Electrical Engineering",
            "Information Science",
        ],
    },
    {
        "university": "Purdue University",
        "state": "IN",
        "control": "Public",
        "size": "Large",
        "majors": [
            "Computer Science",
            "Industrial Engineering",
            "Computer Engineering",
            "Data Science",
        ],
    },
    {
        "university": "The University of Texas at Austin",
        "state": "TX",
        "control": "Public",
        "size": "Large",
        "majors": [
            "Computer Science",
            "Electrical Engineering",
            "Operations Research",
            "Computational Science",
        ],
    },
    {
        "university": "Texas Tech University",
        "state": "TX",
        "control": "Public",
        "size": "Large",
        "majors": ["Computer Science", "Industrial Engineering", "Computer Engineering"],
    },
    {
        "university": "University of North Texas",
        "state": "TX",
        "control": "Public",
        "size": "Large",
        "majors": ["Computer Science", "Data Science", "Information Science"],
    },
    {
        "university": "University of Houston",
        "state": "TX",
        "control": "Public",
        "size": "Large",
        "majors": ["Computer Science", "Industrial Engineering", "Statistics"],
    },
    {
        "university": "The University of Texas at Arlington",
        "state": "TX",
        "control": "Public",
        "size": "Large",
        "majors": ["Computer Science", "Industrial Engineering", "Computer Engineering"],
    },
    {
        "university": "Prairie View A&M University",
        "state": "TX",
        "control": "Public",
        "size": "Medium",
        "majors": ["Computer Science", "Computer Engineering", "Electrical Engineering"],
    },
    {
        "university": "Virginia Tech",
        "state": "VA",
        "control": "Public",
        "size": "Large",
        "majors": [
            "Computer Science",
            "Computer Engineering",
            "Industrial Engineering",
            "Systems Engineering",
        ],
    },
]


#: Sites known to reject automated requests outright. We do not attempt to
#: circumvent them — that is their stated access policy. Obtain the Common Data
#: Set from the institution directly and load it with --file instead.
BLOCKS_AUTOMATION = ("utexas.edu",)


async def fetch(url: str, timeout: float) -> str:
    import httpx

    headers = {"User-Agent": "PathoraAI/0.1 (educational research; contact: you@example.com)"}
    async with httpx.AsyncClient(follow_redirects=True, timeout=timeout, headers=headers) as client:
        response = await client.get(url)
        response.raise_for_status()
        if "pdf" in response.headers.get("content-type", "").lower():
            import pymupdf

            with pymupdf.open(stream=response.content, filetype="pdf") as doc:
                return "\n".join(page.get_text("text") for page in doc)
        return html_to_text(response.text)


def html_to_text(html: str) -> str:
    """Crude tag strip. Swap in trafilatura or readability for production."""
    import re

    html = re.sub(r"<(script|style|nav|footer|header)[^>]*>.*?</\1>", " ", html, flags=re.S | re.I)
    html = re.sub(r"<br\s*/?>|</p>|</div>|</li>|</tr>", "\n", html, flags=re.I)
    html = re.sub(r"<[^>]+>", " ", html)
    html = (
        html.replace("&nbsp;", " ")
        .replace("&amp;", "&")
        .replace("&#8211;", "-")
        .replace("&ndash;", "-")
    )
    return re.sub(r"[ \t]{2,}", " ", html)


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="list sources without fetching")
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "data/seed/real.colleges.json",
        help=(
            "Write the fetched corpus here. Files named *.colleges.json in data/seed "
            "are loaded automatically on startup, which is how the running app sees "
            "these schools. Pass --out '' to skip."
        ),
    )
    args = parser.parse_args()

    if args.dry_run:
        for university, _title, url, source_type, depth in SOURCES:
            print(f"[depth {depth}] {source_type:22s} {university:35s} {url}")
        print(f"\n{len(SOURCES)} sources across {len({s[0] for s in SOURCES})} universities")
        print("Virginia Tech has no public CDS; request it from aiesupport@vt.edu.")
        return

    records, failures = [], []
    for university, title, url, source_type, depth in SOURCES:
        try:
            text = await fetch(url, args.timeout)
        except Exception as exc:  # noqa: BLE001
            failures.append((url, str(exc)))
            if "403" in str(exc) and any(h in url for h in BLOCKS_AUTOMATION):
                print(
                    f"BLOCKED {url}\n"
                    f"        The site rejects automated requests. That is its access\n"
                    f"        policy, not a bug to work around: request the Common Data\n"
                    f"        Set from the institution and load the file directly."
                )
            else:
                print(f"FAILED  {url}\n        {str(exc).splitlines()[0]}")
            continue

        if len(text.strip()) < 400:
            failures.append((url, "page returned too little text (JS-rendered?)"))
            print(f"THIN    {url} — {len(text.strip())} chars, likely JS-rendered")
            continue

        records.append(
            {
                "id": f"{university.lower().replace(' ', '-')}-{source_type}-{depth}",
                "university": university,
                "title": title,
                "text": text,
                "source_url": url,
                "source_type": source_type,
                "depth": depth,
            }
        )
        print(f"OK      {university:35s} {len(text):>7,} chars  {url}")

    if not records:
        print("\nNothing ingested.")
        return

    count = await ingest_records(records, build_store())
    print(f"\nIndexed {count} chunks from {len(records)} documents ({len(failures)} failed).")

    # With VECTOR_BACKEND=memory the index above lives and dies with THIS
    # process. Writing a corpus file is what makes the schools visible to the
    # API and UI, because they rebuild their store from data/seed on startup.
    if str(args.out):
        fetched = {r["university"] for r in records}
        payload = {
            "_source": "Fetched from official university pages by ingest_real_colleges.py",
            "_retrieved_at": datetime.now(UTC).isoformat(),
            "colleges": [c for c in CATALOG if c["university"] in fetched],
            "documents": records,
        }
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(payload, indent=2) + "\n")
        print(f"Wrote {args.out} ({len(payload['colleges'])} colleges, {len(records)} documents).")
        print("Restart the API/UI to pick it up (Streamlit: 'Start over / reload code').")


if __name__ == "__main__":
    asyncio.run(main())
