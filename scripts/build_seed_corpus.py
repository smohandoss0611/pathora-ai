"""Build the synthetic demo corpus.

IMPORTANT: every institution here is FICTIONAL and every number is invented for
demonstration. Section 17 forbids fabricating statistics about real universities,
so the demo corpus deliberately uses institutions that do not exist. Point the
ingestion pipeline (`pathora.rag.ingest`) at real official documents to research
real schools.
"""

from __future__ import annotations

import json
from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "data/seed/colleges.json"

COLLEGES = [
    {
        "university": "Lakeside State University",
        "state": "TX",
        "control": "Public",
        "size": "Large",
        "majors": ["Computer Science", "Data Science", "Industrial Engineering", "Statistics"],
        "admit_rate": "58%",
        "major_admit_rate": "31% (Computer Science, first-year direct admit)",
        "sat_range": "1230-1420 (middle 50%)",
        "act_range": "27-32 (middle 50%)",
        "test_policy": "Test optional through the 2027 entering class",
        "admission_structure": "Direct admission to major for Computer Science; university-wide review for other STEM majors",
        "deadlines": "Priority December 1; regular February 1",
        "transfer_restrictions": "Internal transfer into Computer Science requires a 3.5 college GPA",
        "official_depth": 0,
    },
    {
        "university": "Cedar Valley University",
        "state": "TX",
        "control": "Public",
        "size": "Large",
        "majors": ["Computer Science", "Computer Engineering", "Applied Mathematics"],
        "admit_rate": "44%",
        "major_admit_rate": None,
        "sat_range": "1280-1450 (middle 50%)",
        "act_range": "28-33 (middle 50%)",
        "test_policy": "Scores considered when submitted",
        "admission_structure": "Applicants are admitted to the College of Engineering, then place into major after the first year",
        "deadlines": "Early action November 1; regular January 15",
        "transfer_restrictions": None,
        "official_depth": 0,
    },
    {
        "university": "Northgate Institute of Technology",
        "state": "MA",
        "control": "Private",
        "size": "Medium",
        "majors": ["Computer Science", "Computer Engineering", "Cybersecurity"],
        "admit_rate": None,
        "major_admit_rate": None,
        "sat_range": None,
        "act_range": None,
        "test_policy": None,
        "admission_structure": None,
        "deadlines": "Regular decision January 5 (per program page)",
        "transfer_restrictions": None,
        # No official admissions document exists at any depth: this college
        # drives the retry -> exhausted -> human review path.
        "official_depth": None,
    },
    {
        "university": "Rio Blanco State University",
        "state": "TX",
        "control": "Public",
        "size": "Medium",
        "majors": ["Industrial Engineering", "Systems Engineering", "Statistics"],
        "admit_rate": "71%",
        "major_admit_rate": "62% (Industrial Engineering)",
        "sat_range": "1120-1310 (middle 50%)",
        "act_range": "23-29 (middle 50%)",
        "test_policy": "Test optional",
        "admission_structure": "Direct admission to major",
        "deadlines": "Rolling; priority January 20",
        "transfer_restrictions": None,
        # Official admissions page is only retrieved on a deeper pass: this
        # college drives the Critic "research_more" retry path.
        "official_depth": 1,
    },
    {
        "university": "Harborview University",
        "state": "CA",
        "control": "Private",
        "size": "Medium",
        "majors": ["Data Science", "Statistics", "Applied Mathematics"],
        "admit_rate": "23%",
        "major_admit_rate": "17% (Data Science)",
        "sat_range": "1400-1540 (middle 50%)",
        "act_range": "32-35 (middle 50%)",
        "test_policy": "Test required for the 2027 entering class",
        "admission_structure": "Direct admission to major with a supplemental quantitative essay",
        "deadlines": "Early decision November 1; regular January 5",
        "transfer_restrictions": "Internal transfer into Data Science is closed",
        "official_depth": 0,
    },
    {
        "university": "Prairie Tech University",
        "state": "KS",
        "control": "Public",
        "size": "Medium",
        "majors": ["Computer Engineering", "Electrical Engineering", "Systems Engineering"],
        "admit_rate": "77%",
        "major_admit_rate": "74% (Computer Engineering)",
        "sat_range": "1090-1290 (middle 50%)",
        "act_range": "22-28 (middle 50%)",
        "test_policy": "Test optional",
        "admission_structure": "Direct admission to major",
        "deadlines": "Rolling",
        "transfer_restrictions": None,
        "official_depth": 0,
    },
    {
        "university": "Summit Ridge College",
        "state": "CO",
        "control": "Private",
        "size": "Small",
        "majors": ["Information Science", "Data Science", "Computational Science"],
        "admit_rate": "49%",
        "major_admit_rate": None,
        "sat_range": "1250-1400 (middle 50%)",
        "act_range": "27-31 (middle 50%)",
        "test_policy": "Test optional",
        "admission_structure": "University-wide review; students declare a major in the sophomore year",
        "deadlines": "Early action November 15; regular February 1",
        "transfer_restrictions": None,
        "official_depth": 0,
    },
    {
        "university": "Bayland University",
        "state": "FL",
        "control": "Public",
        "size": "Large",
        "majors": ["Cybersecurity", "Information Science", "Computer Science"],
        "admit_rate": "36%",
        "major_admit_rate": "21% (Computer Science)",
        "sat_range": "1310-1460 (middle 50%)",
        "act_range": "29-33 (middle 50%)",
        "test_policy": "Scores required for merit scholarship consideration",
        "admission_structure": "Limited-access major: separate portfolio review after admission",
        "deadlines": "Regular November 1",
        "transfer_restrictions": "Limited-access majors admit a fixed cohort each fall",
        "official_depth": 0,
    },
    {
        "university": "Ironwood State University",
        "state": "MI",
        "control": "Public",
        "size": "Large",
        "majors": ["Operations Research", "Industrial Engineering", "Statistics"],
        "admit_rate": "64%",
        "major_admit_rate": "55% (Industrial Engineering)",
        "sat_range": "1180-1360 (middle 50%)",
        "act_range": "25-30 (middle 50%)",
        "test_policy": "Test optional",
        "admission_structure": "Direct admission to major",
        "deadlines": "Priority December 1",
        "transfer_restrictions": None,
        "official_depth": 0,
    },
    {
        "university": "Crescent Bay University",
        "state": "CA",
        "control": "Public",
        "size": "Large",
        "majors": ["Computer Science", "Data Science", "Electrical Engineering"],
        "admit_rate": "11%",
        "major_admit_rate": "4% (Computer Science)",
        "sat_range": "1460-1560 (middle 50%)",
        "act_range": "33-35 (middle 50%)",
        "test_policy": "Test blind",
        "admission_structure": "Direct admission to major; capped major with a separate review committee",
        "deadlines": "Regular November 30",
        "transfer_restrictions": "Change-of-major into Computer Science is not permitted",
        "official_depth": 0,
    },
]


def slug(name: str) -> str:
    return "".join(ch.lower() if ch.isalnum() else "-" for ch in name).strip("-")


def build() -> list[dict]:
    documents: list[dict] = []
    for college in COLLEGES:
        base = slug(college["university"])
        domain = f"https://www.{base.replace('-', '')}.example.edu"

        if college["official_depth"] is not None:
            facts = {
                key: college[key]
                for key in (
                    "admit_rate",
                    "sat_range",
                    "act_range",
                    "test_policy",
                    "admission_structure",
                    "deadlines",
                    "transfer_restrictions",
                )
                if college[key]
            }
            documents.append(
                {
                    "id": f"{base}-adm",
                    "university": college["university"],
                    "title": f"{college['university']} First-Year Admission Requirements",
                    "text": (
                        f"{college['university']} first-year admission. "
                        f"Overall admit rate {college['admit_rate']}. "
                        f"Middle 50% SAT {college['sat_range']}. "
                        f"Testing policy: {college['test_policy']}. "
                        f"Application deadlines: {college['deadlines']}."
                    ),
                    "source_url": f"{domain}/admissions/first-year",
                    "source_type": "official_admissions",
                    "published_at": "2026-03-01T00:00:00+00:00",
                    "depth": college["official_depth"],
                    "facts": facts,
                }
            )

        stem_facts = {"admission_structure": college["admission_structure"]}
        if college["major_admit_rate"]:
            stem_facts["major_admit_rate"] = college["major_admit_rate"]
        if college["transfer_restrictions"]:
            stem_facts["transfer_restrictions"] = college["transfer_restrictions"]
        documents.append(
            {
                "id": f"{base}-stem",
                "university": college["university"],
                "title": f"{college['university']} Engineering and Computing Programs",
                "text": (
                    f"Programs offered: {', '.join(college['majors'])}. "
                    f"{college['admission_structure'] or 'Admission structure is not described on this page.'} "
                    f"Application deadline reference: {college['deadlines']}."
                ),
                "source_url": f"{domain}/engineering/programs",
                "source_type": "official_stem_program",
                "published_at": "2026-02-01T00:00:00+00:00",
                "depth": 0,
                "facts": {k: v for k, v in stem_facts.items() if v},
            }
        )

        if college["admit_rate"]:
            documents.append(
                {
                    "id": f"{base}-cds",
                    "university": college["university"],
                    "title": f"{college['university']} Common Data Set 2025-2026 (Section C)",
                    "text": (
                        "Section C: First-time first-year admission. "
                        f"Admit rate {college['admit_rate']}. SAT {college['sat_range']}. "
                        f"ACT {college['act_range']}."
                    ),
                    "source_url": f"{domain}/ir/common-data-set",
                    "source_type": "common_data_set",
                    "published_at": "2025-11-15T00:00:00+00:00",
                    "depth": 1,
                    "facts": {
                        "admit_rate": college["admit_rate"],
                        "sat_range": college["sat_range"],
                        "act_range": college["act_range"],
                    },
                }
            )

    return documents


def main() -> None:
    payload = {
        "_disclaimer": (
            "SYNTHETIC DEMO DATA. All institutions and figures are fictional. "
            "Do not present as real admissions information."
        ),
        "colleges": [
            {
                k: v
                for k, v in c.items()
                if k in {"university", "state", "control", "size", "majors"}
            }
            for c in COLLEGES
        ],
        "documents": build(),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"wrote {OUT} ({len(payload['documents'])} documents)")


if __name__ == "__main__":
    main()
