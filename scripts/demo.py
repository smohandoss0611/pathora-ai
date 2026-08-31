"""End-to-end demo: `python scripts/demo.py`.

Runs the full journey against the offline provider and synthetic corpus, and
prints the retry, critic-rejection and human-in-the-loop paths as they happen.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pathora.config import Settings  # noqa: E402
from pathora.domain.models import HumanResponse, WhatIfScenario  # noqa: E402
from pathora.graph.nodes import Deps  # noqa: E402
from pathora.llm.providers import build_provider  # noqa: E402
from pathora.rag.store import load_seed_payload, seeded_store  # noqa: E402
from pathora.service import PathoraService  # noqa: E402

STUDENT = {
    "testing": {"sat_total": 1450, "sat_math": 760, "sat_verbal": 690},
    "activities": [
        {
            "name": "Robotics Club",
            "role": "Team Captain",
            "years": ["2024-2025", "2025-2026"],
            "hours_per_week": 6,
            "description": "Led the software subteam for the competition robot",
        },
        {
            "name": "Peer Math Tutoring",
            "role": "Volunteer tutor",
            "years": ["2024-2025"],
            "description": "Weekly tutoring for algebra students",
        },
    ],
    "projects": [
        {
            "name": "Bus route delay tracker",
            "description": "Charted delays from published transit data",
            "technologies": ["Python", "SQLite"],
        }
    ],
    "awards": [{"name": "Regional Robotics Finalist", "level": "Regional", "year": "2026"}],
    "stem_interests": ["Computer Science", "Data Science"],
    "career_interests": ["Software Engineer"],
    "preferences": {"locations": ["TX"], "public_private": "Public", "school_size": "Large"},
}


def rule(title: str) -> None:
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


async def main() -> None:
    started = perf_counter()
    settings = Settings()
    store = await seeded_store(settings)
    service = PathoraService(
        Deps(
            provider=build_provider(settings),
            store=store,
            settings=settings,
            catalog=load_seed_payload()["colleges"],
        )
    )

    rule("1. Upload transcript, extract academics, calculate GPA, verify")
    result = await service.start(
        thread_id="demo",
        user_id="demo-user",
        student_id="demo-student",
        transcript_document={"text": (ROOT / "data/seed/sample_transcript.txt").read_text()},
        student_input=STUDENT,
    )

    hitl = 0
    while result.awaiting_human and hitl < 4:
        print(f"\n  HUMAN-IN-THE-LOOP [{result.interrupt['kind']}]")
        print(f"  {result.interrupt['message']}")
        print(f"  options: {result.interrupt['options']} -> choosing 'continue_with_uncertainty'")
        result = await service.resume(
            thread_id="demo", response=HumanResponse(choice="continue_with_uncertainty")
        )
        hitl += 1

    state = result.state
    gpa = state["gpa_result"]
    print(f"\n  GPA {gpa['gpa']} over {gpa['graded_credits']} graded credits ({gpa['method']})")
    print(f"  excluded: {gpa['excluded_courses']}")

    rule("2. STEM discovery")
    for fit in state["stem_fit"]:
        print(f"  {fit['discipline']:28s} {fit['fit']}")

    rule("3. College research, assessment and critic validation")
    print(
        f"  critic loops: {state['critic_loop_count']}   "
        f"research retries: {state['research_retry_count']}"
    )
    print(f"  critic decision: {state['critic_results']['decision']}")
    for issue in state["critic_results"]["issues"]:
        print(f"    ! {issue}")

    rule("4. Explain my match")
    for university in state["college_candidates"]:
        assessment = state["admission_results"].get(university)
        passport = state["evidence_passports"][university]
        if assessment is None:
            print(f"  {university:38s} NOT ASSESSED  evidence gate refused")
            continue
        print(
            f"  {university:38s} {assessment['classification']:13s} "
            f"confidence={assessment['confidence']:8s} evidence={passport['quality']}"
        )

    for university, abstention in state.get("abstentions", {}).items():
        print(f"\n  Gate refused {university}: {', '.join(abstention['failed_checks'])}")
        for item in abstention["what_would_help"]:
            print(f"    - {item}")
    first = next(u for u in state["college_candidates"] if u in state["admission_results"])
    print(f"\n  Why {first} was classified this way:")
    print(f"    {state['admission_results'][first]['rationale_summary']}")
    print("  Gap analysis:")
    for factor in state["gap_analysis"][first]["factors"]:
        print(f"    {factor['factor']:20s} {factor['impact'].upper():7s} {factor['note']}")

    rule("5. What-If Lab: SAT 1450 -> 1520 (selective re-execution)")
    _, whatif = await service.what_if(thread_id="demo", scenario=WhatIfScenario(sat_total=1520))
    for change in whatif.changes:
        arrow = "->" if change.changed else "= "
        print(f"  {change.university:38s} {change.before:13s} {arrow} {change.after}")
    print(f"\n  {whatif.summary}")
    print(f"  recomputed: {', '.join(whatif.nodes_rerun)}")
    print(f"  reused:     {', '.join(whatif.nodes_skipped)}")

    rule("6. Next best actions and roadmap")
    for action in state["next_actions"]:
        print(f"  [{action['priority']:6s}] {action['title']}")
    for section in ("today", "this_week", "this_month", "upcoming"):
        items = state["roadmap"].get(section, [])
        print(f"\n  {section.upper().replace('_', ' ')}")
        for item in items or []:
            print(f"    - {item['title']}")

    print(f"\nCompleted in {perf_counter() - started:.2f}s")


if __name__ == "__main__":
    asyncio.run(main())
