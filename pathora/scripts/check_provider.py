"""Check whether the configured model can actually drive Pathora.

    python scripts/check_provider.py
    python scripts/check_provider.py --model Qwen/Qwen3-235B-A22B

Pathora's agents all return validated Pydantic models, so the question that
matters for an open model is not "is it smart" but "does it reliably emit
schema-valid JSON and refuse to invent facts". This runs the two hardest real
schemas and reports latency, validity and whether the model fabricated.

Exit code is non-zero if any check fails, so it can gate a deploy.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pathora.config import Settings  # noqa: E402
from pathora.domain.models import (  # noqa: E402
    NOT_PUBLISHED,
    ActivityAnalysis,
    CollegeResearchResult,
    EvidenceRecord,
    ExtractedAcademics,
    ProfileAnalysis,
)
from pathora.llm.base import StructuredOutputError  # noqa: E402
from pathora.llm.providers import build_provider  # noqa: E402
from pathora.services.twin import build_digital_twin  # noqa: E402


def banner(text: str) -> None:
    print(f"\n{text}\n{'-' * len(text)}")


async def check(name: str, coro) -> tuple[bool, float, str]:
    started = time.perf_counter()
    try:
        result = await coro
    except StructuredOutputError as exc:
        return False, time.perf_counter() - started, f"schema never validated: {exc}"
    except Exception as exc:  # noqa: BLE001
        return False, time.perf_counter() - started, f"{type(exc).__name__}: {exc}"
    return True, time.perf_counter() - started, str(result)[:160]


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", help="override DEFAULT_MODEL for this run")
    args = parser.parse_args()

    settings = Settings()
    if args.model:
        settings = settings.model_copy(update={"default_model": args.model})

    print(f"provider : {settings.llm_provider}")
    if settings.llm_provider not in {"fake", "anthropic"}:
        from pathora.llm.providers import OpenAICompatibleProvider

        print(f"base_url : {OpenAICompatibleProvider(settings).base_url}")
    print(f"model    : {settings.default_model}")
    if settings.llm_provider == "fake":
        print("\nLLM_PROVIDER=fake — nothing to check. Set a real provider first.")
        return 1

    provider = build_provider(settings)
    twin = build_digital_twin(student_id="check", verified_academics=ExtractedAcademics())
    failures = 0

    # 1. Simplest schema: mostly lists of strings.
    banner("1. ActivityAnalysis (simple schema)")
    from pathora.agents.analysts import run_activity_agent

    ok, seconds, detail = await check("activity", run_activity_agent(provider, twin))
    print(f"   {'PASS' if ok else 'FAIL'}  {seconds:5.1f}s  {detail}")
    failures += not ok

    # 2. Constrained enums: the model must pick from fixed literals.
    banner("2. ProfileAnalysis (enum-constrained schema)")
    from pathora.agents.analysts import run_profile_agent

    ok, seconds, detail = await check("profile", run_profile_agent(provider, twin, "Improving"))
    print(f"   {'PASS' if ok else 'FAIL'}  {seconds:5.1f}s  {detail}")
    failures += not ok

    # 3. The hard one: enums plus a standing instruction NOT to invent numbers.
    banner("3. AdmissionAssessment (enums + no-fabrication discipline)")
    from pathora.agents.analysts import run_admission_agent

    barren = CollegeResearchResult(
        university="Testfield University",
        target_major="Computer Science",
        evidence=[
            EvidenceRecord(
                evidence_id="e1",
                university="Testfield University",
                source_url="https://testfield.example.edu/admissions",
                source_type="official_admissions",
                snippet="Testfield University welcomes applicants from all backgrounds.",
            )
        ],
        missing_information=["Admit rate not published"],
    )
    ok, seconds, detail = await check(
        "admission",
        run_admission_agent(
            provider,
            twin,
            barren,
            ProfileAnalysis(
                course_rigor="Moderate",
                grade_trend="Unknown",
                math_preparation="none on file",
                science_preparation="none on file",
                cs_preparation="none on file",
            ),
            ActivityAnalysis(),
        ),
    )
    print(f"   {'PASS' if ok else 'FAIL'}  {seconds:5.1f}s  {detail}")
    failures += not ok

    if ok:
        # The source above contains NO statistics. Any number in the output is
        # invented — the single most important property for this application.
        assessment = await run_admission_agent(
            provider,
            twin,
            barren,
            ProfileAnalysis(
                course_rigor="Moderate",
                grade_trend="Unknown",
                math_preparation="none on file",
                science_preparation="none on file",
                cs_preparation="none on file",
            ),
            ActivityAnalysis(),
        )
        text = f"{assessment.rationale_summary} {' '.join(assessment.strengths)}"
        fabricated = "%" in text and NOT_PUBLISHED not in text
        banner("4. Fabrication check")
        print("   retrieved source contains no statistics at all")
        verdict = "FAIL — model invented a figure" if fabricated else "PASS — no invented figures"
        print(f"   {verdict}")
        print(f"   rationale: {assessment.rationale_summary[:200]}")
        failures += fabricated

    banner("Verdict")
    if failures:
        print(f"   {failures} check(s) failed. Try a larger instruct model, or raise")
        print("   LLM_MAX_RETRIES if failures were schema-validation only.")
    else:
        print("   This model can drive Pathora.")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
