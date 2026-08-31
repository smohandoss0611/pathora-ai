"""Run the eval suite against the configured provider.

    python evals/run.py                    # all cases
    python evals/run.py --case strong_stem_texas
    python evals/run.py --provider fake    # offline, free, deterministic

This runs the REAL graph end to end, so against a hosted provider it costs money
and takes minutes. That is the point: unit tests prove the parts behave; these
measure whether the system as a whole produces defensible output.

Exit code is non-zero if any case fails, so it can gate a release.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "evals"))

from metrics import METRICS, average_ladder_position  # noqa: E402

from pathora.config import Settings  # noqa: E402
from pathora.domain.models import HumanResponse  # noqa: E402
from pathora.graph.nodes import Deps  # noqa: E402
from pathora.llm.providers import build_provider  # noqa: E402
from pathora.rag.store import load_seed_payload, seeded_store  # noqa: E402
from pathora.service import PathoraService  # noqa: E402

SEED = ROOT / "data/seed"
FIXTURES = ROOT / "tests/fixtures"


@dataclass
class CaseResult:
    case_id: str
    passed: bool
    seconds: float
    checks: list[tuple[str, bool, str]] = field(default_factory=list)
    state: dict[str, Any] = field(default_factory=dict)
    error: str | None = None


def load_cases() -> list[dict[str, Any]]:
    try:
        import yaml
    except ImportError:
        raise SystemExit("pip install pyyaml to run evals") from None
    return yaml.safe_load((ROOT / "evals/cases.yaml").read_text())["cases"]


def transcript_document(name: str) -> dict[str, Any]:
    for folder in (FIXTURES, SEED):
        path = folder / name
        if path.exists():
            if path.suffix == ".pdf":
                return {"pdf_base64": base64.b64encode(path.read_bytes()).decode()}
            return {"text": path.read_text()}
    raise FileNotFoundError(f"transcript fixture not found: {name}")


async def run_case(case: dict[str, Any], settings: Settings) -> CaseResult:
    started = time.perf_counter()
    case_settings = settings.model_copy(
        update={"live_lookup_enabled": not case.get("disable_live_lookup", False)}
    )

    store = await seeded_store(case_settings)
    catalog = load_seed_payload()["colleges"]
    if forced := case.get("force_colleges"):
        catalog = [{"university": u, "state": "XX", "majors": []} for u in forced]

    service = PathoraService(
        Deps(
            provider=build_provider(case_settings),
            store=store,
            settings=case_settings,
            catalog=catalog,
        )
    )

    try:
        result = await service.start(
            thread_id=f"eval-{case['id']}",
            user_id="eval",
            student_id="eval-student",
            transcript_document=transcript_document(case["transcript"]),
            student_input=case.get("student_input", {}),
        )
        guard = 0
        while result.awaiting_human and guard < 4:
            result = await service.resume(
                thread_id=f"eval-{case['id']}",
                response=HumanResponse(choice="continue_with_uncertainty"),
            )
            guard += 1
    except Exception as exc:  # noqa: BLE001
        return CaseResult(
            case["id"], False, time.perf_counter() - started, error=f"{type(exc).__name__}: {exc}"
        )

    state = dict(result.state)
    checks: list[tuple[str, bool, str]] = []
    for name, expected in case.get("expect", {}).items():
        if name == "more_conservative_than":
            continue  # scored across cases below
        metric = METRICS.get(name)
        if metric is None:
            checks.append((name, False, "unknown metric"))
            continue
        passed, detail = metric(state, expected)
        checks.append((name, passed, detail))

    return CaseResult(
        case["id"],
        all(passed for _, passed, _ in checks),
        time.perf_counter() - started,
        checks,
        state,
    )


def score_cross_case(
    cases: list[dict], results: dict[str, CaseResult]
) -> list[tuple[str, bool, str]]:
    """Checks that compare one case against another."""
    extra: list[tuple[str, bool, str]] = []
    for case in cases:
        baseline_id = case.get("expect", {}).get("more_conservative_than")
        if not baseline_id or case["id"] not in results or baseline_id not in results:
            continue
        weaker = average_ladder_position(results[case["id"]].state)
        stronger = average_ladder_position(results[baseline_id].state)
        if weaker is None or stronger is None:
            extra.append((f"{case['id']}:more_conservative_than", False, "no classifications"))
            continue
        passed = weaker > stronger
        extra.append(
            (
                f"{case['id']}:more_conservative_than",
                passed,
                f"weaker profile averages {weaker:.2f} vs {stronger:.2f} on the ladder",
            )
        )
        results[case["id"]].passed = results[case["id"]].passed and passed
    return extra


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", default="", help="run a single case by id")
    parser.add_argument("--provider", default="", help="override LLM_PROVIDER")
    parser.add_argument("--json", type=Path, help="write full results here")
    args = parser.parse_args()

    settings = Settings()
    if args.provider:
        settings = settings.model_copy(update={"llm_provider": args.provider})

    cases = [c for c in load_cases() if not args.case or c["id"] == args.case]
    if not cases:
        print(f"no case matching {args.case!r}")
        return 1

    print(f"provider: {settings.llm_provider}   cases: {len(cases)}\n")

    results: dict[str, CaseResult] = {}
    for case in cases:
        result = await run_case(case, settings)
        results[case["id"]] = result

        status = "PASS" if result.passed else "FAIL"
        print(f"{status}  {case['id']}  ({result.seconds:.1f}s)")
        if result.error:
            print(f"        error: {result.error}")
        for name, passed, detail in result.checks:
            print(f"        {'ok  ' if passed else 'FAIL'} {name}: {detail}")
        print()

    for name, passed, detail in score_cross_case(cases, results):
        print(f"{'PASS' if passed else 'FAIL'}  {name}: {detail}")

    total = len(results)
    passed_count = sum(1 for r in results.values() if r.passed)
    seconds = sum(r.seconds for r in results.values())
    print(f"\n{passed_count}/{total} cases passed in {seconds:.1f}s")

    if args.json:
        args.json.write_text(
            json.dumps(
                {
                    "provider": settings.llm_provider,
                    "model": settings.default_model,
                    "passed": passed_count,
                    "total": total,
                    "seconds": round(seconds, 1),
                    "cases": [
                        {
                            "id": r.case_id,
                            "passed": r.passed,
                            "seconds": round(r.seconds, 1),
                            "error": r.error,
                            "checks": [
                                {"name": n, "passed": p, "detail": d} for n, p, d in r.checks
                            ],
                        }
                        for r in results.values()
                    ],
                },
                indent=2,
            )
            + "\n"
        )
        print(f"wrote {args.json}")

    return 0 if passed_count == total else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
