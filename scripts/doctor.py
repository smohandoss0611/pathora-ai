"""Answer 'why isn't my configuration loading?' in one command.

python scripts/doctor.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pathora.config import config_report, get_settings  # noqa: E402


def main() -> int:
    print(config_report())

    settings = get_settings()
    problems: list[str] = []

    if settings.llm_provider == "fake":
        problems.append(
            "LLM_PROVIDER=fake — the deterministic offline engine is running, not a model."
        )
    elif settings.llm_provider != "anthropic" and not settings.compatible_api_key:
        problems.append("No LLM_API_KEY / OPENAI_API_KEY set for the chosen provider.")
    elif settings.llm_provider == "anthropic" and not settings.anthropic_api_key:
        problems.append("ANTHROPIC_API_KEY is not set.")

    if settings.live_lookup_enabled and not settings.scorecard_api_key:
        problems.append(
            "LIVE_LOOKUP_ENABLED=true but SCORECARD_API_KEY is unset, so colleges with "
            "no indexed documents will abstain. Free key: https://api.data.gov/signup/"
        )

    if settings.college_discovery_mode == "catalog":
        problems.append("COLLEGE_DISCOVERY_MODE=catalog — only indexed colleges can be selected.")

    print()
    if problems:
        print("Issues:")
        for problem in problems:
            print(f"  - {problem}")
    else:
        print("No configuration issues detected.")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
