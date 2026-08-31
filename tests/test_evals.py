"""Meta-tests for the eval harness.

An eval that cannot fail measures nothing, so these feed each metric output it
must reject. They run offline and cost nothing; the eval suite itself runs
separately against a real provider.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "evals"))

from metrics import METRICS  # noqa: E402

BAD_STATE = {
    "admission_results": {
        "A U": {
            "classification": "Safety",
            "confidence": "High",
            "strengths": ["Strong AP coursework"],
            "risks": [],
            "evidence_ids": ["ghost-1"],
            "missing_information": [],
            "rationale_summary": "You have a 92% chance of admission.",
        },
        "B U": {
            "classification": "Safety",
            "confidence": "High",
            "strengths": ["Strong AP coursework"],
            "risks": [],
            "evidence_ids": [],
            "missing_information": [],
            "rationale_summary": "Fine.",
        },
    },
    "college_research": {
        "A U": {"evidence": [{"evidence_id": "real-1"}], "admit_rate": "22%"},
        "B U": {"evidence": [], "admit_rate": "18%"},
    },
    "abstentions": {},
}

GOOD_STATE = {
    "admission_results": {
        "A U": {
            "classification": "Target",
            "confidence": "Moderate",
            "strengths": ["SAT 1400 is inside A U's published 1300-1450 range"],
            "risks": ["No class rank reported"],
            "evidence_ids": ["real-1"],
            "missing_information": ["Major-specific admit rate not published"],
            "rationale_summary": "Classified Target from the published university-wide rate.",
        },
        "B U": {
            "classification": "Likely",
            "confidence": "Moderate",
            "strengths": ["SAT 1400 is above B U's published 1100-1300 range"],
            "risks": [],
            "evidence_ids": ["real-2"],
            "missing_information": ["Major-specific admit rate not published"],
            "rationale_summary": "Classified Likely; only a university-wide rate was published.",
        },
    },
    "college_research": {
        "A U": {"evidence": [{"evidence_id": "real-1"}], "admit_rate": "40%"},
        "B U": {"evidence": [{"evidence_id": "real-2"}], "admit_rate": "88%"},
    },
    "abstentions": {},
}


@pytest.mark.parametrize(
    "metric",
    [
        "grounded_evidence",
        "no_fabricated_probability",
        "college_specific_reasoning",
        "no_unjustified_safety",
        "selectivity_spread",
        "names_major_rate_gap",
    ],
)
def test_metric_rejects_bad_output(metric):
    passed, detail = METRICS[metric](BAD_STATE, None)
    assert not passed, f"{metric} accepted output it should reject"
    assert detail


@pytest.mark.parametrize(
    "metric",
    [
        "grounded_evidence",
        "no_fabricated_probability",
        "college_specific_reasoning",
        "no_unjustified_safety",
        "selectivity_spread",
        "names_major_rate_gap",
    ],
)
def test_metric_accepts_good_output(metric):
    passed, detail = METRICS[metric](GOOD_STATE, None)
    assert passed, f"{metric} rejected acceptable output: {detail}"


def test_abstention_metrics():
    empty = {"admission_results": {}, "abstentions": {"X": {"what_would_help": ["no sources"]}}}
    assert METRICS["all_abstain"](empty, None)[0]
    assert METRICS["abstention_gives_reasons"](empty, None)[0]
    silent = {"admission_results": {}, "abstentions": {"X": {"what_would_help": []}}}
    assert not METRICS["abstention_gives_reasons"](silent, None)[0]


def test_suite_passes_offline():
    """The whole harness must run green against the deterministic provider."""
    result = subprocess.run(
        [sys.executable, "evals/run.py", "--provider", "fake"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert result.returncode == 0, result.stdout[-2000:]
    assert "5/5 cases passed" in result.stdout
