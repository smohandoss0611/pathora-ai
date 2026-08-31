---
title: Pathora AI
emoji: 🎓
colorFrom: indigo
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
short_description: Explainable multi-agent STEM college decision platform
---

# Pathora AI

Understand your profile. Explore your path. Make better decisions.

Upload a high-school transcript and get evidence-backed admission-fit
classifications, an Evidence Passport per college, gap analysis, and a What-If
simulator.

Pathora does not predict or guarantee admission. Every college fact comes from
retrieved official sources; anything unavailable reads "Not officially
published", and colleges without sufficient evidence are refused rather than
labelled.

## Setup

This Space needs two secrets (Settings → Variables and secrets):

| Secret | Purpose |
|---|---|
| `LLM_API_KEY` | Nebius / Groq / OpenRouter key for the agents |
| `SCORECARD_API_KEY` | College Scorecard, free at api.data.gov/signup |

Plus these variables:

```
LLM_PROVIDER=nebius
DEFAULT_MODEL=Qwen/Qwen3-235B-A22B-Instruct-2507
COLLEGE_DISCOVERY_MODE=hybrid
MAX_COLLEGES_PER_ANALYSIS=5
LLM_TIMEOUT_SECONDS=180
```

Without `LLM_API_KEY` the Space still runs: it falls back to a deterministic
offline engine, so the full workflow is demonstrable at zero cost.
