# Pathora AI

**Understand your profile. Explore your path. Make better decisions.**

An explainable multi-agent STEM college decision platform. A student uploads a
transcript and gets back evidence-backed admission-fit classifications, an
Evidence Passport for every claim, a qualitative gap analysis, a What-If
simulator, prioritized next actions, and a roadmap.

> Pathora does **not** predict or guarantee admission and does not output
> admission probabilities. It classifies fit from official published evidence
> and shows you the reasoning behind every classification.

---

## Quickstart

```bash
pip install -e ".[dev,ui]"
pytest -q                 # 142 tests
python scripts/demo.py    # full journey in the terminal, ~0.1s
```

Then run the app:

```bash
make run-api              # http://localhost:8000/docs
make run-ui               # http://localhost:8501
```

Or with infrastructure:

```bash
cp .env.example .env
docker compose up --build # api + ui + postgres + redis
```

**If the UI looks stale**, the container is almost certainly serving code from
the last image build. `docker-compose.override.yml` bind-mounts `src` and turns
on hot reload for exactly this reason, but a container created before that file
existed will not pick it up: `docker compose up --build --force-recreate`. The
Streamlit sidebar shows a build stamp plus a "Start over / reload code" button
that clears the cached service.

### Degraded mode (the default)

The system runs end to end with **no API keys and no network**:

| Component | Default | Production |
|---|---|---|
| LLM | `FakeProvider` — deterministic rule-based reference implementation | `LLM_PROVIDER=anthropic` |
| Vector store | in-memory hybrid retrieval | `VECTOR_BACKEND=pinecone` |
| Database | SQLite | `DATABASE_URL=postgresql+psycopg://…` |
| Checkpoints | in-memory | `CHECKPOINT_BACKEND=postgres` |
| Cache | in-process | `REDIS_URL=redis://…` |

This is deliberate. Every graph node, conditional route, retry loop and
interrupt is exercised in CI without a paid call, and the deterministic provider
doubles as the executable specification of what each agent is allowed to say.

---

## The journey

```
Upload transcript → Extract academics → Calculate GPA → Verify (human)
  → Profile Agent ∥ Activity Agent → Digital Twin → STEM Discovery
  → College Discovery → Parallel research → Admission Assessment
  → Critic validation → Explain My Match → Evidence Passport
  → Gap Analysis → What-If Lab → Next Best Action → Roadmap
```

## The graph

LangGraph is the authoritative orchestrator. There is no free-form agent chat and
no agent chooses who speaks next — every transition is a static edge or an
explicit conditional function over typed state.

```
START → parse_transcript → calculate_gpa → verify_academic_profile ──(interrupt)
                                                    │
                              ┌─────────────────────┴─────────────────────┐
                              ▼                                           ▼
                        profile_agent                              activity_agent
                              └─────────────────────┬─────────────────────┘
                                                    ▼
                                              build_twin
                                                    ▼
                                            stem_fit_agent
                                                    ▼
                                          college_discovery
                                                    ▼
                                   DYNAMIC FAN-OUT (Send, ≤5 concurrent)
                          research_worker × N  ──────────────► collect_research
                                                    ▼
                                            admission_agent
                                                    ▼
                                              critic_agent
                          ┌─────────────────────────┼─────────────────────────┐
                       approve                research_more              human_review
                          │                        ▼                          │(interrupt)
                          │                targeted_research                  │
                          │                        └──► research_worker ──┐   │
                          │                                               │   │
                          └───────────────────────┬───────────────────────┴───┘
                                                  ▼
                                          next_best_action → dynamic_roadmap → END
```

`fan_out_research` and `fan_out_targeted_research` emit LangGraph `Send`
objects, so worker count follows the candidate list at runtime. The retry path
re-researches **only** the colleges the Critic named, with a deeper retrieval
sweep.

### Limits (all env-configurable, none bypassable by an agent)

```
MAX_COLLEGES_PER_ANALYSIS=8
MAX_PARALLEL_COLLEGE_WORKERS=5
MAX_RESEARCH_RETRIES=2
MAX_CRITIC_LOOPS=2
```

Retries are bounded twice over — by the critic loop counter *and* by the research
retry counter — and the graph carries a hard `recursion_limit`. A Critic that
keeps asking for more research is converted to `human_review` rather than
allowed to loop.

---

## Design decisions worth defending

**GPA is never computed by an LLM.** `services/gpa.py` is pure Python with 36
unit tests covering credit weighting, half credits, pass/fail exclusion,
unrecognized grades, numeric scales and configurable mappings. An LLM that
"computes" a GPA is non-reproducible and gets it wrong under load.

**Transcript parsing is deterministic too.** Rule-based extraction with per-field
confidence. Low confidence or a transcript/computed GPA conflict raises a
LangGraph interrupt rather than guessing.

**College facts are copied from evidence, never generated.** The research worker
merges facts from retrieved chunks in source-authority order (official
admissions → official STEM program → Common Data Set → institutional research).
Anything absent stays `"Not officially published"`. The worker never asks a model
to fill a gap — that is how the no-fabrication rule is *enforced* rather than
merely requested in a prompt.

**Evidence is stored separately from prose.** Every `evidence_id` on an
assessment resolves to an `EvidenceRecord` with a URL, source type and retrieval
timestamp. A test asserts assessments can only cite evidence that research
actually returned.

**No fake precision.** Classifications are ordinal (`Safety … High Reach`) with
`Low/Moderate/High` confidence. The gap analyzer emits `High/Medium/Low` impact
and no invented weights. Nothing outputs "72% chance".

**Conservative classification floors.** An unpublished admit rate can never be
scored better than `Target`; a program under 15% admit can never fall below
`Reach` no matter how strong the student looks. Uncertainty resolves against the
student's interests, not toward a flattering answer.

---

## Researching real universities

`scripts/ingest_real_colleges.py` ingests Texas A&M, UT Dallas, Purdue and
Virginia Tech from verified official pages (`--dry-run` lists them without
fetching). Facts come from `agents/fact_extractor.py`, which regexes admit
rates, score ranges, test policies, admission structures and deadlines out of
the retrieved *text* and records the `evidence_id` and matched span for each —
so a real page with no structured metadata still produces traceable facts, and a
page that doesn't state a number produces nothing.

Two things to expect with real schools:

- **Virginia Tech does not publish its Common Data Set.** Its AIE office
  distributes it by email request (aiesupport@vt.edu). Pathora will report
  "Not officially published" for anything only the CDS answers. That is correct
  behaviour, not a gap to paper over.
- **Major-level admit rates usually don't exist.** Universities publish an
  institution-wide rate; a CS-specific rate generally isn't official. The Critic
  flags any assessment that lets a university-wide rate read as major-specific.

### Making a new corpus show up in the app

With `VECTOR_BACKEND=memory`, an index built by a script lives and dies with that
process — the API and UI rebuild their own store on startup. So ingestion writes
a corpus file:

```bash
python scripts/ingest_real_colleges.py          # -> data/seed/real.colleges.json
```

Anything matching `data/seed/*.colleges.json` is discovered and merged with the
demo corpus at startup (later files win on colliding ids). Restart the API, or
hit **Start over / reload code** in the Streamlit sidebar, and the new schools
appear in both the retrieval index and the discovery catalog.

On `VECTOR_BACKEND=pinecone` this is unnecessary — the index is shared, and
ingesting from any process is immediately visible to every other.

The HTML-to-text step is a crude tag strip — swap in trafilatura or readability
before trusting it on JavaScript-heavy admissions sites.

## Demo corpus: synthetic on purpose

`data/seed/colleges.json` contains **ten fictional universities with invented
statistics**. Putting made-up admit rates next to real university names would
violate the system's own core rule, so the demo corpus uses institutions that do
not exist. To research real schools, point `pathora.rag.ingest` at real official
documents — the pipeline is the same.

The corpus is also structured to force the interesting paths:

- **Rio Blanco State University** — official admissions page only surfaces on a
  deep retrieval pass → exercises the Critic's `research_more` retry.
- **Northgate Institute of Technology** — no official admissions source exists at
  any depth → retries exhaust → escalates to `human_review`.

Both fire on every `scripts/demo.py` run.

---

## Layout

```
src/pathora/
  config.py            env-driven settings, graph limits, per-task model routing
  domain/models.py     every inter-node contract (Pydantic v2, extra="forbid")
  services/            gpa · transcript · twin · evidence+gap   (deterministic)
  llm/                 Protocol · Anthropic provider · Fake provider + heuristics
  agents/              analysts (prompted) · college_worker (fan-out)
  rag/                 store (memory | Pinecone) · ingest pipeline
  graph/               state · nodes · build · whatif
  mcp_server/          College MCP server: 4 tools
  persistence/         SQLAlchemy repository · Redis cache/rate-limit/lock
  api/main.py          FastAPI
  ui/app.py            Streamlit
  service.py           application facade used by API, UI and tests
```

The domain and application layers know nothing about HTTP or Streamlit, so
replacing the UI with Next.js or moving to ECS/RDS/ElastiCache means writing new
adapters, not new agents.

### MCP

`mcp_server/college.py` exposes `research_college`, `search_college_documents`,
`get_program_info` and `get_admission_policy`. MCP is a tool transport here, not
the orchestrator — LangGraph still decides what runs next. Transcript parsing and
GPA stay internal because routing them through MCP would buy nothing.

---

## API

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/health` | status + active limits |
| `POST` | `/transcript/extract` | upload PDF/text, get extraction + confidence |
| `POST` | `/analysis/start` | run the workflow |
| `POST` | `/analysis/resume` | answer an interrupt (`confirm` / `edit` / `continue_with_uncertainty` / `cancel`) |
| `GET` | `/analysis/{thread_id}` | current state |
| `GET` | `/analysis/{thread_id}/explain/{university}` | Explain My Match + Evidence Passport + gaps |
| `POST` | `/analysis/what-if` | scenario simulation |

## What-If: selective re-execution

Changing an SAT score reruns the admission agent, critic, next actions and the
affected roadmap sections. It does **not** rerun transcript parsing, GPA
calculation, activity analysis, STEM fit, college discovery or research. The
response reports exactly which nodes were recomputed and which were reused:

```
recomputed: admission_agent, critic_agent, next_best_action, dynamic_roadmap(today,this_week)
reused:     activity_agent, calculate_gpa, college_discovery, parse_transcript,
            profile_agent, research_worker, stem_fit_agent, verify_academic_profile
```

A grade change additionally recomputes GPA (deterministically — the PDF is never
re-parsed) and STEM fit. A preference change re-runs discovery and research.

---

## Testing

```
tests/test_gpa.py                 36  GPA determinism and edge cases
tests/test_transcript.py          16  extraction, confidence, PDF round-trip
tests/test_workflow.py            28  end-to-end: happy path, grounding, retry,
                                      critic, HITL, what-if
tests/test_rag_and_evidence.py    23  ingestion, retrieval, worker, MCP,
                                      passport, gaps
tests/test_api.py                  5  FastAPI journey
tests/test_persistence.py         14  repository, cache/rate-limit/lock,
                                      checkpointer selection, .env drift guard
```

`ruff check`, `ruff format --check` and `mypy src/pathora` all pass clean.

Two of these tests exist to stop silent config rot: `.env.example` is asserted to
match `Settings` field-for-field, and no value may carry an inline comment (some
env parsers, including Docker Compose's `env_file`, do not reliably strip them —
`VECTOR_BACKEND=memory  # memory | pinecone` can arrive as the whole string).

---

## Status: what is and is not proven

**Verified here** — the full journey, the retry path, a Critic rejection, two
human-in-the-loop paths (transcript verification and critic escalation), bounded
fan-out, selective what-if re-execution, the FastAPI surface, the SQLAlchemy
repository against SQLite, lint, types, and 142 tests.

**Checkpointing caveat**: with `CHECKPOINT_BACKEND=memory` (the default), an
analysis paused at a human-in-the-loop interrupt does not survive a process
restart. Set `CHECKPOINT_BACKEND=postgres` for any real deployment — interrupts
are the one feature that is actively unsafe under the in-memory default.

**Written but not executed against live infrastructure** (no credentials or
network in the build environment):

- `AnthropicProvider` — the structured-output/retry/repair loop is implemented
  and typed, but has not made a real API call. Expect prompt tuning before the
  real provider matches the deterministic reference implementation's discipline.
- `PineconeVectorStore` — implemented against the documented client API; the
  hashing embedder in `rag/store.py` is a placeholder and should be swapped for a
  real embedding model before Pinecone is useful.
- Redis, Postgres, the Docker images, `render.yaml` and the MCP `stdio` server —
  written to spec, not run.

**Deliberately not built**: Mem0 (the spec marks it optional and it is not the
system of record); the OpenAI provider (the Protocol is the only integration
point — one `match` arm); LangSmith beyond config flags.

**Note on the spec**: the source document is truncated mid-sentence in §32
("CRIT…"), so sections 33+ were never received. Model routing is implemented as
described through §32; if later sections cover auth, multi-tenancy, deployment
specifics or evaluation harnesses, those are unaddressed.

---

## Ethical posture

- Never states or implies an admission probability.
- Never fabricates a statistic; unavailable facts read "Not officially published".
- Never recommends a major because admission looks easier.
- Never suggests dishonest application strategies.
- Surfaces uncertainty to the student and lets them choose how to proceed rather
  than silently continuing.
