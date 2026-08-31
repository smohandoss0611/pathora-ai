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
pytest -q                 # 285 tests
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

**If the UI shows only grey loading blocks**, check the terminal first: on a
fresh install Streamlit prompts for an email address and blocks until you press
Enter, which serves the page but never runs the script. `mkdir -p ~/.streamlit
&& printf '[general]\nemail = ""\n' > ~/.streamlit/credentials.toml` skips it
permanently.

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
| LLM | `FakeProvider` — deterministic rule-based reference implementation | `LLM_PROVIDER=anthropic`, or `openai` + `LLM_BASE_URL` for any OpenAI-compatible endpoint (Ollama, Groq, OpenRouter, DeepSeek, LM Studio, vLLM) |
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

Real transcripts are *tables*, and PyMuPDF flattens a table to one cell per
line — no line-oriented regex can reassemble a row from that. So
`services/transcript_table.py` reconstructs rows from word coordinates
(words sharing a baseline are one row) and is tried first, with the
line-oriented parser as fallback. It handles semester-plus-final grade columns,
missing finals, `P`/`NG` marks, in-progress senior courses at 0.0000 credit, and
district abbreviations where the programme prefix is fused to the subject
(`APSTATS`, `IBPHYSHL`, `APTACSAM`).

A real district PDF is checked in at `tests/fixtures/` and covered by 28 tests,
including one asserting that plain-text extraction *would* have failed — the
original parser found zero courses in it.

**College facts are copied from evidence, never generated.** The research worker
merges facts from retrieved chunks in source-authority order (official
admissions → official STEM program → Common Data Set → institutional research).
Anything absent stays `"Not officially published"`. The worker never asks a model
to fill a gap — that is how the no-fabrication rule is *enforced* rather than
merely requested in a prompt.

**Abstention is a code decision, not a prompt instruction.** `services/
evidence_gate.py` runs *before* the Admission Agent and decides from retrieval
metadata alone — source count, source authority, fact provenance, staleness,
whether any selectivity anchor exists — whether an assessment may be attempted.
If it refuses, no LLM call is made for that college and an `AssessmentAbstention`
is recorded with the failed checks. The grounding rules in `ADMISSION_SYSTEM`
remain as a second line of defence.

This ordering matters: a model that has already read plausible-looking passages
will nearly always produce an answer, so "abstain if evidence is weak" in a
prompt is a request, not a control. The gate is deterministic and testable —
`test_no_llm_call_is_made_for_a_gated_college` asserts the provider is never
invoked — which is not something a prompt instruction can support.

Note what an abstention is *not*: a classification with `Low` confidence. When
there is no published admit rate, there is no honest Safety/Target/Reach label,
so none is produced.

**Prompts are checked, not trusted.** Three instructions have a deterministic
verifier behind them, because an instruction a model may quietly ignore is not a
control:

| Instruction | Verifier |
|---|---|
| abstain when evidence is thin | `evidence_gate` runs before generation; no LLM call happens |
| pick majors from the ranked STEM fits | `college_discovery` compares against the ranked list and warns |
| make strengths college-specific | `admission_agent` flags identical strength sets across colleges |
| classify from the evidence | `services/classifier.py` computes the label; the agent explains it |

**The model explains the classification, it does not choose it.** Selectivity,
score position against the college's published band, GPA and course rigor feed a
rule engine that produces the label; the agent receives it and writes the
reasoning. If the agent returns a different label it is overridden and the
disagreement is recorded as a warning.

This came from a real inconsistency: Texas Tech was classified `Target` for a
student scoring above its published 75th percentile at a 72.7% admit rate — a
conclusion that did not follow from the evidence the agent had just cited.

Two asymmetries in the engine are deliberate. Missing major-level data caps the
*ceiling* (no `Safety` without knowing how selective the major is) but does not
lower the floor, because absence of data is not evidence of selectivity — the
exception being a broadly accessible institution where the student is above the
published 75th percentile, since refusing `Safety` there is false caution. And
downgrades require concrete retrieved wording such as a capped or limited-access
major, never a missing field.

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

### Federal data: where admit rates actually come from

Scraping admissions websites does not produce admit rates. Those pages are
marketing — "test optional", "we review holistically" — and some institutions
(UT Austin among them) reject automated requests entirely, which is their access
policy and not something to route around.

Two ingestion paths onto the same federal source data:

| Path | Command | When |
|---|---|---|
| **College Scorecard API** | `scripts/ingest_scorecard.py` | default — queryable, one command to refresh |
| IPEDS bulk CSV | `scripts/ingest_ipeds.py` | offline, or when you need fields Scorecard omits |

```bash
export SCORECARD_API_KEY=...          # free at https://api.data.gov/signup/
python scripts/ingest_scorecard.py --state TX --min-size 5000
```

Both write a `*.colleges.json` the app picks up on restart.

The caveat that matters: `admission_rate.overall` is **university-wide**. It is
not the admit rate for Computer Science, and at large publics the major-level
rate is far lower. That caveat is written into the fact string itself, so the
Critic sees it and flags any assessment that treats one as the other. Expect
Low/Moderate confidence from federal data alone until program-level sources are
layered on.

### IPEDS: where admit rates actually come from

Every institution receiving federal Title IV funding must report admissions
counts and score ranges annually. That is IPEDS: authoritative, complete,
consistent across ~6,000 institutions, and free in bulk.

```bash
# Download ADM and HD for the latest year from
#   https://nces.ed.gov/ipeds/datacenter/DataFiles.aspx
python scripts/ingest_ipeds.py --adm adm2025.csv --hd hd2025.csv --state TX
```

Admit rate is computed as ADMSSN / APPLCN — the definition institutions report
against. A school missing either count yields no admit rate rather than an
estimate.

Two gate rules had to change to accommodate it, both defensible on their own:

- **`institutional_research` counts as authoritative.** A mandatory federal
  survey is a stronger source than most admissions pages.
- **A single authoritative dataset carrying a selectivity anchor satisfies the
  corroboration check.** Corroboration exists to catch an unreliable lone page;
  IPEDS is not that, and demanding two URLs would bar the best source available.
- **Annual surveys get a longer freshness window** (`ANNUAL_SURVEY_STALE_AFTER_DAYS`,
  default 3 years) because they publish on a reporting lag. Data older than that
  is still refused: a 2015 survey must not anchor a 2026 classification.

### On-demand lookup

`LIVE_LOOKUP_ENABLED=true` (default) closes the gap that made open discovery
unusable: a college the model names but nobody ingested is fetched from College
Scorecard at research time, indexed, and assessed — instead of abstaining.

```bash
export SCORECARD_API_KEY=...     # free at https://api.data.gov/signup/
```

Properties worth knowing:

- **Triggered by a missing anchor, not just an empty corpus.** A program page
  describing admission structure carries no admit rate; if its presence
  suppressed the lookup, adding evidence would turn a classifiable college into
  an abstention. The lookup runs whenever nothing retrieved can anchor a label.
- **Cached both ways.** A hit is indexed, so the second research reads locally.
  A miss is remembered for `LIVE_LOOKUP_TTL_SECONDS`, so a fictional school is
  not re-queried on every run.
- **Failure degrades to abstention, never to invention.** A 403, a timeout or an
  unknown institution leaves the college unassessed with the gate's reasons
  intact. A failed lookup is not recorded as a research error.
- **A weak match abstains rather than substituting.** Federal records carry
  campus qualifiers, and Scorecard fuzzy-matches: "Texas A&M University-San
  Antonio" came back for "Texas A&M University" and supplied an 840-1070 SAT
  band. Below `SCORECARD_MATCH_THRESHOLD` (0.9) no record is accepted. Names are
  scored in both directions — extra tokens in the query are the caller
  qualifying a name, extra tokens in the candidate mean a different campus, and
  only the latter is penalised.
- **Cached records can be purged.** A record fetched before a matching fix stays
  in the corpus, satisfies the anchor check, and prevents the lookup from ever
  re-running: `python scripts/purge_cache.py --university "Texas A&M University"`.
- **It does not make the numbers fresher.** Admissions figures are annual
  federal reporting. What becomes live is *coverage*, not recency — and the
  major-specific rate remains `Not officially published` either way.

### Letting the model pick the colleges

`COLLEGE_DISCOVERY_MODE=open` lets the model propose real universities from its
own knowledge instead of choosing from the indexed catalog; `hybrid` does
indexed-first then fills the remaining slots.

The division of labour is deliberate. Model knowledge is acceptable for deciding
*which schools to consider* — that is a suggestion the student will review.
It is not acceptable for stating what those schools' admit rates are, so the
discovery prompt explicitly forbids asserting statistics, and anything it
asserts is discarded: facts come only from retrieved documents.

The consequence is worth understanding before switching modes. A school the
model names but that has no indexed documents will be **refused by the evidence
gate**, not assessed from memory:

```
research for an un-ingested school:
  admit_rate : Not officially published
  evidence   : 0
  gate passed: False
  -> No official documents were found for this university.
```

That is the system working. To get real assessments for real schools, ingest
them first with `scripts/ingest_real_colleges.py`, then use `catalog` or
`hybrid` mode.

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
tests/test_transcript_table.py    28  real tabular PDF: rows, grades, subjects
tests/test_workflow.py            28  end-to-end: happy path, grounding, retry,
                                      critic, HITL, what-if
tests/test_rag_and_evidence.py    23  ingestion, retrieval, worker, MCP,
                                      passport, gaps
tests/test_api.py                  5  FastAPI journey
tests/test_persistence.py         14  repository, cache/rate-limit/lock,
                                      checkpointer selection, .env drift guard
tests/test_fact_extractor.py      21  grounded extraction, corpus merging
tests/test_classifier.py          18  deterministic labels, ceilings, floors
tests/test_evidence_gate.py       14  pre-generation gate, incl. proof that no
                                      LLM call occurs when the gate refuses
tests/test_embeddings.py          13  embedding backends, batching, ordering
tests/test_openai_provider.py     28  OpenAI-compatible provider vs. a stub
tests/test_ui.py                   8  Streamlit renders without hanging;
                                      deploy entrypoint and requirements
```

`ruff check`, `ruff format --check` and `mypy src/pathora` all pass clean.

Two of these tests exist to stop silent config rot: `.env.example` is asserted to
match `Settings` field-for-field, and no value may carry an inline comment (some
env parsers, including Docker Compose's `env_file`, do not reliably strip them —
`VECTOR_BACKEND=memory  # memory | pinecone` can arrive as the whole string).

---

## Evals

Unit tests assert deterministic behaviour on fixed inputs. Evals measure whether
the *system* produces defensible output on cases where the answer is a judgment.

```bash
python evals/run.py --provider fake     # offline, free, ~2s
python evals/run.py                     # against the configured provider
python evals/run.py --json report.json
```

Five cases in `evals/cases.yaml`, each stating **properties** rather than
expected strings — an eval that pins prose measures the prompt, not the system:

| Case | What it catches |
|---|---|
| `strong_stem_texas` | happy path: spread of selectivity, grounded citations, per-college reasoning |
| `no_test_score` | a missing score treated as a good one |
| `weaker_record` | a weaker profile not classified more conservatively (scored *across* cases) |
| `uncorroborated_college` | classifying from model memory when nothing is retrievable |
| `unpublished_major_rate` | a university-wide rate passed off as major-specific |

Eleven metrics score each run: citation traceability, fabricated-probability
language, duplicate reasoning across colleges, unjustified `Safety`, abstention
reasons, and others. `tests/test_evals.py` feeds every metric output it must
reject — an eval that cannot fail measures nothing — and asserts the suite runs
green offline. The offline run is part of CI.

## Status: what is and is not proven

**Verified here** — the full journey, the retry path, a Critic rejection, two
human-in-the-loop paths (transcript verification and critic escalation), bounded
fan-out, selective what-if re-execution, the FastAPI surface, the SQLAlchemy
repository against SQLite, lint, types, and 285 tests.

**Checkpointing caveat**: with `CHECKPOINT_BACKEND=memory` (the default), an
analysis paused at a human-in-the-loop interrupt does not survive a process
restart. Set `CHECKPOINT_BACKEND=postgres` for any real deployment — interrupts
are the one feature that is actively unsafe under the in-memory default.

**Written but not executed against live infrastructure** (no credentials or
network in the build environment):

- `AnthropicProvider` and `OpenAICompatibleProvider` — the structured-output,
  retry and repair loops are implemented, typed and tested against stubs, but
  neither has made a real API call. Expect prompt tuning before the
  real provider matches the deterministic reference implementation's discipline.
- `PineconeVectorStore` — implemented against the documented client API, and now
  driven by a real embedder (see below) rather than the hash placeholder. Still
  never run against a live index.
- Redis, Postgres, the Docker images, `render.yaml` and the MCP `stdio` server —
  written to spec, not run.

### Deploying

| Target | Config | Notes |
|---|---|---|
| Hugging Face Spaces | `deploy/huggingface/` | Docker SDK, more memory, runs the container |
| Streamlit Community Cloud | `streamlit_app.py` + `requirements.txt` | ~1GB RAM; lower `MAX_COLLEGES_PER_ANALYSIS` |
| Render | `render.yaml` | API + UI + Postgres + Redis |

Spaces is the better free option for this app: twelve concurrent college
assessments are memory-hungry, and Community Cloud is tight. See
`deploy/huggingface/DEPLOY.md`.

Three things bite on any free host. **Cost is per visitor** — a public deploy
spends your inference credits on whoever opens it; leaving `LLM_PROVIDER` unset
runs the free deterministic engine instead. **Sleeping instances lose state**,
so an analysis paused at a human-verification interrupt is gone on wake unless
`CHECKPOINT_BACKEND=postgres`. And **the corpus is baked into the image**, so
re-ingesting colleges means a rebuild.

### Deploying to Streamlit Community Cloud

Push to GitHub, then in the deploy dialog set the entrypoint to
**`streamlit_app.py`** (not `src/pathora/ui/app.py` — Community Cloud installs
`requirements.txt` but does not pip-install the repo, so the src-layout package
would not be importable).

Paste configuration into App settings -> Secrets; see
`.streamlit/secrets.toml.example`. Keys are uppercased into environment
variables before Settings is first read.

Ollama cannot be used from a deployed app — it listens on localhost. For a
deployed demo either keep `LLM_PROVIDER = "fake"` (zero cost, fully functional,
deterministic) or point `LLM_BASE_URL` at a hosted OpenAI-compatible endpoint.

`requirements.txt` deliberately omits Postgres, Redis, Pinecone and MCP: the
deployed demo runs on SQLite and the in-memory vector store. Two tests guard
this — one asserts every runtime import is pinned, the other asserts the heavy
extras stay out.

### Running with no API cost at all

```bash
brew install ollama && ollama serve
ollama pull qwen2.5:7b-instruct

export LLM_PROVIDER=openai
export LLM_BASE_URL=http://localhost:11434/v1
export DEFAULT_MODEL=qwen2.5:7b-instruct
```

One provider class covers OpenAI, Ollama, LM Studio, Nebius, Groq, OpenRouter,
DeepSeek, Together and vLLM — they differ only by base URL and model name. Local backends
need no key; the provider detects localhost and omits the auth header.

Each vendor is a named `LLM_PROVIDER` carrying its own default endpoint, so
switching backends is one variable:

```bash
export LLM_PROVIDER=nebius            # or groq | openrouter | deepseek |
export LLM_API_KEY=$NEBIUS_API_KEY    #    together | ollama | lmstudio | openai
export DEFAULT_MODEL=<id from tokenfactory.nebius.com/models/catalog>
```

`LLM_BASE_URL` overrides the default when a vendor moves an endpoint —
Nebius AI Studio was rebranded Nebius Token Factory in 2026, so verify the URL
against their current docs if you hit connection errors. `LLM_API_KEY` is
vendor-neutral and takes precedence over `OPENAI_API_KEY`; `ollama` and
`lmstudio` need no key at all.

### Using Pinecone

The default `EMBEDDING_BACKEND=hash` is a bag-of-words hashing function: offline,
free, deterministic and **not semantic** — "acceptance rate" and "admit rate"
land nowhere near each other. That is adequate for the small seeded corpus and
actively wrong for a paid vector database, where you would be paying to store
vectors that carry no meaning.

So switch the embedder before switching the store:

```bash
EMBEDDING_BACKEND=openai_compatible          # reuses LLM_PROVIDER + LLM_API_KEY
EMBEDDING_MODEL=BAAI/bge-multilingual-gemma2 # or Qwen/Qwen3-Embedding-8B
PINECONE_API_KEY=...

python scripts/setup_pinecone.py --dry-run
python scripts/setup_pinecone.py
```

The setup script measures the real vector dimension by embedding a probe string
rather than trusting `EMBEDDING_DIM`, and refuses to run at all on the hash
backend. This matters because Pinecone cannot change an index's dimension after
creation — a mismatch means deleting and rebuilding.

Then set `VECTOR_BACKEND=pinecone` and restart.

### Choosing an open model

Pathora's binding constraint is schema adherence, not general capability: every
agent returns a validated Pydantic model. Prefer **instruct** models over
**reasoning** models — R1/QwQ-style models emit long chains of thought before
the answer, which inflates latency, risks truncation against `max_tokens`, and
conflicts with the standing instruction not to expose reasoning.

`python scripts/check_provider.py` runs the three real schemas against whatever
is configured and reports validity, latency and whether the model invented a
statistic when handed a source containing none. Non-zero exit on failure, so it
can gate a deploy.

```bash
python scripts/check_provider.py --model meta-llama/Meta-Llama-3.3-70B-Instruct
```

Schema enforcement degrades gracefully: OpenAI gets native `json_schema`, other
hosts get `json_object` mode, and everything falls back to the repair loop.
Expect small local models to need more repair rounds than a frontier model — the
loop is bounded by `LLM_MAX_RETRIES`.

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
