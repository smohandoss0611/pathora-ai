# Deploying to Hugging Face Spaces

Spaces gives more memory than Streamlit Community Cloud and can run the
container directly, which matters here: twelve concurrent college assessments
are memory-hungry.

## 1. Create the Space

huggingface.co/new-space → **SDK: Docker** → Blank template.

## 2. Push the code

```bash
git remote add space https://huggingface.co/spaces/<user>/pathora-ai
cp deploy/huggingface/Dockerfile ./Dockerfile.hf
cp deploy/huggingface/README.md ./README.hf.md
```

The Space needs `Dockerfile` and `README.md` (with YAML frontmatter) at the
repo root. Either rename these, or keep a deploy branch where they sit at the
top level — the frontmatter README conflicts with the project README, so a
separate branch is cleaner than overwriting it.

```bash
git checkout -b space
mv README.hf.md README.md
mv Dockerfile.hf Dockerfile
git add -A && git commit -m "Space config"
git push space space:main
```

## 3. Secrets

Settings → Variables and secrets. Add `LLM_API_KEY` and `SCORECARD_API_KEY` as
**secrets**; add the rest as **variables** (they are not sensitive):

```
LLM_PROVIDER=nebius
DEFAULT_MODEL=Qwen/Qwen3-235B-A22B-Instruct-2507
COLLEGE_DISCOVERY_MODE=hybrid
MAX_COLLEGES_PER_ANALYSIS=5
LLM_TIMEOUT_SECONDS=180
```

## Things that will bite

**Cost is per visitor.** A public Space spends your Nebius credits on anyone who
opens it. Set the Space to private for a graded demo, or leave `LLM_PROVIDER`
unset so it runs the free deterministic engine.

**Sleeping Spaces lose state.** Free Spaces sleep after inactivity; the
in-memory checkpointer means any analysis paused at a human-verification
interrupt is gone on wake. Fine for a demo, not for real use — that needs
`CHECKPOINT_BACKEND=postgres`.

**The corpus is baked into the image.** `data/seed/*.colleges.json` ships with
the build, so re-ingesting means a rebuild. Run the ingest scripts locally and
commit the corpus files.

**First request is slow.** A cold Space plus a cold Scorecard cache plus a 235B
model means the first analysis can take several minutes. Warm it before
recording.
