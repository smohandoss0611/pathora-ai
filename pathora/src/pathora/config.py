"""Central configuration. Every graph limit and model choice is env-overridable."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

#: `.env` next to the repo root, plus the working directory. Pydantic-settings
#: resolves a bare ".env" against the CWD, so running a script from anywhere but
#: the repo root silently loaded no configuration at all — the key appeared
#: unset with no error to explain it.
REPO_ROOT = Path(__file__).resolve().parents[2]
ENV_FILES = (REPO_ROOT / ".env", Path(".env"))


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=ENV_FILES, extra="ignore")

    # --- Graph limits (Section 7) -------------------------------------------
    max_colleges_per_analysis: int = 8
    # catalog | open | hybrid
    #   catalog  choose only from the indexed corpus (default; every candidate
    #            is guaranteed to have documents to research)
    #   open     the model proposes real universities from its own knowledge
    #   hybrid   indexed colleges first, model fills the remaining slots
    # In open/hybrid, proposed schools with no indexed documents will be refused
    # by the evidence gate rather than assessed from model memory.
    college_discovery_mode: str = "catalog"
    max_parallel_college_workers: int = 5
    max_research_retries: int = 2
    max_critic_loops: int = 2

    # --- LLM provider (Section 31) ------------------------------------------
    # fake | anthropic | nebius | groq | openrouter | deepseek | together |
    # ollama | lmstudio | openai
    # Everything except fake/anthropic uses the OpenAI-compatible provider; the
    # name selects a default base URL, which LLM_BASE_URL overrides.
    llm_provider: str = "fake"
    llm_base_url: str | None = None
    #: Vendor-neutral key for any OpenAI-compatible endpoint. Takes precedence
    #: over OPENAI_API_KEY so NEBIUS_API_KEY / GROQ_API_KEY / TOGETHER_API_KEY
    #: can be mapped without pretending they are OpenAI keys.
    llm_api_key: str | None = None
    anthropic_api_key: str | None = None
    openai_api_key: str | None = None
    #: Large open models answering with a full digital twin plus research
    #: payload routinely take longer than a minute; 60s produced ReadTimeouts
    #: mid-analysis.
    llm_timeout_seconds: float = 180.0
    llm_max_retries: int = 2

    # --- Model routing (Section 32) -----------------------------------------
    default_model: str = "claude-sonnet-4-6"
    profile_model: str | None = None
    activity_model: str | None = None
    stem_model: str | None = None
    research_model: str | None = None
    admission_model: str | None = None
    critic_model: str | None = None
    action_model: str | None = None
    roadmap_model: str | None = None

    # --- RAG -----------------------------------------------------------------
    vector_backend: str = "memory"  # memory | pinecone
    pinecone_api_key: str | None = None
    pinecone_index: str = "pathora-colleges"
    # hash | openai_compatible | pinecone
    # hash is offline and NOT semantic: fine for the demo corpus, useless in a
    # real vector database. Use a real embedder before paying for Pinecone.
    embedding_backend: str = "hash"
    embedding_model: str = "BAAI/bge-multilingual-gemma2"
    embedding_dim: int = 256
    embedding_base_url: str | None = None
    embedding_batch_size: int = 64

    #: On-demand lookup when a researched college has no indexed documents.
    #: This makes *coverage* live, not the figures: admissions data is annual
    #: federal reporting, so there is no fresher truth to fetch.
    live_lookup_enabled: bool = True
    live_lookup_timeout: float = 15.0
    live_lookup_ttl_seconds: int = 86400
    #: Minimum name-match score to accept a federal record. Deliberately strict:
    #: a branch campus can differ from the flagship by 30 points of admit rate
    #: and 300 points of SAT range, so a near-miss must abstain, not classify.
    scorecard_match_threshold: float = 0.9
    scorecard_api_key: str | None = None

    rag_top_k: int = 6
    rag_rerank_enabled: bool = False
    evidence_stale_after_days: int = 365
    #: Annual survey data (IPEDS, Common Data Set) is *published* on a lag —
    #: the most recent IPEDS year is always one to two cycles behind. Judging it
    #: against the same freshness window as a live admissions page would make
    #: the most complete admissions dataset available permanently unusable.
    annual_survey_stale_after_days: int = 1095

    # --- Infrastructure ------------------------------------------------------
    database_url: str = "sqlite+pysqlite:///./pathora.db"
    redis_url: str | None = None
    checkpoint_backend: str = "memory"  # memory | postgres

    # --- Observability -------------------------------------------------------
    langsmith_tracing: bool = False
    langsmith_api_key: str | None = None
    langsmith_project: str = "pathora-ai"

    # --- Transcript ----------------------------------------------------------
    extraction_confidence_threshold: float = 0.75

    def model_for(self, task: str) -> str:
        """Resolve the model for a task, falling back to the default."""
        return getattr(self, f"{task}_model", None) or self.default_model

    @property
    def compatible_api_key(self) -> str | None:
        return self.llm_api_key or self.openai_api_key


@lru_cache
def get_settings() -> Settings:
    return Settings()


def config_report() -> str:
    """Human-readable answer to 'why is my key not loading?'."""
    settings = get_settings()
    lines = [f"repo root      : {REPO_ROOT}", f"working dir    : {Path.cwd()}"]
    for path in ENV_FILES:
        resolved = path if path.is_absolute() else Path.cwd() / path
        lines.append(f"env file       : {resolved} {'FOUND' if resolved.exists() else 'missing'}")
    lines += [
        f"llm provider   : {settings.llm_provider}",
        f"llm key set    : {bool(settings.compatible_api_key or settings.anthropic_api_key)}",
        f"scorecard key  : {bool(settings.scorecard_api_key)}",
        f"live lookup    : {settings.live_lookup_enabled}",
        f"discovery mode : {settings.college_discovery_mode}",
        f"vector backend : {settings.vector_backend}",
    ]
    return "\n".join(lines)
