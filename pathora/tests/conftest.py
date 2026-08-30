from __future__ import annotations

from pathlib import Path

import pytest

from pathora.config import Settings
from pathora.graph.nodes import Deps
from pathora.llm.fake import FakeProvider
from pathora.rag.store import load_seed_payload, seeded_store
from pathora.service import PathoraService

ROOT = Path(__file__).resolve().parents[1]
SAMPLE_TRANSCRIPT = (ROOT / "data/seed/sample_transcript.txt").read_text()


@pytest.fixture
def settings() -> Settings:
    return Settings(
        llm_provider="fake",
        vector_backend="memory",
        max_colleges_per_analysis=8,
        max_parallel_college_workers=5,
        max_research_retries=2,
        max_critic_loops=2,
    )


@pytest.fixture
async def store(settings):
    return await seeded_store(settings)


@pytest.fixture
def provider() -> FakeProvider:
    return FakeProvider()


@pytest.fixture
async def deps(provider, store, settings) -> Deps:
    return Deps(
        provider=provider,
        store=store,
        settings=settings,
        catalog=load_seed_payload()["colleges"],
    )


@pytest.fixture
async def service(deps) -> PathoraService:
    return PathoraService(deps)


@pytest.fixture
def student_input() -> dict:
    return {
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
                "hours_per_week": 2,
                "description": "Weekly tutoring for algebra students",
            },
        ],
        "projects": [
            {
                "name": "Bus route delay tracker",
                "description": "Scraped published transit data and charted delays",
                "technologies": ["Python", "SQLite"],
            }
        ],
        "awards": [{"name": "Regional Robotics Finalist", "level": "Regional", "year": "2026"}],
        "stem_interests": ["Computer Science", "Data Science"],
        "career_interests": ["Software Engineer"],
        "preferences": {
            "locations": ["TX"],
            "public_private": "Public",
            "school_size": "Large",
            "cost_sensitivity": "High",
        },
    }


@pytest.fixture
def transcript_document() -> dict:
    return {"text": SAMPLE_TRANSCRIPT}
