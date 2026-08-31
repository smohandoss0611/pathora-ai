from __future__ import annotations

import pytest

from pathora.persistence.cache import InMemoryCache
from pathora.persistence.repository import Repository


@pytest.fixture
def repo(tmp_path):
    return Repository(f"sqlite+pysqlite:///{tmp_path / 't.db'}")


class TestRepository:
    def test_saves_and_reloads_an_analysis(self, repo):
        state = {
            "user_id": "u1",
            "student_id": "s1",
            "workflow_status": "complete",
            "student_twin": {"testing": {"sat_total": 1450}, "activities": [], "projects": []},
            "verified_academics": {"graduation_year": 2027},
            "college_candidates": ["Lakeside State University"],
            "admission_results": {"Lakeside State University": {"classification": "Target"}},
            "critic_results": {"decision": "approve"},
            "next_actions": [{"title": "Do the thing", "priority": "High"}],
            "roadmap": {"today": []},
            "college_research": {
                "Lakeside State University": {
                    "evidence": [
                        {
                            "evidence_id": "lakeside-adm",
                            "university": "Lakeside State University",
                            "source_url": "https://lakeside.example.edu/admissions",
                            "source_type": "official_admissions",
                            "snippet": "Admit rate 58%",
                        }
                    ]
                }
            },
        }
        repo.save_analysis(state, thread_id="t1")
        analysis = repo.get_analysis("t1")
        assert analysis is not None
        assert analysis.workflow_status == "complete"
        assert analysis.college_list == ["Lakeside State University"]

    def test_evidence_stored_independently_of_prose(self, repo):
        repo.save_analysis(
            {
                "user_id": "u1",
                "student_id": "s1",
                "college_research": {
                    "U": {
                        "evidence": [
                            {
                                "evidence_id": "e1",
                                "university": "U",
                                "source_url": "https://u.example.edu",
                                "source_type": "common_data_set",
                            }
                        ]
                    }
                },
            },
            thread_id="t2",
        )
        records = repo.evidence_for("t2")
        assert len(records) == 1
        assert records[0].source_type == "common_data_set"

    def test_resave_replaces_evidence_without_duplicating(self, repo):
        state = {
            "user_id": "u1",
            "student_id": "s1",
            "college_research": {
                "U": {
                    "evidence": [
                        {
                            "evidence_id": "e1",
                            "university": "U",
                            "source_url": "https://u.example.edu",
                            "source_type": "official_admissions",
                        }
                    ]
                }
            },
        }
        repo.save_analysis(state, thread_id="t3")
        repo.save_analysis(state, thread_id="t3")
        assert len(repo.evidence_for("t3")) == 1


class TestCache:
    async def test_set_get_roundtrip(self):
        cache = InMemoryCache()
        await cache.set("k", {"a": 1})
        assert await cache.get("k") == {"a": 1}

    async def test_missing_key_returns_none(self):
        assert await InMemoryCache().get("nope") is None

    async def test_rate_limit_blocks_after_limit(self):
        cache = InMemoryCache()
        assert await cache.allow("ip", limit=2, window_seconds=60)
        assert await cache.allow("ip", limit=2, window_seconds=60)
        assert not await cache.allow("ip", limit=2, window_seconds=60)

    async def test_lock_is_exclusive(self):
        cache = InMemoryCache()
        order = []

        async def worker(n):
            async with cache.lock("job"):
                order.append(("in", n))
                order.append(("out", n))

        import asyncio

        await asyncio.gather(worker(1), worker(2))
        assert order in (
            [("in", 1), ("out", 1), ("in", 2), ("out", 2)],
            [("in", 2), ("out", 2), ("in", 1), ("out", 1)],
        )


class TestCheckpointer:
    def test_memory_backend_is_default(self, settings):
        from langgraph.checkpoint.memory import MemorySaver

        from pathora.graph.checkpointer import build_checkpointer

        assert isinstance(build_checkpointer(settings), MemorySaver)

    def test_unknown_backend_fails_loudly(self, settings):
        from pathora.graph.checkpointer import build_checkpointer

        settings.checkpoint_backend = "sqlite"
        with pytest.raises(ValueError, match="unknown CHECKPOINT_BACKEND"):
            build_checkpointer(settings)

    def test_postgres_backend_rejects_a_non_postgres_dsn(self, settings):
        from pathora.graph.checkpointer import build_checkpointer

        settings.checkpoint_backend = "postgres"
        settings.database_url = "sqlite+pysqlite:///./pathora.db"
        with pytest.raises((RuntimeError, ImportError)):
            build_checkpointer(settings)


class TestEnvExample:
    """.env.example is documentation that silently rots. Pin it to Settings."""

    def _declared(self):
        from pathlib import Path

        from dotenv import dotenv_values

        return dotenv_values(Path(__file__).resolve().parents[1] / ".env.example")

    def test_every_setting_is_documented(self):
        from pathora.config import Settings

        declared = {k.upper() for k in self._declared()}
        assert {f.upper() for f in Settings.model_fields} - declared == set()

    def test_no_undocumented_keys(self):
        from pathora.config import Settings

        declared = {k.upper() for k in self._declared()}
        assert declared - {f.upper() for f in Settings.model_fields} == set()

    def test_values_parse_without_inline_comment_contamination(self):
        for key, value in self._declared().items():
            assert value is None or "#" not in value, f"{key} carries an inline comment"


class TestEnvResolution:
    """A bare env_file resolves against the CWD, so running a script from
    anywhere but the repo root silently loaded nothing."""

    def test_env_files_include_an_absolute_repo_path(self):
        from pathora.config import ENV_FILES, REPO_ROOT

        assert any(p.is_absolute() for p in ENV_FILES)
        assert (REPO_ROOT / "pyproject.toml").exists()

    def test_config_report_names_every_env_path_checked(self):
        from pathora.config import config_report

        report = config_report()
        assert "env file" in report
        assert "scorecard key" in report
