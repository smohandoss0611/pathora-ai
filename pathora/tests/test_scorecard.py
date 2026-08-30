"""College Scorecard ingestion, against stubbed API responses.

No live call is made here (or anywhere in this repo's test suite), so these
cover shape handling and the honesty guarantees rather than connectivity.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def scorecard():
    spec = importlib.util.spec_from_file_location(
        "ingest_scorecard", ROOT / "scripts/ingest_scorecard.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def row(**overrides):
    base = {
        "id": 228723,
        "school.name": "Texas A&M University",
        "school.state": "TX",
        "school.ownership": 1,
        "school.school_url": "www.tamu.edu",
        "latest.student.size": 57000,
        "latest.admissions.admission_rate.overall": 0.628,
        "latest.admissions.sat_scores.25th_percentile.critical_reading": 590,
        "latest.admissions.sat_scores.75th_percentile.critical_reading": 690,
        "latest.admissions.sat_scores.25th_percentile.math": 580,
        "latest.admissions.sat_scores.75th_percentile.math": 720,
        "latest.admissions.act_scores.25th_percentile.cumulative": 25,
        "latest.admissions.act_scores.75th_percentile.cumulative": 31,
    }
    return base | overrides


class TestRecordBuilding:
    def test_admission_rate_converted_to_percent(self, scorecard):
        record = scorecard.build_record(row(), "2026")
        assert record["facts"]["admit_rate"].startswith("62.8%")

    def test_university_wide_caveat_is_stated_in_the_fact_itself(self, scorecard):
        """The Critic reads this text; the caveat must survive into the fact."""
        record = scorecard.build_record(row(), "2026")
        assert "university-wide" in record["facts"]["admit_rate"]
        assert "not the admit rate for any individual major" in record["text"]

    def test_sat_sections_are_summed(self, scorecard):
        record = scorecard.build_record(row(), "2026")
        assert record["facts"]["sat_range"].startswith("1170-1410")

    def test_act_range_captured(self, scorecard):
        assert scorecard.build_record(row(), "2026")["facts"]["act_range"].startswith("25-31")

    def test_school_with_no_reported_figures_is_dropped(self, scorecard):
        empty = {k: None for k in row()}
        empty["school.name"] = "Nowhere College"
        empty["id"] = 1
        assert scorecard.build_record(empty, "2026") is None

    def test_zero_admission_rate_is_not_treated_as_published(self, scorecard):
        """0.0 means 'not reported' in Scorecard, not 'admits nobody'."""
        record = scorecard.build_record(
            row(**{"latest.admissions.admission_rate.overall": 0}), "2026"
        )
        assert "admit_rate" not in (record["facts"] if record else {})

    def test_nested_response_shape_is_supported(self, scorecard):
        nested = {
            "id": 1,
            "school": {"name": "Nested U", "state": "TX", "ownership": 1},
            "latest": {"admissions": {"admission_rate": {"overall": 0.5}}},
        }
        record = scorecard.build_record(nested, "2026")
        assert record["university"] == "Nested U"
        assert record["facts"]["admit_rate"].startswith("50.0%")

    def test_source_type_is_institutional_research(self, scorecard):
        assert scorecard.build_record(row(), "2026")["source_type"] == "institutional_research"


class TestCatalogEntry:
    def test_ownership_and_size_mapped(self, scorecard):
        entry = scorecard.catalog_entry(row())
        assert entry["control"] == "Public"
        assert entry["size"] == "Large"

    @pytest.mark.parametrize(
        ("size", "band"),
        [(30000, "Large"), (8000, "Medium"), (1200, "Small"), (None, "NoPreference")],
    )
    def test_size_bands(self, scorecard, size, band):
        assert scorecard.size_band(size) == band


class TestGateIntegration:
    async def test_scorecard_record_clears_the_evidence_gate(self, scorecard, settings):
        from pathora.agents.college_worker import college_research_worker
        from pathora.domain.models import CollegeCandidate
        from pathora.rag.ingest import ingest_records
        from pathora.rag.store import InMemoryVectorStore
        from pathora.services.evidence_gate import evidence_gate

        store = InMemoryVectorStore(settings)
        await ingest_records([scorecard.build_record(row(), "2026")], store)
        result = await college_research_worker(
            CollegeCandidate(university="Texas A&M University", target_major="Computer Science"),
            store,
            settings=settings,
        )
        assert evidence_gate(result, settings=settings).passed

    async def test_major_specific_rate_remains_unpublished(self, scorecard, settings):
        """Scorecard cannot supply it, and nothing may invent it."""
        from pathora.agents.college_worker import college_research_worker
        from pathora.domain.models import NOT_PUBLISHED, CollegeCandidate
        from pathora.rag.ingest import ingest_records
        from pathora.rag.store import InMemoryVectorStore

        store = InMemoryVectorStore(settings)
        await ingest_records([scorecard.build_record(row(), "2026")], store)
        result = await college_research_worker(
            CollegeCandidate(university="Texas A&M University", target_major="Computer Science"),
            store,
            settings=settings,
        )
        assert result.major_admit_rate == NOT_PUBLISHED
