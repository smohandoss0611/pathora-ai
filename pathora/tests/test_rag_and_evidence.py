from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from pathora.agents.college_worker import college_research_worker, research_many
from pathora.domain.models import (
    NOT_PUBLISHED,
    AdmissionAssessment,
    CollegeCandidate,
    CollegeResearchResult,
    EvidenceRecord,
)
from pathora.mcp_server import college as mcp_tools
from pathora.rag.ingest import chunk, clean, to_documents
from pathora.rag.store import Document, InMemoryVectorStore
from pathora.services.evidence import analyze_gap, build_passport
from pathora.services.transcript import parse_transcript_text
from pathora.services.twin import build_digital_twin


class TestIngestion:
    def test_clean_strips_boilerplate_and_whitespace(self):
        assert "cookie" not in clean("Cookie Policy\nReal   content here").lower()

    def test_chunking_respects_size(self):
        text = ". ".join(f"Sentence number {i} about admissions" for i in range(120)) + "."
        chunks = chunk(text, size=300, overlap=40)
        assert len(chunks) > 1
        assert all(len(c) <= 320 for c in chunks)

    def test_short_text_is_one_chunk(self):
        assert len(chunk("A single short sentence.")) == 1

    def test_every_chunk_carries_required_metadata(self):
        docs = to_documents(
            {
                "id": "x",
                "university": "Test U",
                "title": "Admissions",
                "text": "Admit rate is published here. " * 80,
                "source_url": "https://example.edu/a",
                "source_type": "official_admissions",
                "published_at": "2026-01-01T00:00:00+00:00",
            }
        )
        assert len(docs) > 1
        for doc in docs:
            assert doc.university and doc.source_url and doc.source_type
            assert doc.retrieved_at is not None


class TestRetrieval:
    async def test_filters_by_university(self, store):
        chunks = await store.query("admission requirements", university="Harborview University")
        assert chunks
        assert {c.university for c in chunks} == {"Harborview University"}

    async def test_depth_gate_hides_deep_documents(self, store):
        shallow = await store.query(
            "admission", university="Rio Blanco State University", max_depth=0, top_k=10
        )
        deep = await store.query(
            "admission", university="Rio Blanco State University", max_depth=1, top_k=10
        )
        assert len(deep) > len(shallow)

    async def test_rerank_prefers_authoritative_sources(self, settings):
        store = InMemoryVectorStore(settings)
        await store.upsert(
            [
                Document(
                    id="blog",
                    university="U",
                    text="admission admission admission rate",
                    source_url="https://blog.example.com",
                    source_type="other",
                ),
                Document(
                    id="official",
                    university="U",
                    text="admission rate",
                    source_url="https://u.example.edu/admissions",
                    source_type="official_admissions",
                ),
            ]
        )
        default = await store.query("admission rate", top_k=2, rerank=False)
        reranked = await store.query("admission rate", top_k=2, rerank=True)
        assert [c.score for c in default] == sorted((c.score for c in default), reverse=True)
        assert reranked[0].source_type == "official_admissions"


class TestResearchWorker:
    async def test_returns_facts_only_from_evidence(self, store, settings):
        result = await college_research_worker(
            CollegeCandidate(
                university="Lakeside State University", target_major="Computer Science"
            ),
            store,
            settings=settings,
        )
        assert result.admit_rate != NOT_PUBLISHED
        assert result.evidence
        assert all(e.university == "Lakeside State University" for e in result.evidence)

    async def test_missing_facts_are_labelled_not_invented(self, store, settings):
        result = await college_research_worker(
            CollegeCandidate(
                university="Northgate Institute of Technology", target_major="Cybersecurity"
            ),
            store,
            settings=settings,
        )
        assert result.admit_rate == NOT_PUBLISHED
        assert result.sat_range == NOT_PUBLISHED
        assert result.missing_information

    async def test_deep_pass_finds_more_evidence(self, store, settings):
        candidate = CollegeCandidate(
            university="Rio Blanco State University", target_major="Industrial Engineering"
        )
        shallow = await college_research_worker(candidate, store, settings=settings)
        deep = await college_research_worker(candidate, store, deep=True, settings=settings)
        assert len(deep.evidence) > len(shallow.evidence)
        assert any(e.source_type == "official_admissions" for e in deep.evidence)

    async def test_unknown_university_degrades_gracefully(self, store, settings):
        result = await college_research_worker(
            CollegeCandidate(university="Nonexistent U", target_major="Statistics"),
            store,
            settings=settings,
        )
        assert result.evidence == []
        assert result.missing_information

    async def test_fan_out_is_bounded_and_fans_in(self, store, settings):
        settings.max_parallel_college_workers = 2
        candidates = [
            CollegeCandidate(university=u, target_major="Computer Science")
            for u in ("Lakeside State University", "Cedar Valley University", "Bayland University")
        ]
        results = await research_many(candidates, store, settings=settings)
        assert set(results) == {c.university for c in candidates}
        assert all(isinstance(r, CollegeResearchResult) for r in results.values())


class TestMcpTools:
    async def test_all_four_tools_registered(self):
        assert set(mcp_tools.TOOLS) == {
            "research_college",
            "search_college_documents",
            "get_program_info",
            "get_admission_policy",
        }

    async def test_research_college(self, store):
        result = await mcp_tools.research_college(
            "Harborview University", "Data Science", store=store
        )
        assert result["university"] == "Harborview University"
        assert result["evidence"]

    async def test_search_documents_returns_citations(self, store):
        results = await mcp_tools.search_college_documents("test policy", top_k=3, store=store)
        assert results
        assert all(r["source_url"].startswith("https://") for r in results)

    async def test_program_info(self, store):
        info = await mcp_tools.get_program_info(
            "Crescent Bay University", "Computer Science", store=store
        )
        assert info["admission_structure"] != NOT_PUBLISHED

    async def test_admission_policy(self, store):
        policy = await mcp_tools.get_admission_policy("Bayland University", store=store)
        assert policy["test_policy"] != NOT_PUBLISHED
        assert policy["sources"]


def _assessment(**kw) -> AdmissionAssessment:
    base = dict(
        university="Test U",
        recommended_major="Computer Science",
        classification="Target",
        confidence="Moderate",
        evidence_ids=["a"],
    )
    return AdmissionAssessment(**(base | kw))


def _evidence(source_type: str, days_old: int = 30) -> EvidenceRecord:
    return EvidenceRecord(
        evidence_id=source_type,
        university="Test U",
        source_url="https://test.example.edu",
        source_type=source_type,  # type: ignore[arg-type]
        published_at=datetime.now(UTC) - timedelta(days=days_old),
    )


class TestEvidencePassport:
    def test_high_quality_requires_official_admissions_and_no_gaps(self):
        research = CollegeResearchResult(
            university="Test U",
            target_major="Computer Science",
            evidence=[
                _evidence("official_admissions"),
                _evidence("official_stem_program"),
                _evidence("common_data_set"),
            ],
        )
        passport = build_passport(research, _assessment(), profile_verified=True)
        assert passport.quality == "HIGH"

    def test_missing_official_source_downgrades(self):
        research = CollegeResearchResult(
            university="Test U", target_major="CS", evidence=[_evidence("other")]
        )
        passport = build_passport(research, _assessment(), profile_verified=True)
        assert passport.quality in {"LOW", "MEDIUM"}
        assert "No official admissions source retrieved" in passport.missing

    def test_stale_evidence_flagged(self):
        research = CollegeResearchResult(
            university="Test U",
            target_major="CS",
            evidence=[_evidence("official_admissions", days_old=900)],
        )
        passport = build_passport(
            research, _assessment(), profile_verified=True, stale_after_days=365
        )
        assert passport.stale_evidence_ids


class TestGapAnalyzer:
    @pytest.fixture
    def twin(self):
        extracted = parse_transcript_text(
            "Lakeview High School\n"
            "Class of 2027\n"
            "2025-2026\n"
            "AP Calculus AB                    A     1.0\n"
            "AP Physics 1                      B     1.0\n"
        )
        return build_digital_twin(student_id="s", verified_academics=extracted)

    def test_impacts_are_qualitative_only(self, twin):
        research = CollegeResearchResult(university="Test U", target_major="CS")
        analysis = analyze_gap(twin, research, _assessment(classification="Reach"))
        assert all(f.impact in {"High", "Medium", "Low"} for f in analysis.factors)
        assert analysis.primary_constraint

    def test_test_blind_policy_lowers_test_impact(self, twin):
        research = CollegeResearchResult(
            university="Test U", target_major="CS", test_policy="Test blind"
        )
        analysis = analyze_gap(twin, research, _assessment())
        test_factor = next(f for f in analysis.factors if f.factor == "SAT / ACT")
        assert test_factor.impact == "Low"

    def test_capped_major_adds_capacity_constraint(self, twin):
        research = CollegeResearchResult(
            university="Test U",
            target_major="CS",
            admission_structure="Direct admission; capped major with a separate committee",
        )
        analysis = analyze_gap(twin, research, _assessment(classification="Reach"))
        assert any(f.factor == "Major capacity" for f in analysis.factors)
