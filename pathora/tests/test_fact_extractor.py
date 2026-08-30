"""The extractor's job is to never invent. These tests are mostly about absence."""

from __future__ import annotations

from pathora.agents.fact_extractor import extract_facts, extract_from_chunk
from pathora.rag.store import RetrievedChunk


def chunk(
    text: str, *, cid: str = "c1", source_type: str = "official_admissions"
) -> RetrievedChunk:
    return RetrievedChunk(
        id=cid,
        university="Test U",
        title="",
        text=text,
        source_url="https://test.example.edu",
        source_type=source_type,  # type: ignore[arg-type]
    )


def facts_of(text: str) -> dict[str, str]:
    return {f.field: f.value for f in extract_from_chunk(chunk(text))}


class TestNeverInvents:
    def test_empty_text_yields_nothing(self):
        assert extract_from_chunk(chunk("")) == []

    def test_prose_without_numbers_yields_no_stats(self):
        text = "We review every application holistically and welcome students from all backgrounds."
        assert "admit_rate" not in facts_of(text)
        assert "sat_range" not in facts_of(text)

    def test_unrelated_percentage_is_not_read_as_an_admit_rate(self):
        text = "About 92% of our graduates are employed within six months."
        assert "admit_rate" not in facts_of(text)

    def test_a_year_range_is_not_read_as_an_sat_range(self):
        assert "sat_range" not in facts_of("Data covering 2023-2024 enrollment.")

    def test_page_that_only_lists_deadlines_yields_only_deadlines(self):
        result = facts_of("Regular decision deadline is January 15 for all applicants.")
        assert set(result) == {"deadlines"}


class TestExtracts:
    def test_admit_rate_phrasings(self):
        assert facts_of("Admit rate: 43.4%")["admit_rate"] == "43.4%"
        assert facts_of("We admitted 53% of applicants.")["admit_rate"] == "53%"
        assert facts_of("An 11% acceptance rate for the class.")["admit_rate"] == "11%"

    def test_sat_and_act_ranges_with_en_dash(self):
        result = facts_of("Middle 50% SAT 1220\u20131470 and ACT 28\u201334.")
        assert result["sat_range"].startswith("1220-1470")
        assert result["act_range"].startswith("28-34")

    def test_reversed_range_is_normalized(self):
        assert facts_of("SAT 1470-1220").get("sat_range", "").startswith("1220-1470")

    def test_test_policy_variants(self):
        assert facts_of("We are test-optional.")["test_policy"] == "Test optional"
        assert facts_of("This campus is test blind.")["test_policy"] == "Test blind"
        assert facts_of("SAT scores are required.")["test_policy"] == "Test required"

    def test_test_blind_wins_over_optional_when_both_appear(self):
        assert facts_of("Test blind. Formerly test optional.")["test_policy"] == "Test blind"

    def test_admission_structure(self):
        assert facts_of("Direct admission to major.")["admission_structure"] == (
            "Direct admission to major"
        )
        assert facts_of("This is a capped major.")["admission_structure"] == "Capped major"

    def test_transfer_restriction_captured_verbatim(self):
        result = facts_of("Change of major into Computer Science is not permitted.")
        assert "not permitted" in result["transfer_restrictions"]


class TestProvenance:
    def test_every_fact_traces_to_an_evidence_id(self):
        facts, provenance = extract_facts([chunk("Admit rate: 22%. Test optional.", cid="doc-a")])
        assert set(facts) == set(provenance)
        assert all(v == "doc-a" for v in provenance.values())

    def test_more_authoritative_source_wins(self):
        chunks = [
            chunk("Admit rate: 50%", cid="blog", source_type="other"),
            chunk("Admit rate: 22%", cid="cds", source_type="common_data_set"),
        ]
        facts, provenance = extract_facts(chunks)
        assert facts["admit_rate"] == "22%"
        assert provenance["admit_rate"] == "cds"

    def test_matched_span_is_recorded_for_audit(self):
        [fact] = [
            f for f in extract_from_chunk(chunk("Acceptance rate 17%")) if f.field == "admit_rate"
        ]
        assert "17" in fact.matched_text


class TestIngestRegression:
    """A freshly created store is empty, and an empty store must not be falsy."""

    async def test_empty_store_is_truthy(self, settings):
        from pathora.rag.store import InMemoryVectorStore

        assert bool(InMemoryVectorStore(settings)) is True

    async def test_ingest_writes_to_the_store_it_was_given(self, settings):
        from pathora.rag.ingest import ingest_records
        from pathora.rag.store import InMemoryVectorStore

        store = InMemoryVectorStore(settings)
        await ingest_records(
            [
                {
                    "id": "d1",
                    "university": "Test U",
                    "title": "Admissions",
                    "text": "Acceptance rate 41%. Test optional. " * 10,
                    "source_url": "https://test.example.edu",
                    "source_type": "official_admissions",
                }
            ],
            store,
        )
        assert len(store) > 0

    async def test_facts_extracted_from_unstructured_prose(self, settings):
        """Real pages carry no seeded `facts`; the worker must read the text."""
        from pathora.agents.college_worker import college_research_worker
        from pathora.domain.models import CollegeCandidate
        from pathora.rag.ingest import ingest_records
        from pathora.rag.store import InMemoryVectorStore

        store = InMemoryVectorStore(settings)
        await ingest_records(
            [
                {
                    "id": "u-adm",
                    "university": "Test U",
                    "title": "Freshman Admission",
                    "text": (
                        "Test U reviews applications holistically. The university is test "
                        "optional. Acceptance rate 41%. Middle 50 percent SAT 1200-1400. "
                        "Regular decision deadline is January 15. " * 6
                    ),
                    "source_url": "https://test.example.edu/admissions",
                    "source_type": "official_admissions",
                }
            ],
            store,
        )
        result = await college_research_worker(
            CollegeCandidate(university="Test U", target_major="Computer Science"),
            store,
            settings=settings,
        )
        assert result.admit_rate == "41%"
        assert result.test_policy == "Test optional"
        assert result.fact_sources["admit_rate"].startswith("u-adm")

    async def test_major_admit_rate_stays_unpublished_when_only_a_university_rate_exists(
        self, settings
    ):
        from pathora.agents.college_worker import college_research_worker
        from pathora.domain.models import NOT_PUBLISHED, CollegeCandidate
        from pathora.rag.ingest import ingest_records
        from pathora.rag.store import InMemoryVectorStore

        store = InMemoryVectorStore(settings)
        await ingest_records(
            [
                {
                    "id": "u-adm",
                    "university": "Test U",
                    "title": "Admission",
                    "text": "Acceptance rate 41% university-wide. " * 15,
                    "source_url": "https://test.example.edu/admissions",
                    "source_type": "official_admissions",
                }
            ],
            store,
        )
        result = await college_research_worker(
            CollegeCandidate(university="Test U", target_major="Computer Science"),
            store,
            settings=settings,
        )
        assert result.admit_rate == "41%"
        assert result.major_admit_rate == NOT_PUBLISHED


class TestCorpusMerging:
    """Dropping a *.colleges.json into data/seed must reach the running app."""

    def test_extra_corpus_file_is_discovered_and_merged(self, tmp_path):
        import json

        from pathora.rag.store import load_seed_payload

        base = tmp_path / "colleges.json"
        base.write_text(
            json.dumps(
                {
                    "colleges": [{"university": "A", "state": "TX", "majors": ["CS"]}],
                    "documents": [],
                }
            )
        )
        extra = tmp_path / "real.colleges.json"
        extra.write_text(
            json.dumps(
                {
                    "colleges": [{"university": "B", "state": "IN", "majors": ["CS"]}],
                    "documents": [],
                }
            )
        )

        import pathora.rag.store as store_module

        original_dir, original_path = store_module.SEED_DIR, store_module.SEED_PATH
        store_module.SEED_DIR, store_module.SEED_PATH = tmp_path, base
        try:
            merged = load_seed_payload()
            assert {c["university"] for c in merged["colleges"]} == {"A", "B"}
        finally:
            store_module.SEED_DIR, store_module.SEED_PATH = original_dir, original_path

    def test_explicit_path_still_loads_only_that_file(self):
        from pathora.rag.store import SEED_PATH, load_seed_payload

        only = load_seed_payload(SEED_PATH)
        assert len(only["colleges"]) == 10


class TestScoreRelativeToPublishedRange:
    """A global threshold made every college react identically to a score change."""

    def test_above_published_band_is_a_strength(self):
        from pathora.llm.heuristics import score_against_range

        delta, note = score_against_range(1500, "1230-1420 (middle 50%)", "SAT")
        assert delta == 1
        assert "above the published 1230-1420" in note

    def test_inside_published_band_is_neutral(self):
        from pathora.llm.heuristics import score_against_range

        delta, note = score_against_range(1300, "1230-1420 (middle 50%)", "SAT")
        assert delta == 0
        assert "inside the published" in note

    def test_below_published_band_is_a_risk(self):
        from pathora.llm.heuristics import score_against_range

        delta, _ = score_against_range(1100, "1230-1420 (middle 50%)", "SAT")
        assert delta == -1

    def test_same_score_scores_differently_at_different_colleges(self):
        from pathora.llm.heuristics import score_against_range

        selective, _ = score_against_range(1450, "1460-1560 (middle 50%)", "SAT")
        accessible, _ = score_against_range(1450, "1120-1310 (middle 50%)", "SAT")
        assert selective < accessible

    def test_falls_back_to_a_benchmark_when_nothing_is_published(self):
        from pathora.domain.models import NOT_PUBLISHED
        from pathora.llm.heuristics import score_against_range

        delta, note = score_against_range(1500, NOT_PUBLISHED, "SAT")
        assert delta == 1
        assert "no published range" in note

    def test_act_uses_its_own_scale(self):
        from pathora.llm.heuristics import score_against_range

        assert score_against_range(34, "27-32 (middle 50%)", "ACT")[0] == 1
        assert score_against_range(22, "27-32 (middle 50%)", "ACT")[0] == -1
