"""On-demand college lookup.

The gap this closes: in open discovery mode the model names real universities
that nobody ingested, and every one of them abstained. Now an un-indexed college
is fetched from federal data, indexed, and assessed — without any pre-ingestion
step.
"""

from __future__ import annotations

import httpx
import pytest

from pathora.agents import college_worker
from pathora.agents.college_worker import college_research_worker
from pathora.config import Settings
from pathora.domain.models import NOT_PUBLISHED, CollegeCandidate
from pathora.rag.store import InMemoryVectorStore
from pathora.services.evidence_gate import evidence_gate

ROW = {
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


def stub_api(results, *, status: int = 200, calls: list | None = None):
    class _Response:
        status_code = status
        text = "error body"

        def json(self):
            return {"results": results}

    async def _get(self, url, params=None):
        if calls is not None:
            calls.append(params)
        return _Response()

    return _get


@pytest.fixture(autouse=True)
def _clear_cache():
    college_worker._LOOKUP_CACHE.clear()
    yield
    college_worker._LOOKUP_CACHE.clear()


@pytest.fixture
def live_settings():
    return Settings(
        live_lookup_enabled=True,
        scorecard_api_key="test-key",
        vector_backend="memory",
    )


class TestOnDemandLookup:
    async def test_unindexed_college_is_fetched_and_assessed(self, live_settings, monkeypatch):
        monkeypatch.setattr(httpx.AsyncClient, "get", stub_api([ROW]))
        store = InMemoryVectorStore(live_settings)
        assert len(store) == 0

        result = await college_research_worker(
            CollegeCandidate(university="Texas A&M University", target_major="Computer Science"),
            store,
            settings=live_settings,
        )
        assert result.admit_rate.startswith("62.8%")
        assert result.evidence
        assert evidence_gate(result, settings=live_settings).passed

    async def test_fetched_document_is_indexed_for_reuse(self, live_settings, monkeypatch):
        calls: list = []
        monkeypatch.setattr(httpx.AsyncClient, "get", stub_api([ROW], calls=calls))
        store = InMemoryVectorStore(live_settings)
        candidate = CollegeCandidate(
            university="Texas A&M University", target_major="Computer Science"
        )
        await college_research_worker(candidate, store, settings=live_settings)
        assert len(store) > 0
        before = len(calls)
        await college_research_worker(candidate, store, settings=live_settings)
        assert len(calls) == before, "second research should hit the index, not the API"

    async def test_disabled_flag_skips_the_lookup(self, monkeypatch):
        def explode(*a, **k):
            raise AssertionError("no API call should be made when disabled")

        monkeypatch.setattr(httpx.AsyncClient, "get", explode)
        settings = Settings(live_lookup_enabled=False, scorecard_api_key="k")
        result = await college_research_worker(
            CollegeCandidate(university="Nowhere U", target_major="Statistics"),
            InMemoryVectorStore(settings),
            settings=settings,
        )
        assert result.evidence == []

    async def test_unknown_college_abstains_rather_than_inventing(self, live_settings, monkeypatch):
        monkeypatch.setattr(httpx.AsyncClient, "get", stub_api([]))
        result = await college_research_worker(
            CollegeCandidate(university="Fictional Tech", target_major="Statistics"),
            InMemoryVectorStore(live_settings),
            settings=live_settings,
        )
        assert result.admit_rate == NOT_PUBLISHED
        assert not evidence_gate(result, settings=live_settings).passed

    async def test_api_failure_degrades_to_abstention(self, live_settings, monkeypatch):
        monkeypatch.setattr(httpx.AsyncClient, "get", stub_api([], status=403))
        result = await college_research_worker(
            CollegeCandidate(university="Texas A&M University", target_major="CS"),
            InMemoryVectorStore(live_settings),
            settings=live_settings,
        )
        assert result.evidence == []
        assert result.research_error is None, "a failed lookup is not a research error"

    async def test_negative_result_is_cached(self, live_settings, monkeypatch):
        calls: list = []
        monkeypatch.setattr(httpx.AsyncClient, "get", stub_api([], calls=calls))
        store = InMemoryVectorStore(live_settings)
        candidate = CollegeCandidate(university="Fictional Tech", target_major="Statistics")
        await college_research_worker(candidate, store, settings=live_settings)
        await college_research_worker(candidate, store, settings=live_settings)
        assert len(calls) == 1, "a known-missing college should not be re-queried"

    async def test_exact_name_match_is_preferred_over_fuzzy(self, live_settings, monkeypatch):
        satellite = dict(ROW, id=999)
        satellite["school.name"] = "Texas A&M University-Galveston"
        monkeypatch.setattr(httpx.AsyncClient, "get", stub_api([satellite, ROW]))
        result = await college_research_worker(
            CollegeCandidate(university="Texas A&M University", target_major="CS"),
            InMemoryVectorStore(live_settings),
            settings=live_settings,
        )
        assert result.evidence[0].evidence_id.startswith("scorecard-228723")

    async def test_major_rate_still_unpublished_after_live_lookup(self, live_settings, monkeypatch):
        monkeypatch.setattr(httpx.AsyncClient, "get", stub_api([ROW]))
        result = await college_research_worker(
            CollegeCandidate(university="Texas A&M University", target_major="Computer Science"),
            InMemoryVectorStore(live_settings),
            settings=live_settings,
        )
        assert result.major_admit_rate == NOT_PUBLISHED


class TestNameMatching:
    """Federal records carry campus qualifiers that nobody types.

    The failure this covers: `school.name=Texas A&M University` returns HTTP 200
    with zero results, because the institution is registered as "Texas A&M
    University-College Station". Every A&M lookup silently abstained.
    """

    def test_exact_match_scores_highest(self):
        from pathora.rag.scorecard import match_score

        assert match_score("Purdue University", "Purdue University") == 1.0

    def test_campus_qualifier_still_matches(self):
        from pathora.rag.scorecard import match_score

        assert match_score("Texas A&M University", "Texas A&M University-College Station") > 0.6

    def test_different_institution_is_rejected(self):
        from pathora.rag.scorecard import match_score

        assert match_score("Texas A&M University", "Prairie View A&M University") < 0.6

    def test_sibling_campuses_are_distinguished(self):
        from pathora.rag.scorecard import match_score

        austin = match_score("University of Texas at Austin", "The University of Texas at Austin")
        dallas = match_score("University of Texas at Austin", "The University of Texas at Dallas")
        assert austin > 0.9
        assert dallas < 0.6

    async def test_flagship_wins_the_tie_on_enrollment(self, live_settings, monkeypatch):
        """All three A&M campuses score identically; size decides."""
        rows = [
            dict(
                ROW,
                id=1,
                **{
                    "school.name": "Texas A&M University-Galveston",
                    "latest.student.size": 2200,
                },
            ),
            dict(
                ROW,
                id=2,
                **{
                    "school.name": "Texas A&M International University",
                    "latest.student.size": 8000,
                },
            ),
            dict(
                ROW,
                id=3,
                **{
                    "school.name": "Texas A&M University-College Station",
                    "latest.student.size": 74000,
                },
            ),
        ]
        monkeypatch.setattr(httpx.AsyncClient, "get", stub_api(rows))
        from pathora.rag.scorecard import lookup_by_name

        record = await lookup_by_name("Texas A&M University", settings=live_settings)
        assert "College Station" in record["title"]

    async def test_record_is_indexed_under_the_requested_name(self, live_settings, monkeypatch):
        """Retrieval filters on university, so the key must be what was asked for."""
        row = dict(ROW, **{"school.name": "Texas A&M University-College Station"})
        monkeypatch.setattr(httpx.AsyncClient, "get", stub_api([row]))
        from pathora.rag.scorecard import lookup_by_name

        record = await lookup_by_name("Texas A&M University", settings=live_settings)
        assert record["university"] == "Texas A&M University"
        assert "Registered federally as" in record["text"]

    async def test_campus_qualified_college_now_resolves_end_to_end(
        self, live_settings, monkeypatch
    ):
        row = dict(ROW, **{"school.name": "Texas A&M University-College Station"})
        monkeypatch.setattr(httpx.AsyncClient, "get", stub_api([row]))
        result = await college_research_worker(
            CollegeCandidate(university="Texas A&M University", target_major="Data Science"),
            InMemoryVectorStore(live_settings),
            settings=live_settings,
        )
        assert result.admit_rate.startswith("62.8%")
        assert evidence_gate(result, settings=live_settings).passed

    async def test_unrelated_result_is_not_accepted(self, live_settings, monkeypatch):
        wrong = dict(ROW, **{"school.name": "Prairie View A&M University"})
        monkeypatch.setattr(httpx.AsyncClient, "get", stub_api([wrong]))
        from pathora.rag.scorecard import lookup_by_name

        assert await lookup_by_name("Texas A&M University", settings=live_settings) is None


class TestCampusDisambiguation:
    """A branch campus can differ from the flagship by 30 points of admit rate.

    This is the bug that reported 92.8% for Texas A&M, whose College Station
    campus admits around 63%.
    """

    async def test_flagship_beats_branch_campuses(self, live_settings, monkeypatch):
        rows = [
            dict(
                ROW,
                id=1,
                **{
                    "school.name": "Texas A&M University-Commerce",
                    "latest.student.size": 11000,
                    "latest.admissions.admission_rate.overall": 0.928,
                },
            ),
            dict(
                ROW,
                id=2,
                **{
                    "school.name": "Texas A&M University-College Station",
                    "latest.student.size": 74000,
                    "latest.admissions.admission_rate.overall": 0.628,
                },
            ),
        ]
        monkeypatch.setattr(httpx.AsyncClient, "get", stub_api(rows))
        from pathora.rag.scorecard import lookup_by_name

        record = await lookup_by_name("Texas A&M University", settings=live_settings)
        assert "College Station" in record["title"]
        assert record["facts"]["admit_rate"].startswith("62.8%")

    async def test_chosen_campus_is_recorded_in_the_evidence(self, live_settings, monkeypatch):
        """A student must be able to see which campus was matched."""
        row = dict(ROW, **{"school.name": "Texas A&M University-College Station"})
        monkeypatch.setattr(httpx.AsyncClient, "get", stub_api([row]))
        from pathora.rag.scorecard import lookup_by_name

        record = await lookup_by_name("Texas A&M University", settings=live_settings)
        assert "Registered federally as Texas A&M University-College Station" in record["text"]

    def test_no_name_rule_separates_sibling_campuses(self):
        """Documents why enrollment is the tiebreak: the names are equally valid."""
        from pathora.rag.scorecard import match_score

        commerce = match_score("Texas A&M University", "Texas A&M University-Commerce")
        station = match_score("Texas A&M University", "Texas A&M University-College Station")
        assert commerce == station


class TestDateAwareness:
    def test_prompts_state_the_current_academic_year(self):
        """Without this the model calls a completed junior year 'projected'."""
        from pathora.agents.analysts import _today

        text = _today()
        assert "Today's date is" in text
        assert "COMPLETED, not planned" in text


class TestAnchorTriggeredLookup:
    """Indexing a program page must not suppress the federal lookup.

    Regression: ingesting Texas A&M's ETAM page made `seen` non-empty, so the
    Scorecard lookup never ran, the admit rate never arrived, and a college that
    previously classified became an abstention. Adding evidence made the system
    worse.
    """

    PROGRAM_PAGE = {
        "id": "tamu-etam",
        "university": "Texas A&M University",
        "title": "Entry to a Major",
        "text": (
            "Applicants are admitted to the College of Engineering, then place into a "
            "major through Entry to a Major (ETAM) after the first year. " * 6
        ),
        "source_url": "https://engineering.tamu.edu/academics/entry-to-a-major.html",
        "source_type": "official_stem_program",
        "depth": 0,
    }

    async def test_program_page_without_admit_rate_still_triggers_lookup(
        self, live_settings, monkeypatch
    ):
        from pathora.rag.ingest import ingest_records

        monkeypatch.setattr(httpx.AsyncClient, "get", stub_api([ROW]))
        store = InMemoryVectorStore(live_settings)
        await ingest_records([self.PROGRAM_PAGE], store)

        result = await college_research_worker(
            CollegeCandidate(university="Texas A&M University", target_major="Computer Science"),
            store,
            settings=live_settings,
        )
        types = {e.source_type for e in result.evidence}
        assert types == {"official_stem_program", "institutional_research"}
        assert result.admit_rate.startswith("62.8%")
        assert evidence_gate(result, settings=live_settings).passed

    async def test_existing_admit_rate_does_not_trigger_a_lookup(self, live_settings, monkeypatch):
        """Don't spend an API call when the corpus already anchors the label."""
        calls: list = []
        monkeypatch.setattr(httpx.AsyncClient, "get", stub_api([ROW], calls=calls))
        from pathora.rag.ingest import ingest_records

        store = InMemoryVectorStore(live_settings)
        await ingest_records(
            [
                {
                    "id": "cds",
                    "university": "Texas A&M University",
                    "title": "Common Data Set",
                    "text": "Admit rate 62.8%. SAT 1170-1410. " * 10,
                    "source_url": "https://x.example.edu/cds",
                    "source_type": "common_data_set",
                    "depth": 0,
                }
            ],
            store,
        )
        await college_research_worker(
            CollegeCandidate(university="Texas A&M University", target_major="CS"),
            store,
            settings=live_settings,
        )
        assert calls == []


class TestStructureDowngradeUsesRawText:
    """The fact extractor normalises structure text and loses the ETAM signal."""

    async def test_etam_downgrade_survives_normalisation(self, live_settings, monkeypatch):
        from pathora.domain.models import (
            Course,
            ExtractedAcademics,
            ProfileAnalysis,
            TestingProfile,
        )
        from pathora.rag.ingest import ingest_records
        from pathora.services.classifier import classify
        from pathora.services.twin import build_digital_twin

        monkeypatch.setattr(httpx.AsyncClient, "get", stub_api([ROW]))
        store = InMemoryVectorStore(live_settings)
        await ingest_records([TestAnchorTriggeredLookup.PROGRAM_PAGE], store)
        research = await college_research_worker(
            CollegeCandidate(university="Texas A&M University", target_major="Computer Science"),
            store,
            settings=live_settings,
        )

        courses = [
            Course(name=f"AP {i}", grade="90", credits=1.0, level="AP", subject="Math")
            for i in range(20)
        ]
        twin = build_digital_twin(
            student_id="j",
            verified_academics=ExtractedAcademics(courses=courses),
            testing=TestingProfile(sat_total=1330),
        )
        twin.academics.gpa.gpa = 3.62
        profile = ProfileAnalysis(
            course_rigor="Excellent",
            grade_trend="Mixed",
            math_preparation="",
            science_preparation="",
            cs_preparation="",
        )

        result = classify(twin, research, profile)
        assert result.classification == "Target"
        assert any("Entry to a Major" in c for c in result.caps_applied)


class TestQueryResilience:
    """Scorecard returns HTTP 500 for names containing punctuation.

    Observed with "St. Mary's University, Texas". The first attempt raised and
    aborted the fallback loop, so the college abstained despite being findable
    under a simpler name.
    """

    @staticmethod
    def _punctuation_sensitive(attempts: list):
        class _Response:
            def __init__(self, status_code, payload):
                self.status_code = status_code
                self._payload = payload
                self.text = "<html>Internal Server Error</html>"

            def json(self):
                return self._payload

        async def _get(self, url, params=None):
            name = params["school.name"]
            attempts.append(name)
            if any(ch in name for ch in "',"):
                return _Response(500, {})
            return _Response(
                200,
                {
                    "results": [
                        {
                            "id": 1,
                            "school.name": "St. Mary's University",
                            "latest.student.size": 3800,
                            "latest.admissions.admission_rate.overall": 0.81,
                        }
                    ]
                },
            )

        return _get

    async def test_punctuated_name_falls_back_and_succeeds(self, live_settings, monkeypatch):
        attempts: list = []
        monkeypatch.setattr(httpx.AsyncClient, "get", self._punctuation_sensitive(attempts))
        from pathora.rag.scorecard import lookup_by_name

        record = await lookup_by_name("St. Mary's University, Texas", settings=live_settings)
        assert len(attempts) > 1, "the fallback attempt must actually be made"
        assert record is not None
        assert record["facts"]["admit_rate"].startswith("81.0%")

    async def test_credential_errors_still_surface(self, live_settings, monkeypatch):
        """A 403 is a configuration problem, not a bad query — do not swallow it."""

        class _Response:
            status_code = 403
            text = "forbidden"

            def json(self):
                return {}

        async def _get(self, url, params=None):
            return _Response()

        monkeypatch.setattr(httpx.AsyncClient, "get", _get)
        from pathora.rag.scorecard import lookup_by_name

        with pytest.raises(RuntimeError, match="403"):
            await lookup_by_name("Anywhere University", settings=live_settings)

    async def test_worker_degrades_to_abstention_when_all_attempts_fail(
        self, live_settings, monkeypatch
    ):
        class _Response:
            status_code = 500
            text = "server error"

            def json(self):
                return {}

        async def _get(self, url, params=None):
            return _Response()

        monkeypatch.setattr(httpx.AsyncClient, "get", _get)
        result = await college_research_worker(
            CollegeCandidate(university="St. Mary's University, Texas", target_major="CS"),
            InMemoryVectorStore(live_settings),
            settings=live_settings,
        )
        assert result.evidence == []
        assert not evidence_gate(result, settings=live_settings).passed


class TestWeakMatchRejection:
    """A branch campus is a different institution, not a near-enough answer.

    Texas A&M University-San Antonio was matched for "Texas A&M University" and
    supplied an 840-1070 SAT band. The Critic caught it, but the system should
    not have produced it: below the match threshold, abstain.
    """

    @staticmethod
    def _rows(*names):
        return [
            {
                "id": index,
                "school.name": name,
                "latest.student.size": 5000,
                "latest.admissions.admission_rate.overall": 0.8,
            }
            for index, name in enumerate(names)
        ]

    async def test_branch_campus_alone_is_rejected(self, live_settings, monkeypatch):
        monkeypatch.setattr(
            httpx.AsyncClient,
            "get",
            stub_api(self._rows("Texas A&M University-San Antonio")),
        )
        from pathora.rag.scorecard import lookup_by_name

        assert await lookup_by_name("Texas A&M University", settings=live_settings) is None

    async def test_flagship_is_chosen_when_present(self, live_settings, monkeypatch):
        monkeypatch.setattr(
            httpx.AsyncClient,
            "get",
            stub_api(
                self._rows(
                    "Texas A&M University-San Antonio",
                    "Texas A&M University-College Station",
                )
            ),
        )
        from pathora.rag.scorecard import lookup_by_name

        record = await lookup_by_name("Texas A&M University", settings=live_settings)
        assert "College Station" in record["title"]

    async def test_rejection_leads_to_abstention_not_wrong_data(self, live_settings, monkeypatch):
        monkeypatch.setattr(
            httpx.AsyncClient,
            "get",
            stub_api(self._rows("Texas A&M University-San Antonio")),
        )
        result = await college_research_worker(
            CollegeCandidate(university="Texas A&M University", target_major="CS"),
            InMemoryVectorStore(live_settings),
            settings=live_settings,
        )
        assert result.evidence == []
        assert not evidence_gate(result, settings=live_settings).passed

    def test_threshold_is_configurable(self):
        from pathora.config import Settings

        assert Settings().scorecard_match_threshold >= 0.85
        assert Settings(scorecard_match_threshold=0.5).scorecard_match_threshold == 0.5
