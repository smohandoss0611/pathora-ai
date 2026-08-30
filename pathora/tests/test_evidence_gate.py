"""The gate is a control, not a request.

The decisive test is `test_no_llm_call_is_made_for_a_gated_college`: it proves
the model is never handed the retrieved passages when evidence is insufficient.
A prompt instruction cannot be tested this way, which is the whole point.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from pathora.domain.models import (
    NOT_PUBLISHED,
    CollegeResearchResult,
    EvidenceRecord,
)
from pathora.services.evidence_gate import evidence_gate, to_abstention


def evidence(source_type: str, *, eid: str = "e1", url: str | None = None, days: int = 30):
    return EvidenceRecord(
        evidence_id=eid,
        university="Test U",
        source_url=url or f"https://test.example.edu/{eid}",
        source_type=source_type,  # type: ignore[arg-type]
        published_at=datetime.now(UTC) - timedelta(days=days),
    )


def research(**kw) -> CollegeResearchResult:
    base = dict(
        university="Test U",
        target_major="Computer Science",
        admit_rate="42%",
        fact_sources={"admit_rate": "e1"},
        evidence=[
            evidence("official_admissions", eid="e1"),
            evidence("common_data_set", eid="e2"),
        ],
    )
    return CollegeResearchResult(**(base | kw))


class TestGatePasses:
    def test_well_sourced_research_passes(self):
        assert evidence_gate(research()).passed

    def test_all_checks_reported_even_on_success(self):
        result = evidence_gate(research())
        assert len(result.checks) == 7
        assert result.failed_checks == []


class TestGateRefuses:
    def test_no_evidence(self):
        result = evidence_gate(research(evidence=[], admit_rate=NOT_PUBLISHED, fact_sources={}))
        assert not result.passed
        assert "evidence_retrieved" in result.failed_checks

    def test_single_non_authoritative_source_is_not_corroboration(self):
        one = [evidence("other", eid="e1", url="https://blog.example.com/same")]
        result = evidence_gate(research(evidence=one))
        assert not result.passed
        assert "independent_sources" in result.failed_checks

    def test_single_authoritative_source_with_an_anchor_is_accepted(self):
        """IPEDS is one URL. Demanding two would bar the best dataset there is."""
        one = [evidence("institutional_research", eid="ipeds-1")]
        assert evidence_gate(research(evidence=one)).passed

    def test_single_authoritative_source_without_an_anchor_is_refused(self):
        from pathora.domain.models import NOT_PUBLISHED

        one = [evidence("official_admissions", eid="e1")]
        result = evidence_gate(research(evidence=one, admit_rate=NOT_PUBLISHED, fact_sources={}))
        assert not result.passed

    def test_no_authoritative_source(self):
        blogs = [
            evidence("other", eid="b1", url="https://blog1.example.com"),
            evidence("other", eid="b2", url="https://blog2.example.com"),
        ]
        result = evidence_gate(research(evidence=blogs))
        assert not result.passed
        assert "authoritative_source" in result.failed_checks

    def test_no_selectivity_anchor(self):
        result = evidence_gate(research(admit_rate=NOT_PUBLISHED, fact_sources={}))
        assert not result.passed
        assert "selectivity_anchor" in result.failed_checks

    def test_untraced_fact_is_refused(self):
        """A stated fact with no evidence_id behind it must not reach the model."""
        result = evidence_gate(research(fact_sources={}))
        assert not result.passed
        assert "facts_traced" in result.failed_checks

    def test_entirely_stale_evidence(self):
        old = [
            evidence("official_admissions", eid="e1", days=5000),
            evidence("common_data_set", eid="e2", days=5000),
        ]
        result = evidence_gate(research(evidence=old))
        assert not result.passed
        assert "evidence_fresh" in result.failed_checks

    def test_research_error(self):
        result = evidence_gate(research(research_error="timeout"))
        assert not result.passed
        assert "research_succeeded" in result.failed_checks

    def test_reason_names_every_failure(self):
        result = evidence_gate(research(evidence=[], admit_rate=NOT_PUBLISHED, fact_sources={}))
        for name in result.failed_checks:
            assert name in result.reason


class TestAbstention:
    def test_abstention_explains_what_would_help(self):
        result = evidence_gate(research(admit_rate=NOT_PUBLISHED, fact_sources={}))
        abstention = to_abstention(research(admit_rate=NOT_PUBLISHED, fact_sources={}), result)
        assert abstention.what_would_help
        assert abstention.failed_checks == result.failed_checks

    def test_abstention_carries_no_classification(self):
        """No Safety/Target/Reach label is invented when evidence is absent."""
        result = evidence_gate(research(evidence=[], admit_rate=NOT_PUBLISHED, fact_sources={}))
        abstention = to_abstention(
            research(evidence=[], admit_rate=NOT_PUBLISHED, fact_sources={}), result
        )
        assert not hasattr(abstention, "classification")
        assert not hasattr(abstention, "confidence")


class TestGatePrecedesGeneration:
    async def test_no_llm_call_is_made_for_a_gated_college(self, deps, provider):
        """The model must never see passages it was not cleared to reason over."""
        from pathora.graph import nodes

        state = {
            "student_id": "s1",
            "verified_academics": {"courses": []},
            "student_twin": _twin(),
            "profile_analysis": _profile(),
            "activity_analysis": _activity(),
            "college_research": {
                "Test U": research(
                    evidence=[], admit_rate=NOT_PUBLISHED, fact_sources={}
                ).model_dump(mode="json")
            },
        }
        before = [c for c in provider.calls if c[1] == "AdmissionAssessment"]
        result = await nodes.admission_agent(state, deps)  # type: ignore[arg-type]
        after = [c for c in provider.calls if c[1] == "AdmissionAssessment"]

        assert len(after) == len(before), "the gate must run before generation"
        assert "Test U" in result["abstentions"]
        assert "Test U" not in result["admission_results"]

    async def test_llm_is_called_when_the_gate_passes(self, deps, provider):
        from pathora.graph import nodes

        state = {
            "student_id": "s1",
            "verified_academics": {"courses": []},
            "student_twin": _twin(),
            "profile_analysis": _profile(),
            "activity_analysis": _activity(),
            "college_research": {"Test U": research().model_dump(mode="json")},
        }
        result = await nodes.admission_agent(state, deps)  # type: ignore[arg-type]
        assert any(c[1] == "AdmissionAssessment" for c in provider.calls)
        assert "Test U" in result["admission_results"]
        assert result["abstentions"] == {}


def _twin() -> dict:
    from pathora.domain.models import ExtractedAcademics
    from pathora.services.twin import build_digital_twin

    return build_digital_twin(student_id="s1", verified_academics=ExtractedAcademics()).model_dump(
        mode="json"
    )


def _profile() -> dict:
    from pathora.domain.models import ProfileAnalysis

    return ProfileAnalysis(
        course_rigor="Moderate",
        grade_trend="Unknown",
        math_preparation="",
        science_preparation="",
        cs_preparation="",
    ).model_dump(mode="json")


def _activity() -> dict:
    from pathora.domain.models import ActivityAnalysis

    return ActivityAnalysis().model_dump(mode="json")


@pytest.fixture(autouse=True)
def _quiet_settings(settings):
    return settings


class TestOpenDiscoveryMode:
    """Open discovery lets the model name schools; the gate still governs facts."""

    async def test_proposed_school_without_documents_abstains(self, store, settings):
        from pathora.agents.college_worker import college_research_worker
        from pathora.domain.models import CollegeCandidate

        result = await college_research_worker(
            CollegeCandidate(university="Somewhere Not Indexed", target_major="Statistics"),
            store,
            settings=settings,
        )
        gate = evidence_gate(result, settings=settings)
        assert not gate.passed
        assert "evidence_retrieved" in gate.failed_checks

    def test_discovery_modes_are_configurable(self):
        from pathora.config import Settings

        for mode in ("catalog", "open", "hybrid"):
            assert Settings(college_discovery_mode=mode).college_discovery_mode == mode

    async def test_open_mode_prompt_forbids_asserting_statistics(self):
        """The model may name schools; it may not supply their admission facts."""
        from pathora.agents.analysts import DISCOVERY_SYSTEM

        assert "Do NOT state admission rates" in DISCOVERY_SYSTEM


class TestActivityRecognition:
    """Keyword lists that only look for job titles miss real credentials."""

    def _analyse(self, activities=None, projects=None):
        from pathora.domain.models import (
            Activity,
            ExtractedAcademics,
            Project,
        )
        from pathora.llm.heuristics import activity_analysis
        from pathora.services.twin import build_digital_twin

        twin = build_digital_twin(
            student_id="s",
            verified_academics=ExtractedAcademics(),
            activities=[Activity.model_validate(a) for a in (activities or [])],
            projects=[Project.model_validate(p) for p in (projects or [])],
        )
        return activity_analysis({"twin": twin})

    def test_eagle_scout_counts_as_leadership(self):
        result = self._analyse(
            [{"name": "Eagle Scout", "role": "Eagle Scout, Boy Scouts of America"}]
        )
        assert "leadership" in result.themes
        assert result.leadership_evidence

    def test_internship_is_recognized_as_professional_experience(self):
        result = self._analyse([{"name": "EGBI", "role": "Intern"}])
        assert "professional experience" in result.themes

    def test_tutoring_counts_as_service(self):
        result = self._analyse([{"name": "Learn To Be", "role": "Volunteer Tutor"}])
        assert result.service_evidence

    def test_multi_year_activity_reads_as_sustained(self):
        result = self._analyse(
            [{"name": "Learn To Be", "role": "Tutor", "years": ["2023-2024", "2024-2025"]}]
        )
        assert "sustained commitment" in result.themes

    def test_project_supplies_technical_evidence(self):
        result = self._analyse(projects=[{"name": "Trade Analyser", "technologies": ["Python"]}])
        assert result.technical_evidence
        assert not any("technical" in r.lower() for r in result.risks)

    def test_no_technical_evidence_is_flagged_as_a_risk(self):
        result = self._analyse([{"name": "Choir", "role": "Member"}])
        assert any("technical" in r.lower() for r in result.risks)

    def test_nothing_is_invented_from_an_empty_profile(self):
        result = self._analyse()
        assert result.leadership_evidence == []
        assert result.technical_evidence == []
        assert any("no activities" in r.lower() for r in result.risks)


class TestIpedsIngestion:
    """IPEDS is the answer to admissions pages that publish no admit rate."""

    def _record(self, year: str = "2025"):
        import importlib.util
        from pathlib import Path

        root = Path(__file__).resolve().parents[1]
        spec = importlib.util.spec_from_file_location(
            "ingest_ipeds", root / "scripts/ingest_ipeds.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        row = {
            "UNITID": "228723",
            "APPLCN": "54000",
            "ADMSSN": "33000",
            "SATVR25": "590",
            "SATVR75": "690",
            "SATMT25": "580",
            "SATMT75": "720",
            "ACTCM25": "25",
            "ACTCM75": "31",
            "ADMCON7": "3",
        }
        institution = {"INSTNM": "Texas A&M University", "STABBR": "TX", "CONTROL": "1"}
        return module.build_record(row, institution, year)

    def test_admit_rate_is_computed_not_guessed(self):
        assert self._record()["facts"]["admit_rate"] == "61.1%"

    def test_sat_sections_are_combined(self):
        # 580+590 = 1170, 720+690 = 1410
        assert self._record()["facts"]["sat_range"].startswith("1170-1410")

    def test_test_policy_code_is_decoded(self):
        assert "optional" in self._record()["facts"]["test_policy"]

    def test_missing_counts_produce_no_admit_rate(self):
        import importlib.util
        from pathlib import Path

        root = Path(__file__).resolve().parents[1]
        spec = importlib.util.spec_from_file_location(
            "ingest_ipeds", root / "scripts/ingest_ipeds.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        record = module.build_record(
            {"UNITID": "1", "APPLCN": "", "ADMSSN": "", "ADMCON7": "3"},
            {"INSTNM": "Nowhere U", "STABBR": "TX", "CONTROL": "1"},
            "2025",
        )
        assert "admit_rate" not in record["facts"]

    async def test_ipeds_alone_clears_the_gate(self, settings):
        """A single mandatory federal survey is sufficient corroboration."""
        from pathora.agents.college_worker import college_research_worker
        from pathora.domain.models import CollegeCandidate
        from pathora.rag.ingest import ingest_records
        from pathora.rag.store import InMemoryVectorStore

        store = InMemoryVectorStore(settings)
        await ingest_records([self._record()], store)
        result = await college_research_worker(
            CollegeCandidate(university="Texas A&M University", target_major="Computer Science"),
            store,
            settings=settings,
        )
        gate = evidence_gate(result, settings=settings)
        assert gate.passed, gate.failed_checks

    async def test_stale_survey_year_is_still_refused(self, settings):
        """Three-year-old data must not anchor an admissions classification."""
        from pathora.agents.college_worker import college_research_worker
        from pathora.domain.models import CollegeCandidate
        from pathora.rag.ingest import ingest_records
        from pathora.rag.store import InMemoryVectorStore

        store = InMemoryVectorStore(settings)
        await ingest_records([self._record(year="2015")], store)
        result = await college_research_worker(
            CollegeCandidate(university="Texas A&M University", target_major="Computer Science"),
            store,
            settings=settings,
        )
        gate = evidence_gate(result, settings=settings)
        assert not gate.passed
        assert "evidence_fresh" in gate.failed_checks


class TestMajorMatchesRankedFit:
    """A CS-heavy profile steered into a broad engineering major is a failure."""

    def test_prompt_requires_majors_from_the_ranked_list(self):
        from pathora.agents.analysts import DISCOVERY_SYSTEM

        assert "MUST come from the student's ranked STEM" in DISCOVERY_SYSTEM
        assert "easier to be admitted to" in DISCOVERY_SYSTEM

    async def test_off_list_major_is_flagged_as_a_warning(self, deps, monkeypatch):
        from pathora.domain.models import CollegeCandidate, CollegeCandidateList
        from pathora.graph import nodes

        async def fake_discovery(*a, **k):
            return CollegeCandidateList(
                candidates=[
                    CollegeCandidate(
                        university="Texas A&M University",
                        target_major="Industrial Engineering",
                    )
                ]
            )

        monkeypatch.setattr(nodes, "run_college_discovery", fake_discovery)
        state = {
            "student_id": "s",
            "student_twin": _twin(),
            "stem_fit": [
                {"discipline": "Computer Science", "fit": "Excellent"},
                {"discipline": "Data Science", "fit": "Strong"},
            ],
            "college_catalog": [{"university": "Texas A&M University", "majors": []}],
            "warnings": [],
        }
        result = await nodes.college_discovery(state, deps)  # type: ignore[arg-type]
        assert any("not among the student's ranked STEM fits" in w for w in result["warnings"])

    async def test_on_list_major_produces_no_warning(self, deps, monkeypatch):
        from pathora.domain.models import CollegeCandidate, CollegeCandidateList
        from pathora.graph import nodes

        async def fake_discovery(*a, **k):
            return CollegeCandidateList(
                candidates=[
                    CollegeCandidate(
                        university="Texas A&M University", target_major="Computer Science"
                    )
                ]
            )

        monkeypatch.setattr(nodes, "run_college_discovery", fake_discovery)
        state = {
            "student_id": "s",
            "student_twin": _twin(),
            "stem_fit": [{"discipline": "Computer Science", "fit": "Excellent"}],
            "college_catalog": [{"university": "Texas A&M University", "majors": []}],
            "warnings": [],
        }
        result = await nodes.college_discovery(state, deps)  # type: ignore[arg-type]
        assert result["warnings"] == []


class TestPerCollegeReasoning:
    """Eight identical bullets across six universities is boilerplate, not analysis."""

    def test_prompt_forbids_institution_agnostic_bullets(self):
        from pathora.agents.analysts import ADMISSION_SYSTEM

        assert "must be about THIS college" in ADMISSION_SYSTEM
        assert "Do not pad the list with profile praise" in ADMISSION_SYSTEM

    async def test_duplicate_strengths_across_colleges_are_flagged(self, deps, monkeypatch):
        from pathora.domain.models import AdmissionAssessment
        from pathora.graph import nodes

        shared = ["Strong AP coursework", "Demonstrated leadership"]

        async def fake_admission(provider, twin, research, profile, activity, **kw):
            return AdmissionAssessment(
                university=research.university,
                recommended_major=research.target_major,
                classification="Target",
                confidence="Moderate",
                strengths=shared,
                evidence_ids=[e.evidence_id for e in research.evidence],
            )

        monkeypatch.setattr(nodes, "run_admission_agent", fake_admission)
        monkeypatch.setattr(nodes, "evidence_gate", lambda *a, **k: _passing_gate())

        state = {
            "student_id": "s",
            "verified_academics": {"courses": []},
            "student_twin": _twin(),
            "profile_analysis": _profile(),
            "activity_analysis": _activity(),
            "warnings": [],
            "college_research": {
                "A University": _research("A University"),
                "B University": _research("B University"),
            },
        }
        result = await nodes.admission_agent(state, deps)  # type: ignore[arg-type]
        assert any("not college-specific" in w for w in result["warnings"])

    async def test_distinct_strengths_produce_no_warning(self, deps, monkeypatch):
        from pathora.domain.models import AdmissionAssessment
        from pathora.graph import nodes

        async def fake_admission(provider, twin, research, profile, activity, **kw):
            return AdmissionAssessment(
                university=research.university,
                recommended_major=research.target_major,
                classification="Target",
                confidence="Moderate",
                strengths=[f"SAT sits above {research.university}'s published range"],
                evidence_ids=[e.evidence_id for e in research.evidence],
            )

        monkeypatch.setattr(nodes, "run_admission_agent", fake_admission)
        monkeypatch.setattr(nodes, "evidence_gate", lambda *a, **k: _passing_gate())

        state = {
            "student_id": "s",
            "verified_academics": {"courses": []},
            "student_twin": _twin(),
            "profile_analysis": _profile(),
            "activity_analysis": _activity(),
            "warnings": [],
            "college_research": {
                "A University": _research("A University"),
                "B University": _research("B University"),
            },
        }
        result = await nodes.admission_agent(state, deps)  # type: ignore[arg-type]
        assert not any("not college-specific" in w for w in result["warnings"])


def _passing_gate():
    from pathora.domain.models import GateResult

    return GateResult(university="x", passed=True, reason="ok")


def _research(name: str) -> dict:
    from pathora.domain.models import CollegeResearchResult, EvidenceRecord

    return CollegeResearchResult(
        university=name,
        target_major="Computer Science",
        admit_rate="50%",
        fact_sources={"admit_rate": "e1"},
        evidence=[
            EvidenceRecord(
                evidence_id="e1",
                university=name,
                source_url="https://x.example.edu",
                source_type="institutional_research",
            )
        ],
    ).model_dump(mode="json")


class TestAdmissionAgentFailure:
    """The label is deterministic; only the prose needs the model."""

    async def test_provider_failure_still_produces_a_classification(self, deps, monkeypatch):
        from pathora.graph import nodes

        async def exploding(*a, **kw):
            raise TimeoutError("model timed out")

        monkeypatch.setattr(nodes, "run_admission_agent", exploding)
        monkeypatch.setattr(nodes, "evidence_gate", lambda *a, **k: _passing_gate())

        state = {
            "student_id": "s",
            "verified_academics": {"courses": []},
            "student_twin": _twin(),
            "profile_analysis": _profile(),
            "activity_analysis": _activity(),
            "warnings": [],
            "college_research": {"A University": _research("A University")},
        }
        result = await nodes.admission_agent(state, deps)  # type: ignore[arg-type]
        assert "A University" in result["admission_results"]
        assert result["admission_results"]["A University"]["classification"]
        assert any("explanation unavailable" in w for w in result["warnings"])

    async def test_one_failure_does_not_lose_the_other_colleges(self, deps, monkeypatch):
        from pathora.domain.models import AdmissionAssessment
        from pathora.graph import nodes

        async def flaky(provider, twin, research, profile, activity, **kw):
            if research.university == "B University":
                raise TimeoutError("model timed out")
            return AdmissionAssessment(
                university=research.university,
                recommended_major=research.target_major,
                classification="Target",
                confidence="Moderate",
                evidence_ids=[e.evidence_id for e in research.evidence],
            )

        monkeypatch.setattr(nodes, "run_admission_agent", flaky)
        monkeypatch.setattr(nodes, "evidence_gate", lambda *a, **k: _passing_gate())

        state = {
            "student_id": "s",
            "verified_academics": {"courses": []},
            "student_twin": _twin(),
            "profile_analysis": _profile(),
            "activity_analysis": _activity(),
            "warnings": [],
            "college_research": {
                "A University": _research("A University"),
                "B University": _research("B University"),
            },
        }
        result = await nodes.admission_agent(state, deps)  # type: ignore[arg-type]
        assert set(result["admission_results"]) == {"A University", "B University"}


class TestConcurrentAssessment:
    """Twelve sequential model calls made the workflow feel broken."""

    async def test_colleges_are_assessed_concurrently(self, deps, monkeypatch):
        import asyncio

        from pathora.domain.models import AdmissionAssessment
        from pathora.graph import nodes

        active = {"now": 0, "peak": 0}

        async def slow(provider, twin, research, profile, activity, **kw):
            active["now"] += 1
            active["peak"] = max(active["peak"], active["now"])
            await asyncio.sleep(0.05)
            active["now"] -= 1
            return AdmissionAssessment(
                university=research.university,
                recommended_major=research.target_major,
                classification="Target",
                confidence="Moderate",
                evidence_ids=[e.evidence_id for e in research.evidence],
            )

        monkeypatch.setattr(nodes, "run_admission_agent", slow)
        monkeypatch.setattr(nodes, "evidence_gate", lambda *a, **k: _passing_gate())

        state = {
            "student_id": "s",
            "verified_academics": {"courses": []},
            "student_twin": _twin(),
            "profile_analysis": _profile(),
            "activity_analysis": _activity(),
            "warnings": [],
            "college_research": {f"U{i}": _research(f"U{i}") for i in range(6)},
        }
        result = await nodes.admission_agent(state, deps)  # type: ignore[arg-type]
        assert len(result["admission_results"]) == 6
        assert active["peak"] > 1, "assessments ran sequentially"

    async def test_concurrency_respects_the_configured_bound(self, deps, monkeypatch):
        import asyncio

        from pathora.domain.models import AdmissionAssessment
        from pathora.graph import nodes

        deps.settings.max_parallel_college_workers = 2
        active = {"now": 0, "peak": 0}

        async def slow(provider, twin, research, profile, activity, **kw):
            active["now"] += 1
            active["peak"] = max(active["peak"], active["now"])
            await asyncio.sleep(0.05)
            active["now"] -= 1
            return AdmissionAssessment(
                university=research.university,
                recommended_major=research.target_major,
                classification="Target",
                confidence="Moderate",
                evidence_ids=[e.evidence_id for e in research.evidence],
            )

        monkeypatch.setattr(nodes, "run_admission_agent", slow)
        monkeypatch.setattr(nodes, "evidence_gate", lambda *a, **k: _passing_gate())

        state = {
            "student_id": "s",
            "verified_academics": {"courses": []},
            "student_twin": _twin(),
            "profile_analysis": _profile(),
            "activity_analysis": _activity(),
            "warnings": [],
            "college_research": {f"U{i}": _research(f"U{i}") for i in range(6)},
        }
        await nodes.admission_agent(state, deps)  # type: ignore[arg-type]
        assert active["peak"] <= 2


class TestPromptSize:
    def test_admission_payload_omits_raw_course_rows(self):
        """37 course rows per college is tokens the model does not need."""
        from pathora.agents.analysts import _compact_twin

        compact = _compact_twin(
            __import__("pathora.services.twin", fromlist=["build_digital_twin"]).build_digital_twin(
                student_id="s",
                verified_academics=__import__(
                    "pathora.domain.models", fromlist=["ExtractedAcademics"]
                ).ExtractedAcademics(),
            )
        )
        assert "courses" not in compact
        assert "gpa" in compact
        assert "math_progression" in compact
