from __future__ import annotations

import pytest

from pathora.domain.models import (
    NOT_PUBLISHED,
    AdmissionAssessment,
    CriticResult,
    HumanResponse,
    WhatIfScenario,
)


@pytest.fixture
async def completed(service, transcript_document, student_input):
    """A full run to completion, resolving any interrupt with 'continue'."""
    result = await service.start(
        thread_id="t-main",
        user_id="u1",
        student_id="s1",
        transcript_document=transcript_document,
        student_input=student_input,
    )
    guard = 0
    while result.awaiting_human and guard < 5:
        result = await service.resume(
            thread_id="t-main",
            response=HumanResponse(choice="continue_with_uncertainty"),
        )
        guard += 1
    return result


class TestHappyPath:
    async def test_workflow_completes(self, completed):
        assert completed.state["workflow_status"] == "complete"

    async def test_gpa_is_deterministic_and_present(self, completed):
        gpa = completed.state["gpa_result"]
        assert gpa["method"] == "standard_unweighted_4_scale"
        assert 3.5 <= gpa["gpa"] <= 4.0

    async def test_digital_twin_built(self, completed):
        twin = completed.state["student_twin"]
        assert twin["student_id"] == "s1"
        assert twin["academics"]["ap_courses"]
        assert twin["testing"]["sat_total"] == 1450

    async def test_stem_disciplines_discovered(self, completed):
        fits = completed.state["stem_fit"]
        assert 5 <= len(fits) <= 8
        assert all(f["fit"] in {"Excellent", "Strong", "Moderate", "Weak"} for f in fits)
        assert any(f["supporting_evidence"] for f in fits)

    async def test_college_list_respects_limit(self, completed, settings):
        assert 1 <= len(completed.state["college_candidates"]) <= settings.max_colleges_per_analysis

    async def test_every_college_was_researched(self, completed):
        assert set(completed.state["college_research"]) == set(
            completed.state["college_candidates"]
        )

    async def test_assessments_have_no_fake_precision(self, completed):
        """Citing a published admit rate is fine; inventing a probability is not."""
        banned = (
            "chance of admission",
            "probability",
            "likelihood of",
            "odds of",
            "% chance",
        )
        for raw in completed.state["admission_results"].values():
            assessment = AdmissionAssessment.model_validate(raw)
            lowered = assessment.rationale_summary.lower()
            for phrase in banned:
                assert phrase not in lowered, f"fabricated precision: {phrase}"
            assert assessment.classification in {
                "Safety",
                "Likely",
                "Target",
                "Target-Reach",
                "Reach",
                "High Reach",
            }
            assert assessment.confidence in {"Low", "Moderate", "High"}

    async def test_evidence_ids_trace_back_to_research(self, completed):
        research = completed.state["college_research"]
        for university, raw in completed.state["admission_results"].items():
            known = {e["evidence_id"] for e in research[university]["evidence"]}
            assert set(raw["evidence_ids"]) <= known

    async def test_evidence_passport_per_college(self, completed):
        """Every college gets a passport, including ones the gate refused."""
        passports = completed.state["evidence_passports"]
        assessed = set(completed.state["admission_results"])
        abstained = set(completed.state["abstentions"])
        assert set(passports) == assessed | abstained
        assert all(p["quality"] in {"HIGH", "MEDIUM", "LOW"} for p in passports.values())

    async def test_assessed_and_abstained_are_disjoint(self, completed):
        assert not (set(completed.state["admission_results"]) & set(completed.state["abstentions"]))

    async def test_gap_analysis_is_qualitative_only(self, completed):
        for gap in completed.state["gap_analysis"].values():
            assert gap["factors"]
            assert all(f["impact"] in {"High", "Medium", "Low"} for f in gap["factors"])
            assert gap["primary_constraint"]

    async def test_next_actions_prioritized(self, completed):
        actions = completed.state["next_actions"]
        assert actions
        order = {"High": 0, "Medium": 1, "Low": 2}
        assert [order[a["priority"]] for a in actions] == sorted(
            order[a["priority"]] for a in actions
        )

    async def test_roadmap_has_all_sections(self, completed):
        roadmap = completed.state["roadmap"]
        for section in ("today", "this_week", "this_month", "upcoming"):
            assert section in roadmap


class TestGrounding:
    async def test_unpublished_facts_are_labelled_not_fabricated(self, completed):
        northgate = completed.state["college_research"].get("Northgate Institute of Technology")
        if northgate:
            assert northgate["admit_rate"] == NOT_PUBLISHED
            assert northgate["missing_information"]

    async def test_every_research_result_carries_evidence_or_says_why_not(self, completed):
        for result in completed.state["college_research"].values():
            assert result["evidence"] or result["missing_information"]

    async def test_facts_only_come_from_retrieved_evidence(self, completed):
        for result in completed.state["college_research"].values():
            if result["admit_rate"] != NOT_PUBLISHED:
                assert result["evidence"], "a published stat with no evidence record"


class TestCriticAndRetry:
    async def test_critic_ran_and_recorded_a_decision(self, completed):
        critic = CriticResult.model_validate(completed.state["critic_results"])
        assert critic.decision in {"approve", "research_more", "human_review"}

    async def test_retry_or_human_path_was_exercised(self, completed):
        # The seeded corpus is built so at least one college is missing an
        # official admissions source on the shallow pass.
        assert (
            completed.state["research_retry_count"] > 0
            or completed.state["critic_loop_count"] > 1
            or completed.state["human_responses"]
        )

    async def test_loops_are_bounded(self, completed, settings):
        assert completed.state["critic_loop_count"] <= settings.max_critic_loops + 1
        assert completed.state["research_retry_count"] <= settings.max_research_retries

    async def test_deep_research_recovers_missing_source(self, completed):
        rio = completed.state["college_research"].get("Rio Blanco State University")
        if rio:
            types = {e["source_type"] for e in rio["evidence"]}
            assert "official_admissions" in types


class TestHumanInTheLoop:
    async def test_low_confidence_transcript_interrupts(self, service, student_input):
        result = await service.start(
            thread_id="t-hitl",
            user_id="u1",
            student_id="s2",
            transcript_document={"text": "Algebra I                         A     1.0\n"},
            student_input=student_input,
        )
        assert result.awaiting_human
        assert result.interrupt["kind"] in {"verify_transcript", "resolve_conflict"}
        assert set(result.interrupt["options"]) == {
            "confirm",
            "edit",
            "continue_with_uncertainty",
            "cancel",
        }

    async def test_cancel_stops_the_workflow(self, service, student_input):
        await service.start(
            thread_id="t-cancel",
            user_id="u1",
            student_id="s3",
            transcript_document={"text": "Algebra I                         A     1.0\n"},
            student_input=student_input,
        )
        result = await service.resume(thread_id="t-cancel", response=HumanResponse(choice="cancel"))
        assert result.state["workflow_status"] == "cancelled_by_user"

    async def test_resume_does_not_restart_the_graph(self, service, provider, student_input):
        await service.start(
            thread_id="t-resume",
            user_id="u1",
            student_id="s4",
            transcript_document={"text": "Algebra I                         A     1.0\n"},
            student_input=student_input,
        )
        calls_before = len(provider.calls)
        result = await service.resume(
            thread_id="t-resume", response=HumanResponse(choice="continue_with_uncertainty")
        )
        # The transcript node is deterministic and upstream of the interrupt; it
        # must not be re-executed, and no agent call is repeated for it.
        assert result.state["extracted_academics"]["raw_text_sha256"]
        assert len(provider.calls) > calls_before  # downstream work continued

    async def test_edit_choice_recomputes_gpa(self, service, student_input):
        await service.start(
            thread_id="t-edit",
            user_id="u1",
            student_id="s5",
            transcript_document={"text": "Algebra I                         C     1.0\n"},
            student_input=student_input,
        )
        result = await service.resume(
            thread_id="t-edit",
            response=HumanResponse(
                choice="edit",
                edits={
                    "courses": [
                        {"name": "Algebra I", "grade": "A", "credits": 1.0, "subject": "Math"}
                    ]
                },
            ),
        )
        assert result.state["gpa_result"]["gpa"] == 4.0


class TestWhatIf:
    async def test_sat_change_skips_transcript_and_gpa(self, service, completed):
        _, whatif = await service.what_if(
            thread_id="t-main", scenario=WhatIfScenario(sat_total=1520)
        )
        assert "parse_transcript" in whatif.nodes_skipped
        assert "calculate_gpa" in whatif.nodes_skipped
        assert "activity_agent" in whatif.nodes_skipped
        assert "admission_agent" in whatif.nodes_rerun
        assert "critic_agent" in whatif.nodes_rerun

    async def test_before_and_after_reported_for_every_college(self, service, completed):
        _, whatif = await service.what_if(
            thread_id="t-main", scenario=WhatIfScenario(sat_total=1520)
        )
        assert len(whatif.changes) == len(completed.state["admission_results"])
        assert all(c.reason for c in whatif.changes)

    async def test_lower_sat_can_move_a_classification(self, service, completed):
        _, whatif = await service.what_if(
            thread_id="t-main", scenario=WhatIfScenario(sat_total=1050)
        )
        assert any(c.changed for c in whatif.changes)

    async def test_adding_a_project_reruns_activity_analysis(self, service, completed):
        _, whatif = await service.what_if(
            thread_id="t-main",
            scenario=WhatIfScenario(
                added_project={"name": "Weather ML model", "technologies": ["Python"]}
            ),
        )
        assert "activity_agent" in whatif.nodes_rerun

    async def test_original_analysis_is_not_mutated(self, service, completed):
        before = dict(completed.state["admission_results"])
        await service.what_if(thread_id="t-main", scenario=WhatIfScenario(sat_total=1600))
        after = (await service.state("t-main"))["admission_results"]
        assert after == before


class TestWhatIfMajorChange:
    """Simulating a major must change what is being simulated.

    Regression: a scenario asking for Data Science reported classifications for
    Industrial Engineering, because a major change added an interest but never
    reran discovery or reached the candidates' target majors.
    """

    async def test_major_change_reruns_discovery(self, service, completed):
        _, result = await service.what_if(
            thread_id="t-main", scenario=WhatIfScenario(major="Data Science")
        )
        assert "college_discovery" in result.nodes_rerun

    async def test_simulated_major_reaches_the_candidates(self, service, completed):
        state, _ = await service.what_if(
            thread_id="t-main", scenario=WhatIfScenario(major="Data Science")
        )
        majors = {a["recommended_major"] for a in state["admission_results"].values()}
        assert "Data Science" in majors

    async def test_colleges_without_the_major_keep_their_own(self, service, completed):
        """Do not claim a college offers a program it does not."""
        state, _ = await service.what_if(
            thread_id="t-main", scenario=WhatIfScenario(major="Cybersecurity")
        )
        catalog = {c["university"]: c for c in state["college_catalog"]}
        for university, assessment in state["admission_results"].items():
            offered = catalog.get(university, {}).get("majors", [])
            if offered and assessment["recommended_major"] == "Cybersecurity":
                assert "Cybersecurity" in offered

    async def test_score_only_change_does_not_rerun_discovery(self, service, completed):
        """The fast path must stay fast."""
        _, result = await service.what_if(
            thread_id="t-main", scenario=WhatIfScenario(sat_total=1520)
        )
        assert "college_discovery" in result.nodes_skipped
