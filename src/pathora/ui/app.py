"""Streamlit front end.

One college-planning application. Agent names, graph nodes and retry loops are
deliberately not surfaced: the student sees verification, matches, evidence,
gaps, simulations and a roadmap.
"""

from __future__ import annotations

import asyncio
import base64
import uuid
from datetime import datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import streamlit as st

from pathora.config import get_settings
from pathora.domain.models import HumanResponse, WhatIfScenario
from pathora.graph.nodes import Deps
from pathora.llm.providers import build_provider
from pathora.rag.store import load_seed_payload, seeded_store
from pathora.service import PathoraService

st.set_page_config(page_title="Pathora AI", page_icon="🎓", layout="wide")

CLASSIFICATION_COLORS = {
    "Safety": "#1a7f37",
    "Likely": "#2da44e",
    "Target": "#0969da",
    "Target-Reach": "#9a6700",
    "Reach": "#bc4c00",
    "High Reach": "#cf222e",
}


def run(coro):
    """Run a coroutine from Streamlit's script thread.

    Deliberately a fresh loop per call via asyncio.run(). Caching a loop in
    session_state hangs: Streamlit reruns the script on a different thread each
    time, and run_until_complete on a loop owned by another thread blocks
    forever, which shows up as the loading skeleton never resolving. Nothing in
    this stack is loop-bound, so a per-call loop is safe.
    """
    return asyncio.run(coro)


@st.cache_resource
def get_service() -> PathoraService:
    settings = get_settings()
    store = asyncio.run(seeded_store(settings))
    return PathoraService(
        Deps(
            provider=build_provider(settings),
            store=store,
            settings=settings,
            catalog=load_seed_payload()["colleges"],
        )
    )


def build_stamp() -> str:
    """Identify the running code so a stale server/image is obvious."""
    source = Path(__file__).resolve()
    changed = datetime.fromtimestamp(source.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
    try:
        installed = version("pathora-ai")
    except PackageNotFoundError:  # running straight from source
        installed = "source"
    return f"v{installed} · ui file {changed}"


def sidebar_inputs() -> dict[str, Any]:
    st.sidebar.caption(build_stamp())
    if st.sidebar.button("Start over / reload code"):
        st.cache_resource.clear()
        st.session_state.clear()
        st.rerun()

    st.sidebar.header("Your profile")
    sat = st.sidebar.number_input("SAT total (0 = not taken)", 0, 1600, 1450, step=10)
    act = st.sidebar.number_input("ACT composite (0 = not taken)", 0, 36, 0)
    interests = st.sidebar.multiselect(
        "STEM interests",
        [
            "Computer Science",
            "Data Science",
            "Statistics",
            "Applied Mathematics",
            "Industrial Engineering",
            "Systems Engineering",
            "Operations Research",
            "Computer Engineering",
            "Electrical Engineering",
            "Cybersecurity",
            "Information Science",
            "Computational Science",
        ],
        default=["Computer Science", "Data Science"],
    )
    states = st.sidebar.text_input("Preferred states (comma separated)", "TX")
    control = st.sidebar.selectbox("Public or private", ["NoPreference", "Public", "Private"])
    size = st.sidebar.selectbox(
        "Campus size", ["NoPreference", "Small", "Medium", "Large"], index=3
    )

    st.sidebar.subheader("Activities")
    activity_text = st.sidebar.text_area(
        "One per line: name | role | years",
        "Eagle Scout | Eagle Scout, Boy Scouts of America | 2022-2023,2023-2024,2024-2025\n"
        "Learn To Be | Volunteer Tutor | 2023-2024,2024-2025,2025-2026\n"
        "EGBI | Intern | 2025-2026",
        help="Multi-year entries are what the analysis reads as sustained commitment.",
    )
    activities = []
    for line in activity_text.splitlines():
        parts = [p.strip() for p in line.split("|")]
        if parts and parts[0]:
            activities.append(
                {
                    "name": parts[0],
                    "role": parts[1] if len(parts) > 1 else None,
                    "years": parts[2].split(",") if len(parts) > 2 else [],
                }
            )

    st.sidebar.subheader("Projects")
    project_text = st.sidebar.text_area(
        "One per line: name | what it does | technologies",
        "Trade Analyser | Analyses historical price data and charts signals | Python, pandas",
        help=(
            "Describe what the project actually does. The analysis only restates "
            "what you write here — it will not invent scope, users or impact."
        ),
    )
    projects = []
    for line in project_text.splitlines():
        parts = [p.strip() for p in line.split("|")]
        if parts and parts[0]:
            projects.append(
                {
                    "name": parts[0],
                    "description": parts[1] if len(parts) > 1 else None,
                    "technologies": (
                        [t.strip() for t in parts[2].split(",") if t.strip()]
                        if len(parts) > 2
                        else []
                    ),
                }
            )

    st.sidebar.subheader("Awards")
    award_text = st.sidebar.text_area(
        "One per line: name | level | year",
        "",
        placeholder="National Merit Commended | National | 2026",
    )
    awards = []
    for line in award_text.splitlines():
        parts = [p.strip() for p in line.split("|")]
        if parts and parts[0]:
            awards.append(
                {
                    "name": parts[0],
                    "level": parts[1] if len(parts) > 1 else None,
                    "year": parts[2] if len(parts) > 2 else None,
                }
            )

    testing: dict[str, Any] = {}
    if sat:
        testing["sat_total"] = int(sat)
    if act:
        testing["act_composite"] = int(act)

    return {
        "testing": testing,
        "activities": activities,
        "projects": projects,
        "awards": awards,
        "stem_interests": interests,
        "preferences": {
            "locations": [s.strip() for s in states.split(",") if s.strip()],
            "public_private": control,
            "school_size": size,
        },
    }


def render_pending_action(service: PathoraService, pending: dict[str, Any]) -> None:
    st.warning(pending["message"])
    payload = pending.get("payload", {})
    kind = pending.get("kind", "")

    if kind == "critic_human_review":
        # A review payload, not an extraction. Rendering it under "What we
        # extracted" as raw JSON told the student nothing they could act on.
        st.markdown("**What we could not verify**")
        for university in payload.get("colleges", []):
            st.markdown(f"- **{university}**")
        for issue in payload.get("issues", []):
            st.markdown(f"  - {issue}")

        if missing := payload.get("missing_information", []):
            with st.expander(f"Open information gaps ({len(missing)})"):
                for item in missing:
                    st.markdown(f"- {item}")

        st.caption(
            "Continuing keeps these colleges in your list with the gaps recorded "
            "on their Evidence Passport. It does not invent the missing figures."
        )
    else:
        academics = payload.get("extracted_academics", {})
        courses = academics.get("courses", [])
        with st.expander(f"What we extracted from your transcript ({len(courses)} courses)"):
            if reported := payload.get("reported_gpa"):
                st.markdown(
                    f"Transcript reports GPA **{reported}**; we computed "
                    f"**{payload.get('computed_gpa')}** from the courses below."
                )
            if uncertain := academics.get("uncertain_fields", []):
                st.markdown("Could not read with confidence: " + ", ".join(uncertain))
            if courses:
                st.dataframe(
                    [
                        {
                            "Course": c["name"],
                            "Grade": c["grade"],
                            "Credits": c["credits"],
                            "Year": c.get("academic_year") or "-",
                            "Level": c["level"],
                        }
                        for c in courses
                    ],
                    hide_index=True,
                )

    choice = st.radio(
        "How would you like to continue?",
        pending.get("options", ["confirm", "continue_with_uncertainty", "cancel"]),
        horizontal=True,
        format_func=lambda c: c.replace("_", " ").title(),
    )
    if st.button("Continue", type="primary"):
        result = run(
            service.resume(
                thread_id=st.session_state["thread_id"], response=HumanResponse(choice=choice)
            )
        )
        st.session_state["result"] = result
        st.rerun()


def render_match(state: dict[str, Any], university: str) -> None:
    assessment = state["admission_results"][university]
    research = state["college_research"][university]
    passport = state["evidence_passports"][university]
    gap = state["gap_analysis"][university]
    color = CLASSIFICATION_COLORS.get(assessment["classification"], "#57606a")

    st.markdown(f"### {university}")
    st.caption(f"Recommended STEM path: **{assessment['recommended_major']}**")
    st.markdown(
        f"<span style='background:{color};color:white;padding:4px 12px;border-radius:12px;'>"
        f"{assessment['classification']}</span>&nbsp;&nbsp;Confidence: "
        f"**{assessment['confidence']}**",
        unsafe_allow_html=True,
    )

    left, right = st.columns(2)
    with left:
        st.markdown("**Why it fits**")
        for item in assessment["strengths"] or ["No supporting signals were identified."]:
            st.markdown(f"✓ {item}")
    with right:
        st.markdown("**Risks**")
        for item in assessment["risks"] or ["No specific risks were identified."]:
            st.markdown(f"⚠ {item}")

    st.markdown("**Admission structure**")
    st.write(research["admission_structure"])
    st.markdown("**Why this classification**")
    st.write(assessment["rationale_summary"])

    with st.expander(f"Evidence Passport — quality {passport['quality']}"):
        checks = {
            "Official Admissions": passport["has_official_admissions"],
            "Official STEM / Engineering": passport["has_official_stem"],
            "Common Data Set": passport["has_common_data_set"],
            "Verified Student Profile": passport["has_verified_student_profile"],
        }
        for label, present in checks.items():
            st.markdown(f"{'✓' if present else '✗'} {label}")
        if passport["missing"]:
            st.markdown("**Missing**")
            for item in passport["missing"]:
                st.markdown(f"- {item}")
        st.caption(f"Retrieved: {passport['retrieved_at']}")
        for record in research["evidence"]:
            st.markdown(
                f"- [{record['title'] or record['source_url']}]({record['source_url']}) "
                f"· `{record['source_type']}` · id `{record['evidence_id']}`"
            )
            # Federal records carry campus qualifiers. Showing the registered
            # name lets a student catch a wrong-campus match — a branch campus
            # can differ from the flagship by thirty points of admit rate.
            if "Registered federally as" in record.get("snippet", ""):
                official = record["snippet"].split("Registered federally as")[-1].strip()
                st.caption(f"Matched federal record: {official}")

    with st.expander("Gap analysis"):
        for factor in gap["factors"]:
            st.markdown(f"**{factor['factor']}** — {factor['impact'].upper()} IMPACT")
            st.caption(factor["note"])
        st.markdown(f"**Primary constraint:** {gap['primary_constraint']}")
        if gap["still_controllable"]:
            st.markdown("**Still controllable:** " + ", ".join(gap["still_controllable"]))


def render_what_if(service: PathoraService, state: dict[str, Any]) -> None:
    st.subheader("What-If Lab")
    st.caption(
        "Simulations reuse your verified transcript and GPA — only the affected "
        "parts of the analysis are recomputed."
    )
    if not state.get("admission_results") and not state.get("abstentions"):
        st.info("Run an analysis first — there is nothing to simulate against yet.")
        return

    disciplines = [
        "",
        "Computer Science",
        "Data Science",
        "Statistics",
        "Applied Mathematics",
        "Industrial Engineering",
        "Systems Engineering",
        "Operations Research",
        "Computer Engineering",
        "Electrical Engineering",
        "Cybersecurity",
        "Information Science",
        "Computational Science",
    ]

    col1, col2 = st.columns(2)
    with col1:
        current = state.get("student_twin", {}).get("testing", {}).get("sat_total") or 1520
        new_sat = st.number_input("Simulated SAT", 0, 1600, int(current), step=10)
    with col2:
        # A free-text field silently accepted "Dara Science" and produced a run
        # built on a discipline that does not exist.
        new_major = st.selectbox(
            "Simulated intended major",
            disciplines,
            help="Leave blank to simulate the score change alone (much faster).",
        )

    if new_major:
        st.caption(
            "Changing the major reruns STEM fit, discovery and research for a new "
            "college list — expect this to take as long as the original analysis."
        )

    if st.button("Run simulation", type="primary"):
        scenario = WhatIfScenario(sat_total=int(new_sat) or None, major=new_major.strip() or None)
        try:
            with st.spinner("Recomputing the affected parts of your analysis…"):
                _, result = run(
                    service.what_if(thread_id=st.session_state["thread_id"], scenario=scenario)
                )
        except ValueError as exc:
            st.error(f"{exc}. Re-run the analysis and try again.")
            return
        except Exception as exc:  # noqa: BLE001
            # Never fail silently: an unhandled error looked exactly like the
            # button doing nothing at all.
            st.error(f"Simulation failed: {type(exc).__name__}: {exc}")
            return
        # Persist it. Rendering inside the button block meant results vanished
        # on the next script rerun.
        st.session_state["whatif"] = result

    result = st.session_state.get("whatif")
    if result is None:
        return

    st.success(result.summary)
    for change in result.changes:
        arrow = "→" if change.changed else "="
        st.markdown(
            f"**{change.university}**: {change.before} {arrow} {change.after}  \n"
            f"<span style='color:#57606a'>{change.reason}</span>",
            unsafe_allow_html=True,
        )
    st.caption(f"Recomputed: {', '.join(result.nodes_rerun)}")
    st.caption(f"Reused without recomputation: {', '.join(result.nodes_skipped)}")


def main() -> None:
    st.title("Pathora AI")
    st.caption("Understand your profile. Explore your path. Make better decisions.")
    st.info(
        "Pathora does not predict or guarantee admission. It classifies fit from "
        "official published evidence and shows you the reasoning.",
        icon="ℹ️",
    )

    service = get_service()
    student_input = sidebar_inputs()

    upload = st.file_uploader("Upload your transcript (PDF or text)", type=["pdf", "txt"])
    if st.button("Run my analysis", type="primary", disabled=upload is None) and upload is not None:
        payload = upload.read()
        document = (
            {"pdf_base64": base64.b64encode(payload).decode()}
            if upload.name.lower().endswith(".pdf")
            else {"text": payload.decode("utf-8", errors="replace")}
        )
        # A fresh thread per run. Deriving the id from the file meant
        # re-uploading the same transcript resumed the *previous* completed
        # thread and replayed its results instead of analysing again.
        st.session_state["thread_id"] = f"ui-{uuid.uuid4().hex[:12]}"
        with st.spinner("Reading your transcript and researching colleges…"):
            st.session_state["result"] = run(
                service.start(
                    thread_id=st.session_state["thread_id"],
                    user_id="ui-user",
                    student_id="ui-student",
                    transcript_document=document,
                    student_input=student_input,
                )
            )

    result = st.session_state.get("result")
    if result is None:
        st.stop()

    if result.awaiting_human:
        render_pending_action(service, result.interrupt)
        st.stop()

    state = result.state
    if state.get("workflow_status") == "cancelled_by_user":
        st.error("Analysis cancelled. Upload a transcript to start again.")
        st.stop()

    academics = state["student_twin"]["academics"]
    a, b, c, d = st.columns(4)
    a.metric("Unweighted GPA", state["gpa_result"]["gpa"])
    b.metric("Graded credits", state["gpa_result"]["graded_credits"])
    c.metric("AP / IB courses", len(academics["ap_courses"]) + len(academics["ib_courses"]))
    d.metric("Colleges analyzed", len(state["college_candidates"]))

    if state.get("warnings"):
        for warning in state["warnings"]:
            st.warning(warning, icon="⚠️")

    tabs = st.tabs(["STEM paths", "My matches", "What-If Lab", "Next steps", "Roadmap"])

    with tabs[0]:
        for fit in state["stem_fit"]:
            st.markdown(f"**{fit['discipline']}** — {fit['fit']} fit")
            for item in fit["supporting_evidence"]:
                st.caption(f"✓ {item}")
            for item in fit["concerns"]:
                st.caption(f"⚠ {item}")
            st.caption("Careers: " + ", ".join(fit["career_paths"]))
            st.divider()

    with tabs[1]:
        for university in state["college_candidates"]:
            if university in state["admission_results"]:
                render_match(state, university)
                st.divider()

        if abstentions := state.get("abstentions", {}):
            st.subheader("Not enough evidence to classify")
            st.caption(
                "These were not assessed. Rather than label them from incomplete "
                "sources, Pathora declined and recorded why."
            )
            for university, abstention in abstentions.items():
                st.markdown(f"**{university}** — {abstention['recommended_major']}")
                for item in abstention["what_would_help"]:
                    st.markdown(f"- {item}")
                with st.expander("Gate detail"):
                    for check in (
                        state.get("gate_results", {}).get(university, {}).get("checks", [])
                    ):
                        st.markdown(
                            f"{'✓' if check['passed'] else '✗'} `{check['name']}` — "
                            f"{check['detail']}"
                        )
                st.divider()

    with tabs[2]:
        render_what_if(service, state)

    with tabs[3]:
        for action in state["next_actions"]:
            st.markdown(f"**[{action['priority']}] {action['title']}**")
            st.caption(action["reason"])
            if action["related_colleges"]:
                st.caption("Applies to: " + ", ".join(action["related_colleges"]))

    with tabs[4]:
        roadmap = state["roadmap"]
        for key, label in [
            ("today", "TODAY"),
            ("this_week", "THIS WEEK"),
            ("this_month", "THIS MONTH"),
            ("upcoming", "UPCOMING"),
        ]:
            st.subheader(label)
            items = roadmap.get(key, [])
            if not items:
                st.caption("Nothing scheduled.")
            for item in items:
                st.markdown(f"- **{item['title']}** — {item['detail']}")


if __name__ == "__main__":
    main()
