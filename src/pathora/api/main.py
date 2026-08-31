"""FastAPI surface.

Thin transport over ``PathoraService``. All domain logic lives below this layer,
so swapping Streamlit for Next.js means writing a new client, not new agents.
"""

from __future__ import annotations

import base64
import uuid
from typing import Any

from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel, Field

from pathora.config import get_settings
from pathora.domain.models import (
    AdmissionAssessment,
    HumanResponse,
    WhatIfScenario,
)
from pathora.service import PathoraService, get_service
from pathora.services.transcript import parse_transcript_pdf, parse_transcript_text

app = FastAPI(
    title="Pathora AI",
    description="Understand your profile. Explore your path. Make better decisions.",
    version="0.1.0",
)


class StartRequest(BaseModel):
    user_id: str = "demo-user"
    student_id: str = "demo-student"
    thread_id: str | None = None
    transcript_text: str | None = None
    transcript_pdf_base64: str | None = None
    student_input: dict[str, Any] = Field(default_factory=dict)


class ResumeRequest(BaseModel):
    thread_id: str
    response: HumanResponse


class WhatIfRequest(BaseModel):
    thread_id: str
    scenario: WhatIfScenario


def _envelope(result) -> dict[str, Any]:
    return {
        "thread_id": result.thread_id,
        "workflow_status": result.state.get("workflow_status"),
        "awaiting_human": result.awaiting_human,
        "pending_human_action": result.interrupt,
        "state": result.state,
    }


@app.get("/health")
async def health() -> dict[str, Any]:
    settings = get_settings()
    return {
        "status": "ok",
        "llm_provider": settings.llm_provider,
        "vector_backend": settings.vector_backend,
        "limits": {
            "max_colleges_per_analysis": settings.max_colleges_per_analysis,
            "max_parallel_college_workers": settings.max_parallel_college_workers,
            "max_research_retries": settings.max_research_retries,
            "max_critic_loops": settings.max_critic_loops,
        },
    }


@app.post("/transcript/extract")
async def extract_transcript(file: UploadFile = File(...)) -> dict[str, Any]:
    """Step 2 of the journey: extract, then let the student verify."""
    payload = await file.read()
    try:
        extracted = (
            parse_transcript_pdf(payload)
            if file.filename and file.filename.lower().endswith(".pdf")
            else parse_transcript_text(payload.decode("utf-8", errors="replace"))
        )
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=422, detail=f"could not parse transcript: {exc}") from exc

    return {
        "extracted_academics": extracted.model_dump(mode="json"),
        "needs_human_verification": extracted.needs_human_verification(
            get_settings().extraction_confidence_threshold
        ),
        "transcript_pdf_base64": base64.b64encode(payload).decode()
        if file.filename and file.filename.lower().endswith(".pdf")
        else None,
    }


@app.post("/analysis/start")
async def start_analysis(request: StartRequest) -> dict[str, Any]:
    service: PathoraService = await get_service()
    thread_id = request.thread_id or f"thread-{uuid.uuid4().hex[:12]}"

    document: dict[str, Any] | None = None
    if request.transcript_text:
        document = {"text": request.transcript_text}
    elif request.transcript_pdf_base64:
        document = {"pdf_base64": request.transcript_pdf_base64}

    result = await service.start(
        thread_id=thread_id,
        user_id=request.user_id,
        student_id=request.student_id,
        transcript_document=document,
        student_input=request.student_input,
    )
    return _envelope(result)


@app.post("/analysis/resume")
async def resume_analysis(request: ResumeRequest) -> dict[str, Any]:
    service: PathoraService = await get_service()
    result = await service.resume(thread_id=request.thread_id, response=request.response)
    return _envelope(result)


@app.get("/analysis/{thread_id}")
async def get_analysis(thread_id: str) -> dict[str, Any]:
    service: PathoraService = await get_service()
    state = await service.state(thread_id)
    if not state:
        raise HTTPException(status_code=404, detail="unknown thread")
    return {"thread_id": thread_id, "state": state}


@app.get("/analysis/{thread_id}/explain/{university}")
async def explain_match(thread_id: str, university: str) -> dict[str, Any]:
    """Explain My Match + Evidence Passport + Gap Analyzer for one college."""
    service: PathoraService = await get_service()
    state = await service.state(thread_id)
    raw = state.get("admission_results", {}).get(university)
    if raw is None:
        raise HTTPException(status_code=404, detail=f"no assessment for {university}")

    assessment = AdmissionAssessment.model_validate(raw)
    research = state["college_research"][university]
    evidence_by_id = {e["evidence_id"]: e for e in research["evidence"]}

    return {
        "university": university,
        "recommended_major": assessment.recommended_major,
        "classification": assessment.classification,
        "confidence": assessment.confidence,
        "why_it_fits": assessment.strengths,
        "risks": assessment.risks,
        "admission_structure": research["admission_structure"],
        "why_this_classification": assessment.rationale_summary,
        "missing_information": assessment.missing_information,
        "evidence_passport": state["evidence_passports"][university],
        "evidence": [evidence_by_id[i] for i in assessment.evidence_ids if i in evidence_by_id],
        "gap_analysis": state["gap_analysis"][university],
    }


@app.post("/analysis/what-if")
async def what_if(request: WhatIfRequest) -> dict[str, Any]:
    service: PathoraService = await get_service()
    try:
        state, result = await service.what_if(
            thread_id=request.thread_id, scenario=request.scenario
        )
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return {"result": result.model_dump(mode="json"), "state": state}
