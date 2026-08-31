from __future__ import annotations

import pytest
from httpx import ASGITransport, AsyncClient

from pathora.api.main import app
from pathora.service import PathoraService


@pytest.fixture
async def client(deps, monkeypatch):
    service = PathoraService(deps)

    async def _get_service():
        return service

    monkeypatch.setattr("pathora.api.main.get_service", _get_service)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


async def test_health_reports_limits(client):
    response = await client.get("/health")
    assert response.status_code == 200
    assert response.json()["limits"]["max_colleges_per_analysis"] == 8


async def test_transcript_extract_endpoint(client, transcript_document):
    response = await client.post(
        "/transcript/extract",
        files={"file": ("transcript.txt", transcript_document["text"], "text/plain")},
    )
    assert response.status_code == 200
    body = response.json()
    assert len(body["extracted_academics"]["courses"]) == 18


async def test_full_journey_through_the_api(client, transcript_document, student_input):
    start = await client.post(
        "/analysis/start",
        json={
            "thread_id": "api-1",
            "user_id": "u",
            "student_id": "s",
            "transcript_text": transcript_document["text"],
            "student_input": student_input,
        },
    )
    assert start.status_code == 200
    body = start.json()

    guard = 0
    while body["awaiting_human"] and guard < 5:
        resumed = await client.post(
            "/analysis/resume",
            json={
                "thread_id": "api-1",
                "response": {"choice": "continue_with_uncertainty", "edits": {}},
            },
        )
        assert resumed.status_code == 200
        body = resumed.json()
        guard += 1

    assert body["workflow_status"] == "complete"
    university = body["state"]["college_candidates"][0]

    explain = await client.get(f"/analysis/api-1/explain/{university}")
    assert explain.status_code == 200
    detail = explain.json()
    assert detail["classification"]
    assert detail["why_this_classification"]
    assert detail["evidence_passport"]["quality"] in {"HIGH", "MEDIUM", "LOW"}
    assert detail["gap_analysis"]["factors"]

    whatif = await client.post(
        "/analysis/what-if",
        json={"thread_id": "api-1", "scenario": {"sat_total": 1560}},
    )
    assert whatif.status_code == 200
    result = whatif.json()["result"]
    assert "parse_transcript" in result["nodes_skipped"]
    assert result["changes"]


async def test_explain_unknown_college_is_404(client, transcript_document, student_input):
    await client.post(
        "/analysis/start",
        json={
            "thread_id": "api-2",
            "transcript_text": transcript_document["text"],
            "student_input": student_input,
        },
    )
    response = await client.get("/analysis/api-2/explain/Nowhere University")
    assert response.status_code == 404


async def test_what_if_before_analysis_is_conflict(client):
    response = await client.post(
        "/analysis/what-if", json={"thread_id": "missing", "scenario": {"sat_total": 1500}}
    )
    assert response.status_code == 409
