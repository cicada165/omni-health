"""
tests/test_web.py — FastAPI integration tests for Appointment Scheduler scheduling agent (Milestone 3).

Uses FastAPI TestClient and mocked Gemini client.
Covers:
- Session creation & isolation
- Text messaging & structured pending proposals
- Proposals without DB mutation
- Confirm endpoint & booking ID retrieval
- Cancel endpoint without DB change
- 404 for unknown session IDs
- Concurrent confirmations on the same session
- Two sessions competing for the same slot
- Gemini API failure handling
- Tool-round-exhaustion cleanup
- Health check and static HTML serving
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient
from google.genai import types

from scheduler import db
from scheduler.web import app, sessions


# ---------------------------------------------------------------------------
# Helpers for Mocking Gemini
# ---------------------------------------------------------------------------


def _make_text_response(text: str) -> MagicMock:
    part = types.Part.from_text(text=text)
    content = types.Content(role="model", parts=[part])
    candidate = MagicMock()
    candidate.content = content
    response = MagicMock()
    response.candidates = [candidate]
    response.function_calls = []
    response.text = text
    return response


def _make_fn_call_response(
    calls: list[tuple[str, dict[str, Any]] | tuple[str, dict[str, Any], str]],
    text_alongside: str | None = None,
) -> MagicMock:
    parts = []
    if text_alongside:
        parts.append(types.Part.from_text(text=text_alongside))

    fn_call_objects = []
    for call_tuple in calls:
        if len(call_tuple) == 3:
            name, args, call_id = call_tuple  # type: ignore[misc]
        else:
            name, args = call_tuple  # type: ignore[misc]
            call_id = None
        fc = types.FunctionCall(name=name, args=args, id=call_id)
        fn_call_objects.append(fc)
        parts.append(types.Part(function_call=fc))

    content = types.Content(role="model", parts=parts)
    candidate = MagicMock()
    candidate.content = content
    response = MagicMock()
    response.candidates = [candidate]
    response.function_calls = fn_call_objects
    response.text = text_alongside or ""
    return response


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def client(tmp_path: Path) -> TestClient:
    test_db = tmp_path / "web_test.db"
    db.init_db(test_db)
    db.seed(test_db)

    # Attach test database to app state
    app.state.db_path = test_db
    # Clear in-memory session registry between tests
    sessions.clear()

    # Default mock client
    mock_client = MagicMock()
    mock_client.models.generate_content.return_value = _make_text_response(
        "Welcome to Appointment Scheduler! How can I help you today?"
    )
    app.state.gemini_client = mock_client

    with TestClient(app) as test_client:
        yield test_client

    sessions.clear()


# ---------------------------------------------------------------------------
# Tests: Health & HTML
# ---------------------------------------------------------------------------


def test_health_check(client: TestClient) -> None:
    res = client.get("/api/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "ok"
    assert data["app"] == "Appointment Scheduler"


def test_index_html_served(client: TestClient) -> None:
    res = client.get("/")
    assert res.status_code == 200
    assert "text/html" in res.headers["content-type"]
    assert "Appointment Scheduler" in res.text
    assert "Alex Taylor" in res.text


# ---------------------------------------------------------------------------
# Tests: Session Creation & Isolation
# ---------------------------------------------------------------------------


def test_create_session(client: TestClient) -> None:
    res = client.post("/api/sessions")
    assert res.status_code == 201
    data = res.json()
    assert "session_id" in data
    assert len(data["session_id"]) >= 16
    assert data["patient_name"] == "Alex Taylor"
    assert data["session_id"] in sessions


def test_independent_sessions(client: TestClient) -> None:
    res1 = client.post("/api/sessions")
    res2 = client.post("/api/sessions")
    sid1 = res1.json()["session_id"]
    sid2 = res2.json()["session_id"]
    assert sid1 != sid2

    # Verify they have distinct Agent instances and history
    agent1 = sessions[sid1].agent
    agent2 = sessions[sid2].agent
    assert agent1 is not agent2
    assert agent1.history is not agent2.history


# ---------------------------------------------------------------------------
# Tests: Messaging & Staging
# ---------------------------------------------------------------------------


def test_successful_text_message(client: TestClient) -> None:
    res_sess = client.post("/api/sessions")
    sid = res_sess.json()["session_id"]

    mock_client = MagicMock()
    mock_client.models.generate_content.return_value = _make_text_response(
        "Dr. Lovelace is available tomorrow morning."
    )
    sessions[sid].agent.client = mock_client

    res = client.post(f"/api/sessions/{sid}/messages", json={"text": "Who is available?"})
    assert res.status_code == 200
    data = res.json()
    assert "Dr. Lovelace" in data["reply"]
    assert data["pending"] is None


def test_proposal_without_database_mutation(client: TestClient) -> None:
    res_sess = client.post("/api/sessions")
    sid = res_sess.json()["session_id"]
    test_db: Path = app.state.db_path

    slots = db.list_available_slots(db_path=test_db)
    target = slots[0]

    mock_client = MagicMock()
    mock_client.models.generate_content.side_effect = [
        _make_fn_call_response([("propose_booking", {"slot_id": target.id})]),
        _make_text_response("I have staged slot 1. Please confirm."),
    ]
    sessions[sid].agent.client = mock_client

    res = client.post(f"/api/sessions/{sid}/messages", json={"text": "Book slot 1"})
    assert res.status_code == 200
    data = res.json()

    # Structured pending details returned to client
    assert data["pending"] is not None
    assert data["pending"]["slot_id"] == target.id
    assert data["pending"]["doctor_name"] == target.doctor_name
    assert data["pending"]["status"] == "awaiting_confirmation"

    # CRITICAL: Verify NO database write occurred during proposal
    conn = db._connect(test_db)
    count = conn.execute("SELECT count(*) FROM bookings").fetchone()[0]
    conn.close()
    assert count == 0


# ---------------------------------------------------------------------------
# Tests: Explicit Confirm & Cancel
# ---------------------------------------------------------------------------


def test_confirm_and_retrieve_booking_id(client: TestClient) -> None:
    res_sess = client.post("/api/sessions")
    sid = res_sess.json()["session_id"]
    test_db: Path = app.state.db_path

    slots = db.list_available_slots(db_path=test_db)
    target = slots[0]

    mock_client = MagicMock()
    mock_client.models.generate_content.side_effect = [
        _make_fn_call_response([("propose_booking", {"slot_id": target.id})]),
        _make_text_response("Staged."),
    ]
    sessions[sid].agent.client = mock_client

    # 1. Stage proposal
    client.post(f"/api/sessions/{sid}/messages", json={"text": "Book slot 1"})

    # 2. Explicit confirmation
    res_confirm = client.post(f"/api/sessions/{sid}/confirm")
    assert res_confirm.status_code == 200
    cdata = res_confirm.json()
    assert cdata["success"] is True
    assert cdata["status"] == "confirmed"
    assert cdata["booking_id"] is not None

    # Verify committed in SQLite
    conn = db._connect(test_db)
    row = conn.execute("SELECT * FROM bookings WHERE id = ?", (cdata["booking_id"],)).fetchone()
    conn.close()
    assert row is not None
    assert row["slot_id"] == target.id
    assert row["status"] == "confirmed"


def test_cancel_without_booking(client: TestClient) -> None:
    res_sess = client.post("/api/sessions")
    sid = res_sess.json()["session_id"]
    test_db: Path = app.state.db_path

    slots = db.list_available_slots(db_path=test_db)
    target = slots[0]

    mock_client = MagicMock()
    mock_client.models.generate_content.side_effect = [
        _make_fn_call_response([("propose_booking", {"slot_id": target.id})]),
        _make_text_response("Staged."),
    ]
    sessions[sid].agent.client = mock_client

    # Stage
    client.post(f"/api/sessions/{sid}/messages", json={"text": "Book slot 1"})

    # Cancel
    res_cancel = client.post(f"/api/sessions/{sid}/cancel")
    assert res_cancel.status_code == 200
    assert res_cancel.json()["was_pending"] is True

    # Verify DB has 0 rows
    conn = db._connect(test_db)
    count = conn.execute("SELECT count(*) FROM bookings").fetchone()[0]
    conn.close()
    assert count == 0


# ---------------------------------------------------------------------------
# Tests: Error & Validation Handling
# ---------------------------------------------------------------------------


def test_invalid_session_returns_404(client: TestClient) -> None:
    assert client.post("/api/sessions/unknown-123/messages", json={"text": "Hi"}).status_code == 404
    assert client.post("/api/sessions/unknown-123/confirm").status_code == 404
    assert client.post("/api/sessions/unknown-123/cancel").status_code == 404


def test_empty_message_returns_422(client: TestClient) -> None:
    res_sess = client.post("/api/sessions")
    sid = res_sess.json()["session_id"]
    res = client.post(f"/api/sessions/{sid}/messages", json={"text": "   "})
    assert res.status_code == 422


# ---------------------------------------------------------------------------
# Tests: Concurrency & Slot Competition
# ---------------------------------------------------------------------------


def test_same_session_concurrent_confirmation(client: TestClient) -> None:
    """Duplicate/concurrent confirm requests on the same session do not duplicate bookings."""
    res_sess = client.post("/api/sessions")
    sid = res_sess.json()["session_id"]
    test_db: Path = app.state.db_path

    slots = db.list_available_slots(db_path=test_db)
    target = slots[0]

    mock_client = MagicMock()
    mock_client.models.generate_content.side_effect = [
        _make_fn_call_response([("propose_booking", {"slot_id": target.id})]),
        _make_text_response("Staged."),
    ]
    sessions[sid].agent.client = mock_client

    client.post(f"/api/sessions/{sid}/messages", json={"text": "Book slot 1"})

    # Send confirmation twice
    c1 = client.post(f"/api/sessions/{sid}/confirm")
    c2 = client.post(f"/api/sessions/{sid}/confirm")

    assert c1.status_code == 200
    assert c2.status_code == 200
    assert c1.json()["success"] is True
    assert c2.json()["success"] is True
    assert c1.json()["booking_id"] == c2.json()["booking_id"]

    # Exactly 1 row in DB
    conn = db._connect(test_db)
    count = conn.execute("SELECT count(*) FROM bookings").fetchone()[0]
    conn.close()
    assert count == 1


def test_two_sessions_competing_for_same_slot(client: TestClient) -> None:
    """Two different user sessions try to book the same slot: first wins, second fails safely."""
    test_db: Path = app.state.db_path
    slots = db.list_available_slots(db_path=test_db)
    target = slots[0]

    # Session 1
    sid1 = client.post("/api/sessions").json()["session_id"]
    mc1 = MagicMock()
    mc1.models.generate_content.side_effect = [
        _make_fn_call_response([("propose_booking", {"slot_id": target.id})]),
        _make_text_response("Staged."),
    ]
    sessions[sid1].agent.client = mc1

    # Session 2
    sid2 = client.post("/api/sessions").json()["session_id"]
    mc2 = MagicMock()
    mc2.models.generate_content.side_effect = [
        _make_fn_call_response([("propose_booking", {"slot_id": target.id})]),
        _make_text_response("Staged."),
    ]
    sessions[sid2].agent.client = mc2

    # Both stage the same slot
    client.post(f"/api/sessions/{sid1}/messages", json={"text": "Book slot 1"})
    client.post(f"/api/sessions/{sid2}/messages", json={"text": "Book slot 1"})

    # Session 1 confirms first
    c1 = client.post(f"/api/sessions/{sid1}/confirm").json()
    assert c1["success"] is True

    # Session 2 confirms second -> conflict: slot taken
    c2 = client.post(f"/api/sessions/{sid2}/confirm").json()
    assert c2["success"] is False
    assert "no longer available" in c2["message"]

    # Only 1 booking committed
    conn = db._connect(test_db)
    count = conn.execute("SELECT count(*) FROM bookings").fetchone()[0]
    conn.close()
    assert count == 1


# ---------------------------------------------------------------------------
# Tests: Reliability, Gemini Failures & Tool Exhaustion
# ---------------------------------------------------------------------------


def test_gemini_failure_handling(client: TestClient) -> None:
    res_sess = client.post("/api/sessions")
    sid = res_sess.json()["session_id"]

    mock_client = MagicMock()
    mock_client.models.generate_content.side_effect = TimeoutError("Gemini API connection timed out")
    sessions[sid].agent.client = mock_client

    res = client.post(f"/api/sessions/{sid}/messages", json={"text": "Hello"})
    assert res.status_code == 200
    data = res.json()
    assert "trouble connecting" in data["reply"]
    assert data["pending"] is None


def test_tool_round_exhaustion_cleanup(client: TestClient) -> None:
    res_sess = client.post("/api/sessions")
    sid = res_sess.json()["session_id"]

    # Mock continuous function calls that never return plain text
    mock_client = MagicMock()
    mock_client.models.generate_content.return_value = _make_fn_call_response([("list_doctors", {})])
    sessions[sid].agent.client = mock_client

    res = client.post(f"/api/sessions/{sid}/messages", json={"text": "Loop forever"})
    assert res.status_code == 200
    assert "(max tool rounds reached)" in res.json()["reply"]

    # Pending booking must be None and confirm must fail safely
    assert sessions[sid].agent.pending_booking is None
    c_res = client.post(f"/api/sessions/{sid}/confirm").json()
    assert c_res["success"] is False
    assert "No pending appointment" in c_res["message"]
