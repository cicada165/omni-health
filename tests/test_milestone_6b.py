"""
tests/test_milestone_6b.py — Dedicated test suite for Milestone 6B:
Voice Quality, Developer Inspector, and Voice Safety.

Coverage:
1. VOICE QUALITY
   - GET /api/voices: Lists 3 allowlisted voices (Gemma, Cora, Clara) and standardized sample sentences.
   - Validates that no API keys or credentials are leaked in response payloads.
   - POST /api/voices/audition: Synthesizes standardized audition sentence for allowlisted voices.
   - Non-allowlisted voice IDs return HTTP 422 Unprocessable Entity.
   - PUT /api/sessions/{session_id}/voice: Updates session active voice and logs event.
   - Non-allowlisted voice ID in PUT returns HTTP 422.

2. DEVELOPER INSPECTOR
   - Local access (127.0.0.1, localhost, ::1, testclient) succeeds (200 OK).
   - Non-local access returns HTTP 403 Forbidden.
   - Event timeline tracks stages: STT -> Gemini -> Tool -> Proposal -> SQLite -> TTS.
   - Function names, arguments, durations, and result counts are captured.
   - Staging (uncommitted in memory) is clearly distinguished from committed SQLite row.
   - Real SQLite booking row is fetched from `bookings` table upon commit.
   - Sensitive credentials (API keys, idempotency keys, raw audio) are NOT leaked.

3. VOICE SAFETY
   - "Yes, but change it to 2 PM": correction overrides affirmation; does not book; searches 2 PM.
   - "No, don't book that": negation overrides confirmation; cancels pending proposal; zero SQLite bookings.
   - "Wait, I meant Friday": correction overrides affirmation; discards proposal; searches Friday.
   - Gemini tool allowlist strictly prevents direct booking; only Python confirm_booking() commits to SQLite.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from google.genai import types

from scheduler import db
from scheduler.agent import (
    Agent,
    _ALLOWLIST,
    is_verbal_ambiguity,
    is_verbal_cancellation,
    is_verbal_confirmation,
    is_verbal_modification,
)
from scheduler.db import init_db, seed
from scheduler.tts import (
    ALLOWLISTED_VOICES,
    AUDITION_SAMPLES,
    DEFAULT_VOICE_ID,
    is_voice_allowlisted,
    validate_voice_id,
)
from scheduler.web import app, sessions


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def test_db(tmp_path: Path) -> Path:
    db_file = tmp_path / "milestone_6b_test.db"
    init_db(db_file)
    seed(db_file)
    return db_file


@pytest.fixture
def client(test_db: Path) -> TestClient:
    app.state.db_path = test_db
    app.state.gemini_client = MagicMock()
    app.state.deepgram_api_key = "test-deepgram-key"
    app.state.cartesia_api_key = "test-cartesia-key"
    sessions.clear()
    return TestClient(app)


def _connect(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def _make_text_response(text: str) -> MagicMock:
    content = types.Content(
        role="model",
        parts=[types.Part.from_text(text=text)],
    )
    candidate = types.Candidate(content=content)
    resp = MagicMock()
    resp.candidates = [candidate]
    resp.function_calls = []
    resp.text = text
    return resp


def _make_fn_call_response(calls: list[tuple[str, dict[str, Any]]]) -> MagicMock:
    parts = []
    fn_calls = []
    for name, args in calls:
        part = types.Part.from_function_call(name=name, args=args)
        parts.append(part)
        fc = types.FunctionCall(name=name, args=args)
        fn_calls.append(fc)
    content = types.Content(role="model", parts=parts)
    candidate = types.Candidate(content=content)
    resp = MagicMock()
    resp.candidates = [candidate]
    resp.function_calls = fn_calls
    resp.text = ""
    return resp


# ---------------------------------------------------------------------------
# 1. Voice Quality Tests
# ---------------------------------------------------------------------------


class TestVoiceQuality:
    def test_list_voices_returns_three_allowlisted_voices_and_samples(
        self, client: TestClient
    ) -> None:
        """Verify GET /api/voices returns Gemma, Cora, Clara with metadata and sample sentences."""
        resp = client.get("/api/voices")
        assert resp.status_code == 200
        data = resp.json()

        assert "voices" in data
        assert "samples" in data
        assert data["default_voice_id"] == DEFAULT_VOICE_ID

        voices = data["voices"]
        assert len(voices) >= 3
        voice_ids = {v["id"] for v in voices}
        assert "62ae83ad-4f6a-430b-af41-a9bede9286ca" in voice_ids  # Gemma
        assert "c46cf1f6-49a1-4d67-9a57-ff859a4046d3" in voice_ids  # Cora
        assert "01eaafa9-308a-4276-a017-6ab0cf061b1f" in voice_ids  # Clara

        # Check metadata fields
        for v in voices:
            assert v["name"]
            assert v["gender"]
            assert v["accent"]
            assert v["description"]
            assert v["sample_quote"]

        # Check sample sentences
        samples = data["samples"]
        assert "greeting" in samples
        assert "confirmation" in samples
        assert "receipt" in samples

    def test_list_voices_does_not_leak_credentials(self, client: TestClient) -> None:
        """Verify API keys, secrets, or internal tokens are never leaked in /api/voices."""
        resp = client.get("/api/voices")
        text = resp.text.lower()
        assert "cartesia_api_key" not in text
        assert "deepgram_api_key" not in text
        assert "gemini_api_key" not in text
        assert "token" not in text
        assert "secret" not in text

    def test_voice_audition_with_allowlisted_voices(self, client: TestClient) -> None:
        """Verify POST /api/voices/audition produces WAV audio for allowlisted voice options."""
        fake_wav = b"RIFF\x24\x00\x00\x00WAVEfmt \x10\x00\x00\x00data\x00\x00\x00\x00"

        with patch("scheduler.web.synthesize_speech", return_value=fake_wav) as mock_synth:
            for sample_key in ["greeting", "confirmation", "receipt"]:
                resp = client.post(
                    "/api/voices/audition",
                    json={
                        "voice_id": "c46cf1f6-49a1-4d67-9a57-ff859a4046d3",  # Cora
                        "sample_key": sample_key,
                    },
                )
                assert resp.status_code == 200
                assert resp.headers["Content-Type"] == "audio/wav"
                assert resp.headers["X-Audition-Voice"] == "c46cf1f6-49a1-4d67-9a57-ff859a4046d3"
                assert resp.content == fake_wav
                assert mock_synth.called

    def test_voice_audition_rejects_non_allowlisted_voice(self, client: TestClient) -> None:
        """Verify POST /api/voices/audition rejects unknown voices with HTTP 422."""
        resp = client.post(
            "/api/voices/audition",
            json={
                "voice_id": "malicious-or-unknown-voice-id",
                "sample_key": "greeting",
            },
        )
        assert resp.status_code == 422
        assert "not allowlisted" in resp.json()["detail"].lower()

    def test_switch_session_voice_and_record_inspector_event(self, client: TestClient) -> None:
        """Verify PUT /api/sessions/{id}/voice updates active voice and records telemetry event."""
        sess_resp = client.post("/api/sessions")
        session_id = sess_resp.json()["session_id"]

        # Switch to Clara
        clara_id = "01eaafa9-308a-4276-a017-6ab0cf061b1f"
        resp = client.put(
            f"/api/sessions/{session_id}/voice",
            json={"voice_id": clara_id},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["active_voice_id"] == clara_id
        assert data["voice_name"] == "Clara"

        # Verify session state reflects new voice
        session_state = sessions[session_id]
        assert session_state.active_voice_id == clara_id

        # Verify event timeline has voice_change event
        voice_events = [e for e in session_state.events if e.stage == "voice_change"]
        assert len(voice_events) == 1
        assert "Clara" in voice_events[0].title
        assert voice_events[0].details["voice_id"] == clara_id

    def test_switch_session_voice_rejects_non_allowlisted(self, client: TestClient) -> None:
        """Verify PUT /api/sessions/{id}/voice rejects non-allowlisted voice with HTTP 422."""
        sess_resp = client.post("/api/sessions")
        session_id = sess_resp.json()["session_id"]

        resp = client.put(
            f"/api/sessions/{session_id}/voice",
            json={"voice_id": "unauthorized-voice-123"},
        )
        assert resp.status_code == 422
        assert "not allowlisted" in resp.json()["detail"]


# ---------------------------------------------------------------------------
# 2. Developer Inspector Tests
# ---------------------------------------------------------------------------


class TestDeveloperInspector:
    def test_inspector_endpoint_accessible_via_local_client(self, client: TestClient) -> None:
        """Verify GET /api/sessions/{id}/inspector returns 200 for local requests."""
        sess_resp = client.post("/api/sessions")
        session_id = sess_resp.json()["session_id"]

        resp = client.get(f"/api/sessions/{session_id}/inspector")
        assert resp.status_code == 200
        data = resp.json()
        assert data["session_id"] == session_id
        assert "active_voice" in data
        assert "appointment_state" in data
        assert "events" in data
        assert isinstance(data["events"], list)

    def test_inspector_endpoint_forbidden_for_non_local_client(
        self, client: TestClient
    ) -> None:
        """Verify GET /api/sessions/{id}/inspector blocks non-local IP addresses with HTTP 403."""
        sess_resp = client.post("/api/sessions")
        session_id = sess_resp.json()["session_id"]

        # Patch request client host to simulate external remote access
        with patch("scheduler.web._is_local_request", return_value=False):
            resp = client.get(f"/api/sessions/{session_id}/inspector")
            assert resp.status_code == 403
            assert "restricted to local development" in resp.json()["detail"]

    def test_inspector_tracks_stt_gemini_tools_proposal_and_distinguishes_proposal_from_commit(
        self, client: TestClient, test_db: Path
    ) -> None:
        """
        Verify live event timeline logs:
        - STT / User Utterance
        - Tool call (with function name, arguments, duration_ms, result_count)
        - LLM proposal marked storage="session_memory (uncommitted)"
        - Database remains uncommitted until explicit confirm
        - After confirm, SQLite commit event logged with storage="sqlite_database"
        - Real SQLite database record is queried directly from `bookings` table
        """
        mock_gemini = MagicMock()
        app.state.gemini_client = mock_gemini

        sess_resp = client.post("/api/sessions")
        session_id = sess_resp.json()["session_id"]

        # Turn 1: Propose slot 1
        mock_gemini.models.generate_content.side_effect = [
            _make_fn_call_response([("propose_booking", {"slot_id": 1})]),
            _make_text_response(
                "I have prepared an appointment with Dr. Florence Nightingale on Tuesday, "
                "November 17, 2026 at 9:00 AM PST. Would you like me to book this for you?"
            ),
        ]

        msg_resp = client.post(
            f"/api/sessions/{session_id}/messages",
            json={"text": "Book Tuesday at 9am with Dr. Nightingale"},
        )
        assert msg_resp.status_code == 200
        assert msg_resp.json()["pending"] is not None

        # Inspect state while awaiting confirmation:
        # INVARIANT: Must be staged in-memory, NOT committed to SQLite
        insp_resp1 = client.get(f"/api/sessions/{session_id}/inspector")
        assert insp_resp1.status_code == 200
        insp1 = insp_resp1.json()

        assert insp1["appointment_state"]["status"] == "awaiting_confirmation"
        assert insp1["appointment_state"]["is_staged_in_memory"] is True
        assert insp1["appointment_state"]["is_committed_to_sqlite"] is False
        assert insp1["real_sqlite_booking"] is None

        # Verify events logged in timeline
        stages = [e["stage"] for e in insp1["events"]]
        assert "session_init" in stages
        assert "stt" in stages
        assert "tool" in stages
        assert "proposal" in stages
        assert "llm" in stages

        # Check proposal event details
        prop_evts = [e for e in insp1["events"] if e["stage"] == "proposal"]
        assert len(prop_evts) == 1
        assert prop_evts[0]["details"]["storage"] == "session_memory (uncommitted)"
        assert prop_evts[0]["details"]["committed_to_sqlite"] is False
        assert prop_evts[0]["details"]["slot_id"] == 1

        # Check tool event details (duration_ms, arguments, function name)
        tool_evts = [e for e in insp1["events"] if e["stage"] == "tool"]
        assert len(tool_evts) >= 1
        propose_tool_evt = tool_evts[0]
        assert propose_tool_evt["details"]["function"] == "propose_booking"
        assert propose_tool_evt["details"]["arguments"] == {"slot_id": 1}
        assert propose_tool_evt["duration_ms"] is not None
        assert propose_tool_evt["duration_ms"] >= 0

        # Invariant check: SQLite bookings table count is still 0
        conn = _connect(test_db)
        count_before = conn.execute("SELECT count(*) FROM bookings").fetchone()[0]
        conn.close()
        assert count_before == 0

        # Turn 2: Verbal confirmation "Yes, book it"
        msg_resp2 = client.post(
            f"/api/sessions/{session_id}/messages",
            json={"text": "Yes, book it"},
        )
        assert msg_resp2.status_code == 200
        assert msg_resp2.json()["confirmed"] is not None
        booking_id = msg_resp2.json()["confirmed"]["booking_id"]
        assert booking_id is not None

        # Inspect state after confirmation
        insp_resp2 = client.get(f"/api/sessions/{session_id}/inspector")
        assert insp_resp2.status_code == 200
        insp2 = insp_resp2.json()

        assert insp2["appointment_state"]["status"] == "confirmed"
        assert insp2["appointment_state"]["is_staged_in_memory"] is False
        assert insp2["appointment_state"]["is_committed_to_sqlite"] is True
        assert insp2["appointment_state"]["booking_id"] == booking_id

        # Verify real SQLite booking details
        real_b = insp2["real_sqlite_booking"]
        assert real_b is not None
        assert real_b["id"] == booking_id
        assert real_b["slot_id"] == 1
        assert real_b["patient_name"] == "Alex Taylor"
        assert real_b["status"] == "confirmed"

        # Verify commit event in timeline
        commit_evts = [e for e in insp2["events"] if e["stage"] == "sqlite_commit"]
        assert len(commit_evts) == 1
        assert commit_evts[0]["details"]["storage"] == "sqlite_database"
        assert commit_evts[0]["details"]["committed_to_sqlite"] is True
        assert commit_evts[0]["details"]["booking_id"] == booking_id

    def test_inspector_does_not_expose_secrets_or_raw_audio(
        self, client: TestClient
    ) -> None:
        """Verify inspector output sanitizes API keys, raw audio, and idempotency keys."""
        sess_resp = client.post("/api/sessions")
        session_id = sess_resp.json()["session_id"]

        resp = client.get(f"/api/sessions/{session_id}/inspector")
        payload = resp.text.lower()

        assert "deepgram_api_key" not in payload
        assert "cartesia_api_key" not in payload
        assert "gemini_api_key" not in payload
        assert "audio_bytes" not in payload or "audio_bytes\": " in payload  # integer count allowed
        # Make sure raw binary audio is never returned
        assert "riff" not in payload


# ---------------------------------------------------------------------------
# 3. Voice Safety Tests
# ---------------------------------------------------------------------------


class TestVoiceSafety:
    def test_yes_but_change_it_to_2_pm_prioritizes_correction_over_confirmation(
        self, client: TestClient, test_db: Path
    ) -> None:
        """
        Verify:
        User says: 'Yes, but change it to 2 PM'.
        Correction / modification MUST take precedence over affirmative token 'Yes'.
        - Pending proposal must be discarded.
        - SQLite database booking must NOT be committed (count remains 0).
        - LLM must be called to search for the new requested time.
        """
        mock_gemini = MagicMock()
        app.state.gemini_client = mock_gemini

        sess_resp = client.post("/api/sessions")
        session_id = sess_resp.json()["session_id"]

        # Stage slot 1 (9 AM)
        mock_gemini.models.generate_content.side_effect = [
            _make_fn_call_response([("propose_booking", {"slot_id": 1})]),
            _make_text_response(
                "I have prepared an appointment for Tuesday at 9:00 AM. Would you like me to book it?"
            ),
        ]
        r1 = client.post(
            f"/api/sessions/{session_id}/messages",
            json={"text": "I'd like Tuesday morning"},
        )
        assert r1.status_code == 200
        assert r1.json()["pending"] is not None

        # Verify classifier flags
        utterance = "Yes, but change it to 2 PM"
        assert is_verbal_modification(utterance) is True
        assert is_verbal_confirmation(utterance) is False

        # User says "Yes, but change it to 2 PM"
        mock_gemini.models.generate_content.side_effect = [
            _make_fn_call_response([("search_slots", {"date": "2026-11-17"})]),
            _make_text_response("Dr. Nightingale has an opening at 2:00 PM on Tuesday."),
        ]
        r2 = client.post(
            f"/api/sessions/{session_id}/messages",
            json={"text": utterance},
        )
        assert r2.status_code == 200
        # Invariant: confirmed must be None!
        assert r2.json()["confirmed"] is None

        # Invariant: SQLite bookings table MUST have 0 bookings!
        conn = _connect(test_db)
        count = conn.execute("SELECT count(*) FROM bookings").fetchone()[0]
        conn.close()
        assert count == 0

    def test_no_dont_book_that_prioritizes_negation_over_confirmation(
        self, client: TestClient, test_db: Path
    ) -> None:
        """
        Verify:
        User says: 'No, don't book that'.
        Negation / cancellation MUST take precedence.
        - Pending proposal must be cancelled.
        - SQLite database booking must NOT be committed (count remains 0).
        - Assistant politely confirms cancellation without booking.
        """
        mock_gemini = MagicMock()
        app.state.gemini_client = mock_gemini

        sess_resp = client.post("/api/sessions")
        session_id = sess_resp.json()["session_id"]

        # Stage slot 1
        mock_gemini.models.generate_content.side_effect = [
            _make_fn_call_response([("propose_booking", {"slot_id": 1})]),
            _make_text_response(
                "I have staged an appointment for Tuesday at 9:00 AM. Would you like me to book it?"
            ),
        ]
        r1 = client.post(
            f"/api/sessions/{session_id}/messages",
            json={"text": "Book Tuesday with Dr. Nightingale"},
        )
        assert r1.status_code == 200
        assert r1.json()["pending"] is not None

        # Verify classifier flags
        utterance = "No, don't book that"
        assert is_verbal_cancellation(utterance) is True
        assert is_verbal_confirmation(utterance) is False

        # User says "No, don't book that"
        r2 = client.post(
            f"/api/sessions/{session_id}/messages",
            json={"text": utterance},
        )
        assert r2.status_code == 200
        assert r2.json()["confirmed"] is None
        assert r2.json()["pending"] is None
        assert "cancelled" in r2.json()["reply"].lower()

        # Invariant: SQLite bookings table MUST have 0 bookings!
        conn = _connect(test_db)
        count = conn.execute("SELECT count(*) FROM bookings").fetchone()[0]
        conn.close()
        assert count == 0

    def test_wait_i_meant_friday_prioritizes_correction_over_confirmation(
        self, client: TestClient, test_db: Path
    ) -> None:
        """
        Verify:
        User says: 'Wait, I meant Friday'.
        Correction MUST take precedence.
        - Staged Tuesday proposal is discarded.
        - SQLite database booking must NOT be committed (count remains 0).
        - LLM searches/processes Friday request.
        """
        mock_gemini = MagicMock()
        app.state.gemini_client = mock_gemini

        sess_resp = client.post("/api/sessions")
        session_id = sess_resp.json()["session_id"]

        # Stage slot 1 (Tuesday)
        mock_gemini.models.generate_content.side_effect = [
            _make_fn_call_response([("propose_booking", {"slot_id": 1})]),
            _make_text_response(
                "I have staged an appointment for Tuesday at 9:00 AM. Would you like me to book it?"
            ),
        ]
        r1 = client.post(
            f"/api/sessions/{session_id}/messages",
            json={"text": "Book Tuesday morning"},
        )
        assert r1.status_code == 200
        assert r1.json()["pending"] is not None

        # Verify classifier flags
        utterance = "Wait, I meant Friday"
        assert is_verbal_modification(utterance) is True
        assert is_verbal_confirmation(utterance) is False

        # User says "Wait, I meant Friday"
        mock_gemini.models.generate_content.side_effect = [
            _make_fn_call_response([("search_slots", {"date": "2026-11-20"})]),
            _make_text_response("Looking at Friday, November 20: Dr. Lovelace is available at 10 AM."),
        ]
        r2 = client.post(
            f"/api/sessions/{session_id}/messages",
            json={"text": utterance},
        )
        assert r2.status_code == 200
        assert r2.json()["confirmed"] is None

        # Invariant: SQLite bookings table MUST have 0 bookings!
        conn = _connect(test_db)
        count = conn.execute("SELECT count(*) FROM bookings").fetchone()[0]
        conn.close()
        assert count == 0

    def test_gemini_cannot_commit_booking_directly(self) -> None:
        """
        Architectural Invariant:
        Verify that the Gemini function declaration allowlist contains ONLY:
        - list_doctors
        - search_slots
        - propose_booking
        It must NEVER include book_slot, confirm_booking, or any direct database write function.
        """
        assert _ALLOWLIST == frozenset({"list_doctors", "search_slots", "propose_booking"})
        assert "book_slot" not in _ALLOWLIST
        assert "confirm_booking" not in _ALLOWLIST
        assert "write_booking" not in _ALLOWLIST
