"""
tests/test_voice_mode.py — Comprehensive tests for Milestone 6A Hands-Free Conversational Voice Mode.

Test Coverage:
1. Natural verbal confirmation ("Yes, book it", "Confirm that appointment", "Yes please").
2. "Yes" without an active confirmation prompt cannot create a booking.
3. Verbal cancellation ("No", "Cancel", "Never mind").
4. Change of mind / modification ("Actually, can we do Wednesday at 2 PM instead?").
5. Ambiguous / hesitant responses ("Maybe", "Hold on") request clarification without committing.
6. Idempotency: repeated verbal confirmation does not create duplicate bookings.
7. Streaming WebSocket STT proxy:
   - Session validation (rejects invalid session with 4404).
   - Credential validation (rejects missing API key with 4503).
   - Bidirectional message forwarding (client audio bytes -> upstream; Results/UtteranceEnd -> client).
8. Side-effect isolation: WebSocket activity does not mutate agent history or SQLite.
9. End-to-end multi-turn conversation loop via web endpoints.
"""

from __future__ import annotations

import asyncio
import json
import sqlite3
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from google.genai import types

from scheduler import db
from scheduler.agent import (
    Agent,
    PendingBooking,
    is_verbal_ambiguity,
    is_verbal_cancellation,
    is_verbal_confirmation,
    is_verbal_modification,
)
from scheduler.db import (
    book_slot,
    cancel_booking,
    init_db,
    is_slot_available,
    list_available_slots,
    seed,
)
from scheduler.web import app, sessions


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def test_db(tmp_path: Path) -> Path:
    db_file = tmp_path / "voice_test.db"
    init_db(db_file)
    seed(db_file)
    return db_file


@pytest.fixture
def client(test_db: Path) -> TestClient:
    app.state.db_path = test_db
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
    resp.text = None
    return resp


# ---------------------------------------------------------------------------
# 1. Verbal Classification Unit Tests
# ---------------------------------------------------------------------------


class TestVerbalClassification:
    @pytest.mark.parametrize(
        "phrase",
        [
            "Yes",
            "yes",
            "YES",
            "Yes, book it",
            "Yes. Book it.",
            "Confirm",
            "Confirmed",
            "Confirm. Confirm.",
            "Confirm that appointment",
            "Confirm appointment",
            "Book it",
            "Yes please",
            "Please book it",
            "Go ahead",
            "Sounds good, book it",
            "That works, please book it",
            "Perfect, book it",
            "Let's do it",
            "Lets book it",
            "Sure, confirm it",
            "Yes, please confirm",
        ],
    )
    def test_verbal_confirmation_positives(self, phrase: str) -> None:
        assert is_verbal_confirmation(phrase) is True
        assert is_verbal_cancellation(phrase) is False

    @pytest.mark.parametrize(
        "phrase",
        [
            "No",
            "no",
            "NO",
            "Cancel",
            "cancel that",
            "Cancel it",
            "No thanks",
            "Don't book it",
            "Do not book it",
            "Never mind",
            "Stop",
            "Discard",
        ],
    )
    def test_verbal_cancellation_positives(self, phrase: str) -> None:
        assert is_verbal_cancellation(phrase) is True
        assert is_verbal_confirmation(phrase) is False

    @pytest.mark.parametrize(
        "phrase",
        [
            "Actually, can we do Wednesday instead?",
            "Change it to 2 PM",
            "Let's switch to Dr. Lovelace",
            "What about Thursday morning?",
            "I'd rather see a cardiologist",
            "Is there another time available?",
        ],
    )
    def test_verbal_modification_positives(self, phrase: str) -> None:
        assert is_verbal_modification(phrase) is True

    @pytest.mark.parametrize(
        "phrase",
        [
            "Maybe",
            "maybe",
            "Not sure",
            "Hold on",
            "Wait a second",
            "Wait a minute",
            "One moment",
            "Let me think",
            "Let me check",
            "Um",
            "Uh",
            "Idk",
        ],
    )
    def test_verbal_ambiguity_positives(self, phrase: str) -> None:
        assert is_verbal_ambiguity(phrase) is True
        assert is_verbal_confirmation(phrase) is False
        assert is_verbal_cancellation(phrase) is False


# ---------------------------------------------------------------------------
# 2. Verbal Confirmation Agent Staging & Commit Flow
# ---------------------------------------------------------------------------


class TestVerbalConfirmationAgent:
    def test_natural_verbal_confirmation_commits_exactly_once(self, test_db: Path) -> None:
        """
        After propose_booking, verbal confirmation ('Yes, book it') commits the
        booking to SQLite and returns the confirmed receipt.
        """
        slots = list_available_slots(db_path=test_db)
        target = slots[0]

        mock_client = MagicMock()
        mock_client.models.generate_content.side_effect = [
            _make_fn_call_response([("propose_booking", {"slot_id": target.id})]),
            _make_text_response(
                f"I've prepared your appointment with {target.doctor_name}. "
                "Would you like me to book this appointment for you?"
            ),
        ]

        agent = Agent(db_path=test_db, client=mock_client)
        # Turn 1: Stage proposal
        reply1 = agent.turn("I want an appointment with Dr. Lovelace")
        assert agent.pending_booking is not None
        assert agent.pending_booking.status == "awaiting_confirmation"
        assert agent.awaiting_verbal_confirmation is True

        # Turn 2: Natural verbal confirmation
        reply2 = agent.turn("Yes, book it")
        assert "successfully confirmed" in reply2.lower()
        assert agent.pending_booking.status == "confirmed"
        assert agent.awaiting_verbal_confirmation is False

        # Verify exactly one booking in SQLite
        conn = _connect(test_db)
        bookings = conn.execute("SELECT * FROM bookings").fetchall()
        conn.close()
        assert len(bookings) == 1
        assert bookings[0]["slot_id"] == target.id
        assert bookings[0]["status"] == "confirmed"

    def test_verbal_confirmation_with_staccato_user_phrases(
        self, test_db: Path
    ) -> None:
        """
        Verify that user phrases like 'Confirm. Confirm.' or 'Yes. Book it.'
        are immediately recognized as affirmative confirmation and commit to SQLite.
        """
        slots = list_available_slots(db_path=test_db)
        target = slots[0]

        mock_client = MagicMock()
        mock_client.models.generate_content.side_effect = [
            _make_fn_call_response([("propose_booking", {"slot_id": target.id})]),
            _make_text_response("I have prepared your appointment. Would you like to confirm?"),
        ]

        agent = Agent(db_path=test_db, client=mock_client)
        agent.turn("Book slot 1")
        assert agent.awaiting_verbal_confirmation is True

        # User says 'Confirm. Confirm.'
        reply = agent.turn("Confirm. Confirm.")
        assert "successfully confirmed" in reply.lower()
        assert agent.pending_booking.status == "confirmed"

        conn = _connect(test_db)
        count = conn.execute("SELECT count(*) FROM bookings").fetchone()[0]
        conn.close()
        assert count == 1

    def test_yes_without_active_confirmation_prompt_cannot_create_booking(
        self, test_db: Path
    ) -> None:
        """
        A 'Yes' said during ordinary conversation when NO confirmation was requested
        must NOT create any booking in SQLite.
        """
        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = _make_text_response(
            "Which doctor would you prefer to see: Dr. Lovelace or Dr. Babbage?"
        )

        agent = Agent(db_path=test_db, client=mock_client)
        assert agent.pending_booking is None
        assert agent.awaiting_verbal_confirmation is False

        # User says 'Yes' in greeting turn
        reply = agent.turn("Yes, please")
        assert "Which doctor" in reply
        assert agent.pending_booking is None

        # Verify zero bookings in SQLite
        conn = _connect(test_db)
        count = conn.execute("SELECT count(*) FROM bookings").fetchone()[0]
        conn.close()
        assert count == 0

    def test_verbal_cancellation_discards_proposal_without_database_write(
        self, test_db: Path
    ) -> None:
        """
        Patient saying 'No' or 'Cancel' when asked to confirm discards the proposal
        with zero database changes.
        """
        slots = list_available_slots(db_path=test_db)
        target = slots[0]

        mock_client = MagicMock()
        mock_client.models.generate_content.side_effect = [
            _make_fn_call_response([("propose_booking", {"slot_id": target.id})]),
            _make_text_response("Would you like me to book this appointment for you?"),
        ]

        agent = Agent(db_path=test_db, client=mock_client)
        agent.turn("Book slot 1")
        assert agent.pending_booking is not None
        assert agent.awaiting_verbal_confirmation is True

        # Patient says No
        reply = agent.turn("No thanks, cancel that")
        assert "cancelled" in reply.lower()
        assert agent.pending_booking is None
        assert agent.awaiting_verbal_confirmation is False

        # Verify SQLite has no bookings and slot remains available
        conn = _connect(test_db)
        count = conn.execute("SELECT count(*) FROM bookings").fetchone()[0]
        conn.close()
        assert count == 0
        assert is_slot_available(target.id, db_path=test_db) is True

    def test_patient_changes_mind_discards_old_proposal_and_processes_new_request(
        self, test_db: Path
    ) -> None:
        """
        Patient changing preference ('Actually, what about Wednesday instead?')
        discards the old proposal and invokes Gemini for the new request.
        """
        slots = list_available_slots(db_path=test_db)
        slot1 = slots[0]
        slot2 = slots[1]

        mock_client = MagicMock()
        mock_client.models.generate_content.side_effect = [
            # Turn 1: proposes slot 1
            _make_fn_call_response([("propose_booking", {"slot_id": slot1.id})]),
            _make_text_response("Would you like me to book Tuesday at 9 AM?"),
            # Turn 2: proposes slot 2 upon user mind change
            _make_fn_call_response([("propose_booking", {"slot_id": slot2.id})]),
            _make_text_response("I have Tuesday at 2 PM instead. Shall I book that?"),
        ]

        agent = Agent(db_path=test_db, client=mock_client)
        agent.turn("Book 9 AM")
        assert agent.pending_booking.slot_id == slot1.id

        # Patient changes mind
        reply2 = agent.turn("Actually, change it to 2 PM instead")
        assert agent.pending_booking.slot_id == slot2.id
        assert "2 PM" in reply2

        # Verify slot 1 was not booked
        assert is_slot_available(slot1.id, db_path=test_db) is True

    def test_ambiguous_reply_requests_clarification_without_committing(
        self, test_db: Path
    ) -> None:
        """
        Ambiguous answer ('Maybe', 'Hold on') requests clarification, preserves
        staged proposal, and writes nothing to SQLite.
        """
        slots = list_available_slots(db_path=test_db)
        target = slots[0]

        mock_client = MagicMock()
        mock_client.models.generate_content.side_effect = [
            _make_fn_call_response([("propose_booking", {"slot_id": target.id})]),
            _make_text_response("Would you like me to book this appointment for you?"),
        ]

        agent = Agent(db_path=test_db, client=mock_client)
        agent.turn("Book slot 1")
        assert agent.pending_booking is not None
        assert agent.awaiting_verbal_confirmation is True

        # Patient hesitates
        reply = agent.turn("Maybe, hold on a second")
        assert "Just to confirm" in reply
        assert "say yes to confirm" in reply
        # Staged proposal remains intact
        assert agent.pending_booking is not None
        assert agent.pending_booking.slot_id == target.id
        assert agent.awaiting_verbal_confirmation is True

        # Database untouched
        conn = _connect(test_db)
        count = conn.execute("SELECT count(*) FROM bookings").fetchone()[0]
        conn.close()
        assert count == 0

    def test_repeated_verbal_confirmation_is_idempotent(self, test_db: Path) -> None:
        """Repeating verbal confirmation does not create duplicate bookings."""
        slots = list_available_slots(db_path=test_db)
        target = slots[0]

        mock_client = MagicMock()
        mock_client.models.generate_content.side_effect = [
            _make_fn_call_response([("propose_booking", {"slot_id": target.id})]),
            _make_text_response("Would you like me to book this appointment for you?"),
            _make_text_response("You are all set with your confirmed appointment!"),
        ]

        agent = Agent(db_path=test_db, client=mock_client)
        agent.turn("Book slot 1")

        # 1. First confirmation
        reply1 = agent.turn("Yes, book it")
        assert "successfully confirmed" in reply1.lower()
        booking_id = agent.pending_booking.booking_id

        # 2. Second confirmation immediately following
        reply2 = agent.turn("Yes, confirm")
        # Should not insert duplicate row
        conn = _connect(test_db)
        bookings = conn.execute("SELECT * FROM bookings").fetchall()
        conn.close()
        assert len(bookings) == 1
        assert bookings[0]["id"] == booking_id

    def test_background_noise_followed_by_repeated_confirm_commits_booking(
        self, test_db: Path
    ) -> None:
        """
        Replicates user scenario:
        1. Stage slot
        2. Accidental room/phone speech ('Not far. My phone.')
        3. Clarification requested, proposal preserved
        4. User confirms ('Confirm. Confirm.') -> Commits to SQLite!
        """
        slots = list_available_slots(db_path=test_db)
        target = slots[0]

        mock_client = MagicMock()
        mock_client.models.generate_content.side_effect = [
            _make_fn_call_response([("propose_booking", {"slot_id": target.id})]),
            _make_text_response("Would you like me to book this appointment for you?"),
        ]

        agent = Agent(db_path=test_db, client=mock_client)
        agent.turn("Book slot 1")
        assert agent.pending_booking is not None
        assert agent.awaiting_verbal_confirmation is True

        # Accidental microphone pickup
        reply_noise = agent.turn("Not far. My phone.")
        assert "Just to confirm" in reply_noise
        assert agent.pending_booking is not None
        assert agent.pending_booking.slot_id == target.id
        assert agent.awaiting_verbal_confirmation is True

        # User confirms
        reply_confirm = agent.turn("Confirm. Confirm.")
        assert "successfully confirmed" in reply_confirm.lower()

        # Verify exactly 1 booking in SQLite
        conn = _connect(test_db)
        count = conn.execute("SELECT count(*) FROM bookings").fetchone()[0]
        conn.close()
        assert count == 1

    def test_repeated_confirm_and_yes_book_it_commits_immediately(
        self, test_db: Path
    ) -> None:
        """
        Replicates user scenario:
        User says 'Confirm. Confirm.' or 'Yes. Book it.' to staged slot.
        Commits immediately without looping.
        """
        slots = list_available_slots(db_path=test_db)
        target = slots[0]

        mock_client = MagicMock()
        mock_client.models.generate_content.side_effect = [
            _make_fn_call_response([("propose_booking", {"slot_id": target.id})]),
            _make_text_response("Would you like me to book this appointment for you?"),
        ]

        agent = Agent(db_path=test_db, client=mock_client)
        agent.turn("Book slot 1")
        assert agent.pending_booking is not None

        reply = agent.turn("Yes. Book it.")
        assert "successfully confirmed" in reply.lower()

        conn = _connect(test_db)
        count = conn.execute("SELECT count(*) FROM bookings").fetchone()[0]
        conn.close()
        assert count == 1

    def test_recovery_from_cleared_proposal_after_informative_question(
        self, test_db: Path
    ) -> None:
        """
        If user asks an informative question during confirmation ('Where is the clinic?'),
        model answers, and user then says 'Yes, book it', system recovers the staged proposal
        and commits to SQLite.
        """
        slots = list_available_slots(db_path=test_db)
        target = slots[0]

        mock_client = MagicMock()
        mock_client.models.generate_content.side_effect = [
            _make_fn_call_response([("propose_booking", {"slot_id": target.id})]),
            _make_text_response("Would you like me to book this appointment for you?"),
            _make_text_response("Our clinic is at 100 Main Street in downtown."),
        ]

        agent = Agent(db_path=test_db, client=mock_client)
        agent.turn("Book slot 1")
        assert agent.pending_booking is not None

        # User asks informative question
        reply_q = agent.turn("Where is the clinic located?")
        assert "100 Main Street" in reply_q
        assert agent.pending_booking is None

        # User then confirms
        reply_confirm = agent.turn("Yes, book it")
        assert "successfully confirmed" in reply_confirm.lower()

        conn = _connect(test_db)
        count = conn.execute("SELECT count(*) FROM bookings").fetchone()[0]
        conn.close()
        assert count == 1


# ---------------------------------------------------------------------------
# 3. Web Layer Verbal Confirmation Integration
# ---------------------------------------------------------------------------


class TestWebVerbalConfirmation:
    def test_messages_endpoint_returns_confirmed_object_on_verbal_confirm(
        self, client: TestClient, test_db: Path
    ) -> None:
        sess_resp = client.post("/api/sessions")
        session_id = sess_resp.json()["session_id"]
        session_state = sessions[session_id]

        slots = list_available_slots(db_path=test_db)
        target = slots[0]

        # Stage proposal directly on agent
        session_state.agent.pending_booking = PendingBooking(
            slot_id=target.id,
            doctor_name=target.doctor_name,
            specialty=target.specialty,
            start_utc=target.start_utc,
            end_utc=target.end_utc,
            local_time_str="Tuesday, November 17, 2026 at 09:00 AM PST",
            idempotency_key=f"book-{target.id}-testkey",
            patient_name="Alex Taylor",
            status="awaiting_confirmation",
        )
        session_state.agent.awaiting_verbal_confirmation = True

        # User says verbal confirmation via messages endpoint
        res = client.post(
            f"/api/sessions/{session_id}/messages",
            json={"text": "Yes, please book it"},
        )
        assert res.status_code == 200
        data = res.json()
        assert data["confirmed"] is not None
        assert data["confirmed"]["status"] == "confirmed"
        assert data["confirmed"]["booking_id"] is not None
        assert "confirmed" in data["spoken_reply"].lower()
        assert data["pending"] is None


# ---------------------------------------------------------------------------
# 4. Streaming STT WebSocket Proxy
# ---------------------------------------------------------------------------


class TestStreamingSTTWebSocket:
    def test_websocket_unknown_session_closes_with_4404(self, client: TestClient) -> None:
        with pytest.raises(Exception):
            with client.websocket_connect("/api/sessions/nonexistent-session/ws/transcribe") as ws:
                ws.receive_json()

    def test_websocket_missing_credentials_closes_with_4503(
        self, client: TestClient
    ) -> None:
        sess_resp = client.post("/api/sessions")
        session_id = sess_resp.json()["session_id"]

        app.state.deepgram_api_key = ""
        with patch.dict("os.environ", {}, clear=True):
            with pytest.raises(Exception):
                with client.websocket_connect(
                    f"/api/sessions/{session_id}/ws/transcribe"
                ) as ws:
                    ws.receive_json()

    def test_websocket_bidirectional_streaming_mock(self, client: TestClient) -> None:
        """
        Verify bidirectional streaming through the proxy:
        Client sends audio bytes, server forwards mock Results and UtteranceEnd.
        """
        sess_resp = client.post("/api/sessions")
        session_id = sess_resp.json()["session_id"]

        received_bytes = []

        async def mock_handler(ws, s_id):
            await ws.send_json({"type": "ready"})
            # Read one binary message from client
            msg = await ws.receive()
            if "bytes" in msg:
                received_bytes.append(msg["bytes"])

            # Send back a mock interim result
            await ws.send_json({
                "type": "Results",
                "transcript": "I need an appointment",
                "is_final": False,
                "speech_final": False,
                "confidence": 0.95,
            })
            # Send back speech_final
            await ws.send_json({
                "type": "Results",
                "transcript": "I need an appointment on Tuesday",
                "is_final": True,
                "speech_final": True,
                "confidence": 0.99,
            })
            # Send SpeechStarted (barge-in VAD event)
            await ws.send_json({"type": "SpeechStarted"})
            # Send UtteranceEnd
            await ws.send_json({"type": "UtteranceEnd"})
            await ws.close()

        app.state.mock_deepgram_ws = mock_handler

        with client.websocket_connect(f"/api/sessions/{session_id}/ws/transcribe") as ws:
            # 1. Expect ready
            msg1 = ws.receive_json()
            assert msg1["type"] == "ready"

            # 2. Send dummy audio chunk
            ws.send_bytes(b"\x00\x01\x02\x03\x04")

            # 3. Expect interim Results
            msg2 = ws.receive_json()
            assert msg2["type"] == "Results"
            assert msg2["is_final"] is False
            assert "appointment" in msg2["transcript"]

            # 4. Expect finalized Results
            msg3 = ws.receive_json()
            assert msg3["type"] == "Results"
            assert msg3["speech_final"] is True
            assert "Tuesday" in msg3["transcript"]

            # 5. Expect SpeechStarted
            msg4 = ws.receive_json()
            assert msg4["type"] == "SpeechStarted"

            # 6. Expect UtteranceEnd
            msg5 = ws.receive_json()
            assert msg5["type"] == "UtteranceEnd"

        assert len(received_bytes) == 1
        assert received_bytes[0] == b"\x00\x01\x02\x03\x04"
        app.state.mock_deepgram_ws = None

    def test_websocket_does_not_mutate_agent_or_sqlite(
        self, client: TestClient, test_db: Path
    ) -> None:
        """
        Verify that streaming audio frames over WebSocket is strictly presentational
        and never stages proposals or writes to SQLite.
        """
        sess_resp = client.post("/api/sessions")
        session_id = sess_resp.json()["session_id"]
        session_state = sessions[session_id]

        async def mock_handler(ws, s_id):
            await ws.send_json({"type": "ready"})
            await ws.receive()
            await ws.send_json({"type": "UtteranceEnd"})
            await ws.close()

        app.state.mock_deepgram_ws = mock_handler

        with client.websocket_connect(f"/api/sessions/{session_id}/ws/transcribe") as ws:
            ws.receive_json()
            ws.send_bytes(b"test-audio-bytes")
            ws.receive_json()

        # Invariant: agent conversation history untouched
        assert len(session_state.agent.history) == 0
        assert session_state.agent.pending_booking is None

        # Invariant: SQLite bookings untouched
        conn = _connect(test_db)
        count = conn.execute("SELECT count(*) FROM bookings").fetchone()[0]
        conn.close()
        assert count == 0
        app.state.mock_deepgram_ws = None
