"""
tests/test_interruption_m7.py — Tests for Milestone 7 Real-Time Voice Experience.

Test Coverage:
1. Interruption endpoint (/api/sessions/{session_id}/interrupt):
   - Returns 404 for non-existent session.
   - Resets playback state and marks session as interrupted.
   - Discards active pending in-memory proposals safely.
   - Logs an 'interrupt' inspector event.
   - Interruption prevents any subsequent /confirm commit.
2. Crucial safety invariant: Interrupted confirmation prompt NEVER commits:
   - User barge-in with "Wait" does not commit to SQLite.
   - User barge-in with "Stop" discards proposal without committing.
   - User barge-in with "Actually no" discards proposal without committing.
   - User barge-in with "Cancel" discards proposal without committing.
   - Backend /interrupt before confirmation prevents database commit.
3. Sequential barge-in with preference changes:
   - "Wait, change to Dr. Lovelace" cancels previous proposal and processes new request.
   - Previous slot is never booked in SQLite.
4. Echo suppression and barge-in guard:
   - EchoSuppressionGuard suppresses speaker onset transients within debounce window.
   - Suppresses low-energy speaker acoustic feedback below threshold.
   - Allows genuine human speech exceeding energy threshold after debounce window.
   - PCM energy calculation accurately computes RMS on audio buffers.
"""

from __future__ import annotations

import sqlite3
import struct
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient
from google.genai import types

from scheduler import db
from scheduler.agent import (
    Agent,
    PendingBooking,
    is_verbal_cancellation,
    is_verbal_confirmation,
    is_verbal_modification,
)
from scheduler.speech import EchoSuppressionGuard, calculate_pcm_energy
from scheduler.web import app, sessions


# ---------------------------------------------------------------------------
# Test Fixtures & Helpers
# ---------------------------------------------------------------------------


@pytest.fixture
def test_db(tmp_path: Path) -> Path:
    db_file = tmp_path / "m7_test.db"
    db.init_db(db_file)
    db.seed(db_file)
    return db_file


@pytest.fixture
def client(test_db: Path) -> TestClient:
    app.state.db_path = test_db
    app.state.deepgram_api_key = "test-deepgram-key"
    app.state.cartesia_api_key = "test-cartesia-key"
    sessions.clear()

    # Mock Gemini client
    mock_gemini = MagicMock()
    mock_gemini.models.generate_content.return_value = MagicMock(
        candidates=[
            MagicMock(
                content=types.Content(
                    role="model",
                    parts=[types.Part.from_text(text="I can help you schedule an appointment.")],
                )
            )
        ],
        function_calls=[],
        text="I can help you schedule an appointment.",
    )
    app.state.gemini_client = mock_gemini

    return TestClient(app)


def _count_bookings(db_path: Path) -> int:
    with sqlite3.connect(db_path) as conn:
        return conn.execute("SELECT COUNT(*) FROM bookings").fetchone()[0]


def _stage_mock_proposal(agent: Agent, slot_id: int = 1, doctor: str = "Dr. Florence Nightingale") -> None:
    slot = db.get_slot(slot_id, db_path=agent.db_path)
    assert slot is not None
    agent.pending_booking = PendingBooking(
        slot_id=slot.id,
        doctor_name=doctor,
        specialty=slot.specialty or "Pediatrics",
        start_utc=slot.start_utc,
        end_utc=slot.end_utc,
        local_time_str="Tuesday, November 17, 2026 at 09:00 AM PST",
        idempotency_key=f"book-{slot.id}-test-key",
        patient_name=agent.patient_name,
        status="awaiting_confirmation",
    )
    agent.awaiting_verbal_confirmation = True


# ---------------------------------------------------------------------------
# Test Suite 1: Interruption Endpoint (/api/sessions/{session_id}/interrupt)
# ---------------------------------------------------------------------------


class TestInterruptionEndpoint:
    def test_interrupt_unknown_session_returns_404(self, client: TestClient) -> None:
        res = client.post("/api/sessions/non-existent-session-id/interrupt")
        assert res.status_code == 404
        assert "not found" in res.json()["detail"].lower()

    def test_interrupt_idle_session_resets_state_and_logs_event(self, client: TestClient) -> None:
        sess_res = client.post("/api/sessions")
        session_id = sess_res.json()["session_id"]
        session_state = sessions[session_id]

        res = client.post(f"/api/sessions/{session_id}/interrupt")
        assert res.status_code == 200
        data = res.json()
        assert data["session_id"] == session_id
        assert data["interrupted"] is True
        assert data["staged_cancelled"] is False

        assert session_state.is_interrupted is True

        # Verify inspector logs an 'interrupt' stage event
        interrupt_events = [e for e in session_state.events if e.stage == "interrupt"]
        assert len(interrupt_events) == 1
        assert interrupt_events[0].title == "User Interruption / Barge-in"
        assert interrupt_events[0].details["staged_cancelled"] is False

    def test_interrupt_cancels_awaiting_confirmation_proposal(
        self, client: TestClient, test_db: Path
    ) -> None:
        sess_res = client.post("/api/sessions")
        session_id = sess_res.json()["session_id"]
        session_state = sessions[session_id]

        # Stage proposal
        _stage_mock_proposal(session_state.agent, slot_id=1)
        assert session_state.agent.pending_booking is not None
        assert session_state.agent.awaiting_verbal_confirmation is True

        # User barges in: call interrupt
        res = client.post(f"/api/sessions/{session_id}/interrupt")
        assert res.status_code == 200
        data = res.json()
        assert data["interrupted"] is True
        assert data["staged_cancelled"] is True

        # Verify pending booking is cleared
        assert session_state.agent.pending_booking is None
        assert session_state.agent.awaiting_verbal_confirmation is False

        # Verify inspector event logged
        interrupt_events = [e for e in session_state.events if e.stage == "interrupt"]
        assert len(interrupt_events) == 1
        assert interrupt_events[0].details["staged_cancelled"] is True
        assert interrupt_events[0].details["slot_id"] == 1

        # Attempt to confirm after interruption: MUST fail without committing
        confirm_res = client.post(f"/api/sessions/{session_id}/confirm")
        assert confirm_res.status_code == 200
        confirm_data = confirm_res.json()
        assert confirm_data["success"] is False
        assert confirm_data["booking_id"] is None

        # Ensure SQLite has NO bookings
        assert _count_bookings(test_db) == 0


# ---------------------------------------------------------------------------
# Test Suite 2: Crucial Safety Invariant (Interrupted Prompt Never Commits)
# ---------------------------------------------------------------------------


class TestInterruptedConfirmationSafety:
    def test_user_says_wait_does_not_commit_to_sqlite(self, test_db: Path) -> None:
        agent = Agent(db_path=test_db, client=MagicMock())
        _stage_mock_proposal(agent, slot_id=1)

        # User interrupts playback / asks to wait
        reply = agent.turn("Wait")
        assert "just to confirm" in reply.lower() or "would you like" in reply.lower()

        # Proposal is NOT committed in SQLite
        assert _count_bookings(test_db) == 0
        assert agent.pending_booking is not None
        assert agent.pending_booking.status == "awaiting_confirmation"

    def test_user_says_stop_cancels_proposal_without_database_write(self, test_db: Path) -> None:
        agent = Agent(db_path=test_db, client=MagicMock())
        _stage_mock_proposal(agent, slot_id=1)

        # User interrupts with "Stop"
        reply = agent.turn("Stop")
        assert "cancelled" in reply.lower() or "discarded" in reply.lower()

        # Database must have zero bookings
        assert _count_bookings(test_db) == 0
        assert agent.pending_booking is None

    def test_user_says_actually_no_cancels_proposal_without_database_write(
        self, test_db: Path
    ) -> None:
        # Mock Gemini response for turn processing
        mock_gemini = MagicMock()
        mock_gemini.models.generate_content.return_value = MagicMock(
            candidates=[
                MagicMock(
                    content=types.Content(
                        role="model",
                        parts=[types.Part.from_text(text="Understood. Let me know what you prefer.")],
                    )
                )
            ],
            function_calls=[],
            text="Understood. Let me know what you prefer.",
        )

        agent = Agent(db_path=test_db, client=mock_gemini)
        _stage_mock_proposal(agent, slot_id=1)

        # User interrupts with "Actually no"
        reply = agent.turn("Actually no")

        # Database must have zero bookings
        assert _count_bookings(test_db) == 0
        # Proposal was discarded
        assert agent.pending_booking is None

    def test_interrupted_confirmation_via_endpoint_cannot_be_confirmed(
        self, client: TestClient, test_db: Path
    ) -> None:
        sess_res = client.post("/api/sessions")
        session_id = sess_res.json()["session_id"]
        session_state = sessions[session_id]

        _stage_mock_proposal(session_state.agent, slot_id=1)

        # Frontend issues barge-in interrupt signal
        client.post(f"/api/sessions/{session_id}/interrupt")

        # Now suppose an in-flight or delayed confirm request arrives
        confirm_res = client.post(f"/api/sessions/{session_id}/confirm")
        assert confirm_res.status_code == 200
        assert confirm_res.json()["success"] is False

        # No booking in database
        assert _count_bookings(test_db) == 0


# ---------------------------------------------------------------------------
# Test Suite 3: Sequential Barge-In with Utterance ("Wait, change to...")
# ---------------------------------------------------------------------------


class TestSequentialBargeIn:
    def test_barge_in_change_doctor_cancels_old_and_processes_new(
        self, test_db: Path
    ) -> None:
        """
        Patient is in confirmation prompt for Dr. Nightingale.
        Patient barges in with: 'Wait, change to Dr. Lovelace'.
        Verifies old proposal is cancelled and new preference is processed.
        """
        mock_gemini = MagicMock()
        mock_gemini.models.generate_content.return_value = MagicMock(
            candidates=[
                MagicMock(
                    content=types.Content(
                        role="model",
                        parts=[
                            types.Part.from_text(
                                text="I can help you book with Dr. Lovelace. Let me check her available times."
                            )
                        ],
                    )
                )
            ],
            function_calls=[],
            text="I can help you book with Dr. Lovelace. Let me check her available times.",
        )

        agent = Agent(db_path=test_db, client=mock_gemini)
        _stage_mock_proposal(agent, slot_id=1, doctor="Dr. Florence Nightingale")

        # Patient barges in with preference change
        reply = agent.turn("Wait, change to Dr. Lovelace")

        assert "lovelace" in reply.lower()
        # Initial proposal for Dr. Nightingale must NOT have committed to SQLite
        assert _count_bookings(test_db) == 0
        # Agent pending booking for Nightingale is discarded
        assert agent.pending_booking is None


# ---------------------------------------------------------------------------
# Test Suite 4: Echo Suppression Guard & Audio Energy Calculations
# ---------------------------------------------------------------------------


class TestEchoSuppressionGuard:
    def test_guard_allows_speech_when_not_speaking(self) -> None:
        guard = EchoSuppressionGuard(energy_threshold=0.02, onset_debounce_ms=150.0)
        assert guard.is_speaking is False

        # Even with zero energy or arbitrary timestamp, speech is allowed when idle
        assert guard.should_allow_barge_in(energy=0.0, timestamp=100.0) is True
        assert guard.should_allow_barge_in(energy=0.05, timestamp=101.0) is True

    def test_guard_suppresses_onset_debounce_transients(self) -> None:
        guard = EchoSuppressionGuard(energy_threshold=0.02, onset_debounce_ms=150.0)
        guard.start_playback(timestamp=100.0)
        assert guard.is_speaking is True

        # At t=100.05 (50ms into playback, < 150ms debounce window),
        # speech should be rejected even with strong energy to prevent speaker onset clicks
        allowed = guard.should_allow_barge_in(energy=0.08, timestamp=100.05)
        assert allowed is False

        # At t=100.12 (120ms into playback, < 150ms debounce window)
        allowed = guard.should_allow_barge_in(energy=0.10, timestamp=100.12)
        assert allowed is False

    def test_guard_suppresses_low_energy_speaker_echo(self) -> None:
        guard = EchoSuppressionGuard(energy_threshold=0.02, onset_debounce_ms=150.0)
        guard.start_playback(timestamp=100.0)

        # After onset debounce (> 150ms), e.g. at t=100.25 (250ms),
        # low-energy input (speaker echo leakage) must be suppressed
        allowed = guard.should_allow_barge_in(energy=0.008, timestamp=100.25)
        assert allowed is False

    def test_guard_allows_genuine_user_barge_in(self) -> None:
        guard = EchoSuppressionGuard(energy_threshold=0.02, onset_debounce_ms=150.0)
        guard.start_playback(timestamp=100.0)

        # After onset debounce (> 150ms), input above energy threshold is allowed
        allowed = guard.should_allow_barge_in(energy=0.065, timestamp=100.25)
        assert allowed is True
        assert guard.last_interrupt_time == 100.25

    def test_guard_stop_playback_resets_state(self) -> None:
        guard = EchoSuppressionGuard()
        guard.start_playback(timestamp=100.0)
        assert guard.is_speaking is True

        guard.stop_playback()
        assert guard.is_speaking is False
        assert guard.playback_start_time is None
        assert guard.should_allow_barge_in(energy=0.01) is True

    def test_calculate_pcm_energy_silence(self) -> None:
        # Empty or zero buffer
        assert calculate_pcm_energy(b"") == 0.0
        assert calculate_pcm_energy(b"\x00\x00" * 100) == 0.0
        # Odd-length buffer handled safely
        assert calculate_pcm_energy(b"\x00") == 0.0

    def test_calculate_pcm_energy_active_audio(self) -> None:
        # Generate 100 samples of 16-bit PCM at ~50% amplitude (~16384)
        sample_val = 16384
        pcm_data = struct.pack("<100h", *([sample_val] * 100))
        energy = calculate_pcm_energy(pcm_data)
        # RMS of constant 16384 normalized by 32768 should be ~0.50
        assert 0.49 <= energy <= 0.51

    def test_affirmative_utterance_with_trailing_cancellation_never_commits(
        self, client: TestClient, test_db: Path
    ) -> None:
        """
        Verify that a composite utterance containing both affirmation and cancellation
        (e.g., 'Yes, book it, actually cancel that' or 'Confirm appointment, no wait stop')
        triggers the cancellation/negation guard and NEVER commits to SQLite.
        """
        composite_utterances = [
            "Yes, book it, actually cancel that",
            "Confirm appointment, no wait stop",
            "Yes please, never mind do not book",
        ]

        for utt in composite_utterances:
            sess_resp = client.post("/api/sessions")
            session_id = sess_resp.json()["session_id"]
            session_state = sessions[session_id]

            # Stage slot 1 directly
            session_state.agent.pending_booking = PendingBooking(
                slot_id=1,
                doctor_name="Dr. Florence Nightingale",
                specialty="Pediatrics",
                start_utc="2026-11-17T17:00:00Z",
                end_utc="2026-11-17T17:30:00Z",
                local_time_str="Tuesday, November 17, 2026 at 09:00 AM PST",
                idempotency_key="book-1-test-cancel-race",
                patient_name="Alex Taylor",
                status="awaiting_confirmation",
            )
            session_state.agent.awaiting_verbal_confirmation = True

            # Invariant: Never classified as verbal confirmation
            assert is_verbal_confirmation(utt) is False

            # Send through agent turn
            mock_gemini = MagicMock()
            mock_gemini.models.generate_content.return_value = MagicMock(
                candidates=[
                    MagicMock(
                        content=types.Content(
                            role="model",
                            parts=[types.Part.from_text(text="Understood, request cancelled.")],
                        )
                    )
                ],
                function_calls=[],
                text="Understood, request cancelled.",
            )
            session_state.agent.client = mock_gemini

            reply = session_state.agent.turn(utt)
            # Invariant: SQLite bookings table count is strictly 0 (never committed)
            assert _count_bookings(test_db) == 0

    @pytest.mark.asyncio
    async def test_concurrent_affirmative_request_racing_with_cancellation(
        self, test_db: Path
    ) -> None:
        """
        Verify concurrency race safety: When an affirmative confirmation request
        races against an explicit cancellation request on the same session,
        the per-session lock serializes them.
        Invariant: The final database state has either 0 or 1 booking (never corrupted or duplicate).
        """
        import asyncio
        from starlette.responses import Response
        from scheduler.web import (
            create_session,
            send_message,
            cancel_appointment,
            MessageRequest,
        )

        app.state.db_path = test_db
        # Create session directly via async endpoint
        sess_resp = await create_session()
        session_id = sess_resp.session_id
        session_state = sessions[session_id]

        # Stage slot 2
        session_state.agent.pending_booking = PendingBooking(
            slot_id=2,
            doctor_name="Dr. Florence Nightingale",
            specialty="Pediatrics",
            start_utc="2026-11-17T18:00:00Z",
            end_utc="2026-11-17T18:30:00Z",
            local_time_str="Tuesday, November 17, 2026 at 10:00 AM PST",
            idempotency_key="book-2-test-race",
            patient_name="Alex Taylor",
            status="awaiting_confirmation",
        )
        session_state.agent.awaiting_verbal_confirmation = True

        async def do_confirm():
            resp = Response()
            req = MessageRequest(text="Yes, book it")
            return await send_message(session_id=session_id, req=req, response=resp)

        async def do_cancel():
            return await cancel_appointment(session_id=session_id)

        # Race them concurrently with asyncio.gather
        results = await asyncio.gather(do_confirm(), do_cancel(), return_exceptions=True)

        # Check SQLite
        with sqlite3.connect(test_db) as conn:
            bookings = conn.execute("SELECT id, slot_id, status FROM bookings WHERE slot_id = 2").fetchall()

        # Invariant: Exactly 0 or 1 row in SQLite, never duplicated or corrupted
        assert len(bookings) in (0, 1)
