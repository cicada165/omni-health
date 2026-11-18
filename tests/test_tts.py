"""
tests/test_tts.py — Unit and integration tests for Cartesia text-to-speech (Milestone 4B).

Tests Cartesia client, configuration, WAV format validation, error handling,
FastAPI /speak endpoint, audio headers, and side-effect isolation guarantees.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from cartesia import APITimeoutError, AuthenticationError, CartesiaError
from fastapi.testclient import TestClient
from google.genai import types

from scheduler import db
from scheduler.tts import (
    DEFAULT_MODEL,
    DEFAULT_SAMPLE_RATE,
    DEFAULT_VOICE_ID,
    MAX_TEXT_LENGTH,
    TTSCredentialsError,
    TTSError,
    TTSProviderError,
    TTSTimeoutError,
    format_spoken_receipt,
    format_spoken_text,
    get_cartesia_api_key,
    get_cartesia_model,
    get_cartesia_voice_id,
    synthesize_speech,
)
from scheduler.web import app, sessions


# ---------------------------------------------------------------------------
# Helpers & Fixtures
# ---------------------------------------------------------------------------


def _mock_cartesia_generate_response(wav_bytes: bytes = b"RIFF....WAVEfmt ....data....") -> MagicMock:
    resp = MagicMock()
    resp.read.return_value = wav_bytes
    return resp


@pytest.fixture()
def client(tmp_path: Path) -> TestClient:
    test_db = tmp_path / "tts_test.db"
    db.init_db(test_db)
    db.seed(test_db)

    app.state.db_path = test_db
    app.state.cartesia_api_key = "test-cartesia-api-key"
    sessions.clear()

    # Mock Gemini so session creation works without API calls
    mock_gemini = MagicMock()
    mock_gemini.models.generate_content.return_value = MagicMock(
        candidates=[
            MagicMock(
                content=types.Content(
                    role="model",
                    parts=[types.Part.from_text(text="Welcome! How can I help?")],
                )
            )
        ],
        function_calls=[],
        text="Welcome! How can I help?",
    )
    app.state.gemini_client = mock_gemini

    return TestClient(app)


# ---------------------------------------------------------------------------
# Unit Tests: Cartesia TTS Client (scheduler/tts.py)
# ---------------------------------------------------------------------------


class TestCartesiaClient:
    def test_default_configuration_helpers(self) -> None:
        with patch.dict("os.environ", {}, clear=True):
            assert get_cartesia_model() == DEFAULT_MODEL
            assert get_cartesia_voice_id() == DEFAULT_VOICE_ID
            with pytest.raises(TTSCredentialsError, match="CARTESIA_API_KEY is not configured"):
                get_cartesia_api_key()

        with patch.dict(
            "os.environ",
            {
                "CARTESIA_API_KEY": "custom-key",
                "CARTESIA_MODEL": "sonic-custom",
                "CARTESIA_VOICE_ID": "voice-1234",
            },
            clear=True,
        ):
            assert get_cartesia_api_key() == "custom-key"
            assert get_cartesia_model() == "sonic-custom"
            assert get_cartesia_voice_id() == "voice-1234"

    def test_synthesize_speech_success(self) -> None:
        mock_cartesia = MagicMock()
        mock_response = _mock_cartesia_generate_response(b"RIFF\x24\x00\x00\x00WAVEfmt ")
        mock_cartesia.tts.generate.return_value = mock_response

        wav_bytes = synthesize_speech(
            text="Your appointment is confirmed.",
            api_key="test-key",
            model_id="sonic-3.6",
            voice_id="stock-voice-id",
            client=mock_cartesia,
        )

        assert wav_bytes == b"RIFF\x24\x00\x00\x00WAVEfmt "
        mock_cartesia.tts.generate.assert_called_once()
        call_kwargs = mock_cartesia.tts.generate.call_args[1]
        assert call_kwargs["model_id"] == "sonic-3.6"
        assert call_kwargs["transcript"] == "Your appointment is confirmed."
        assert call_kwargs["voice"] == "stock-voice-id"
        assert call_kwargs["output_format"] == {
            "container": "wav",
            "encoding": "pcm_s16le",
            "sample_rate": DEFAULT_SAMPLE_RATE,
        }

    def test_synthesize_speech_empty_text_raises(self) -> None:
        with pytest.raises(ValueError, match="Text cannot be empty"):
            synthesize_speech("", api_key="test-key")
        with pytest.raises(ValueError, match="Text cannot be empty"):
            synthesize_speech("   \n\t  ", api_key="test-key")

    def test_synthesize_speech_oversized_text_raises(self) -> None:
        oversized = "x" * (MAX_TEXT_LENGTH + 1)
        with pytest.raises(ValueError, match="exceeds maximum allowed"):
            synthesize_speech(oversized, api_key="test-key")

    def test_synthesize_speech_missing_api_key_raises(self) -> None:
        with patch.dict("os.environ", {}, clear=True):
            with pytest.raises(TTSCredentialsError, match="CARTESIA_API_KEY is not configured"):
                synthesize_speech("Hello", api_key=None)

    def test_synthesize_speech_timeout_raises(self) -> None:
        mock_cartesia = MagicMock()
        mock_cartesia.tts.generate.side_effect = APITimeoutError(MagicMock())

        with pytest.raises(TTSTimeoutError, match="timed out"):
            synthesize_speech("Hello", api_key="test-key", client=mock_cartesia)

    def test_synthesize_speech_auth_error_raises(self) -> None:
        mock_cartesia = MagicMock()
        mock_cartesia.tts.generate.side_effect = AuthenticationError(
            message="Invalid API Key",
            response=MagicMock(status_code=401),
            body=None,
        )

        with pytest.raises(TTSCredentialsError, match="authentication failed"):
            synthesize_speech("Hello", api_key="bad-key", client=mock_cartesia)

    def test_synthesize_speech_provider_error_raises(self) -> None:
        mock_cartesia = MagicMock()
        mock_cartesia.tts.generate.side_effect = CartesiaError("Service unavailable")

        with pytest.raises(TTSProviderError, match="Cartesia synthesis error"):
            synthesize_speech("Hello", api_key="test-key", client=mock_cartesia)

    def test_synthesize_speech_empty_audio_stream_raises(self) -> None:
        mock_cartesia = MagicMock()
        mock_response = MagicMock()
        mock_response.read.return_value = b""
        mock_cartesia.tts.generate.return_value = mock_response

        with pytest.raises(TTSProviderError, match="empty audio stream"):
            synthesize_speech("Hello", api_key="test-key", client=mock_cartesia)


# ---------------------------------------------------------------------------
# Integration Tests: FastAPI /speak Endpoint (scheduler/web.py)
# ---------------------------------------------------------------------------


class TestSpeakEndpoint:
    def test_speak_valid_request_success(self, client: TestClient) -> None:
        # Create session
        sess_resp = client.post("/api/sessions")
        assert sess_resp.status_code == 201
        session_id = sess_resp.json()["session_id"]

        # Mock Cartesia on app.state
        mock_cartesia = MagicMock()
        expected_wav = b"RIFF....WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00"
        mock_cartesia.tts.generate.return_value = _mock_cartesia_generate_response(expected_wav)
        app.state.cartesia_client = mock_cartesia

        req_body = {"text": "Your appointment with Dr. Chen is confirmed."}
        res = client.post(f"/api/sessions/{session_id}/speak", json=req_body)

        assert res.status_code == 200
        assert res.headers["content-type"] == "audio/wav"
        assert res.headers["cache-control"] == "no-store"
        assert res.content == expected_wav

    def test_speak_unknown_session_returns_404(self, client: TestClient) -> None:
        res = client.post(
            "/api/sessions/unknown-session-id/speak",
            json={"text": "Hello world."},
        )
        assert res.status_code == 404
        assert "not found" in res.json()["detail"].lower()

    def test_speak_empty_text_returns_422(self, client: TestClient) -> None:
        sess_resp = client.post("/api/sessions")
        session_id = sess_resp.json()["session_id"]

        res = client.post(
            f"/api/sessions/{session_id}/speak",
            json={"text": "    \n   "},
        )
        assert res.status_code == 422
        assert "empty" in res.json()["detail"].lower()

    def test_speak_oversized_text_returns_422(self, client: TestClient) -> None:
        sess_resp = client.post("/api/sessions")
        session_id = sess_resp.json()["session_id"]

        oversized_str = "a" * (MAX_TEXT_LENGTH + 50)
        res = client.post(
            f"/api/sessions/{session_id}/speak",
            json={"text": oversized_str},
        )
        assert res.status_code == 422

    def test_speak_missing_credentials_returns_503(self, client: TestClient) -> None:
        sess_resp = client.post("/api/sessions")
        session_id = sess_resp.json()["session_id"]

        app.state.cartesia_api_key = ""
        with patch.dict("os.environ", {}, clear=True):
            res = client.post(
                f"/api/sessions/{session_id}/speak",
                json={"text": "Testing missing credentials."},
            )
            assert res.status_code == 503
            assert "not configured" in res.json()["detail"].lower()

    def test_speak_timeout_returns_504(self, client: TestClient) -> None:
        sess_resp = client.post("/api/sessions")
        session_id = sess_resp.json()["session_id"]

        mock_cartesia = MagicMock()
        mock_cartesia.tts.generate.side_effect = APITimeoutError(MagicMock())
        app.state.cartesia_client = mock_cartesia

        res = client.post(
            f"/api/sessions/{session_id}/speak",
            json={"text": "This request will time out."},
        )
        assert res.status_code == 504
        assert "timed out" in res.json()["detail"].lower()

    def test_speak_provider_error_returns_502(self, client: TestClient) -> None:
        sess_resp = client.post("/api/sessions")
        session_id = sess_resp.json()["session_id"]

        mock_cartesia = MagicMock()
        mock_cartesia.tts.generate.side_effect = CartesiaError("Backend failure")
        app.state.cartesia_client = mock_cartesia

        res = client.post(
            f"/api/sessions/{session_id}/speak",
            json={"text": "Backend error test."},
        )
        assert res.status_code == 502
        assert "error" in res.json()["detail"].lower()

    def test_speak_does_not_alter_history_or_booking_state(
        self, client: TestClient
    ) -> None:
        """
        Verify that /speak is pure presentation: never alters conversation
        history, never stages pending appointments, and never mutates bookings.
        """
        sess_resp = client.post("/api/sessions")
        session_id = sess_resp.json()["session_id"]
        session_state = sessions[session_id]

        history_before = list(session_state.agent.history)
        pending_before = session_state.agent.pending_booking

        with sqlite3.connect(app.state.db_path) as conn:
            booking_count_before = conn.execute("SELECT COUNT(*) FROM bookings").fetchone()[0]

        mock_cartesia = MagicMock()
        mock_cartesia.tts.generate.return_value = _mock_cartesia_generate_response(b"RIFFWAVEaudio")
        app.state.cartesia_client = mock_cartesia

        res = client.post(
            f"/api/sessions/{session_id}/speak",
            json={"text": "Dr. Chen is available next Monday."},
        )
        assert res.status_code == 200

        # Assert no state was altered
        assert session_state.agent.history == history_before
        assert session_state.agent.pending_booking == pending_before

        with sqlite3.connect(app.state.db_path) as conn:
            booking_count_after = conn.execute("SELECT COUNT(*) FROM bookings").fetchone()[0]
        assert booking_count_after == booking_count_before

    def test_speak_endpoint_cleans_markdown_and_slot_ids(self, client: TestClient) -> None:
        sess_resp = client.post("/api/sessions")
        session_id = sess_resp.json()["session_id"]

        mock_cartesia = MagicMock()
        mock_cartesia.tts.generate.return_value = _mock_cartesia_generate_response(b"RIFFWAVEaudio")
        app.state.cartesia_client = mock_cartesia

        # Text containing markdown, interface instructions, and slot ID
        raw_text = "**Dr. Ada Lovelace** (Slot ID: 5). Please type /confirm to finalize."
        res = client.post(
            f"/api/sessions/{session_id}/speak",
            json={"text": raw_text},
        )
        assert res.status_code == 200

        # Verify Cartesia received cleaned speech without markdown, slot IDs, or UI commands
        call_kwargs = mock_cartesia.tts.generate.call_args[1]
        synthesized_text = call_kwargs["transcript"]
        assert "**" not in synthesized_text
        assert "Slot ID" not in synthesized_text
        assert "/confirm" not in synthesized_text
        assert "Dr. Ada Lovelace" in synthesized_text

    def test_speak_endpoint_measures_latency_header(self, client: TestClient) -> None:
        sess_resp = client.post("/api/sessions")
        session_id = sess_resp.json()["session_id"]

        mock_cartesia = MagicMock()
        mock_cartesia.tts.generate.return_value = _mock_cartesia_generate_response(b"RIFFWAVEaudio")
        app.state.cartesia_client = mock_cartesia

        res = client.post(
            f"/api/sessions/{session_id}/speak",
            json={"text": "Quick speech turn."},
        )
        assert res.status_code == 200
        assert "X-Latency-MS" in res.headers
        assert "Server-Timing" in res.headers
        assert "tts;dur=" in res.headers["Server-Timing"]

    def test_messages_endpoint_provides_spoken_reply_and_latency(self, client: TestClient) -> None:
        sess_resp = client.post("/api/sessions")
        session_id = sess_resp.json()["session_id"]

        res = client.post(
            f"/api/sessions/{session_id}/messages",
            json={"text": "Hello receptionist"},
        )
        assert res.status_code == 200
        data = res.json()
        assert "reply" in data
        assert "spoken_reply" in data
        assert data["latency_ms"] is not None
        assert "Server-Timing" in res.headers
        assert "llm;dur=" in res.headers["Server-Timing"]

    def test_confirm_endpoint_provides_spoken_message(self, client: TestClient) -> None:
        sess_resp = client.post("/api/sessions")
        session_id = sess_resp.json()["session_id"]
        session_state = sessions[session_id]

        # Stage a pending booking directly
        session_state.agent.pending_booking = db.Slot(
            id=1,
            doctor_id=1,
            start_utc="2026-11-20T17:00:00Z",
            end_utc="2026-11-20T17:30:00Z",
            doctor_name="Dr. Ada Lovelace",
            specialty="General Practice",
        )
        # Use Agent's proposed staging model
        from scheduler.agent import PendingBooking
        session_state.agent.pending_booking = PendingBooking(
            slot_id=1,
            doctor_name="Dr. Ada Lovelace",
            specialty="General Practice",
            start_utc="2026-11-20T17:00:00Z",
            end_utc="2026-11-20T17:30:00Z",
            local_time_str="Friday, Nov 20 at 9:00 AM PST",
            idempotency_key="test-key-123",
            patient_name="Alex Taylor",
        )

        res = client.post(f"/api/sessions/{session_id}/confirm")
        assert res.status_code == 200
        data = res.json()
        assert data["success"] is True
        assert data["spoken_message"] is not None
        assert "Dr. Ada Lovelace" in data["spoken_message"]
        assert "confirmed" in data["spoken_message"].lower()
        assert "booking number" in data["spoken_message"].lower()


# ---------------------------------------------------------------------------
# Unit Tests: Speech Formatting & SDK Fallbacks
# ---------------------------------------------------------------------------


class TestSpeechFormattingAndFallbacks:
    def test_format_spoken_text_strips_markdown_and_banners(self) -> None:
        raw = (
            "[Pending Confirmation] Your slot with **Dr. Ada Lovelace**\n"
            "### Appointment Options:\n"
            "- Monday, Nov 18 at 9:00 AM PST (Slot ID: 10)\n"
            "- Monday, Nov 18 at 2:00 PM PST (Slot #11)\n"
            "> Note: Please type /confirm to finalize or /cancel to discard.\n"
            "Click Confirm Appointment below. ✔ 📅"
        )
        spoken = format_spoken_text(raw)
        assert "[Pending Confirmation]" not in spoken
        assert "**" not in spoken
        assert "###" not in spoken
        assert "Slot ID" not in spoken
        assert "Slot #11" not in spoken
        assert "/confirm" not in spoken
        assert "Click Confirm" not in spoken
        assert "✔" not in spoken
        assert "📅" not in spoken
        assert "Dr. Ada Lovelace" in spoken
        assert "Monday, Nov 18 at 9:00 AM PST" in spoken

    def test_format_spoken_receipt_deterministic(self) -> None:
        receipt = format_spoken_receipt(
            doctor_name="Dr. Ada Lovelace",
            specialty="General Practice",
            local_time_str="Tuesday, November 17, 2026 at 09:00 AM PST",
            booking_id=42,
        )
        assert receipt == (
            "Your appointment with Dr. Ada Lovelace for General Practice on "
            "Tuesday, November 17, 2026 at 09:00 AM PST has been successfully confirmed. "
            "Your booking number is 42."
        )

    def test_cartesia_sdk_missing_raises_provider_error(self) -> None:
        with patch("scheduler.tts.Cartesia", None):
            with pytest.raises(TTSProviderError, match="Cartesia SDK is not installed"):
                synthesize_speech("Hello", api_key="dummy-key", client=None)

