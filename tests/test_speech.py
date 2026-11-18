"""
tests/test_speech.py — Unit and integration tests for speech-to-text (Milestone 4A).

Tests Deepgram REST client, supported audio MIME validation, error handling,
FastAPI /transcribe endpoint, and side-effect isolation guarantees.
"""

from __future__ import annotations

import io
import sqlite3
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import httpx
import pytest
from fastapi.testclient import TestClient
from google.genai import types

from scheduler import db
from scheduler.speech import (
    SpeechCredentialsError,
    SpeechError,
    SpeechProviderError,
    SpeechTimeoutError,
    get_deepgram_api_key,
    is_supported_audio_mime,
    transcribe_audio,
)
from scheduler.web import MAX_AUDIO_SIZE_BYTES, app, sessions


# ---------------------------------------------------------------------------
# Helpers & Mock Fixtures
# ---------------------------------------------------------------------------


def _mock_deepgram_json(transcript: str = "I would like to see Dr. Chen next Monday.", confidence: float = 0.985) -> dict[str, Any]:
    return {
        "results": {
            "channels": [
                {
                    "alternatives": [
                        {
                            "transcript": transcript,
                            "confidence": confidence,
                        }
                    ]
                }
            ]
        }
    }


@pytest.fixture()
def client(tmp_path: Path) -> TestClient:
    test_db = tmp_path / "speech_test.db"
    db.init_db(test_db)
    db.seed(test_db)

    app.state.db_path = test_db
    app.state.deepgram_api_key = "test-deepgram-api-key"
    sessions.clear()

    # Mock Gemini client so session creation does not fail
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
# Unit Tests: Deepgram Client (scheduler/speech.py)
# ---------------------------------------------------------------------------


class TestDeepgramClient:
    def test_supported_audio_mime_detection(self) -> None:
        assert is_supported_audio_mime("audio/webm") is True
        assert is_supported_audio_mime("audio/webm;codecs=opus") is True
        assert is_supported_audio_mime("AUDIO/WEBM") is True
        assert is_supported_audio_mime("audio/mp4") is True
        assert is_supported_audio_mime("audio/wav") is True
        assert is_supported_audio_mime("audio/ogg") is True
        assert is_supported_audio_mime("audio/mpeg") is True
        assert is_supported_audio_mime("audio/aac") is True
        assert is_supported_audio_mime("video/webm") is True

        assert is_supported_audio_mime("text/plain") is False
        assert is_supported_audio_mime("application/json") is False
        assert is_supported_audio_mime("image/png") is False
        assert is_supported_audio_mime("") is False

    def test_transcribe_audio_success(self) -> None:
        fake_response = MagicMock()
        fake_response.status_code = 200
        fake_response.json.return_value = _mock_deepgram_json(
            transcript="I need an appointment with Dr. Chen.",
            confidence=0.9876,
        )

        mock_http = MagicMock()
        mock_http.post.return_value = fake_response

        res = transcribe_audio(
            audio_bytes=b"dummy webm audio content",
            content_type="audio/webm",
            api_key="test-key",
            client=mock_http,
        )

        assert res["transcript"] == "I need an appointment with Dr. Chen."
        assert res["confidence"] == 0.9876

        # Verify auth and request payload
        mock_http.post.assert_called_once()
        call_kwargs = mock_http.post.call_args[1]
        assert call_kwargs["headers"]["Authorization"] == "Token test-key"
        assert call_kwargs["headers"]["Content-Type"] == "audio/webm"
        assert call_kwargs["content"] == b"dummy webm audio content"

    def test_transcribe_audio_empty_transcript(self) -> None:
        fake_response = MagicMock()
        fake_response.status_code = 200
        fake_response.json.return_value = _mock_deepgram_json(
            transcript="",
            confidence=0.0,
        )

        mock_http = MagicMock()
        mock_http.post.return_value = fake_response

        res = transcribe_audio(
            audio_bytes=b"silence",
            content_type="audio/webm",
            api_key="test-key",
            client=mock_http,
        )
        assert res["transcript"] == ""
        assert res["confidence"] == 0.0

    def test_transcribe_audio_empty_bytes_raises(self) -> None:
        with pytest.raises(SpeechProviderError, match="Audio buffer is empty"):
            transcribe_audio(b"", "audio/webm", api_key="test-key")

    def test_transcribe_audio_missing_api_key_raises(self) -> None:
        with patch.dict("os.environ", {}, clear=True):
            with pytest.raises(SpeechCredentialsError, match="DEEPGRAM_API_KEY is not configured"):
                transcribe_audio(b"some audio", "audio/webm", api_key=None)

    def test_transcribe_audio_timeout_raises(self) -> None:
        mock_http = MagicMock()
        mock_http.post.side_effect = httpx.TimeoutException("Read timed out")

        with pytest.raises(SpeechTimeoutError, match="timed out"):
            transcribe_audio(
                b"audio bytes",
                "audio/webm",
                api_key="test-key",
                client=mock_http,
            )

    def test_transcribe_audio_provider_http_error_raises(self) -> None:
        fake_response = MagicMock()
        fake_response.status_code = 401

        mock_http = MagicMock()
        mock_http.post.return_value = fake_response

        with pytest.raises(SpeechProviderError, match="failed with status code 401"):
            transcribe_audio(
                b"audio bytes",
                "audio/webm",
                api_key="bad-key",
                client=mock_http,
            )

    def test_transcribe_audio_invalid_json_raises(self) -> None:
        fake_response = MagicMock()
        fake_response.status_code = 200
        fake_response.json.side_effect = ValueError("Invalid JSON")

        mock_http = MagicMock()
        mock_http.post.return_value = fake_response

        with pytest.raises(SpeechProviderError, match="Invalid JSON"):
            transcribe_audio(
                b"audio bytes",
                "audio/webm",
                api_key="test-key",
                client=mock_http,
            )


# ---------------------------------------------------------------------------
# FastAPI /transcribe Endpoint Tests
# ---------------------------------------------------------------------------


class TestTranscribeEndpoint:
    def test_transcribe_valid_upload_success(self, client: TestClient) -> None:
        # Create a session
        sess_resp = client.post("/api/sessions")
        assert sess_resp.status_code == 201
        session_id = sess_resp.json()["session_id"]

        # Mock Deepgram client on app state
        mock_http = MagicMock()
        fake_response = MagicMock()
        fake_response.status_code = 200
        fake_response.json.return_value = _mock_deepgram_json(
            transcript="I want to book an appointment with Dr. Chen",
            confidence=0.99,
        )
        mock_http.post.return_value = fake_response
        app.state.deepgram_client = mock_http

        audio_payload = b"RIFF....WAVEfmt ...."
        files = {"file": ("audio.wav", io.BytesIO(audio_payload), "audio/wav")}

        res = client.post(f"/api/sessions/{session_id}/transcribe", files=files)
        assert res.status_code == 200
        data = res.json()
        assert data["transcript"] == "I want to book an appointment with Dr. Chen"
        assert data["confidence"] == 0.99

    def test_transcribe_unknown_session_returns_404(self, client: TestClient) -> None:
        files = {"file": ("recording.webm", io.BytesIO(b"audio data"), "audio/webm")}
        res = client.post("/api/sessions/nonexistent-session/transcribe", files=files)
        assert res.status_code == 404
        assert "not found" in res.json()["detail"].lower()

    def test_transcribe_unsupported_mime_returns_415(self, client: TestClient) -> None:
        sess_resp = client.post("/api/sessions")
        session_id = sess_resp.json()["session_id"]

        files = {"file": ("document.pdf", io.BytesIO(b"%PDF-1.4"), "application/pdf")}
        res = client.post(f"/api/sessions/{session_id}/transcribe", files=files)
        assert res.status_code == 415
        assert "unsupported audio format" in res.json()["detail"].lower()

    def test_transcribe_empty_file_returns_400(self, client: TestClient) -> None:
        sess_resp = client.post("/api/sessions")
        session_id = sess_resp.json()["session_id"]

        files = {"file": ("empty.webm", io.BytesIO(b""), "audio/webm")}
        res = client.post(f"/api/sessions/{session_id}/transcribe", files=files)
        assert res.status_code == 400
        assert "empty" in res.json()["detail"].lower()

    def test_transcribe_oversized_file_returns_413(self, client: TestClient) -> None:
        sess_resp = client.post("/api/sessions")
        session_id = sess_resp.json()["session_id"]

        oversized_bytes = b"0" * (MAX_AUDIO_SIZE_BYTES + 10)
        files = {"file": ("huge.webm", io.BytesIO(oversized_bytes), "audio/webm")}
        res = client.post(f"/api/sessions/{session_id}/transcribe", files=files)
        assert res.status_code == 413
        assert "exceeds maximum allowed size" in res.json()["detail"].lower()

    def test_transcribe_missing_credentials_returns_503(self, client: TestClient) -> None:
        sess_resp = client.post("/api/sessions")
        session_id = sess_resp.json()["session_id"]

        app.state.deepgram_api_key = ""
        with patch.dict("os.environ", {}, clear=True):
            files = {"file": ("test.webm", io.BytesIO(b"audio bytes"), "audio/webm")}
            res = client.post(f"/api/sessions/{session_id}/transcribe", files=files)
            assert res.status_code == 503
            assert "not configured" in res.json()["detail"].lower()

    def test_transcribe_timeout_returns_504(self, client: TestClient) -> None:
        sess_resp = client.post("/api/sessions")
        session_id = sess_resp.json()["session_id"]

        mock_http = MagicMock()
        mock_http.post.side_effect = httpx.TimeoutException("Network timeout")
        app.state.deepgram_client = mock_http

        files = {"file": ("test.webm", io.BytesIO(b"audio bytes"), "audio/webm")}
        res = client.post(f"/api/sessions/{session_id}/transcribe", files=files)
        assert res.status_code == 504
        assert "timed out" in res.json()["detail"].lower()

    def test_transcribe_provider_error_returns_502(self, client: TestClient) -> None:
        sess_resp = client.post("/api/sessions")
        session_id = sess_resp.json()["session_id"]

        mock_http = MagicMock()
        fake_response = MagicMock()
        fake_response.status_code = 502
        mock_http.post.return_value = fake_response
        app.state.deepgram_client = mock_http

        files = {"file": ("test.webm", io.BytesIO(b"audio bytes"), "audio/webm")}
        res = client.post(f"/api/sessions/{session_id}/transcribe", files=files)
        assert res.status_code == 502
        assert "error" in res.json()["detail"].lower()

    def test_transcribe_does_not_alter_history_or_propose_bookings(
        self, client: TestClient
    ) -> None:
        """
        Verify that /transcribe endpoint is strictly read-only for conversation
        history and database state.
        """
        sess_resp = client.post("/api/sessions")
        session_id = sess_resp.json()["session_id"]
        session_state = sessions[session_id]

        # Initial state checks
        history_before = list(session_state.agent.history)
        pending_before = session_state.agent.pending_booking

        # Count bookings in DB before
        with sqlite3.connect(app.state.db_path) as conn:
            count_before = conn.execute("SELECT COUNT(*) FROM bookings").fetchone()[0]

        # Mock Deepgram client
        mock_http = MagicMock()
        fake_response = MagicMock()
        fake_response.status_code = 200
        fake_response.json.return_value = _mock_deepgram_json("Please book Dr. Chen at 9am")
        mock_http.post.return_value = fake_response
        app.state.deepgram_client = mock_http

        files = {"file": ("audio.webm", io.BytesIO(b"spoken words"), "audio/webm")}
        res = client.post(f"/api/sessions/{session_id}/transcribe", files=files)
        assert res.status_code == 200

        # Verify agent history was NOT altered
        assert session_state.agent.history == history_before
        assert session_state.agent.pending_booking == pending_before

        # Verify DB bookings table was NOT altered
        with sqlite3.connect(app.state.db_path) as conn:
            count_after = conn.execute("SELECT COUNT(*) FROM bookings").fetchone()[0]
        assert count_after == count_before

    def test_confirm_endpoint_returns_booking_id_only_when_confirmed(
        self, client: TestClient
    ) -> None:
        """
        Verify that /confirm returns booking_id strictly when success is True
        and status is 'confirmed'; otherwise booking_id is None.
        """
        sess_resp = client.post("/api/sessions")
        session_id = sess_resp.json()["session_id"]

        # Confirm with nothing pending
        res = client.post(f"/api/sessions/{session_id}/confirm")
        assert res.status_code == 200
        data = res.json()
        assert data["success"] is False
        assert data["booking_id"] is None

