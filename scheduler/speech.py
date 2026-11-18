"""
scheduler/speech.py — Deepgram speech-to-text client for Appointment Scheduler agent (Milestone 4A).

Uses Deepgram's pre-recorded REST API:
POST https://api.deepgram.com/v1/listen?model=nova-3&smart_format=true

Security & Privacy
------------------
- Audio recordings are non-persistent at the application level (never stored on disk or retained).
- Note: Web frameworks (e.g. Starlette/FastAPI UploadFile) may spool payloads
  exceeding 1MB to temporary storage during HTTP body parsing, but no recordings
  are saved or retained after request completion.
- Never logs raw audio bytes, patient speech, or API keys.
"""

from __future__ import annotations

import logging
import math
import os
import struct
import time
from typing import Any, Optional

import httpx

logger = logging.getLogger("appointment_scheduler.speech")

DEEPGRAM_API_URL = "https://api.deepgram.com/v1/listen"
DEFAULT_MODEL = "nova-3"
DEFAULT_TIMEOUT_SECONDS = 15.0

# Supported browser audio formats for pre-recorded transcription
SUPPORTED_AUDIO_MIMES = frozenset({
    "audio/webm",
    "audio/mp4",
    "audio/wav",
    "audio/wave",
    "audio/x-wav",
    "audio/ogg",
    "audio/mpeg",
    "audio/mp3",
    "audio/aac",
    "audio/m4a",
    "audio/x-m4a",
    "video/webm",  # Chrome sometimes tags audio-only MediaRecorder blobs as video/webm
    "video/mp4",   # Safari sometimes tags audio-only MediaRecorder blobs as video/mp4
})


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class SpeechError(Exception):
    """Base exception for speech-to-text operations."""


class SpeechCredentialsError(SpeechError):
    """Raised when the DEEPGRAM_API_KEY is missing or invalid."""


class SpeechTimeoutError(SpeechError):
    """Raised when the Deepgram service times out."""


class SpeechProviderError(SpeechError):
    """Raised when the Deepgram service returns an error or rejects audio."""


# ---------------------------------------------------------------------------
# Core Transcription Client
# ---------------------------------------------------------------------------


def get_deepgram_api_key() -> str:
    """Retrieve DEEPGRAM_API_KEY from environment or raise SpeechCredentialsError."""
    api_key = os.getenv("DEEPGRAM_API_KEY", "").strip()
    if not api_key:
        raise SpeechCredentialsError("DEEPGRAM_API_KEY is not configured on the server.")
    return api_key


def is_supported_audio_mime(content_type: str) -> bool:
    """Check if the MIME type is in the allowlist of supported audio formats."""
    base_mime = content_type.split(";")[0].strip().lower()
    return base_mime in SUPPORTED_AUDIO_MIMES


def transcribe_audio(
    audio_bytes: bytes,
    content_type: str,
    api_key: Optional[str] = None,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
    client: Optional[httpx.Client] = None,
) -> dict[str, Any]:
    """
    Transcribe audio bytes using Deepgram's pre-recorded REST API.

    Parameters
    ----------
    audio_bytes : bytes
        Raw audio data recorded by browser or client.
    content_type : str
        MIME type of the audio (e.g. 'audio/webm', 'audio/mp4', 'audio/wav').
    api_key : str, optional
        Deepgram API key. If omitted, reads from DEEPGRAM_API_KEY environment variable.
    timeout : float
        HTTP request timeout in seconds.
    client : httpx.Client, optional
        Injectable HTTP client for testing.

    Returns
    -------
    dict with keys:
        - "transcript": str (the transcribed text)
        - "confidence": Optional[float] (confidence score, if provided by model)

    Raises
    ------
    SpeechCredentialsError
        If no API key is configured.
    SpeechTimeoutError
        If the transcription request times out.
    SpeechProviderError
        If Deepgram rejects the audio or returns a non-200 status code.
    """
    if not audio_bytes:
        raise SpeechProviderError("Audio buffer is empty.")

    key = api_key or get_deepgram_api_key()

    headers = {
        "Authorization": f"Token {key}",
        "Content-Type": content_type,
    }
    params = {
        "model": DEFAULT_MODEL,
        "smart_format": "true",
    }

    logger.debug(
        "Sending audio to Deepgram (size=%d bytes, content_type=%s, model=%s)",
        len(audio_bytes),
        content_type,
        DEFAULT_MODEL,
    )

    try:
        if client is not None:
            response = client.post(
                DEEPGRAM_API_URL,
                headers=headers,
                params=params,
                content=audio_bytes,
                timeout=timeout,
            )
        else:
            with httpx.Client() as default_client:
                response = default_client.post(
                    DEEPGRAM_API_URL,
                    headers=headers,
                    params=params,
                    content=audio_bytes,
                    timeout=timeout,
                )
    except httpx.TimeoutException as exc:
        logger.error("Deepgram transcription request timed out after %.1f seconds", timeout)
        raise SpeechTimeoutError("Deepgram transcription service timed out.") from exc
    except httpx.RequestError as exc:
        logger.error("Network error connecting to Deepgram: %s", exc.__class__.__name__)
        raise SpeechProviderError(f"Failed to connect to speech transcription service: {exc.__class__.__name__}") from exc

    if response.status_code != 200:
        logger.error("Deepgram returned HTTP status %d", response.status_code)
        raise SpeechProviderError(
            f"Deepgram transcription failed with status code {response.status_code}."
        )

    try:
        data = response.json()
    except Exception as exc:
        logger.error("Failed to parse JSON response from Deepgram")
        raise SpeechProviderError("Invalid JSON returned by transcription service.") from exc

    # Parse Deepgram response structure: results.channels[0].alternatives[0]
    channels = data.get("results", {}).get("channels", [])
    if not channels:
        return {"transcript": "", "confidence": None}

    alternatives = channels[0].get("alternatives", [])
    if not alternatives:
        return {"transcript": "", "confidence": None}

    alt = alternatives[0]
    transcript = str(alt.get("transcript", "")).strip()
    confidence = alt.get("confidence")

    if confidence is not None:
        try:
            confidence = round(float(confidence), 4)
        except (ValueError, TypeError):
            confidence = None

    logger.debug("Successfully transcribed audio (length=%d characters)", len(transcript))
    return {
        "transcript": transcript,
        "confidence": confidence,
    }


# ---------------------------------------------------------------------------
# Echo Suppression & Barge-In Guard (Milestone 7)
# ---------------------------------------------------------------------------


def calculate_pcm_energy(pcm_bytes: bytes) -> float:
    """
    Calculate the normalized RMS (root mean square) energy of 16-bit PCM audio samples.
    Returns a float in [0.0, 1.0].
    """
    if not pcm_bytes:
        return 0.0
    num_samples = len(pcm_bytes) // 2
    if num_samples == 0:
        return 0.0
    fmt = f"<{num_samples}h"
    try:
        samples = struct.unpack(fmt, pcm_bytes[: num_samples * 2])
    except Exception:
        return 0.0
    sum_squares = sum(s * s for s in samples)
    mean_square = sum_squares / num_samples
    rms = math.sqrt(mean_square)
    return min(1.0, rms / 32768.0)


class EchoSuppressionGuard:
    """
    Echo suppression and barge-in guard for real-time voice experience.

    Protects against acoustic feedback and speaker echo false barge-ins:
    1. Tracks speaker playback state (is_speaking, playback_start_time).
    2. Enforces speaker-onset debounce (prevents self-interruption from speaker onset transients).
    3. Analyzes audio frame energy / RMS against an adaptive speech threshold.
    4. Distinguishes genuine user barge-in utterances from residual speaker echo.
    """

    def __init__(
        self,
        energy_threshold: float = 0.02,
        onset_debounce_ms: float = 150.0,
        min_speech_duration_ms: float = 100.0,
    ) -> None:
        self.energy_threshold = energy_threshold
        self.onset_debounce_ms = onset_debounce_ms
        self.min_speech_duration_ms = min_speech_duration_ms
        self.is_speaking: bool = False
        self.playback_start_time: Optional[float] = None
        self.last_interrupt_time: Optional[float] = None

    def start_playback(self, timestamp: Optional[float] = None) -> None:
        """Mark that the assistant has started audio playback through device speakers."""
        self.is_speaking = True
        self.playback_start_time = timestamp if timestamp is not None else time.perf_counter()

    def stop_playback(self) -> None:
        """Mark that assistant audio playback has completed or been halted."""
        self.is_speaking = False
        self.playback_start_time = None

    def should_allow_barge_in(
        self,
        energy: float,
        timestamp: Optional[float] = None,
        has_speech_started: bool = True,
    ) -> bool:
        """
        Evaluate whether an incoming speech event should trigger barge-in interruption.
        Returns False if the event is determined to be speaker echo or onset transient.
        """
        if not self.is_speaking:
            # If assistant is not speaking, user speech is always allowed
            return True

        now = timestamp if timestamp is not None else time.perf_counter()

        # 1. Onset debounce: ignore acoustic echo during the initial speaker onset
        if self.playback_start_time is not None:
            elapsed_ms = (now - self.playback_start_time) * 1000.0
            if elapsed_ms < self.onset_debounce_ms:
                logger.debug(
                    "Barge-in suppressed: within onset debounce (%.1f ms < %.1f ms)",
                    elapsed_ms,
                    self.onset_debounce_ms,
                )
                return False

        # 2. Energy check: microphone signal must exceed speaker echo threshold
        if energy < self.energy_threshold:
            logger.debug(
                "Barge-in suppressed: energy %.4f below threshold %.4f",
                energy,
                self.energy_threshold,
            )
            return False

        self.last_interrupt_time = now
        return True

