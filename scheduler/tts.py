"""
scheduler/tts.py — Cartesia Text-to-Speech client for Appointment Scheduler agent (Milestone 4B).

Synthesizes speech audio using Cartesia's official Python SDK:
- Model: sonic-3.6 (configurable via CARTESIA_MODEL)
- Audio format: WAV, PCM 16-bit little-endian (pcm_s16le), 44.1 kHz
- Stock Voice ID: 62ae83ad-4f6a-430b-af41-a9bede9286ca (Gemma - conversational receptionist)
  (configurable via CARTESIA_VOICE_ID)

Security & Privacy Guarantees
-----------------------------
- Audio is synthesized strictly in memory and returned as binary WAV bytes.
- Never logs API credentials, patient text, or raw audio bytes.
- Generated speech is never stored on disk.
- Credentials remain on the server and are never exposed to browser clients.
- Presentation-only: never calls Gemini, database queries, or booking actions.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import logging
import os
import re
from typing import Any, Optional

try:
    from cartesia import (
        APIConnectionError,
        APITimeoutError,
        AuthenticationError,
        Cartesia,
        CartesiaError,
    )
except ImportError:  # pragma: no cover
    Cartesia = None  # type: ignore[assignment,misc]

    class CartesiaError(Exception):  # type: ignore[no-redef]
        """Fallback placeholder when cartesia package is not installed."""

    class AuthenticationError(CartesiaError):  # type: ignore[no-redef]
        pass

    class APITimeoutError(CartesiaError):  # type: ignore[no-redef]
        pass

    class APIConnectionError(CartesiaError):  # type: ignore[no-redef]
        pass

logger = logging.getLogger("appointment_scheduler.tts")

# Official Cartesia Sonic configuration defaults
DEFAULT_MODEL = "sonic-3.6"
DEFAULT_VOICE_ID = "62ae83ad-4f6a-430b-af41-a9bede9286ca"  # Gemma (friendly conversational assistant)
DEFAULT_SAMPLE_RATE = 44100
DEFAULT_TIMEOUT_SECONDS = 15.0
MAX_TEXT_LENGTH = 2000


# ---------------------------------------------------------------------------
# Allowlisted Receptionist Voices & A/B/C Audition Samples (Milestone 6B)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class VoiceOption:
    id: str
    name: str
    gender: str
    accent: str
    description: str
    sample_quote: str


ALLOWLISTED_VOICES: dict[str, VoiceOption] = {
    "62ae83ad-4f6a-430b-af41-a9bede9286ca": VoiceOption(
        id="62ae83ad-4f6a-430b-af41-a9bede9286ca",
        name="Gemma",
        gender="Feminine",
        accent="British",
        description="Confident, polished, and warm British delivery ideal for professional healthcare assistance.",
        sample_quote="Hello, thank you for calling Appointment Scheduler. How may I assist with your appointment today?",
    ),
    "c46cf1f6-49a1-4d67-9a57-ff859a4046d3": VoiceOption(
        id="c46cf1f6-49a1-4d67-9a57-ff859a4046d3",
        name="Cora",
        gender="Feminine",
        accent="American",
        description="Helpful, articulate, and patient American tone well-suited for service-oriented patient navigation.",
        sample_quote="I have an opening with Dr. Charles Babbage for Tuesday at 9:00 AM PST. Would you like me to book it?",
    ),
    "01eaafa9-308a-4276-a017-6ab0cf061b1f": VoiceOption(
        id="01eaafa9-308a-4276-a017-6ab0cf061b1f",
        name="Clara",
        gender="Feminine",
        accent="American",
        description="Clear enunciation and reassuring, measured pacing providing reassuring clinical authority.",
        sample_quote="Your appointment has been successfully confirmed. We look forward to seeing you at the clinic.",
    ),
}

AUDITION_SAMPLES: dict[str, str] = {
    "greeting": (
        "Hello, thank you for calling Appointment Scheduler. "
        "I have two openings with Dr. Charles Babbage in Cardiology: "
        "Tuesday, November 17 at 9:00 AM, or Wednesday, November 18 at 2:00 PM PST. "
        "Which of those times works best for you?"
    ),
    "confirmation": (
        "I have prepared an appointment for you with Dr. Florence Nightingale in Pediatrics "
        "for Tuesday, November 17, 2026 at 9:00 AM PST. "
        "Would you like me to book this appointment for you?"
    ),
    "receipt": (
        "Your appointment with Dr. Ada Lovelace for General Practice on "
        "Wednesday, November 18, 2026 at 9:00 AM PST has been successfully confirmed. "
        "Your booking number is 42. We look forward to seeing you!"
    ),
}


def is_voice_allowlisted(voice_id: str) -> bool:
    """Return True if voice_id is in the server allowlist or matches server configured env."""
    if not voice_id:
        return False
    v = str(voice_id).strip()
    if v in ALLOWLISTED_VOICES or v == "stock-voice-id":
        return True
    env_voice = os.getenv("CARTESIA_VOICE_ID", "").strip()
    if env_voice and v == env_voice:
        return True
    return False


def validate_voice_id(voice_id: Optional[str]) -> str:
    """
    Validate that voice_id is allowlisted.
    Returns default voice ID if voice_id is None/empty.
    Raises ValueError if voice_id is not allowlisted.
    """
    if not voice_id:
        return get_cartesia_voice_id()
    v = str(voice_id).strip()
    if not is_voice_allowlisted(v):
        raise ValueError(
            f"Voice ID '{v}' is not allowlisted. "
            f"Choose from: {list(ALLOWLISTED_VOICES.keys())}"
        )
    return v


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class TTSError(Exception):
    """Base exception for text-to-speech operations."""


class TTSCredentialsError(TTSError):
    """Raised when CARTESIA_API_KEY is missing or invalid."""


class TTSTimeoutError(TTSError):
    """Raised when the Cartesia service times out."""


class TTSProviderError(TTSError):
    """Raised when Cartesia rejects the request or returns an error."""


# ---------------------------------------------------------------------------
# Configuration Helpers
# ---------------------------------------------------------------------------


def get_cartesia_api_key() -> str:
    """Retrieve CARTESIA_API_KEY from environment or raise TTSCredentialsError."""
    api_key = os.getenv("CARTESIA_API_KEY", "").strip()
    if not api_key:
        raise TTSCredentialsError("CARTESIA_API_KEY is not configured on the server.")
    return api_key


def get_cartesia_model() -> str:
    """Retrieve model ID from environment or default to sonic-3.6."""
    return os.getenv("CARTESIA_MODEL", DEFAULT_MODEL).strip() or DEFAULT_MODEL


def get_cartesia_voice_id() -> str:
    """Retrieve voice ID from environment or default to stock Gemma voice."""
    return os.getenv("CARTESIA_VOICE_ID", DEFAULT_VOICE_ID).strip() or DEFAULT_VOICE_ID


# ---------------------------------------------------------------------------
# Spoken Text Formatting
# ---------------------------------------------------------------------------


def format_spoken_text(text: str) -> str:
    """
    Format text for natural, conversational speech synthesis.

    - Removes technical UI banners (e.g. [Pending Confirmation] blocks).
    - Strips UI instructions (e.g. 'type /confirm to finalize or /cancel to discard').
    - Removes slot IDs (e.g. 'Slot ID: 12' or 'Slot #12').
    - Strips markdown formatting (headers, bold, italics, bullets, backticks).
    - Removes checkmarks, arrows, symbols, emojis, and normalizes whitespace.
    """
    if not text:
        return ""
    s = str(text)

    # 1. Strip structured confirmation block: [Pending Confirmation]\n• Doctor ...
    s = re.sub(
        r"\[(?:Pending Confirmation|Awaiting Confirmation)\]\s*(?:\n\s*•\s*[\w\s]+:[^\n]*)+\s*(?:➡[^\n]*)?",
        "",
        s,
        flags=re.IGNORECASE,
    )

    # 2. Remove standalone UI banners
    s = re.sub(
        r"\[(Pending Confirmation|Awaiting Confirmation|Confirmed|Cancelled)\]",
        "",
        s,
        flags=re.IGNORECASE,
    )

    # 3. Remove interface instructions and click/type commands referencing /confirm or /cancel
    s = re.sub(
        r"(?i)(?:^|(?<=[.!?]))[^.!?\n]*?(?:/confirm|/cancel|confirm\s+button|cancel\s+button)[^.!?\n]*[.!?]?",
        "",
        s,
    )
    s = re.sub(
        r"(?i)(?:you can\s+)?(?:confirm|cancel)\s+(?:it\s+)?(?:by\s+)?(?:typing|clicking|entering|saying)\s+[^.\n]*[.!?]?",
        "",
        s,
    )
    s = re.sub(
        r"(?i)(?:please\s+)?(?:type|enter|say|send)\s+(?:/confirm|/cancel)[^.\n]*\.?",
        "",
        s,
    )
    s = re.sub(
        r"(?i)(?:please\s+)?click\s+(?:the\s+)?(?:confirm|cancel)[^.\n]*\.?",
        "",
        s,
    )
    s = re.sub(r"(?i)\s*/(?:confirm|cancel)\b", "", s)

    # 4. Remove slot IDs like (Slot ID: 12) or Slot #12
    s = re.sub(r"(?i)\(?(?:slot\s*id|slot\s*#|slot)\s*[:#]?\s*\d+\)?", "", s)

    # 5. Remove markdown headers and blockquotes
    s = re.sub(r"^\s*#{1,6}\s*", "", s, flags=re.MULTILINE)
    s = re.sub(r"^\s*>\s*", "", s, flags=re.MULTILINE)

    # 6. Remove markdown bold/italic/code
    s = re.sub(r"\*\*([^*]+)\*\*", r"\1", s)
    s = re.sub(r"\*([^*]+)\*", r"\1", s)
    s = re.sub(r"__([^_]+)__", r"\1", s)
    s = re.sub(r"`([^`]+)`", r"\1", s)

    # 7. Remove bullet markers
    s = re.sub(r"^\s*[-*•]\s+", "", s, flags=re.MULTILINE)

    # 8. Remove checkmarks, symbols, arrows, and emojis
    s = re.sub(r"[✔📅🎙️⚡❌•➡→➤]", "", s)

    # 9. Clean up whitespace
    s = re.sub(r"\s+", " ", s).strip()
    return s


def format_spoken_receipt(
    doctor_name: str,
    specialty: str,
    local_time_str: str,
    booking_id: int,
) -> str:
    """
    Format a concise, natural spoken confirmation receipt.
    Never speaks raw markdown, bullet points, or checkmark symbols.
    """
    return (
        f"Your appointment with {doctor_name} for {specialty} on "
        f"{local_time_str} has been successfully confirmed. "
        f"Your booking number is {booking_id}."
    )


# ---------------------------------------------------------------------------
# Core Speech Synthesis
# ---------------------------------------------------------------------------


def synthesize_speech(
    text: str,
    api_key: Optional[str] = None,
    model_id: Optional[str] = None,
    voice_id: Optional[str] = None,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
    client: Optional[Any] = None,
) -> bytes:
    """
    Synthesize text into WAV audio bytes using Cartesia's Python SDK.

    Parameters
    ----------
    text : str
        Text to synthesize (non-empty, up to MAX_TEXT_LENGTH chars).
    api_key : str, optional
        Cartesia API key. Defaults to CARTESIA_API_KEY environment variable.
    model_id : str, optional
        Cartesia model identifier. Defaults to CARTESIA_MODEL or 'sonic-3.6'.
    voice_id : str, optional
        Cartesia voice identifier. Defaults to CARTESIA_VOICE_ID or stock voice.
    timeout : float
        HTTP request timeout in seconds.
    client : Any, optional
        Injectable Cartesia client instance (useful for unit tests).

    Returns
    -------
    bytes
        Binary WAV audio bytes (PCM 16-bit little-endian, 44.1 kHz).

    Raises
    ------
    ValueError
        If text is empty or exceeds MAX_TEXT_LENGTH.
    TTSCredentialsError
        If no API key is configured or authentication fails.
    TTSTimeoutError
        If the Cartesia API request times out.
    TTSProviderError
        If the Cartesia API returns an error or fails.
    """
    if text is None or not str(text).strip():
        raise ValueError("Text cannot be empty.")
    cleaned_text = format_spoken_text(str(text))
    if not cleaned_text:
        raise ValueError("Text cannot be empty.")
    if len(cleaned_text) > MAX_TEXT_LENGTH:
        raise ValueError(
            f"Text length ({len(cleaned_text)}) exceeds maximum allowed ({MAX_TEXT_LENGTH} characters)."
        )

    key = api_key or get_cartesia_api_key()
    model = model_id or get_cartesia_model()
    voice = validate_voice_id(voice_id or get_cartesia_voice_id())

    output_format = {
        "container": "wav",
        "encoding": "pcm_s16le",
        "sample_rate": DEFAULT_SAMPLE_RATE,
    }

    logger.debug(
        "Synthesizing speech via Cartesia (text_length=%d, model=%s, voice=%s)",
        len(cleaned_text),
        model,
        voice,
    )

    try:
        cartesia_client = client
        if cartesia_client is None:
            if Cartesia is None:  # pragma: no cover
                raise TTSProviderError("Cartesia SDK is not installed.")
            cartesia_client = Cartesia(api_key=key, timeout=timeout)

        # Generate audio using official SDK method
        response = cartesia_client.tts.generate(
            model_id=model,
            transcript=cleaned_text,
            voice=voice,
            output_format=output_format,
        )

        if hasattr(response, "read"):
            audio_bytes = response.read()
        elif isinstance(response, bytes):
            audio_bytes = response
        elif hasattr(response, "content"):
            audio_bytes = response.content
        else:
            audio_bytes = bytes(response)

        if not audio_bytes:
            raise TTSProviderError("Cartesia returned an empty audio stream.")

        logger.debug("Successfully synthesized speech (size=%d bytes)", len(audio_bytes))
        return audio_bytes

    except APITimeoutError as exc:
        logger.error("Cartesia TTS request timed out after %.1f seconds", timeout)
        raise TTSTimeoutError("Cartesia text-to-speech service timed out.") from exc
    except AuthenticationError as exc:
        logger.error("Cartesia authentication failed")
        raise TTSCredentialsError("Cartesia authentication failed: invalid API key.") from exc
    except (APIConnectionError, CartesiaError) as exc:
        logger.error("Cartesia API error: %s", exc.__class__.__name__)
        raise TTSProviderError(f"Cartesia synthesis error: {exc.__class__.__name__}") from exc
    except (ValueError, TTSError):
        raise
    except Exception as exc:
        logger.error("Unexpected error during Cartesia synthesis: %s", exc.__class__.__name__)
        raise TTSProviderError(f"Failed to synthesize speech: {exc.__class__.__name__}") from exc
