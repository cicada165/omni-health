"""
scheduler/web.py — FastAPI web application for Appointment Scheduler scheduling agent (Milestone 3).

Endpoints
---------
POST /api/sessions
    Create an in-memory session with an isolated Agent instance and fixed
    fictional demo patient identity (Alex Taylor). Returns session_id.

POST /api/sessions/{id}/messages
    Send free-text user message. Calls Agent.turn(). Returns assistant reply
    plus structured pending appointment details (if any).

POST /api/sessions/{id}/confirm
    Explicit confirmation: calls Agent.confirm_booking() directly without LLM
    authorization. Returns committed booking status and ID.

POST /api/sessions/{id}/cancel
    Explicit cancellation: calls Agent.cancel_pending() directly.

GET /api/health
    Basic application health check.

GET /
    Serves the minimal vanilla HTML/JS browser interface.

Session & Concurrency Architecture
----------------------------------
- In-memory session registry mapping opaque session IDs to SessionState.
- Each session holds its own Agent instance and an asyncio.Lock.
- Concurrent requests for the same session are serialized by the per-session lock.
- Blocking Gemini and SQLite operations are offloaded via run_in_threadpool.
- Credentials and idempotency keys are kept server-side; never exposed to browsers.
- Local-only demo constraint: In-memory sessions reset when server restarts.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import websockets
from dotenv import load_dotenv
from fastapi import (
    FastAPI,
    File,
    HTTPException,
    Request,
    Response,
    UploadFile,
    WebSocket,
    WebSocketDisconnect,
    status,
)
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from scheduler import db
from scheduler.agent import Agent, DEFAULT_MODEL, DEFAULT_PATIENT_NAME
from scheduler.speech import (
    DEFAULT_TIMEOUT_SECONDS,
    EchoSuppressionGuard,
    SpeechCredentialsError,
    SpeechError,
    SpeechProviderError,
    SpeechTimeoutError,
    calculate_pcm_energy,
    is_supported_audio_mime,
    transcribe_audio,
)
from scheduler.tts import (
    ALLOWLISTED_VOICES,
    AUDITION_SAMPLES,
    DEFAULT_VOICE_ID,
    MAX_TEXT_LENGTH,
    TTSCredentialsError,
    TTSError,
    TTSProviderError,
    TTSTimeoutError,
    format_spoken_receipt,
    format_spoken_text,
    is_voice_allowlisted,
    synthesize_speech,
    validate_voice_id,
)

logger = logging.getLogger("appointment_scheduler.web")

# ---------------------------------------------------------------------------
# Paths, Limits and Environment
# ---------------------------------------------------------------------------

load_dotenv()

STATIC_DIR = Path(__file__).parent / "static"
INDEX_FILE = STATIC_DIR / "index.html"
MAX_AUDIO_SIZE_BYTES = int(os.getenv("MAX_AUDIO_SIZE_BYTES", str(10 * 1024 * 1024)))  # 10 MB limit


def get_default_db_path() -> Path:
    return Path(os.getenv("DB_PATH", "appointment_scheduler.db"))


# ---------------------------------------------------------------------------
# Session State Model
# ---------------------------------------------------------------------------


@dataclass
class InspectorEvent:
    id: str
    timestamp: str
    stage: str  # "stt" | "llm" | "tool" | "proposal" | "sqlite_commit" | "tts" | "cancel" | "voice_change" | "interrupt"
    title: str
    details: dict[str, Any]
    duration_ms: Optional[float] = None
    status: str = "ok"


class SessionState:
    """Holds a dedicated Agent, event history, active voice, and synchronization lock for a single web user."""

    def __init__(self, agent: Agent, active_voice_id: str = DEFAULT_VOICE_ID) -> None:
        self.agent = agent
        self.lock = asyncio.Lock()
        self.created_at = datetime.now(timezone.utc)
        self.active_voice_id = active_voice_id
        self.events: list[InspectorEvent] = []
        self.is_interrupted: bool = False
        self.echo_guard: EchoSuppressionGuard = EchoSuppressionGuard()

    def record_event(
        self,
        stage: str,
        title: str,
        details: dict[str, Any],
        duration_ms: Optional[float] = None,
        status: str = "ok",
    ) -> InspectorEvent:
        evt = InspectorEvent(
            id=f"evt-{len(self.events) + 1}-{uuid.uuid4().hex[:6]}",
            timestamp=datetime.now(timezone.utc).isoformat(),
            stage=stage,
            title=title,
            details=details,
            duration_ms=round(duration_ms, 2) if duration_ms is not None else None,
            status=status,
        )
        self.events.append(evt)
        return evt


def _is_local_request(request: Request) -> bool:
    """Return True if request originates from localhost or test client."""
    client_host = request.client.host if request.client else ""
    return client_host in ("127.0.0.1", "::1", "localhost", "testclient")


# In-memory session registry: session_id -> SessionState
# Limitation: In-memory registry does not persist across server restarts or multi-worker deployments.
sessions: dict[str, SessionState] = {}


# ---------------------------------------------------------------------------
# Pydantic Request & Response Schemas
# ---------------------------------------------------------------------------


class CreateSessionResponse(BaseModel):
    session_id: str
    patient_name: str
    created_at: str


class MessageRequest(BaseModel):
    text: str = Field(..., min_length=1, description="User input message text.")


class PendingAppointmentInfo(BaseModel):
    slot_id: int
    doctor_name: str
    specialty: str
    local_time_str: str
    patient_name: str
    status: str


class ConfirmedBookingInfo(BaseModel):
    booking_id: int
    doctor_name: str
    specialty: str
    local_time_str: str
    patient_name: str
    status: str = "confirmed"


class MessageResponse(BaseModel):
    reply: str
    spoken_reply: Optional[str] = None
    pending: Optional[PendingAppointmentInfo] = None
    confirmed: Optional[ConfirmedBookingInfo] = None
    latency_ms: Optional[float] = None


class ConfirmResponse(BaseModel):
    success: bool
    message: str
    spoken_message: Optional[str] = None
    booking_id: Optional[int] = None
    status: Optional[str] = None


class CancelResponse(BaseModel):
    was_pending: bool
    message: str


class InterruptResponse(BaseModel):
    session_id: str
    interrupted: bool = True
    message: str
    staged_cancelled: bool = False


class TranscribeResponse(BaseModel):
    transcript: str
    confidence: Optional[float] = None
    latency_ms: Optional[float] = None


class SpeakRequest(BaseModel):
    text: str = Field(
        ...,
        min_length=1,
        max_length=MAX_TEXT_LENGTH,
        description="Text to synthesize to speech audio.",
    )
    voice_id: Optional[str] = Field(
        default=None,
        description="Optional allowlisted Cartesia voice ID.",
    )


class VoiceAuditionRequest(BaseModel):
    voice_id: str = Field(..., description="Allowlisted Cartesia voice ID to audition.")
    sample_key: Optional[str] = Field(
        default="confirmation",
        description="Sample sentence identifier: 'greeting', 'confirmation', or 'receipt'.",
    )


class SetVoiceRequest(BaseModel):
    voice_id: str = Field(..., description="Allowlisted Cartesia voice ID to activate.")


class HealthResponse(BaseModel):
    status: str
    app: str


# ---------------------------------------------------------------------------
# FastAPI Application & Lifespan
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize and seed database on startup without clearing existing reservations
    db_path = getattr(app.state, "db_path", get_default_db_path())
    db.init_db(db_path)
    db.seed(db_path)
    yield


app = FastAPI(
    title="Appointment Scheduler Scheduling Agent",
    description="Fictional medical appointment voice/chat scheduling backend.",
    version="1.0.0",
    lifespan=lifespan,
)


def _get_session(session_id: str) -> SessionState:
    """Retrieve session from registry or raise 404."""
    session = sessions.get(session_id)
    if session is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session '{session_id}' not found.",
        )
    return session


# ---------------------------------------------------------------------------
# API Routes
# ---------------------------------------------------------------------------


@app.get("/api/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Basic health check endpoint."""
    return HealthResponse(status="ok", app="Appointment Scheduler")


@app.post(
    "/api/sessions",
    response_model=CreateSessionResponse,
    status_code=status.HTTP_201_CREATED,
)
async def create_session() -> CreateSessionResponse:
    """
    Create a new isolated session with its own Agent instance and
    fixed fictional demo patient identity.
    """
    session_id = uuid.uuid4().hex

    # Support testing overrides attached to app.state
    db_path = getattr(app.state, "db_path", get_default_db_path())
    gemini_client = getattr(app.state, "gemini_client", None)
    model = getattr(app.state, "model", None)

    session_events: list[InspectorEvent] = []

    def on_agent_event(
        stage: str,
        title: str,
        details: dict[str, Any],
        duration_ms: Optional[float] = None,
        status: str = "ok",
    ) -> None:
        evt = InspectorEvent(
            id=f"evt-{len(session_events) + 1}-{uuid.uuid4().hex[:6]}",
            timestamp=datetime.now(timezone.utc).isoformat(),
            stage=stage,
            title=title,
            details=details,
            duration_ms=round(duration_ms, 2) if duration_ms is not None else None,
            status=status,
        )
        session_events.append(evt)

    agent = Agent(
        db_path=db_path,
        model=model,
        client=gemini_client,
        patient_name=DEFAULT_PATIENT_NAME,
        event_callback=on_agent_event,
    )

    session = SessionState(agent=agent, active_voice_id=DEFAULT_VOICE_ID)
    session.events = session_events
    sessions[session_id] = session

    session.record_event(
        "session_init",
        "Session Initialized",
        {
            "session_id": session_id,
            "patient_name": agent.patient_name,
            "active_voice_id": DEFAULT_VOICE_ID,
        },
    )

    return CreateSessionResponse(
        session_id=session_id,
        patient_name=agent.patient_name,
        created_at=datetime.now(timezone.utc).isoformat(),
    )


@app.post("/api/sessions/{session_id}/messages", response_model=MessageResponse)
async def send_message(
    session_id: str,
    req: MessageRequest,
    response: Response,
) -> MessageResponse:
    """
    Accept user text, run agent turn with per-session synchronization lock,
    and return reply with structured pending appointment details if staged.
    """
    user_text = req.text.strip()
    if not user_text:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Message text cannot be empty.",
        )

    session = _get_session(session_id)

    t0 = time.perf_counter()
    # Per-session lock serializes concurrent calls to the same session's conversation history
    async with session.lock:
        session.record_event(
            "stt",
            "Speech / User Utterance Finalized",
            {"utterance": user_text},
        )
        reply = await run_in_threadpool(session.agent.turn, user_text)
        latency_ms = round((time.perf_counter() - t0) * 1000, 2)
        logger.info("Agent message turn completed in %.1f ms", latency_ms)

        spoken_reply = format_spoken_text(reply)

        pending_info: Optional[PendingAppointmentInfo] = None
        confirmed_info: Optional[ConfirmedBookingInfo] = None
        pb = session.agent.pending_booking
        if pb is not None:
            if pb.status == "awaiting_confirmation":
                pending_info = PendingAppointmentInfo(
                    slot_id=pb.slot_id,
                    doctor_name=pb.doctor_name,
                    specialty=pb.specialty,
                    local_time_str=pb.local_time_str,
                    patient_name=pb.patient_name,
                    status=pb.status,
                )
            elif pb.status == "confirmed" and pb.booking_id is not None:
                confirmed_info = ConfirmedBookingInfo(
                    booking_id=pb.booking_id,
                    doctor_name=pb.doctor_name,
                    specialty=pb.specialty,
                    local_time_str=pb.local_time_str,
                    patient_name=pb.patient_name,
                    status="confirmed",
                )
                spoken_reply = format_spoken_receipt(
                    doctor_name=pb.doctor_name,
                    specialty=pb.specialty,
                    local_time_str=pb.local_time_str,
                    booking_id=pb.booking_id,
                )

        session.record_event(
            "llm",
            "Gemini Turn Response",
            {
                "user_text": user_text,
                "reply_summary": reply[:120] + ("..." if len(reply) > 120 else ""),
                "staged_slot": pb.slot_id if pb else None,
                "staged_status": pb.status if pb else "none",
            },
            duration_ms=latency_ms,
        )

        response.headers["Server-Timing"] = f"llm;dur={latency_ms}"
        return MessageResponse(
            reply=reply,
            spoken_reply=spoken_reply,
            pending=pending_info,
            confirmed=confirmed_info,
            latency_ms=latency_ms,
        )


@app.post("/api/sessions/{session_id}/confirm", response_model=ConfirmResponse)
async def confirm_appointment(session_id: str) -> ConfirmResponse:
    """
    Explicit confirmation: commits the pending appointment directly in SQLite
    via Agent.confirm_booking(). Never asks Gemini for authorization.
    """
    session = _get_session(session_id)

    async with session.lock:
        success, message = await run_in_threadpool(session.agent.confirm_booking)

        booking_id = None
        curr_status = None
        spoken_message = None
        if session.agent.pending_booking is not None:
            curr_status = session.agent.pending_booking.status
            if success and curr_status == "confirmed":
                booking_id = session.agent.pending_booking.booking_id
                pb = session.agent.pending_booking
                spoken_message = format_spoken_receipt(
                    doctor_name=pb.doctor_name,
                    specialty=pb.specialty,
                    local_time_str=pb.local_time_str,
                    booking_id=booking_id or 0,
                )
            else:
                spoken_message = format_spoken_text(message)
        else:
            spoken_message = format_spoken_text(message)

        return ConfirmResponse(
            success=success,
            message=message,
            spoken_message=spoken_message,
            booking_id=booking_id,
            status=curr_status,
        )


@app.post("/api/sessions/{session_id}/cancel", response_model=CancelResponse)
async def cancel_appointment(session_id: str) -> CancelResponse:
    """
    Explicit cancellation: discards the pending appointment staging.
    Never alters database bookings.
    """
    session = _get_session(session_id)

    async with session.lock:
        was_pending, message = await run_in_threadpool(session.agent.cancel_pending)
        return CancelResponse(was_pending=was_pending, message=message)


@app.post("/api/sessions/{session_id}/interrupt", response_model=InterruptResponse)
async def interrupt_session(session_id: str) -> InterruptResponse:
    """
    Interrupt assistant speech/playback and invalidate in-flight actions.

    Guarantees & Invariants:
    - Resets playback state and marks session as interrupted.
    - Discards any unconfirmed in-memory proposal staged in agent state.
    - Safety invariant: an interrupted confirmation prompt NEVER commits to SQLite.
    - Records an 'interrupt' inspector event for full developer observability.
    """
    session = _get_session(session_id)
    async with session.lock:
        staged_cancelled = False
        pb = session.agent.pending_booking
        slot_id = pb.slot_id if pb else None

        if pb is not None and pb.status == "awaiting_confirmation":
            await run_in_threadpool(session.agent.cancel_pending)
            session.agent.awaiting_verbal_confirmation = False
            session.agent._last_staged_booking = None
            staged_cancelled = True
        elif session.agent.awaiting_verbal_confirmation:
            session.agent.awaiting_verbal_confirmation = False
            session.agent._last_staged_booking = None

        session.is_interrupted = True
        session.echo_guard.stop_playback()

        session.record_event(
            "interrupt",
            "User Interruption / Barge-in",
            {
                "session_id": session_id,
                "staged_cancelled": staged_cancelled,
                "slot_id": slot_id,
                "reason": "user_barge_in",
            },
        )
        logger.info(
            "Session %s interrupted (staged_cancelled=%s, slot_id=%s)",
            session_id,
            staged_cancelled,
            slot_id,
        )
        return InterruptResponse(
            session_id=session_id,
            interrupted=True,
            message="Playback interrupted and pending actions invalidated.",
            staged_cancelled=staged_cancelled,
        )


@app.post("/api/sessions/{session_id}/transcribe", response_model=TranscribeResponse)
async def transcribe_audio_endpoint(
    session_id: str,
    response: Response,
    file: UploadFile = File(...),
) -> TranscribeResponse:
    """
    Accept multipart audio upload from browser, validate format and size,
    and transcribe using Deepgram pre-recorded REST API.

    Guarantees & Invariants:
    - Audio recordings are non-persistent at the application level (never stored on disk or retained;
      underlying framework UploadFile may spool files >1MB to temporary buffers during parsing).
    - Never logs audio bytes, patient speech, or API credentials.
    - Does NOT call Gemini, propose bookings, confirm bookings, or alter conversation history.
    """
    # 1. Validate session existence
    session = _get_session(session_id)

    # 2. Validate MIME type
    content_type = file.content_type or ""
    if not is_supported_audio_mime(content_type):
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Unsupported audio format '{content_type}'. Supported formats: WebM, MP4, WAV, Ogg, MP3, AAC.",
        )

    # 3. Read bytes up to limit + 1 to detect oversized files safely
    audio_bytes = await file.read(MAX_AUDIO_SIZE_BYTES + 1)
    if len(audio_bytes) > MAX_AUDIO_SIZE_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"Audio file exceeds maximum allowed size of {MAX_AUDIO_SIZE_BYTES // (1024 * 1024)}MB.",
        )
    if len(audio_bytes) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded audio file is empty.",
        )

    # 4. Transcribe using Deepgram client (supports testing injection via app.state)
    deepgram_client = getattr(app.state, "deepgram_client", None)
    deepgram_key = getattr(app.state, "deepgram_api_key", None)

    t0 = time.perf_counter()
    try:
        res = await run_in_threadpool(
            transcribe_audio,
            audio_bytes=audio_bytes,
            content_type=content_type,
            api_key=deepgram_key,
            client=deepgram_client,
        )
    except SpeechCredentialsError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Speech recognition service is not configured on the server.",
        ) from exc
    except SpeechTimeoutError as exc:
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail="Speech recognition service timed out.",
        ) from exc
    except SpeechProviderError as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Speech recognition service error: {str(exc)}",
        ) from exc
    except SpeechError as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Speech recognition service error.",
        ) from exc

    latency_ms = round((time.perf_counter() - t0) * 1000, 2)
    logger.info("STT transcription completed in %.1f ms", latency_ms)
    response.headers["Server-Timing"] = f"stt;dur={latency_ms}"

    return TranscribeResponse(
        transcript=res.get("transcript", ""),
        confidence=res.get("confidence"),
        latency_ms=latency_ms,
    )


@app.websocket("/api/sessions/{session_id}/ws/transcribe")
async def websocket_transcribe(websocket: WebSocket, session_id: str):
    """
    Streaming speech-to-text proxy using Deepgram WebSocket (Milestone 6A).

    Guarantees & Invariants:
    - Proxied through FastAPI so credentials (DEEPGRAM_API_KEY) stay strictly server-side.
    - Validates session existence (rejects with close code 4404 if unknown).
    - Forwards binary audio chunks to Deepgram.
    - Forwards interim results (is_final=False) and finalized utterances (speech_final=True, UtteranceEnd).
    - Does NOT call Gemini, stage proposals, or write bookings.
    """
    # 1. Validate session existence
    if session_id not in sessions:
        await websocket.close(code=4404, reason="Session not found")
        return

    # 2. Validate Deepgram credentials
    api_key = getattr(app.state, "deepgram_api_key", None) or os.getenv("DEEPGRAM_API_KEY", "").strip()
    if not api_key:
        await websocket.close(code=4503, reason="DEEPGRAM_API_KEY not configured on server")
        return

    # Check for test injection of mock WebSocket handler
    mock_ws_handler = getattr(app.state, "mock_deepgram_ws", None)
    if mock_ws_handler is not None:
        await websocket.accept()
        await mock_ws_handler(websocket, session_id)
        return

    deepgram_ws_url = getattr(app.state, "deepgram_ws_url", None) or (
        "wss://api.deepgram.com/v1/listen?"
        "model=nova-3&smart_format=true&interim_results=true&"
        "endpointing=300&utterance_end_ms=1000&vad_events=true"
    )

    await websocket.accept()

    headers = {"Authorization": f"Token {api_key}"}
    try:
        async with websockets.connect(deepgram_ws_url, additional_headers=headers) as dg_ws:
            # Inform client that the streaming session is ready
            await websocket.send_json({"type": "ready"})

            async def forward_client_to_deepgram():
                try:
                    while True:
                        msg = await websocket.receive()
                        if "bytes" in msg and msg["bytes"]:
                            await dg_ws.send(msg["bytes"])
                        elif "text" in msg and msg["text"]:
                            try:
                                data = json.loads(msg["text"])
                                if data.get("type") == "close":
                                    await dg_ws.send(json.dumps({"type": "CloseStream"}))
                                    break
                            except Exception:
                                pass
                except WebSocketDisconnect:
                    pass
                except Exception as exc:
                    logger.debug("Client to Deepgram forwarding terminated: %s", exc)

            async def forward_deepgram_to_client():
                try:
                    async for dg_msg in dg_ws:
                        if isinstance(dg_msg, str):
                            try:
                                data = json.loads(dg_msg)
                            except Exception:
                                continue

                            msg_type = data.get("type")
                            if msg_type == "Results":
                                is_final = data.get("is_final", False)
                                speech_final = data.get("speech_final", False)
                                channel = data.get("channel", {})
                                alternatives = channel.get("alternatives", [{}])
                                transcript = alternatives[0].get("transcript", "") if alternatives else ""
                                confidence = alternatives[0].get("confidence") if alternatives else None
                                await websocket.send_json({
                                    "type": "Results",
                                    "transcript": transcript,
                                    "is_final": is_final,
                                    "speech_final": speech_final,
                                    "confidence": confidence,
                                })
                            elif msg_type == "SpeechStarted":
                                await websocket.send_json({"type": "SpeechStarted"})
                            elif msg_type == "UtteranceEnd":
                                await websocket.send_json({"type": "UtteranceEnd"})
                except WebSocketDisconnect:
                    pass
                except Exception as exc:
                    logger.debug("Deepgram to client forwarding terminated: %s", exc)

            task_client = asyncio.create_task(forward_client_to_deepgram())
            task_dg = asyncio.create_task(forward_deepgram_to_client())

            done, pending = await asyncio.wait(
                [task_client, task_dg],
                return_when=asyncio.FIRST_COMPLETED,
            )
            for t in pending:
                t.cancel()

    except Exception as exc:
        logger.error("Deepgram streaming WebSocket error: %s", exc)
        try:
            await websocket.send_json({"type": "error", "message": str(exc)})
        except Exception:
            pass
    finally:
        try:
            await websocket.close()
        except Exception:
            pass


@app.post("/api/sessions/{session_id}/speak")
async def speak_text_endpoint(
    session_id: str,
    req: SpeakRequest,
) -> Response:
    """
    Synthesize text into WAV audio using Cartesia TTS (Milestone 4B).

    Guarantees & Invariants:
    - Session existence validated (404 on missing session).
    - Pure presentation: does NOT invoke Gemini, alter history, or stage/confirm bookings.
    - Never logs patient text, raw audio, or API credentials.
    - Generated audio is never stored on disk.
    - Returns binary WAV stream with Content-Type: audio/wav, Cache-Control: no-store.
    """
    session = _get_session(session_id)

    cleaned_text = req.text.strip()
    if not cleaned_text:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Text cannot be empty.",
        )
    if len(cleaned_text) > MAX_TEXT_LENGTH:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Text exceeds maximum allowed length of {MAX_TEXT_LENGTH} characters.",
        )

    # Determine voice: optional request override or session default
    active_voice = req.voice_id or session.active_voice_id or DEFAULT_VOICE_ID
    if not is_voice_allowlisted(active_voice):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Voice ID '{active_voice}' is not allowlisted.",
        )
    session.active_voice_id = active_voice

    cartesia_client = getattr(app.state, "cartesia_client", None)
    cartesia_key = getattr(app.state, "cartesia_api_key", None)

    t0 = time.perf_counter()
    try:
        wav_bytes = await run_in_threadpool(
            synthesize_speech,
            text=cleaned_text,
            voice_id=active_voice,
            api_key=cartesia_key,
            client=cartesia_client,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc
    except TTSCredentialsError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Speech synthesis service is not configured on the server.",
        ) from exc
    except TTSTimeoutError as exc:
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail="Speech synthesis service timed out.",
        ) from exc
    except (TTSProviderError, TTSError) as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Speech synthesis service error: {str(exc)}",
        ) from exc

    latency_ms = round((time.perf_counter() - t0) * 1000, 2)
    logger.info("TTS speech synthesis completed in %.1f ms (voice=%s)", latency_ms, active_voice)

    session.is_interrupted = False
    session.echo_guard.start_playback()

    voice_info = ALLOWLISTED_VOICES.get(active_voice)
    voice_name = voice_info.name if voice_info else active_voice

    session.record_event(
        "tts",
        f"Cartesia Speech Synthesis ({voice_name})",
        {
            "voice_id": active_voice,
            "voice_name": voice_name,
            "text": cleaned_text[:120] + ("..." if len(cleaned_text) > 120 else ""),
            "audio_bytes": len(wav_bytes),
        },
        duration_ms=latency_ms,
    )

    return Response(
        content=wav_bytes,
        media_type="audio/wav",
        headers={
            "Cache-Control": "no-store",
            "Content-Type": "audio/wav",
            "X-Latency-MS": str(latency_ms),
            "Server-Timing": f"tts;dur={latency_ms}",
            "X-Voice-Id": active_voice,
        },
    )


@app.get("/api/voices")
async def list_voices() -> dict[str, Any]:
    """
    Return all allowlisted Cartesia voices and standardized appointment audition samples.
    Guarantees no API keys or credentials are exposed.
    """
    return {
        "default_voice_id": DEFAULT_VOICE_ID,
        "voices": [
            {
                "id": v.id,
                "name": v.name,
                "gender": v.gender,
                "accent": v.accent,
                "description": v.description,
                "sample_quote": v.sample_quote,
            }
            for v in ALLOWLISTED_VOICES.values()
        ],
        "samples": AUDITION_SAMPLES,
    }


@app.post("/api/voices/audition")
async def audition_voice(req: VoiceAuditionRequest) -> Response:
    """
    Synthesize identical appointment-related sample sentences for A/B/C comparison.
    Rejects any non-allowlisted voice ID with HTTP 422.
    """
    if not is_voice_allowlisted(req.voice_id):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Voice ID '{req.voice_id}' is not allowlisted.",
        )

    sample_text = AUDITION_SAMPLES.get(
        req.sample_key or "confirmation", AUDITION_SAMPLES["confirmation"]
    )

    cartesia_client = getattr(app.state, "cartesia_client", None)
    cartesia_key = getattr(app.state, "cartesia_api_key", None)

    t0 = time.perf_counter()
    try:
        audio_bytes = await run_in_threadpool(
            synthesize_speech,
            text=sample_text,
            voice_id=req.voice_id,
            api_key=cartesia_key,
            client=cartesia_client,
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Audition speech synthesis failed: {str(exc)}",
        )

    latency_ms = round((time.perf_counter() - t0) * 1000, 2)
    return Response(
        content=audio_bytes,
        media_type="audio/wav",
        headers={
            "Cache-Control": "no-store",
            "Content-Type": "audio/wav",
            "X-Audition-Voice": req.voice_id,
            "X-Latency-MS": str(latency_ms),
            "Server-Timing": f"tts;dur={latency_ms}",
        },
    )


@app.put("/api/sessions/{session_id}/voice")
async def set_session_voice(session_id: str, req: SetVoiceRequest) -> dict[str, Any]:
    """
    Switch active voice for a specific session.
    Keeps voice IDs strictly allowlisted.
    """
    session = _get_session(session_id)
    if not is_voice_allowlisted(req.voice_id):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Voice ID '{req.voice_id}' is not allowlisted.",
        )

    session.active_voice_id = req.voice_id
    voice = ALLOWLISTED_VOICES.get(req.voice_id)
    voice_name = voice.name if voice else req.voice_id

    session.record_event(
        "voice_change",
        f"Voice Switched to {voice_name}",
        {
            "voice_id": req.voice_id,
            "name": voice_name,
            "accent": voice.accent if voice else "Custom",
        },
    )

    return {
        "session_id": session_id,
        "active_voice_id": req.voice_id,
        "voice_name": voice_name,
    }


@app.get("/api/sessions/{session_id}/inspector")
async def get_session_inspector(session_id: str, request: Request) -> dict[str, Any]:
    """
    Developer Inspector debug endpoint.
    Restricted to local development (localhost, 127.0.0.1, ::1, testclient).

    Provides:
    - Real-time event timeline (STT -> Gemini -> tool calls -> SQLite -> TTS)
    - Distinguishes in-memory proposals from real database commits
    - Queries real SQLite row for confirmed bookings
    - Sanitizes API keys, idempotency keys, and raw audio
    """
    if not _is_local_request(request):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Developer inspector endpoints are restricted to local development.",
        )

    session = _get_session(session_id)
    pb = session.agent.pending_booking

    # Distinguish in-memory proposal from real database commit
    appointment_state = {
        "status": pb.status if pb else "idle",
        "is_staged_in_memory": pb is not None and pb.status == "awaiting_confirmation",
        "is_committed_to_sqlite": pb is not None and pb.status == "confirmed",
        "slot_id": pb.slot_id if pb else None,
        "doctor": pb.doctor_name if pb else None,
        "specialty": pb.specialty if pb else None,
        "time": pb.local_time_str if pb else None,
        "booking_id": pb.booking_id if pb else None,
    }

    # Query real SQLite database row if committed
    real_booking = None
    if pb and pb.booking_id:
        try:
            conn = db.get_connection(session.agent.db_path)
            row = conn.execute(
                "SELECT * FROM bookings WHERE id = ?", (pb.booking_id,)
            ).fetchone()
            conn.close()
            if row:
                real_booking = {
                    "id": row["id"],
                    "slot_id": row["slot_id"],
                    "doctor_name": pb.doctor_name,
                    "specialty": pb.specialty,
                    "patient_name": row["patient_name"],
                    "status": row["status"],
                    "created_at": row["created_utc"],
                }
        except Exception as exc:
            logger.warning("Failed to query SQLite booking row for inspector: %s", exc)

    active_voice = ALLOWLISTED_VOICES.get(session.active_voice_id)

    return {
        "session_id": session_id,
        "active_voice": {
            "id": active_voice.id if active_voice else session.active_voice_id,
            "name": active_voice.name if active_voice else "Custom",
            "accent": active_voice.accent if active_voice else "Standard",
            "gender": active_voice.gender if active_voice else "Unknown",
        },
        "appointment_state": appointment_state,
        "real_sqlite_booking": real_booking,
        "events": [
            {
                "id": e.id,
                "timestamp": e.timestamp,
                "stage": e.stage,
                "title": e.title,
                "details": e.details,
                "duration_ms": e.duration_ms,
                "status": e.status,
            }
            for e in session.events
        ],
    }


@app.get("/", response_class=FileResponse)
async def get_index() -> FileResponse:
    """Serve the minimal browser interface."""
    if not INDEX_FILE.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Static frontend index.html not found.",
        )
    return FileResponse(INDEX_FILE, media_type="text/html")
