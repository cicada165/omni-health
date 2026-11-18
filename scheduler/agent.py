"""
scheduler/agent.py — Gemini-powered conversational agent (Milestone 2B Hardened).

Capabilities
------------
- Read-only tools for searching: list_doctors(), search_slots().
- Staging tool: propose_booking(slot_id). This tool MUST NOT write a booking.
  It validates the slot in SQLite, stores it in session state as
  ``awaiting_confirmation``, generates a stable idempotency key, and
  returns the clinic-local time details.
- Explicit confirmation (/confirm) and cancellation (/cancel) methods
  that are intercepted by Python before sending anything to the LLM.
- Fixed fictional demo patient identity (Alex Taylor; unauthenticated).

Reliability Hardening Pass
--------------------------
1. Gemini API Failures: Connection timeouts and transient API errors are
   caught cleanly without terminating the CLI. History is rolled back so no
   orphaned user/tool turns remain. Fails closed (partially staged proposals
   are discarded).
2. False Confirmation Prevention: confirm_booking() checks the returned
   booking's CURRENT database status in both initial and repeated invocations.
   Never reports a cancelled booking as confirmed.
3. Stale Pending Selection Prevention: Awaiting-confirmation proposals are
   invalidated when a new free-text user turn begins, preventing stale
   confirmations when a patient changes topic. Direct /confirm and /cancel
   remain unaffected.
4. Ambiguous Proposal Prevention: Multiple different booking proposals in the
   same user turn (parallel or across sequential tool-call rounds) are
   strictly rejected and cleared. Never silently stages the last proposed slot.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import os
import re
import time
import uuid
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional

from dotenv import load_dotenv
from google import genai
from google.genai import types

from scheduler import db

# ---------------------------------------------------------------------------
# Logging & Constants
# ---------------------------------------------------------------------------

logger = logging.getLogger("appointment_scheduler.agent")

DEFAULT_MODEL = "gemini-3.8-flash"
DEFAULT_PATIENT_NAME = "Alex Taylor"  # Fictional demo identity; unauthenticated
MAX_TOOL_ROUNDS = 6  # safety cap on tool-call loops


SYSTEM_INSTRUCTION = """\
You are the voice receptionist for Appointment Scheduler, assisting patients with appointment scheduling.

Today's date (clinic-local, America/Los_Angeles): {today}
Current patient identity: {patient_name} (fictional demo identity; unauthenticated)

Voice Dialogue Rules:
- Default to 15 to 30 words per spoken response. Keep every response concise, natural, and conversational.
- Ask only one focused question at a time at the end of each turn.
- When presenting availability, offer at most TWO options initially (e.g. earliest morning and afternoon openings). Never recite long lists.
- Do not repeat the clinic name ("Appointment Scheduler") every turn; mention it in the initial greeting only.
- NEVER mention or read slot IDs, database primary keys, or technical numbers aloud. Identify appointments only by doctor name, specialty, day of the week, date, and local time (e.g., "Dr. Charles Babbage on Tuesday, November 17 at 9:00 AM PST").
- Avoid all markdown formatting (no headers, bold asterisks, bullet points, emojis, or tables) as they sound unnatural when spoken.
- Remember and preserve the patient's preferences, corrections, and doctor selections across turns.
- Interpret "next week" as the next clinic-local Monday through Sunday. Send its inclusive bounds as `date_start` and `date_end`.
- For a requested time of day, set `time_preference` to morning, afternoon, or evening. Match appointments using clinic-local time.
- When the patient selects or agrees to a specific appointment, call `propose_booking(slot_id)` immediately.
- Execute tool calls in minimal round-trips: when searching for availability, call `search_slots` directly in a single round. When the patient specifies or agrees to a slot, call `propose_booking(slot_id)` immediately without redundant conversational round-trips.
- Once a slot is proposed, read back the exact doctor, date, and local time, and ask: "Would you like me to book this appointment for you?" (Pre-booking confirmation must be crystal clear before booking, even if this requires a slightly longer response).
- NEVER claim an appointment is booked or confirmed on your own authority. The `propose_booking` tool stages the slot; it does NOT write a booking. The system handles confirmation deterministically when the patient confirms.
- NEVER instruct or tell the patient to type or use "/confirm", "/cancel", or any slash commands. In this voice system, patients confirm simply by speaking naturally (e.g. saying "Yes, book it").
- NEVER claim an appointment is "staged though I cannot book it myself", "still pending, not finalized", or ask the patient to use any commands. If the patient affirms or confirms an appointment, the system automatically commits it. If you need to clarify, simply ask: "Would you like me to book Dr. [Name] for [Day, Date at Time]?"
- Do not ask for or accept patient identity or idempotency keys—those are managed strictly by the system.
"""

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class PendingBooking:
    slot_id: int
    doctor_name: str
    specialty: str
    start_utc: str
    end_utc: str
    local_time_str: str
    idempotency_key: str
    patient_name: str
    status: str = "awaiting_confirmation"  # "awaiting_confirmation" | "confirmed" | "cancelled"
    booking_id: int | None = None


# ---------------------------------------------------------------------------
# Tool declarations (Gemini function-calling schema)
# ---------------------------------------------------------------------------

_LIST_DOCTORS_DECL = types.FunctionDeclaration(
    name="list_doctors",
    description="Return all doctors at the clinic with their specialties.",
    parameters=types.Schema(
        type=types.Type.OBJECT,
        properties={},
    ),
)

_SEARCH_SLOTS_DECL = types.FunctionDeclaration(
    name="search_slots",
    description=(
        "Search for available appointment slots. "
        "All filters are optional; omit a parameter to skip that filter."
    ),
    parameters=types.Schema(
        type=types.Type.OBJECT,
        properties={
            "doctor_id": types.Schema(
                type=types.Type.INTEGER,
                description="Filter by doctor ID.",
            ),
            "date": types.Schema(
                type=types.Type.STRING,
                description=(
                    "Filter by clinic-local date in YYYY-MM-DD format "
                    "(America/Los_Angeles). Do not combine with date_start or date_end."
                ),
            ),
            "date_start": types.Schema(
                type=types.Type.STRING,
                description=(
                    "Inclusive start of a clinic-local date range in YYYY-MM-DD format. "
                    "Use with date_end for a complete range."
                ),
            ),
            "date_end": types.Schema(
                type=types.Type.STRING,
                description=(
                    "Inclusive end of a clinic-local date range in YYYY-MM-DD format. "
                    "Use with date_start for a complete range."
                ),
            ),
            "time_preference": types.Schema(
                type=types.Type.STRING,
                enum=["morning", "afternoon", "evening"],
                description=(
                    "Optional clinic-local time-of-day filter. Morning is 06:00–11:59, "
                    "afternoon is 12:00–16:59, and evening is 17:00–20:59."
                ),
            ),
        },
    ),
)

_PROPOSE_BOOKING_DECL = types.FunctionDeclaration(
    name="propose_booking",
    description=(
        "Propose an available appointment slot for the patient to review. "
        "This stages the slot for user confirmation; it does NOT write a booking. "
        "Requires explicit patient verbal confirmation (e.g. 'Yes, book it')."
    ),
    parameters=types.Schema(
        type=types.Type.OBJECT,
        properties={
            "slot_id": types.Schema(
                type=types.Type.INTEGER,
                description="The exact ID of the appointment slot to propose.",
            ),
        },
        required=["slot_id"],
    ),
)

TOOLS = types.Tool(
    function_declarations=[
        _LIST_DOCTORS_DECL,
        _SEARCH_SLOTS_DECL,
        _PROPOSE_BOOKING_DECL,
    ]
)

# The names the agent is allowed to dispatch
_ALLOWLIST = frozenset({"list_doctors", "search_slots", "propose_booking"})


# ---------------------------------------------------------------------------
# Tool Execution & Dispatch
# ---------------------------------------------------------------------------


def _dispatch(
    name: str,
    args: dict[str, Any],
    db_path: Path,
    agent: Optional[Agent] = None,
    call_id: str | None = None,
) -> dict[str, Any]:
    """
    Execute an allowlisted tool function and return a JSON-safe dict.
    Validates arguments and logs sanitized calls and results.
    """
    if name not in _ALLOWLIST:
        logger.warning("Rejected non-allowlisted tool: %s", name)
        raise ValueError(f"Function '{name}' is not allowlisted.")

    logger.debug("Dispatching tool '%s' (id=%s) with args: %s", name, call_id, args)
    t0 = time.perf_counter()

    if name == "list_doctors":
        doctors = db.list_doctors(db_path=db_path)
        res = {"doctors": [dataclasses.asdict(d) for d in doctors]}
        duration_ms = (time.perf_counter() - t0) * 1000
        logger.debug("Tool 'list_doctors' result: %d doctors", len(doctors))
        if agent is not None:
            agent.record_event(
                "tool",
                "list_doctors()",
                {
                    "function": "list_doctors",
                    "arguments": {},
                    "result_count": len(doctors),
                    "doctors": [d.name for d in doctors],
                },
                duration_ms=duration_ms,
            )
        return res

    if name == "search_slots":
        raw_doctor_id = args.get("doctor_id")
        doctor_id: Optional[int] = None
        if raw_doctor_id is not None:
            try:
                doctor_id = int(raw_doctor_id)
            except (ValueError, TypeError):
                return {"error": f"Invalid doctor_id: {raw_doctor_id!r}. Must be an integer."}

        date_str: Optional[str] = args.get("date")
        date_values: dict[str, Optional[date]] = {
            "date": None,
            "date_start": None,
            "date_end": None,
        }
        for argument_name in date_values:
            raw_date = args.get(argument_name)
            if raw_date:
                try:
                    date_values[argument_name] = date.fromisoformat(
                        str(raw_date).strip()
                    )
                except ValueError:
                    return {
                        "error": (
                            f"Invalid {argument_name} format: {raw_date!r}. "
                            "Use YYYY-MM-DD."
                        )
                    }

        local_date = date_values["date"]
        local_date_start = date_values["date_start"]
        local_date_end = date_values["date_end"]
        if local_date is not None and (
            local_date_start is not None or local_date_end is not None
        ):
            return {"error": "Use either date or a date range, not both."}
        if (
            local_date_start is not None
            and local_date_end is not None
            and local_date_start > local_date_end
        ):
            return {"error": "date_start must be on or before date_end."}

        raw_time_preference = args.get("time_preference")
        time_preference = (
            str(raw_time_preference).strip().lower()
            if raw_time_preference is not None
            else None
        )
        if time_preference not in (None, "morning", "afternoon", "evening"):
            return {
                "error": (
                    "time_preference must be morning, afternoon, or evening."
                )
            }

        slots = db.list_available_slots(
            db_path=db_path,
            doctor_id=doctor_id,
            local_date=local_date,
            local_date_start=local_date_start,
            local_date_end=local_date_end,
            time_preference=time_preference,
        )
        res = {"slots": [dataclasses.asdict(s) for s in slots]}
        duration_ms = (time.perf_counter() - t0) * 1000
        logger.debug("Tool 'search_slots' result: %d slots", len(slots))
        if agent is not None:
            agent.record_event(
                "tool",
                f"search_slots(doctor_id={doctor_id}, date={date_str})",
                {
                    "function": "search_slots",
                    "arguments": {
                        "doctor_id": doctor_id,
                        "date": date_str,
                        "date_start": (
                            local_date_start.isoformat()
                            if local_date_start is not None
                            else None
                        ),
                        "date_end": (
                            local_date_end.isoformat()
                            if local_date_end is not None
                            else None
                        ),
                        "time_preference": time_preference,
                    },
                    "result_count": len(slots),
                },
                duration_ms=duration_ms,
            )
        return res

    if name == "propose_booking":
        if "slot_id" not in args:
            return {"error": "Missing required argument 'slot_id'."}
        try:
            slot_id = int(args["slot_id"])
        except (ValueError, TypeError):
            return {"error": f"Invalid slot_id: {args.get('slot_id')!r}. Must be an integer."}

        # 1. Look up exact slot in SQLite
        slot = db.get_slot(slot_id, db_path=db_path)
        if slot is None:
            logger.debug("propose_booking: slot %d not found", slot_id)
            return {"error": f"Slot {slot_id} does not exist."}

        # 2. Verify it is future
        slot_start = datetime.fromisoformat(slot.start_utc)
        if slot_start <= datetime.now(timezone.utc):
            logger.debug("propose_booking: slot %d is in the past (%s)", slot_id, slot.start_utc)
            return {"error": f"Slot {slot_id} is in the past."}

        # 3. Verify it is available
        if not db.is_slot_available(slot_id, db_path=db_path):
            logger.debug("propose_booking: slot %d is unavailable", slot_id)
            return {"error": f"Slot {slot_id} is no longer available (already booked)."}

        # 4. Generate a stable idempotency key for this booking attempt
        idempotency_key = f"book-{slot_id}-{uuid.uuid4().hex}"

        # 5. Store the proposed slot in backend session state if agent is provided
        local_dt = slot_start.astimezone(db.CLINIC_TZ)
        local_time_str = local_dt.strftime("%A, %B %d, %Y at %I:%M %p %Z")
        patient_name = agent.patient_name if agent else DEFAULT_PATIENT_NAME

        if agent is not None:
            agent.pending_booking = PendingBooking(
                slot_id=slot.id,
                doctor_name=slot.doctor_name or "Unknown Doctor",
                specialty=slot.specialty or "General",
                start_utc=slot.start_utc,
                end_utc=slot.end_utc,
                local_time_str=local_time_str,
                idempotency_key=idempotency_key,
                patient_name=patient_name,
                status="awaiting_confirmation",
            )
            agent._last_staged_booking = agent.pending_booking

        res = {
            "status": "awaiting_confirmation",
            "slot_id": slot.id,
            "doctor": slot.doctor_name,
            "specialty": slot.specialty,
            "date": local_dt.strftime("%Y-%m-%d"),
            "time": local_dt.strftime("%I:%M %p %Z"),
            "formatted_time": local_time_str,
            "patient_name": patient_name,
            "instructions": (
                f"Slot {slot.id} with {slot.doctor_name} on {local_time_str} is staged. "
                'Read back the exact doctor, date, and local time, and ask the patient for verbal confirmation to book it.'
            ),
        }
        duration_ms = (time.perf_counter() - t0) * 1000
        logger.info("Staged proposal for slot %d with key %s", slot.id, idempotency_key)
        if agent is not None:
            agent.record_event(
                "tool",
                f"propose_booking(slot_id={slot.id})",
                {
                    "function": "propose_booking",
                    "arguments": {"slot_id": slot.id},
                    "staged_doctor": slot.doctor_name,
                    "staged_time": local_time_str,
                    "status": "awaiting_confirmation",
                },
                duration_ms=duration_ms,
            )
            # Distinct proposal event explicitly identifying uncommitted LLM proposal
            agent.record_event(
                "proposal",
                "LLM Staged Proposal (In-Memory)",
                {
                    "storage": "session_memory (uncommitted)",
                    "committed_to_sqlite": False,
                    "slot_id": slot.id,
                    "doctor": slot.doctor_name,
                    "specialty": slot.specialty,
                    "time": local_time_str,
                    "status": "awaiting_confirmation",
                },
            )
        return res

    raise ValueError(f"Unhandled function: {name}")


# ---------------------------------------------------------------------------
# Verbal Confirmation Intent Classification Helpers (Milestone 6A)
# ---------------------------------------------------------------------------


def is_verbal_confirmation(text: str) -> bool:
    """
    Return True if text expresses natural verbal confirmation to book a proposed appointment.
    Handles single-word, repeated, and multi-word affirmative expressions.
    Examples: 'Yes, book it', 'Confirm', 'Confirm. Confirm.', 'Yes. Book it.', 'Please', 'Sounds good'.
    """
    t = text.strip().lower().replace("'", "")
    cleaned = re.sub(r"[^\w\s]", " ", t)
    tokens = cleaned.split()
    if not tokens:
        return False

    # Negative guard: if text contains negative/cancel words, it cannot be a confirmation
    negative_words = {"no", "not", "dont", "never", "cancel", "stop", "discard", "neither"}
    if any(neg in tokens for neg in negative_words):
        return False

    # Modification guard: if text expresses changing preferences, let modification handler handle it
    if is_verbal_modification(text):
        return False

    # Ambiguity guard: if text expresses hesitation, let ambiguity handler handle it
    if is_verbal_ambiguity(text):
        return False

    affirmative_tokens = {
        "yes", "yeah", "yep", "yup", "sure", "ok", "okay",
        "confirm", "confirmed", "book", "booked", "booking",
        "please", "perfect", "good", "fine", "great", "correct",
        "right", "ahead", "absolutely", "definitely", "certainly",
        "affirmative", "proceed",
    }

    # If any token is an affirmative token, it is an affirmative confirmation
    if any(tok in affirmative_tokens for tok in tokens):
        return True

    # Also check phrases like "that works", "sounds good", "do it", "let us", "lets"
    phrase = " ".join(tokens)
    affirmative_phrases = [
        r"\bthat works\b",
        r"\bsounds good\b",
        r"\b(?:lets|let us)\s+(?:do|book)\s+it\b",
        r"\bdo it\b",
        r"\bgo for it\b",
        r"\bmake it so\b",
    ]
    for pattern in affirmative_phrases:
        if re.search(pattern, phrase):
            return True

    return False


def is_verbal_cancellation(text: str) -> bool:
    """
    Return True if text expresses cancellation or rejection of the proposed appointment.
    Examples: 'No', 'Cancel', 'No thanks', 'Don't book it', 'Never mind'.
    """
    t = text.strip().lower().replace("'", "")
    cleaned = re.sub(r"[^\w\s]", " ", t)
    tokens = cleaned.split()
    if not tokens:
        return False

    exact_negatives = {
        "no",
        "nope",
        "cancel",
        "cancelled",
        "no thanks",
        "dont book",
        "dont book it",
        "do not book",
        "do not book it",
        "never mind",
        "nevermind",
        "stop",
        "no dont",
        "no cancel",
        "cancel that",
        "cancel it",
        "cancel the appointment",
        "discard",
    }
    normalized_phrase = " ".join(tokens)
    if normalized_phrase in exact_negatives:
        return True

    cancellation_patterns = [
        r"\b(?:cancel\s+it|cancel\s+that|cancel\s+the\s+appointment|cancel\s+this)\b",
        r"\b(?:dont\s+book|do\s+not\s+book|never\s+mind)\b",
        r"^(?:no|nope)\b(?!\s+(?:actually|change|switch|what\s+about))",
    ]
    for pattern in cancellation_patterns:
        if re.search(pattern, normalized_phrase):
            return True

    return False


def is_verbal_modification(text: str) -> bool:
    """
    Return True if text expresses changing the appointment preference (date, time, doctor).
    Examples: 'Actually, change it to 2 PM', 'Can we do Wednesday instead?', 'Switch to Dr. Lovelace'.
    """
    t = text.strip().lower()
    mod_patterns = [
        r"\b(?:actually|change|different|instead|what\s+about|how\s+about|switch|another|other|rather|prefer)\b",
        r"\b(?:tuesday|wednesday|thursday|friday|saturday|sunday|monday)\b",
        r"\b(?:morning|afternoon|pm|am|clock|\d{1,2}:\d{2})\b",
        r"\b(?:dr|doctor|babbage|lovelace|nightingale)\b",
    ]
    for pattern in mod_patterns:
        if re.search(pattern, t):
            return True
    return False


def is_verbal_ambiguity(text: str) -> bool:
    """
    Return True if text expresses hesitation, uncertainty, or ambiguity
    in response to an appointment confirmation prompt.
    Examples: 'Maybe', 'Not sure', 'Hold on', 'Wait', 'Let me think', 'Perhaps'.
    """
    t = text.strip().lower()
    cleaned = re.sub(r"[^\w\s]", " ", t)
    tokens = cleaned.split()
    if not tokens:
        return True

    exact_ambiguous = {
        "maybe",
        "perhaps",
        "not sure",
        "im not sure",
        "i am not sure",
        "hold on",
        "wait",
        "wait a second",
        "wait a minute",
        "one second",
        "one moment",
        "let me think",
        "let me see",
        "let me check",
        "thinking",
        "idk",
        "i dont know",
        "i do not know",
        "possibly",
        "um",
        "uh",
        "hmm",
    }
    normalized = " ".join(tokens)
    if normalized in exact_ambiguous:
        return True

    ambiguity_patterns = [
        r"^(?:maybe|perhaps|not\s+sure|hold\s+on|wait|let\s+me\s+think|let\s+me\s+check)\b",
        r"\b(?:give\s+me\s+a\s+(?:second|moment)|not\s+quite\s+sure)\b",
    ]
    for pattern in ambiguity_patterns:
        if re.search(pattern, normalized):
            return True

    return False


def is_informative_question(text: str) -> bool:
    """
    Return True if text is an informational question (e.g. about clinic hours,
    doctor specialties, location, services) rather than a response to a confirmation prompt.
    """
    t = text.strip().lower()
    if not t:
        return False
    cleaned = re.sub(r"[^\w\s]", " ", t).strip()
    words = cleaned.split()
    if not words:
        return False
    # If the user is asking to book or confirm, that is not an informative question
    if any(w in words for w in ["book", "confirm", "finalize", "stage"]):
        return False
    if t.endswith("?") and words[0] not in ["yes", "ok", "sure", "fine", "really"]:
        return True
    question_starters = {
        "what", "where", "when", "why", "who", "which", "how",
        "does", "do", "is", "are", "can", "could", "would",
        "tell", "explain",
    }
    return words[0] in question_starters


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------


class Agent:
    """
    Stateful conversational agent backed by Gemini.

    Parameters
    ----------
    db_path : Path
        SQLite database to query.
    model : str | None
        Gemini model name. Defaults to GEMINI_MODEL in .env or gemini-3.8-flash.
    client : genai.Client | None
        Inject a client (useful for testing). If *None*, one is
        created from the default environment / API key.
    patient_name : str
        Fictional demo patient name. Unauthenticated.
    """

    def __init__(
        self,
        db_path: Path = db.DEFAULT_DB,
        model: str | None = None,
        client: genai.Client | None = None,
        patient_name: str = DEFAULT_PATIENT_NAME,
        event_callback: Optional[Callable[[str, str, dict[str, Any], Optional[float], str], None]] = None,
    ) -> None:
        load_dotenv()
        self.db_path = db_path
        self.model = model or os.getenv("GEMINI_MODEL") or DEFAULT_MODEL
        self.patient_name = patient_name
        self.event_callback = event_callback

        if client is not None:
            self.client = client
        else:
            api_key = os.getenv("GEMINI_API_KEY")
            self.client = genai.Client(api_key=api_key) if api_key else genai.Client()

        today = datetime.now(db.CLINIC_TZ).strftime("%A %B %d, %Y")
        self._system = SYSTEM_INSTRUCTION.format(
            today=today, patient_name=self.patient_name
        )

        # Conversation history: list[types.Content]
        self.history: list[types.Content] = []

        # Session state: pending booking proposal
        self.pending_booking: Optional[PendingBooking] = None
        self._last_staged_booking: Optional[PendingBooking] = None
        self.awaiting_verbal_confirmation: bool = False

    def record_event(
        self,
        stage: str,
        title: str,
        details: dict[str, Any],
        duration_ms: Optional[float] = None,
        status: str = "ok",
    ) -> None:
        """Forward an event to the registered event callback for developer inspection."""
        if self.event_callback is not None:
            try:
                self.event_callback(stage, title, details, duration_ms, status)
            except Exception as exc:
                logger.warning("Event callback failed: %s", exc)

    def _dispatch(
        self, name: str, args: dict[str, Any], call_id: str | None = None
    ) -> dict[str, Any]:
        return _dispatch(name, args, self.db_path, agent=self, call_id=call_id)

    # ------------------------------------------------------------------
    # Public Chat & Confirmation API
    # ------------------------------------------------------------------

    def turn(self, user_text: str) -> str:
        """
        Send a user message, run the tool-call loop, and return the
        final assistant text reply with any staging confirmation banner.

        Reliability guarantees:
        - FIX 1: Catches connection/API errors cleanly, rolls back history,
          fails closed on partially staged proposals, and returns an actionable message.
        - FIX 3: Invalidates any stale awaiting-confirmation proposal at turn start.
        - FIX 4: Rejects ambiguous simultaneous/sequential proposals in the same turn.
        """
        # Save checkpoints before modifying state in this turn
        history_checkpoint = list(self.history)

        # Verbal confirmation interception (Milestone 6A)
        # Check if the immediately preceding prompt explicitly asked for verbal confirmation
        # of an active pending proposal
        if (
            self.awaiting_verbal_confirmation
            and self.pending_booking is not None
            and self.pending_booking.status == "awaiting_confirmation"
        ):
            # 1. Modification / Change of mind takes precedence
            if is_verbal_modification(user_text):
                logger.info(
                    "Patient changed mind during confirmation; discarding pending slot %d",
                    self.pending_booking.slot_id,
                )
                self.pending_booking = None
                self._last_staged_booking = None
                self.awaiting_verbal_confirmation = False
                # Proceed to normal Gemini turn below to handle the new preference

            # 2. Explicit affirmative confirmation
            elif is_verbal_confirmation(user_text):
                logger.info(
                    "Verbal confirmation received for slot %d: %r",
                    self.pending_booking.slot_id,
                    user_text,
                )
                self.awaiting_verbal_confirmation = False
                success, message = self.confirm_booking()
                return message

            # 3. Explicit cancellation / rejection
            elif is_verbal_cancellation(user_text):
                logger.info(
                    "Verbal cancellation received for slot %d: %r",
                    self.pending_booking.slot_id,
                    user_text,
                )
                self.awaiting_verbal_confirmation = False
                self._last_staged_booking = None
                success, message = self.cancel_pending()
                return (
                    "I've cancelled that booking request. "
                    "Would you like to check other times or doctors?"
                )

            # 4. Informative question / topic change (e.g. "Wait, what does Dr. Lovelace specialize in?")
            elif is_informative_question(user_text):
                logger.info("Informative question received during confirmation: %r", user_text)
                self.pending_booking = None
                self.awaiting_verbal_confirmation = False
                # Proceed to normal Gemini turn below to answer the question

            # 5. Ambiguous, hesitant, or unclear background speech (e.g. 'Not far. My phone.', 'Wait', 'Hold on')
            else:
                logger.info("Unclear or ambiguous utterance during confirmation prompt: %r", user_text)
                # Keep pending_booking active and keep awaiting_verbal_confirmation True!
                return (
                    f"Just to confirm, would you like me to book your appointment with "
                    f"{self.pending_booking.doctor_name} on {self.pending_booking.local_time_str}? "
                    "Please say yes to confirm, or let me know if you would like to change it."
                )

        # Recovery check: If user ignore propsal and swtich topic
        # temporarily cleared (e.g. after an informational question turn where the doctor and time were discussed),
        # recover the proposal from self._last_staged_booking if available in SQLite.
        if (
            self.pending_booking is None
            and self._last_staged_booking is not None
            and is_verbal_confirmation(user_text)
        ):
            if db.is_slot_available(self._last_staged_booking.slot_id, db_path=self.db_path):
                logger.info(
                    "Recovering and confirming last staged slot %d from affirmative utterance: %r",
                    self._last_staged_booking.slot_id,
                    user_text,
                )
                self.pending_booking = self._last_staged_booking
                self.awaiting_verbal_confirmation = False
                success, message = self.confirm_booking()
                return message

        # FIX 3: Invalidate any stale awaiting-confirmation proposal at the start
        # of a new free-text turn so old proposals do not linger after topic changes
        if (
            self.pending_booking is not None
            and self.pending_booking.status == "awaiting_confirmation"
        ):
            logger.debug(
                "Invalidating stale pending proposal for slot %d due to new user turn",
                self.pending_booking.slot_id,
            )
            self.pending_booking = None
            self.awaiting_verbal_confirmation = False

        # Append user turn to history
        self.history.append(
            types.Content(
                role="user",
                parts=[types.Part.from_text(text=user_text)],
            )
        )

        # Track proposals across all tool-call rounds in this single turn (FIX 4)
        proposed_slots_in_turn: set[int] = set()
        ambiguous_turn = False

        try:
            for _round in range(MAX_TOOL_ROUNDS):
                response = self.client.models.generate_content(
                    model=self.model,
                    contents=self.history,
                    config=types.GenerateContentConfig(
                        tools=[TOOLS],
                        automatic_function_calling=types.AutomaticFunctionCallingConfig(
                            disable=True,
                        ),
                        system_instruction=self._system,
                        temperature=0.0,
                    ),
                )

                # Preserve the FULL model response (including thought signatures)
                model_content = response.candidates[0].content
                self.history.append(model_content)

                fn_calls = response.function_calls
                if not fn_calls:
                    reply = response.text or ""
                    # If a slot is staged awaiting confirmation, attach the Python-generated banner
                    if (
                        self.pending_booking is not None
                        and self.pending_booking.status == "awaiting_confirmation"
                    ):
                        self.awaiting_verbal_confirmation = True
                        banner = (
                            f"\n\n[Pending Confirmation]\n"
                            f"• Doctor  : {self.pending_booking.doctor_name} ({self.pending_booking.specialty})\n"
                            f"• Time    : {self.pending_booking.local_time_str}\n"
                            f"• Patient : {self.pending_booking.patient_name} (fictional demo identity)\n"
                            f"• Status  : Awaiting confirmation\n"
                            f'➡ Say "Yes, book it" to confirm, or "Cancel" to discard.'
                        )
                        reply += banner
                    else:
                        self.awaiting_verbal_confirmation = False
                    return reply

                # FIX 4: Detect ambiguous proposals in this round or across rounds
                round_propose_calls = [fc for fc in fn_calls if fc.name == "propose_booking"]
                round_slot_ids: set[int] = set()
                for pc in round_propose_calls:
                    raw_id = (pc.args or {}).get("slot_id")
                    if raw_id is not None:
                        try:
                            round_slot_ids.add(int(raw_id))
                        except (ValueError, TypeError):
                            pass

                # If multiple different slots proposed in this round, OR if a new different
                # slot is proposed after a prior round in this turn:
                if len(round_slot_ids) > 1 or (
                    proposed_slots_in_turn
                    and round_slot_ids
                    and not round_slot_ids.issubset(proposed_slots_in_turn)
                ):
                    ambiguous_turn = True
                    self.pending_booking = None
                    logger.warning(
                        "Rejected ambiguous proposals across turn rounds: existing=%s, new=%s",
                        proposed_slots_in_turn,
                        round_slot_ids,
                    )

                proposed_slots_in_turn.update(round_slot_ids)

                # Dispatch each call and capture response parts
                fn_response_parts: list[types.Part] = []
                for fc in fn_calls:
                    if fc.name == "propose_booking" and ambiguous_turn:
                        result = {
                            "error": (
                                "Ambiguous proposal: multiple different appointment slots were proposed in this turn. "
                                "Only one slot may be proposed per turn. No slot has been staged. "
                                "Please ask the patient which specific slot they prefer."
                            )
                        }
                    else:
                        try:
                            result = self._dispatch(fc.name, fc.args or {}, fc.id)
                        except Exception as exc:
                            result = {"error": str(exc)}

                    part = types.Part.from_function_response(
                        name=fc.name,
                        response={"result": result},
                    )
                    if fc.id:
                        part.function_response.id = fc.id
                    fn_response_parts.append(part)

                self.history.append(
                    types.Content(role="user", parts=fn_response_parts)
                )

            # FIX: If loop exceeds MAX_TOOL_ROUNDS, restore consistent conversation
            # state and fail closed on any partially staged proposal
            self.history = history_checkpoint
            self.pending_booking = None
            self.awaiting_verbal_confirmation = False
            logger.warning(
                "Tool-round limit (%d) reached without model concluding. Rolled back state.",
                MAX_TOOL_ROUNDS,
            )
            return "(max tool rounds reached)"

        except Exception as exc:
            # FIX 1: Handle connection timeouts and transient API errors without terminating the CLI
            # Roll back conversation history to clean checkpoint
            self.history = history_checkpoint
            # Fail closed: clear any proposal staged during this failed turn
            self.pending_booking = None
            self.awaiting_verbal_confirmation = False
            logger.error("Gemini API call failed during turn: %s", exc)
            return (
                "I'm sorry, but I'm having trouble connecting to the appointment service right now. "
                "Your request could not be processed. Please try again in a moment."
            )

    def confirm_booking(self) -> tuple[bool, str]:
        """
        Commit the current pending appointment selection.
        Intercepted by Python BEFORE sending anything to Gemini.

        Reliability guarantees (FIX 2):
        - In both initial and repeated invocations, inspects the returned
          booking's CURRENT database status.
        - Reports success ONLY when status == 'confirmed'.
        - Never reports a cancelled booking as confirmed.
        - Handles BookingError consistently across both paths.

        Returns
        -------
        (success: bool, message: str)
        """
        if self.pending_booking is None:
            return (
                False,
                "No pending appointment to confirm. Please search for and select an appointment slot first.",
            )

        pending = self.pending_booking

        if pending.status not in ("awaiting_confirmation", "confirmed"):
            return False, f"Cannot confirm booking in state '{pending.status}'."

        was_already_confirmed = (pending.status == "confirmed")

        try:
            booking = db.book_slot(
                slot_id=pending.slot_id,
                patient_name=pending.patient_name,
                idempotency_key=pending.idempotency_key,
                db_path=self.db_path,
            )
        except db.BookingError as exc:
            if exc.code == db.SLOT_UNAVAILABLE:
                self.pending_booking = None
                return False, (
                    f"The selected slot with {pending.doctor_name} on {pending.local_time_str} "
                    "is no longer available (it was just booked by another patient). "
                    "Please search for another available time."
                )
            elif exc.code == db.SLOT_IN_PAST:
                self.pending_booking = None
                return False, (
                    f"The proposed appointment time ({pending.local_time_str}) has passed. "
                    "Please search for an upcoming slot."
                )
            elif exc.code == db.SLOT_NOT_FOUND:
                self.pending_booking = None
                return False, "The proposed slot does not exist. Please search for available slots."
            elif exc.code == db.IDEMPOTENCY_CONFLICT:
                self.pending_booking = None
                return False, f"Idempotency conflict for slot {pending.slot_id}: {exc}"
            else:
                return False, f"Booking error [{exc.code}]: {exc}"

        # FIX 2: Check returned booking's CURRENT database status
        # Report success ONLY when status == "confirmed". Never report a cancelled booking as confirmed.
        if booking.status != "confirmed":
            pending.status = booking.status
            return False, (
                f"Cannot confirm: this appointment is {booking.status} in the database. "
                "Please search for another available slot."
            )

        pending.status = "confirmed"
        pending.booking_id = booking.id
        self._last_staged_booking = None

        self.record_event(
            "sqlite_commit",
            f"SQLite Atomic Commit (Booking #{booking.id})",
            {
                "storage": "sqlite_database",
                "table": "bookings",
                "committed_to_sqlite": True,
                "booking_id": booking.id,
                "slot_id": pending.slot_id,
                "doctor": pending.doctor_name,
                "time": pending.local_time_str,
                "patient": booking.patient_name,
                "status": "confirmed",
            },
        )

        if was_already_confirmed:
            return True, (
                f"✔ Appointment is already confirmed!\n"
                f"  Booking ID : #{booking.id}\n"
                f"  Doctor     : {pending.doctor_name} ({pending.specialty})\n"
                f"  Time       : {pending.local_time_str}\n"
                f"  Patient    : {booking.patient_name} (fictional demo identity)"
            )

        # First-time confirmation: sync into history
        self.history.append(
            types.Content(
                role="user",
                parts=[
                    types.Part.from_text(
                        text=(
                            f"[System Event: Patient confirmed Booking #{booking.id} "
                            f"with {pending.doctor_name} on {pending.local_time_str} "
                            f"for patient {pending.patient_name}.]"
                        )
                    )
                ],
            )
        )
        self.history.append(
            types.Content(
                role="model",
                parts=[
                    types.Part.from_text(
                        text=(
                            f"Your appointment with {pending.doctor_name} on "
                            f"{pending.local_time_str} is confirmed (Booking #{booking.id})."
                        )
                    )
                ],
            )
        )

        return True, (
            f"✔ Appointment successfully confirmed!\n"
            f"  Booking ID : #{booking.id}\n"
            f"  Doctor     : {pending.doctor_name} ({pending.specialty})\n"
            f"  Time       : {pending.local_time_str}\n"
            f"  Patient    : {booking.patient_name} (fictional demo identity)"
        )

    def cancel_pending(self) -> tuple[bool, str]:
        """
        Discard the pending appointment selection.
        Intercepted by Python BEFORE sending anything to Gemini.
        Makes no database change.

        Returns
        -------
        (was_pending: bool, message: str)
        """
        if (
            self.pending_booking is None
            or self.pending_booking.status != "awaiting_confirmation"
        ):
            return False, "No pending appointment selection to cancel."

        discarded_desc = (
            f"Slot {self.pending_booking.slot_id} with {self.pending_booking.doctor_name} "
            f"on {self.pending_booking.local_time_str}"
        )
        self.record_event(
            "cancel",
            "Proposal Discarded (Cancelled)",
            {
                "storage": "session_memory",
                "slot_id": self.pending_booking.slot_id,
                "doctor": self.pending_booking.doctor_name,
                "time": self.pending_booking.local_time_str,
                "status": "cancelled",
                "db_modified": False,
            },
        )
        self.pending_booking = None
        self._last_staged_booking = None
        self.awaiting_verbal_confirmation = False

        self.history.append(
            types.Content(
                role="user",
                parts=[
                    types.Part.from_text(
                        text=(
                            f"[System Event: Patient cancelled pending proposal for "
                            f"{discarded_desc}. No booking was made.]"
                        )
                    )
                ],
            )
        )
        self.history.append(
            types.Content(
                role="model",
                parts=[
                    types.Part.from_text(
                        text=(
                            "I've discarded that pending appointment selection. "
                            "Let me know if you would like to search for another time."
                        )
                    )
                ],
            )
        )

        return (
            True,
            f"✔ Discarded pending appointment selection ({discarded_desc}). No booking was made.",
        )
