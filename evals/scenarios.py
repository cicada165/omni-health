"""
evals/scenarios.py — Realistic Conversational Scheduling Scenarios (Milestone 8).

Defines 42 realistic conversational scheduling scenarios across the 9 required categories:
1. Normal booking (single doctor, multiple doctors, specific dates, next available).
2. Doctor and time corrections (changing day, changing time, switching doctor mid-conversation).
3. Ambiguous verbal confirmations ("maybe", "hold on", "let me think", "uh I guess").
4. Negation and cancellation ("no", "don't book that", "cancel", "never mind").
5. Interruption during confirmation ("wait stop", "actually change to 3 PM").
6. Duplicate and competing booking requests (concurrent requests for same slot, race condition safety, repeated confirmation idempotency).
7. STT transcription errors (fuzzy doctor names, misheard dates, homophones, garbled background noise).
8. Provider timeouts and failures (Gemini timeout/503, Deepgram disconnection, Cartesia failure).
9. Session isolation and state recovery (multi-session state leakage prevention, recovery after informational turns).
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Optional, Union
from unittest.mock import MagicMock

from google.genai import types

from scheduler import db


# ---------------------------------------------------------------------------
# Enums & Data Classes
# ---------------------------------------------------------------------------


class ScenarioCategory(str, Enum):
    NORMAL_BOOKING = "Normal Booking"
    CORRECTIONS = "Doctor and Time Corrections"
    AMBIGUOUS_CONFIRMATIONS = "Ambiguous Verbal Confirmations"
    NEGATION_CANCELLATION = "Negation and Cancellation"
    INTERRUPTIONS = "Interruption During Confirmation"
    DUPLICATE_COMPETING = "Duplicate and Competing Requests"
    STT_ERRORS = "STT Transcription Errors"
    PROVIDER_FAILURES = "Provider Timeouts and Failures"
    SESSION_ISOLATION_RECOVERY = "Session Isolation and State Recovery"


@dataclass
class TurnSpec:
    """Specification of a single conversational turn in an evaluation scenario."""
    user_input: str
    mock_responses: Optional[Union[list[Any], Exception, Callable[[Path, int], list[Any]]]] = None
    expected_pending_after: Optional[bool] = None
    expected_awaiting_after: Optional[bool] = None
    expected_reply_contains: list[str] = field(default_factory=list)
    reply_must_not_contain: list[str] = field(default_factory=list)
    description: str = ""


@dataclass
class ScenarioSpec:
    """Specification of a complete evaluation scenario."""
    id: str
    name: str
    category: ScenarioCategory
    description: str
    turns: list[TurnSpec] = field(default_factory=list)
    expected_booking: bool = False  # Should an active booking exist in SQLite at the end?
    expected_booking_count: Optional[int] = None
    expected_doctor: Optional[str] = None
    expected_specialty: Optional[str] = None
    expected_slot_id: Optional[int] = None
    is_recovery_scenario: bool = False
    custom_runner: Optional[Callable[[ScenarioSpec, Path], Any]] = None


# ---------------------------------------------------------------------------
# Mock Helper Functions for Gemini
# ---------------------------------------------------------------------------


def make_text_response(text: str) -> MagicMock:
    """Build a mock GenerateContentResponse containing only text."""
    part = types.Part.from_text(text=text)
    content = types.Content(role="model", parts=[part])
    candidate = MagicMock()
    candidate.content = content
    response = MagicMock()
    response.candidates = [candidate]
    response.function_calls = []
    response.text = text
    return response


def make_fn_call_response(
    calls: list[tuple[str, dict[str, Any]] | tuple[str, dict[str, Any], str]],
    text_alongside: str | None = None,
) -> MagicMock:
    """Build a mock response containing function calls and optional alongside text."""
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


def make_propose_flow(slot_id: int, doctor_name: str, date_time_str: str) -> list[Any]:
    """Return standard two-round Gemini response: propose_booking tool call, followed by confirmation question."""
    fc_round = make_fn_call_response([("propose_booking", {"slot_id": slot_id})])
    text_round = make_text_response(
        f"I found an opening with {doctor_name} on {date_time_str}. Would you like me to book this appointment for you?"
    )
    return [fc_round, text_round]


def make_search_and_propose_flow(slot_id: int, doctor_id: int, doctor_name: str, date_time_str: str) -> list[Any]:
    """Return three-round Gemini response: search_slots, propose_booking, then confirmation question."""
    search_round = make_fn_call_response([("search_slots", {"doctor_id": doctor_id})])
    propose_round = make_fn_call_response([("propose_booking", {"slot_id": slot_id})])
    text_round = make_text_response(
        f"I found an opening with {doctor_name} on {date_time_str}. Would you like me to book this appointment for you?"
    )
    return [search_round, propose_round, text_round]


# ---------------------------------------------------------------------------
# Scenario Definitions (42 Total)
# ---------------------------------------------------------------------------


def _get_target_slot(db_path: Path, doctor_id: int, slot_index: int = 0) -> db.Slot:
    """Helper to dynamically resolve available slot from seeded database."""
    slots = db.list_available_slots(db_path=db_path, doctor_id=doctor_id)
    if not slots or slot_index >= len(slots):
        raise ValueError(f"No available slot for doctor {doctor_id} at index {slot_index}")
    return slots[slot_index]


def build_scenarios() -> list[ScenarioSpec]:
    scenarios: list[ScenarioSpec] = []

    # =======================================================================
    # CATEGORY 1: Normal Booking (5 Scenarios)
    # =======================================================================

    # NORM-01: Single Doctor Specific Date
    def norm_01_mocks(db_path: Path, turn_idx: int) -> list[Any]:
        slot = _get_target_slot(db_path, doctor_id=2, slot_index=0)  # Dr. Charles Babbage
        return make_propose_flow(slot.id, "Dr. Charles Babbage", "Tuesday, November 17 at 9:00 AM PST")

    scenarios.append(
        ScenarioSpec(
            id="NORM-01",
            name="Single Doctor Specific Date Booking",
            category=ScenarioCategory.NORMAL_BOOKING,
            description="Patient requests Dr. Charles Babbage on Tuesday morning, agent stages slot and confirms on affirmative reply.",
            turns=[
                TurnSpec(
                    user_input="I would like to book an appointment with Dr. Charles Babbage on Tuesday morning.",
                    mock_responses=norm_01_mocks,
                    expected_pending_after=True,
                    expected_awaiting_after=True,
                    expected_reply_contains=["Dr. Charles Babbage", "Pending Confirmation"],
                    description="Patient asks for Dr. Babbage, agent stages slot 9:00 AM.",
                ),
                TurnSpec(
                    user_input="Yes, book it.",
                    mock_responses=[],
                    expected_pending_after=True,
                    expected_awaiting_after=False,
                    expected_reply_contains=["Appointment successfully confirmed", "Dr. Charles Babbage"],
                    description="Patient confirms verbally, booking is committed to SQLite.",
                ),
            ],
            expected_booking=True,
            expected_doctor="Dr. Charles Babbage",
            expected_specialty="Cardiology",
        )
    )

    # NORM-02: Multiple Doctors Availability Comparison
    def norm_02_turn1_mocks(db_path: Path, turn_idx: int) -> list[Any]:
        return [
            make_fn_call_response([("list_doctors", {})]),
            make_text_response("We have Dr. Ada Lovelace in General Practice and Dr. Florence Nightingale in Pediatrics available on Wednesday."),
        ]

    def norm_02_turn2_mocks(db_path: Path, turn_idx: int) -> list[Any]:
        slot = _get_target_slot(db_path, doctor_id=1, slot_index=1)  # Dr. Ada Lovelace afternoon
        return make_propose_flow(slot.id, "Dr. Ada Lovelace", "Wednesday at 2:00 PM PST")

    scenarios.append(
        ScenarioSpec(
            id="NORM-02",
            name="Multiple Doctors Comparison and Selection",
            category=ScenarioCategory.NORMAL_BOOKING,
            description="Patient asks who is available, chooses Dr. Ada Lovelace, and confirms booking.",
            turns=[
                TurnSpec(
                    user_input="What doctors are available on Wednesday?",
                    mock_responses=norm_02_turn1_mocks,
                    expected_pending_after=False,
                    expected_awaiting_after=False,
                    expected_reply_contains=["Dr. Ada Lovelace", "Dr. Florence Nightingale"],
                ),
                TurnSpec(
                    user_input="Let's go with Dr. Ada Lovelace in the afternoon at 2 PM.",
                    mock_responses=norm_02_turn2_mocks,
                    expected_pending_after=True,
                    expected_awaiting_after=True,
                    expected_reply_contains=["Dr. Ada Lovelace", "Pending Confirmation"],
                ),
                TurnSpec(
                    user_input="Sounds good, please confirm it.",
                    mock_responses=[],
                    expected_pending_after=True,
                    expected_awaiting_after=False,
                    expected_reply_contains=["Appointment successfully confirmed", "Dr. Ada Lovelace"],
                ),
            ],
            expected_booking=True,
            expected_doctor="Dr. Ada Lovelace",
            expected_specialty="General Practice",
        )
    )

    # NORM-03: Next Available Appointment
    def norm_03_mocks(db_path: Path, turn_idx: int) -> list[Any]:
        slot = db.list_available_slots(db_path=db_path)[0]
        return make_propose_flow(slot.id, slot.doctor_name or "Dr. Ada Lovelace", "tomorrow at 9:00 AM PST")

    scenarios.append(
        ScenarioSpec(
            id="NORM-03",
            name="Next Available Slot Booking",
            category=ScenarioCategory.NORMAL_BOOKING,
            description="Patient requests the earliest next available appointment with any doctor.",
            turns=[
                TurnSpec(
                    user_input="Can I get the earliest available appointment with any doctor?",
                    mock_responses=norm_03_mocks,
                    expected_pending_after=True,
                    expected_awaiting_after=True,
                    expected_reply_contains=["Pending Confirmation"],
                ),
                TurnSpec(
                    user_input="That works for me.",
                    mock_responses=[],
                    expected_pending_after=True,
                    expected_awaiting_after=False,
                    expected_reply_contains=["Appointment successfully confirmed"],
                ),
            ],
            expected_booking=True,
        )
    )

    # NORM-04: Specific Afternoon Slot by Doctor Name
    def norm_04_mocks(db_path: Path, turn_idx: int) -> list[Any]:
        slot = _get_target_slot(db_path, doctor_id=3, slot_index=1)  # Dr. Florence Nightingale
        return make_propose_flow(slot.id, "Dr. Florence Nightingale", "Thursday at 2:00 PM PST")

    scenarios.append(
        ScenarioSpec(
            id="NORM-04",
            name="Specific Afternoon Slot with Pediatrician",
            category=ScenarioCategory.NORMAL_BOOKING,
            description="Patient books Dr. Florence Nightingale in the afternoon.",
            turns=[
                TurnSpec(
                    user_input="I need to see Dr. Florence Nightingale on Thursday afternoon.",
                    mock_responses=norm_04_mocks,
                    expected_pending_after=True,
                    expected_awaiting_after=True,
                    expected_reply_contains=["Dr. Florence Nightingale", "Pending Confirmation"],
                ),
                TurnSpec(
                    user_input="Confirm that appointment.",
                    mock_responses=[],
                    expected_pending_after=True,
                    expected_awaiting_after=False,
                    expected_reply_contains=["Appointment successfully confirmed", "Dr. Florence Nightingale"],
                ),
            ],
            expected_booking=True,
            expected_doctor="Dr. Florence Nightingale",
            expected_specialty="Pediatrics",
        )
    )

    # NORM-05: Direct Specialty Search
    def norm_05_mocks(db_path: Path, turn_idx: int) -> list[Any]:
        slot = _get_target_slot(db_path, doctor_id=2, slot_index=0)  # Cardiology
        return [
            make_fn_call_response([("list_doctors", {})]),
            make_fn_call_response([("propose_booking", {"slot_id": slot.id})]),
            make_text_response("Dr. Charles Babbage is our cardiologist. I have an opening Tuesday at 9 AM. Shall I book it?"),
        ]

    scenarios.append(
        ScenarioSpec(
            id="NORM-05",
            name="Direct Specialty Search (Cardiology)",
            category=ScenarioCategory.NORMAL_BOOKING,
            description="Patient requests a cardiologist, agent resolves Dr. Babbage, proposes, and confirms.",
            turns=[
                TurnSpec(
                    user_input="I need to see a cardiologist sometime this week.",
                    mock_responses=norm_05_mocks,
                    expected_pending_after=True,
                    expected_awaiting_after=True,
                    expected_reply_contains=["Dr. Charles Babbage", "Cardiology"],
                ),
                TurnSpec(
                    user_input="Yes please.",
                    mock_responses=[],
                    expected_pending_after=True,
                    expected_awaiting_after=False,
                    expected_reply_contains=["Appointment successfully confirmed"],
                ),
            ],
            expected_booking=True,
            expected_doctor="Dr. Charles Babbage",
            expected_specialty="Cardiology",
        )
    )

    # =======================================================================
    # CATEGORY 2: Doctor and Time Corrections (5 Scenarios)
    # =======================================================================

    # CORR-01: Day Correction Pre-Confirmation
    def corr_01_turn1_mocks(db_path: Path, turn_idx: int) -> list[Any]:
        slot1 = _get_target_slot(db_path, doctor_id=2, slot_index=0)  # Tuesday
        return make_propose_flow(slot1.id, "Dr. Charles Babbage", "Tuesday at 9:00 AM PST")

    def corr_01_turn2_mocks(db_path: Path, turn_idx: int) -> list[Any]:
        slot2 = _get_target_slot(db_path, doctor_id=2, slot_index=2)  # Wednesday
        return make_propose_flow(slot2.id, "Dr. Charles Babbage", "Wednesday at 9:00 AM PST")

    scenarios.append(
        ScenarioSpec(
            id="CORR-01",
            name="Day Correction Pre-Confirmation",
            category=ScenarioCategory.CORRECTIONS,
            description="Patient changes day from Tuesday to Wednesday before confirming; verifies only Wednesday is booked.",
            is_recovery_scenario=True,
            turns=[
                TurnSpec(
                    user_input="Book with Dr. Babbage on Tuesday.",
                    mock_responses=corr_01_turn1_mocks,
                    expected_pending_after=True,
                    expected_awaiting_after=True,
                    expected_reply_contains=["Tuesday"],
                ),
                TurnSpec(
                    user_input="Actually, can we do Wednesday instead?",
                    mock_responses=corr_01_turn2_mocks,
                    expected_pending_after=True,
                    expected_awaiting_after=True,
                    expected_reply_contains=["Wednesday"],
                ),
                TurnSpec(
                    user_input="Yes, book that.",
                    mock_responses=[],
                    expected_pending_after=True,
                    expected_awaiting_after=False,
                    expected_reply_contains=["Appointment successfully confirmed"],
                ),
            ],
            expected_booking=True,
            expected_doctor="Dr. Charles Babbage",
        )
    )

    # CORR-02: Time of Day Correction
    def corr_02_turn1_mocks(db_path: Path, turn_idx: int) -> list[Any]:
        slot1 = _get_target_slot(db_path, doctor_id=1, slot_index=0)  # 9 AM
        return make_propose_flow(slot1.id, "Dr. Ada Lovelace", "Tuesday at 9:00 AM PST")

    def corr_02_turn2_mocks(db_path: Path, turn_idx: int) -> list[Any]:
        slot2 = _get_target_slot(db_path, doctor_id=1, slot_index=1)  # 2 PM
        return make_propose_flow(slot2.id, "Dr. Ada Lovelace", "Tuesday at 2:00 PM PST")

    scenarios.append(
        ScenarioSpec(
            id="CORR-02",
            name="Time of Day Correction (Morning to Afternoon)",
            category=ScenarioCategory.CORRECTIONS,
            description="Patient initially proposed 9 AM, changes mind to 2 PM, confirms 2 PM.",
            is_recovery_scenario=True,
            turns=[
                TurnSpec(
                    user_input="Book Dr. Lovelace on Tuesday morning.",
                    mock_responses=corr_02_turn1_mocks,
                    expected_pending_after=True,
                    expected_awaiting_after=True,
                ),
                TurnSpec(
                    user_input="Actually I prefer afternoon at 2 PM.",
                    mock_responses=corr_02_turn2_mocks,
                    expected_pending_after=True,
                    expected_awaiting_after=True,
                ),
                TurnSpec(
                    user_input="Perfect, book it.",
                    mock_responses=[],
                    expected_pending_after=True,
                    expected_awaiting_after=False,
                    expected_reply_contains=["Appointment successfully confirmed"],
                ),
            ],
            expected_booking=True,
            expected_doctor="Dr. Ada Lovelace",
        )
    )

    # CORR-03: Doctor Switch Mid-Conversation
    def corr_03_turn1_mocks(db_path: Path, turn_idx: int) -> list[Any]:
        slot1 = _get_target_slot(db_path, doctor_id=2, slot_index=0)  # Dr. Babbage
        return make_propose_flow(slot1.id, "Dr. Charles Babbage", "Tuesday at 9:00 AM PST")

    def corr_03_turn2_mocks(db_path: Path, turn_idx: int) -> list[Any]:
        slot2 = _get_target_slot(db_path, doctor_id=3, slot_index=0)  # Dr. Nightingale
        return make_propose_flow(slot2.id, "Dr. Florence Nightingale", "Tuesday at 9:00 AM PST")

    scenarios.append(
        ScenarioSpec(
            id="CORR-03",
            name="Doctor Switch Mid-Conversation",
            category=ScenarioCategory.CORRECTIONS,
            description="Patient proposes Dr. Babbage, then switches to Dr. Florence Nightingale before confirming.",
            is_recovery_scenario=True,
            turns=[
                TurnSpec(
                    user_input="Book Dr. Charles Babbage on Tuesday.",
                    mock_responses=corr_03_turn1_mocks,
                    expected_pending_after=True,
                    expected_awaiting_after=True,
                ),
                TurnSpec(
                    user_input="Wait, can I switch to Dr. Florence Nightingale instead?",
                    mock_responses=corr_03_turn2_mocks,
                    expected_pending_after=True,
                    expected_awaiting_after=True,
                    expected_reply_contains=["Dr. Florence Nightingale"],
                ),
                TurnSpec(
                    user_input="Yes, that works.",
                    mock_responses=[],
                    expected_pending_after=True,
                    expected_awaiting_after=False,
                    expected_reply_contains=["Appointment successfully confirmed", "Dr. Florence Nightingale"],
                ),
            ],
            expected_booking=True,
            expected_doctor="Dr. Florence Nightingale",
        )
    )

    # CORR-04: Multiple Sequential Corrections
    def corr_04_turn1_mocks(db_path: Path, turn_idx: int) -> list[Any]:
        slot1 = _get_target_slot(db_path, doctor_id=2, slot_index=0)
        return make_propose_flow(slot1.id, "Dr. Charles Babbage", "Tuesday morning")

    def corr_04_turn2_mocks(db_path: Path, turn_idx: int) -> list[Any]:
        slot2 = _get_target_slot(db_path, doctor_id=2, slot_index=3)
        return make_propose_flow(slot2.id, "Dr. Charles Babbage", "Thursday afternoon")

    def corr_04_turn3_mocks(db_path: Path, turn_idx: int) -> list[Any]:
        slot3 = _get_target_slot(db_path, doctor_id=1, slot_index=3)
        return make_propose_flow(slot3.id, "Dr. Ada Lovelace", "Thursday afternoon")

    scenarios.append(
        ScenarioSpec(
            id="CORR-04",
            name="Multiple Sequential Corrections (Day, Time, Doctor)",
            category=ScenarioCategory.CORRECTIONS,
            description="Patient changes day -> changes time -> changes doctor across sequential turns.",
            is_recovery_scenario=True,
            turns=[
                TurnSpec(
                    user_input="I want Tuesday morning.",
                    mock_responses=corr_04_turn1_mocks,
                    expected_pending_after=True,
                    expected_awaiting_after=True,
                ),
                TurnSpec(
                    user_input="Actually make it Thursday afternoon.",
                    mock_responses=corr_04_turn2_mocks,
                    expected_pending_after=True,
                    expected_awaiting_after=True,
                ),
                TurnSpec(
                    user_input="Can we switch to Dr. Lovelace for that time?",
                    mock_responses=corr_04_turn3_mocks,
                    expected_pending_after=True,
                    expected_awaiting_after=True,
                    expected_reply_contains=["Dr. Ada Lovelace"],
                ),
                TurnSpec(
                    user_input="Yes, please confirm.",
                    mock_responses=[],
                    expected_pending_after=True,
                    expected_awaiting_after=False,
                    expected_reply_contains=["Appointment successfully confirmed", "Dr. Ada Lovelace"],
                ),
            ],
            expected_booking=True,
            expected_doctor="Dr. Ada Lovelace",
        )
    )

    # CORR-05: Correction Promptly at Confirmation Question
    def corr_05_turn1_mocks(db_path: Path, turn_idx: int) -> list[Any]:
        slot1 = _get_target_slot(db_path, doctor_id=2, slot_index=0)
        return make_propose_flow(slot1.id, "Dr. Charles Babbage", "Tuesday at 9:00 AM PST")

    def corr_05_turn2_mocks(db_path: Path, turn_idx: int) -> list[Any]:
        slot2 = _get_target_slot(db_path, doctor_id=2, slot_index=4)  # Friday
        return make_propose_flow(slot2.id, "Dr. Charles Babbage", "Friday at 9:00 AM PST")

    scenarios.append(
        ScenarioSpec(
            id="CORR-05",
            name="Correction Directly at Confirmation Prompt",
            category=ScenarioCategory.CORRECTIONS,
            description="When agent asks for confirmation, patient immediately says 'Actually let's do Friday instead'.",
            is_recovery_scenario=True,
            turns=[
                TurnSpec(
                    user_input="Book Dr. Babbage on Tuesday.",
                    mock_responses=corr_05_turn1_mocks,
                    expected_pending_after=True,
                    expected_awaiting_after=True,
                ),
                TurnSpec(
                    user_input="Actually let's do Friday instead.",
                    mock_responses=corr_05_turn2_mocks,
                    expected_pending_after=True,
                    expected_awaiting_after=True,
                ),
                TurnSpec(
                    user_input="Yes, book it.",
                    mock_responses=[],
                    expected_pending_after=True,
                    expected_awaiting_after=False,
                    expected_reply_contains=["Appointment successfully confirmed"],
                ),
            ],
            expected_booking=True,
            expected_doctor="Dr. Charles Babbage",
        )
    )

    # =======================================================================
    # CATEGORY 3: Ambiguous Verbal Confirmations (4 Scenarios)
    # =======================================================================

    # AMBIG-01: Hesitant "Maybe"
    def ambig_setup(db_path: Path, turn_idx: int) -> list[Any]:
        slot = _get_target_slot(db_path, doctor_id=1, slot_index=0)
        return make_propose_flow(slot.id, "Dr. Ada Lovelace", "Tuesday at 9:00 AM PST")

    scenarios.append(
        ScenarioSpec(
            id="AMBIG-01",
            name="Hesitant Confirmation ('Maybe')",
            category=ScenarioCategory.AMBIGUOUS_CONFIRMATIONS,
            description="Patient says 'Maybe, I need to check my calendar'. Agent clarifies without booking.",
            is_recovery_scenario=True,
            turns=[
                TurnSpec(
                    user_input="Book an appointment with Dr. Ada Lovelace on Tuesday.",
                    mock_responses=ambig_setup,
                    expected_pending_after=True,
                    expected_awaiting_after=True,
                ),
                TurnSpec(
                    user_input="Maybe, I need to check my calendar first.",
                    mock_responses=[],
                    expected_pending_after=True,
                    expected_awaiting_after=True,
                    expected_reply_contains=["Just to confirm", "would you like me to book"],
                    reply_must_not_contain=["successfully confirmed", "Booking #"],
                ),
            ],
            expected_booking=False,
        )
    )

    # AMBIG-02: Pause "Hold on"
    scenarios.append(
        ScenarioSpec(
            id="AMBIG-02",
            name="Pause and Delay ('Hold on a second')",
            category=ScenarioCategory.AMBIGUOUS_CONFIRMATIONS,
            description="Patient says 'Hold on a second'. System preserves proposal without committing.",
            is_recovery_scenario=True,
            turns=[
                TurnSpec(
                    user_input="Book an appointment with Dr. Ada Lovelace on Tuesday.",
                    mock_responses=ambig_setup,
                    expected_pending_after=True,
                    expected_awaiting_after=True,
                ),
                TurnSpec(
                    user_input="Hold on a second.",
                    mock_responses=[],
                    expected_pending_after=True,
                    expected_awaiting_after=True,
                    expected_reply_contains=["Just to confirm"],
                    reply_must_not_contain=["successfully confirmed", "Booking #"],
                ),
            ],
            expected_booking=False,
        )
    )

    # AMBIG-03: Hesitation "Let me think"
    scenarios.append(
        ScenarioSpec(
            id="AMBIG-03",
            name="Hesitation ('Let me think')",
            category=ScenarioCategory.AMBIGUOUS_CONFIRMATIONS,
            description="Patient says 'Let me think about that for a minute'. Must not commit.",
            is_recovery_scenario=True,
            turns=[
                TurnSpec(
                    user_input="Book an appointment with Dr. Ada Lovelace on Tuesday.",
                    mock_responses=ambig_setup,
                    expected_pending_after=True,
                    expected_awaiting_after=True,
                ),
                TurnSpec(
                    user_input="Let me think about that for a minute.",
                    mock_responses=[],
                    expected_pending_after=True,
                    expected_awaiting_after=True,
                    expected_reply_contains=["Just to confirm"],
                    reply_must_not_contain=["successfully confirmed", "Booking #"],
                ),
            ],
            expected_booking=False,
        )
    )

    # AMBIG-04: Uncertainty "Not sure yet"
    scenarios.append(
        ScenarioSpec(
            id="AMBIG-04",
            name="Uncertainty ('Not sure yet')",
            category=ScenarioCategory.AMBIGUOUS_CONFIRMATIONS,
            description="Patient expresses hesitation ('Not sure yet.'). System asks for clarification.",
            is_recovery_scenario=True,
            turns=[
                TurnSpec(
                    user_input="Book an appointment with Dr. Ada Lovelace on Tuesday.",
                    mock_responses=ambig_setup,
                    expected_pending_after=True,
                    expected_awaiting_after=True,
                ),
                TurnSpec(
                    user_input="Not sure yet.",
                    mock_responses=[],
                    expected_pending_after=True,
                    expected_awaiting_after=True,
                    expected_reply_contains=["Just to confirm"],
                    reply_must_not_contain=["successfully confirmed", "Booking #"],
                ),
            ],
            expected_booking=False,
        )
    )

    # =======================================================================
    # CATEGORY 4: Negation and Cancellation (4 Scenarios)
    # =======================================================================

    # CANC-01: Direct "No"
    scenarios.append(
        ScenarioSpec(
            id="CANC-01",
            name="Direct Verbal Negation ('No')",
            category=ScenarioCategory.NEGATION_CANCELLATION,
            description="Patient says 'No'. System discards proposal, zero database writes.",
            is_recovery_scenario=True,
            turns=[
                TurnSpec(
                    user_input="Book an appointment with Dr. Ada Lovelace on Tuesday.",
                    mock_responses=ambig_setup,
                    expected_pending_after=True,
                    expected_awaiting_after=True,
                ),
                TurnSpec(
                    user_input="No.",
                    mock_responses=[],
                    expected_pending_after=False,
                    expected_awaiting_after=False,
                    expected_reply_contains=["cancelled that booking request"],
                    reply_must_not_contain=["successfully confirmed", "Booking #"],
                ),
            ],
            expected_booking=False,
        )
    )

    # CANC-02: Explicit "Don't book that"
    scenarios.append(
        ScenarioSpec(
            id="CANC-02",
            name="Explicit Verbal Cancellation ('Don't book that')",
            category=ScenarioCategory.NEGATION_CANCELLATION,
            description="Patient says 'Don't book that appointment'. Proposal discarded cleanly.",
            is_recovery_scenario=True,
            turns=[
                TurnSpec(
                    user_input="Book an appointment with Dr. Ada Lovelace on Tuesday.",
                    mock_responses=ambig_setup,
                    expected_pending_after=True,
                    expected_awaiting_after=True,
                ),
                TurnSpec(
                    user_input="Don't book that appointment.",
                    mock_responses=[],
                    expected_pending_after=False,
                    expected_awaiting_after=False,
                    expected_reply_contains=["cancelled that booking request"],
                    reply_must_not_contain=["successfully confirmed"],
                ),
            ],
            expected_booking=False,
        )
    )

    # CANC-03: Conversational "Never mind"
    scenarios.append(
        ScenarioSpec(
            id="CANC-03",
            name="Conversational Dismissal ('Never mind')",
            category=ScenarioCategory.NEGATION_CANCELLATION,
            description="Patient says 'Never mind, I will call back later'. Proposal discarded.",
            is_recovery_scenario=True,
            turns=[
                TurnSpec(
                    user_input="Book an appointment with Dr. Ada Lovelace on Tuesday.",
                    mock_responses=ambig_setup,
                    expected_pending_after=True,
                    expected_awaiting_after=True,
                ),
                TurnSpec(
                    user_input="Never mind.",
                    mock_responses=[],
                    expected_pending_after=False,
                    expected_awaiting_after=False,
                    expected_reply_contains=["cancelled that booking request"],
                ),
            ],
            expected_booking=False,
        )
    )

    # CANC-04: Explicit Cancel via API
    def canc_04_runner(spec: ScenarioSpec, db_path: Path) -> Any:
        from scheduler.agent import Agent, PendingBooking
        slot = _get_target_slot(db_path, doctor_id=1, slot_index=0)
        mock_client = MagicMock()
        agent = Agent(db_path=db_path, client=mock_client)
        agent.pending_booking = PendingBooking(
            slot_id=slot.id,
            doctor_name="Dr. Ada Lovelace",
            specialty="General Practice",
            start_utc=slot.start_utc,
            end_utc=slot.end_utc,
            local_time_str="Tuesday 9 AM",
            idempotency_key="test-key",
            patient_name="Alex Taylor",
        )
        was_pending, msg = agent.cancel_pending()
        assert was_pending is True
        assert "Discarded pending appointment" in msg
        assert agent.pending_booking is None
        conn = db._connect(db_path)
        count = conn.execute("SELECT count(*) FROM bookings").fetchone()[0]
        conn.close()
        assert count == 0
        return True

    scenarios.append(
        ScenarioSpec(
            id="CANC-04",
            name="Deterministic Cancel Method (cancel_pending)",
            category=ScenarioCategory.NEGATION_CANCELLATION,
            description="Calls agent.cancel_pending() directly; verifies clean proposal removal and zero bookings.",
            custom_runner=canc_04_runner,
            expected_booking=False,
        )
    )

    # =======================================================================
    # CATEGORY 5: Interruption During Confirmation (4 Scenarios)
    # =======================================================================

    # INTR-01: "Wait stop" during confirmation
    scenarios.append(
        ScenarioSpec(
            id="INTR-01",
            name="Interruption ('Wait stop, do not book')",
            category=ScenarioCategory.INTERRUPTIONS,
            description="Patient interrupts confirmation with 'Wait stop, do not book that'. Proposal discarded.",
            is_recovery_scenario=True,
            turns=[
                TurnSpec(
                    user_input="Book an appointment with Dr. Ada Lovelace on Tuesday.",
                    mock_responses=ambig_setup,
                    expected_pending_after=True,
                    expected_awaiting_after=True,
                ),
                TurnSpec(
                    user_input="Wait stop, do not book that.",
                    mock_responses=[],
                    expected_pending_after=False,
                    expected_awaiting_after=False,
                    expected_reply_contains=["cancelled that booking request"],
                    reply_must_not_contain=["successfully confirmed"],
                ),
            ],
            expected_booking=False,
        )
    )

    # INTR-02: "Actually change to 2 PM" during confirmation
    def intr_02_mocks(db_path: Path, turn_idx: int) -> list[Any]:
        slot2 = _get_target_slot(db_path, doctor_id=1, slot_index=1)
        return make_propose_flow(slot2.id, "Dr. Ada Lovelace", "Tuesday at 2:00 PM PST")

    scenarios.append(
        ScenarioSpec(
            id="INTR-02",
            name="Interruption with Slot Change ('Actually change to 2 PM')",
            category=ScenarioCategory.INTERRUPTIONS,
            description="Patient interrupts confirmation prompt with a modification to 2 PM.",
            is_recovery_scenario=True,
            turns=[
                TurnSpec(
                    user_input="Book an appointment with Dr. Ada Lovelace on Tuesday morning.",
                    mock_responses=ambig_setup,
                    expected_pending_after=True,
                    expected_awaiting_after=True,
                ),
                TurnSpec(
                    user_input="Wait, actually change to 2 PM instead.",
                    mock_responses=intr_02_mocks,
                    expected_pending_after=True,
                    expected_awaiting_after=True,
                    expected_reply_contains=["2:00 PM"],
                ),
                TurnSpec(
                    user_input="Yes, book it.",
                    mock_responses=[],
                    expected_pending_after=True,
                    expected_awaiting_after=False,
                    expected_reply_contains=["Appointment successfully confirmed"],
                ),
            ],
            expected_booking=True,
            expected_doctor="Dr. Ada Lovelace",
        )
    )

    # INTR-03: Informational Interruption ("Wait, what is her specialty?")
    def intr_03_mocks(db_path: Path, turn_idx: int) -> list[Any]:
        return [
            make_text_response("Dr. Ada Lovelace specializes in General Practice and preventive healthcare.")
        ]

    scenarios.append(
        ScenarioSpec(
            id="INTR-03",
            name="Informational Interruption ('Wait, what is her specialty?')",
            category=ScenarioCategory.INTERRUPTIONS,
            description="Patient asks a clinical question during confirmation. Slot remains uncommitted.",
            is_recovery_scenario=True,
            turns=[
                TurnSpec(
                    user_input="Book an appointment with Dr. Ada Lovelace on Tuesday.",
                    mock_responses=ambig_setup,
                    expected_pending_after=True,
                    expected_awaiting_after=True,
                ),
                TurnSpec(
                    user_input="Wait, what does Dr. Lovelace specialize in?",
                    mock_responses=intr_03_mocks,
                    expected_pending_after=False,
                    expected_awaiting_after=False,
                    expected_reply_contains=["General Practice"],
                    reply_must_not_contain=["successfully confirmed", "Booking #"],
                ),
            ],
            expected_booking=False,
        )
    )

    # INTR-04: Barge-in Noise / Garbled Speech During Confirmation
    scenarios.append(
        ScenarioSpec(
            id="INTR-04",
            name="Barge-in Noise During Confirmation",
            category=ScenarioCategory.INTERRUPTIONS,
            description="Ambient noise or irrelevant words spoken during prompt; clarifies without booking.",
            is_recovery_scenario=True,
            turns=[
                TurnSpec(
                    user_input="Book an appointment with Dr. Ada Lovelace on Tuesday.",
                    mock_responses=ambig_setup,
                    expected_pending_after=True,
                    expected_awaiting_after=True,
                ),
                TurnSpec(
                    user_input="cough phone keys jacket car",
                    mock_responses=[],
                    expected_pending_after=True,
                    expected_awaiting_after=True,
                    expected_reply_contains=["Just to confirm"],
                    reply_must_not_contain=["successfully confirmed"],
                ),
            ],
            expected_booking=False,
        )
    )

    # =======================================================================
    # CATEGORY 6: Duplicate and Competing Booking Requests (5 Scenarios)
    # =======================================================================

    # COMP-01: Concurrent Two-Patient Race for Same Slot
    def comp_01_runner(spec: ScenarioSpec, db_path: Path) -> Any:
        from scheduler.agent import Agent, PendingBooking
        slot = _get_target_slot(db_path, doctor_id=1, slot_index=0)

        agent1 = Agent(db_path=db_path, client=MagicMock(), patient_name="Patient 1")
        agent1.pending_booking = PendingBooking(
            slot_id=slot.id,
            doctor_name=slot.doctor_name or "Dr. Ada Lovelace",
            specialty=slot.specialty or "General Practice",
            start_utc=slot.start_utc,
            end_utc=slot.end_utc,
            local_time_str="Tuesday 9:00 AM",
            idempotency_key="key-patient-1",
            patient_name="Patient 1",
        )

        agent2 = Agent(db_path=db_path, client=MagicMock(), patient_name="Patient 2")
        agent2.pending_booking = PendingBooking(
            slot_id=slot.id,
            doctor_name=slot.doctor_name or "Dr. Ada Lovelace",
            specialty=slot.specialty or "General Practice",
            start_utc=slot.start_utc,
            end_utc=slot.end_utc,
            local_time_str="Tuesday 9:00 AM",
            idempotency_key="key-patient-2",
            patient_name="Patient 2",
        )

        s1, m1 = agent1.confirm_booking()
        s2, m2 = agent2.confirm_booking()

        assert s1 is True, f"Patient 1 should succeed: {m1}"
        assert s2 is False, f"Patient 2 should fail on booked slot: {m2}"
        assert "no longer available" in m2

        conn = db._connect(db_path)
        count = conn.execute("SELECT count(*) FROM bookings WHERE slot_id = ?", (slot.id,)).fetchone()[0]
        conn.close()
        assert count == 1, "Exactly one booking must exist in SQLite"
        return True

    scenarios.append(
        ScenarioSpec(
            id="COMP-01",
            name="Concurrent Two-Patient Race for Same Slot",
            category=ScenarioCategory.DUPLICATE_COMPETING,
            description="Two agents race to confirm the same slot. First succeeds, second receives clean unavailability message.",
            custom_runner=comp_01_runner,
            expected_booking=True,
            is_recovery_scenario=True,
        )
    )

    # COMP-02: Repeated Verbal Confirmation Idempotency
    def comp_02_runner(spec: ScenarioSpec, db_path: Path) -> Any:
        from scheduler.agent import Agent
        slot = _get_target_slot(db_path, doctor_id=1, slot_index=0)
        mock_client = MagicMock()
        mock_client.models.generate_content.side_effect = make_propose_flow(slot.id, "Dr. Ada Lovelace", "Tuesday at 9:00 AM PST")
        agent = Agent(db_path=db_path, client=mock_client)
        agent.turn("Book Dr. Lovelace")
        assert agent.pending_booking is not None
        ok1, msg1 = agent.confirm_booking()
        assert ok1 is True
        assert "successfully confirmed" in msg1
        ok2, msg2 = agent.confirm_booking()
        assert ok2 is True
        assert "already confirmed" in msg2
        conn = db._connect(db_path)
        count = conn.execute("SELECT count(*) FROM bookings").fetchone()[0]
        conn.close()
        assert count == 1
        return True

    scenarios.append(
        ScenarioSpec(
            id="COMP-02",
            name="Repeated Verbal Confirmation Idempotency",
            category=ScenarioCategory.DUPLICATE_COMPETING,
            description="Patient repeats confirmation; system returns existing receipt without duplicate rows.",
            custom_runner=comp_02_runner,
            expected_booking=True,
            expected_doctor="Dr. Ada Lovelace",
        )
    )

    # COMP-03: Same Slot Re-booking After Prior Cancellation
    def comp_03_runner(spec: ScenarioSpec, db_path: Path) -> Any:
        slot = _get_target_slot(db_path, doctor_id=1, slot_index=0)
        # 1. First booking
        b1 = db.book_slot(slot.id, "Alex Taylor", "key-first", db_path=db_path)
        assert b1.status == "confirmed"
        # 2. Cancel it
        b_canc = db.cancel_booking(b1.id, patient_name="Alex Taylor", db_path=db_path)
        assert b_canc.status == "cancelled"
        # 3. Verify slot is available again
        assert db.is_slot_available(slot.id, db_path=db_path) is True
        # 4. Re-book with new key
        b2 = db.book_slot(slot.id, "Alex Taylor", "key-second", db_path=db_path)
        assert b2.status == "confirmed"
        assert b2.id != b1.id
        return True

    scenarios.append(
        ScenarioSpec(
            id="COMP-03",
            name="Same Slot Re-booking After Prior Cancellation",
            category=ScenarioCategory.DUPLICATE_COMPETING,
            description="Verifies a slot can be booked again after an earlier booking for that slot was cancelled.",
            custom_runner=comp_03_runner,
            expected_booking=True,
        )
    )

    # COMP-04: Idempotency Key Conflict Protection
    def comp_04_runner(spec: ScenarioSpec, db_path: Path) -> Any:
        slots = db.list_available_slots(db_path=db_path)
        slot1, slot2 = slots[0], slots[1]
        shared_key = "idemp-shared-key"
        b1 = db.book_slot(slot1.id, "Alex Taylor", shared_key, db_path=db_path)
        assert b1.id is not None
        try:
            db.book_slot(slot2.id, "Alex Taylor", shared_key, db_path=db_path)
            raise AssertionError("Expected IDEMPOTENCY_CONFLICT")
        except db.BookingError as exc:
            assert exc.code == db.IDEMPOTENCY_CONFLICT
        return True

    scenarios.append(
        ScenarioSpec(
            id="COMP-04",
            name="Idempotency Key Conflict Protection",
            category=ScenarioCategory.DUPLICATE_COMPETING,
            description="Attempting to book a different slot with an existing idempotency key fails with IDEMPOTENCY_CONFLICT.",
            custom_runner=comp_04_runner,
            expected_booking=True,
        )
    )

    # COMP-05: Multi-Threaded 10-Worker Race on Last Slot
    def comp_05_runner(spec: ScenarioSpec, db_path: Path) -> Any:
        import concurrent.futures
        slot = _get_target_slot(db_path, doctor_id=1, slot_index=0)
        results = []

        def worker(i: int) -> tuple[str, int]:
            try:
                b = db.book_slot(slot.id, f"Patient {i}", f"key-worker-{i}", db_path=db_path)
                return ("success", b.id)
            except db.BookingError as exc:
                return ("error", exc.code)  # type: ignore[return-value]

        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as ex:
            futures = [ex.submit(worker, i) for i in range(10)]
            for f in concurrent.futures.as_completed(futures):
                results.append(f.result())

        successes = [r for r in results if r[0] == "success"]
        errors = [r for r in results if r[0] == "error"]

        assert len(successes) == 1, f"Expected 1 winner, got {len(successes)}"
        assert len(errors) == 9, f"Expected 9 errors, got {len(errors)}"
        assert all(code == db.SLOT_UNAVAILABLE for _, code in errors)

        conn = db._connect(db_path)
        count = conn.execute("SELECT count(*) FROM bookings WHERE slot_id = ?", (slot.id,)).fetchone()[0]
        conn.close()
        assert count == 1
        return True

    scenarios.append(
        ScenarioSpec(
            id="COMP-05",
            name="Multi-Threaded 10-Worker Race Condition Safety",
            category=ScenarioCategory.DUPLICATE_COMPETING,
            description="10 threads simultaneously attempt to book the same slot; atomic index guarantees exactly 1 booking.",
            custom_runner=comp_05_runner,
            expected_booking=True,
        )
    )

    # =======================================================================
    # CATEGORY 7: STT Transcription Errors (5 Scenarios)
    # =======================================================================

    # STT-01: Fuzzy Doctor Name ("Dr. Cabbage" -> Dr. Charles Babbage)
    def stt_01_mocks(db_path: Path, turn_idx: int) -> list[Any]:
        slot = _get_target_slot(db_path, doctor_id=2, slot_index=0)
        return [
            make_fn_call_response([("search_slots", {"doctor_id": 2})]),
            make_fn_call_response([("propose_booking", {"slot_id": slot.id})]),
            make_text_response("I believe you meant Dr. Charles Babbage. I have an opening Tuesday at 9:00 AM. Would you like me to book it?"),
        ]

    scenarios.append(
        ScenarioSpec(
            id="STT-01",
            name="Fuzzy Doctor Name ('Dr. Cabbage')",
            category=ScenarioCategory.STT_ERRORS,
            description="Transcribed audio says 'Dr. Cabbage'. Agent maps phonetically to Dr. Charles Babbage and proposes slot.",
            turns=[
                TurnSpec(
                    user_input="I want an appointment with Dr. Cabbage on Tuesday.",
                    mock_responses=stt_01_mocks,
                    expected_pending_after=True,
                    expected_awaiting_after=True,
                    expected_reply_contains=["Dr. Charles Babbage"],
                ),
                TurnSpec(
                    user_input="Yes, book it.",
                    mock_responses=[],
                    expected_pending_after=True,
                    expected_awaiting_after=False,
                    expected_reply_contains=["Appointment successfully confirmed", "Dr. Charles Babbage"],
                ),
            ],
            expected_booking=True,
            expected_doctor="Dr. Charles Babbage",
        )
    )

    # STT-02: Misheard Date Homophone ("to stay" -> Tuesday)
    def stt_02_mocks(db_path: Path, turn_idx: int) -> list[Any]:
        slot = _get_target_slot(db_path, doctor_id=1, slot_index=0)
        return make_propose_flow(slot.id, "Dr. Ada Lovelace", "Tuesday morning at 9:00 AM PST")

    scenarios.append(
        ScenarioSpec(
            id="STT-02",
            name="Misheard Date Homophone ('to stay' -> Tuesday)",
            category=ScenarioCategory.STT_ERRORS,
            description="Transcribed utterance 'to stay in the morning' interpreted as Tuesday morning.",
            turns=[
                TurnSpec(
                    user_input="Can I see Dr. Lovelace to stay in the morning?",
                    mock_responses=stt_02_mocks,
                    expected_pending_after=True,
                    expected_awaiting_after=True,
                ),
                TurnSpec(
                    user_input="Yes please.",
                    mock_responses=[],
                    expected_pending_after=True,
                    expected_awaiting_after=False,
                    expected_reply_contains=["Appointment successfully confirmed"],
                ),
            ],
            expected_booking=True,
            expected_doctor="Dr. Ada Lovelace",
        )
    )

    # STT-03: Phonetic Doctor Name ("Dr. Loveless" -> Dr. Ada Lovelace)
    scenarios.append(
        ScenarioSpec(
            id="STT-03",
            name="Phonetic Doctor Name ('Dr. Loveless')",
            category=ScenarioCategory.STT_ERRORS,
            description="Transcribed utterance 'Dr. Loveless' correctly matches Dr. Ada Lovelace.",
            turns=[
                TurnSpec(
                    user_input="Book with Dr. Loveless.",
                    mock_responses=stt_02_mocks,
                    expected_pending_after=True,
                    expected_awaiting_after=True,
                    expected_reply_contains=["Dr. Ada Lovelace"],
                ),
                TurnSpec(
                    user_input="Sounds good, book it.",
                    mock_responses=[],
                    expected_pending_after=True,
                    expected_awaiting_after=False,
                    expected_reply_contains=["Appointment successfully confirmed", "Dr. Ada Lovelace"],
                ),
            ],
            expected_booking=True,
            expected_doctor="Dr. Ada Lovelace",
        )
    )

    # STT-04: Garbled Background Noise
    def stt_04_mocks(db_path: Path, turn_idx: int) -> list[Any]:
        return [
            make_text_response("I'm sorry, I couldn't quite hear that. Which doctor or appointment time were you looking for?")
        ]

    scenarios.append(
        ScenarioSpec(
            id="STT-04",
            name="Garbled Background Noise Utterance",
            category=ScenarioCategory.STT_ERRORS,
            description="Transcribed audio contains corrupted gibberish; agent requests clarification without booking.",
            turns=[
                TurnSpec(
                    user_input="kzzz wrrr bzzz",
                    mock_responses=stt_04_mocks,
                    expected_pending_after=False,
                    expected_awaiting_after=False,
                    expected_reply_contains=["couldn't quite hear that"],
                    reply_must_not_contain=["successfully confirmed", "Booking #"],
                ),
            ],
            expected_booking=False,
        )
    )

    # STT-05: Unsupported Audio Mime Type / Corrupted Bytes
    def stt_05_runner(spec: ScenarioSpec, db_path: Path) -> Any:
        from scheduler.speech import transcribe_audio, SpeechProviderError, is_supported_audio_mime
        assert is_supported_audio_mime("image/png") is False
        assert is_supported_audio_mime("audio/wav") is True

        mock_resp = MagicMock()
        mock_resp.status_code = 415
        mock_resp.text = "Unsupported media type"
        mock_client = MagicMock()
        mock_client.post.return_value = mock_resp

        try:
            transcribe_audio(b"corrupted-bytes", content_type="image/png", api_key="test-key", client=mock_client)
            raise AssertionError("Expected SpeechProviderError for 415 response")
        except SpeechProviderError as exc:
            assert "415" in str(exc) or "Unsupported" in str(exc) or "rejected" in str(exc).lower()
        return True

    scenarios.append(
        ScenarioSpec(
            id="STT-05",
            name="Unsupported Audio MIME Type Error Handling",
            category=ScenarioCategory.STT_ERRORS,
            description="Verifies transcribe_audio cleanly rejects unsupported MIME types without unhandled crashes.",
            custom_runner=stt_05_runner,
            expected_booking=False,
        )
    )

    # =======================================================================
    # CATEGORY 8: Provider Timeouts and Failures (5 Scenarios)
    # =======================================================================

    # FAIL-01: Gemini API 503 Service Unavailable
    scenarios.append(
        ScenarioSpec(
            id="FAIL-01",
            name="Gemini API 503 Service Unavailable",
            category=ScenarioCategory.PROVIDER_FAILURES,
            description="Gemini API raises 503. Agent catches error, rolls back history, and returns friendly message.",
            is_recovery_scenario=True,
            turns=[
                TurnSpec(
                    user_input="I would like to schedule an appointment.",
                    mock_responses=Exception("503 Service Unavailable: High server load"),
                    expected_pending_after=False,
                    expected_awaiting_after=False,
                    expected_reply_contains=["trouble connecting to the appointment service", "Please try again"],
                    reply_must_not_contain=["successfully confirmed"],
                ),
            ],
            expected_booking=False,
        )
    )

    # FAIL-02: Gemini API Request Timeout
    scenarios.append(
        ScenarioSpec(
            id="FAIL-02",
            name="Gemini API Request Timeout",
            category=ScenarioCategory.PROVIDER_FAILURES,
            description="Gemini API call times out. State is rolled back cleanly without orphaned proposals.",
            is_recovery_scenario=True,
            turns=[
                TurnSpec(
                    user_input="Find an opening with Dr. Babbage.",
                    mock_responses=TimeoutError("Request deadline exceeded after 10.0s"),
                    expected_pending_after=False,
                    expected_awaiting_after=False,
                    expected_reply_contains=["trouble connecting to the appointment service"],
                ),
            ],
            expected_booking=False,
        )
    )

    # FAIL-03: Deepgram Speech STT Timeout
    def fail_03_runner(spec: ScenarioSpec, db_path: Path) -> Any:
        import httpx
        from scheduler.speech import transcribe_audio, SpeechTimeoutError

        mock_client = MagicMock()
        mock_client.post.side_effect = httpx.TimeoutException("Deepgram connection timed out")

        try:
            transcribe_audio(b"fake-audio-bytes", content_type="audio/wav", api_key="test-key", timeout=1.0, client=mock_client)
            raise AssertionError("Expected SpeechTimeoutError")
        except SpeechTimeoutError as exc:
            assert "timed out" in str(exc).lower()
        return True

    scenarios.append(
        ScenarioSpec(
            id="FAIL-03",
            name="Deepgram Speech STT Timeout",
            category=ScenarioCategory.PROVIDER_FAILURES,
            description="Upstream speech-to-text service times out; SpeechTimeoutError is caught cleanly.",
            custom_runner=fail_03_runner,
            is_recovery_scenario=True,
            expected_booking=False,
        )
    )

    # FAIL-04: Cartesia TTS Service Failure
    def fail_04_runner(spec: ScenarioSpec, db_path: Path) -> Any:
        from scheduler.tts import synthesize_speech, TTSProviderError

        mock_client = MagicMock()
        mock_client.tts.generate.side_effect = RuntimeError("Cartesia 500 Internal Server Error")

        try:
            synthesize_speech("Hello", api_key="test-key", client=mock_client)
            raise AssertionError("Expected TTSProviderError")
        except TTSProviderError as exc:
            assert "cartesia" in str(exc).lower() or "synthesize" in str(exc).lower()
        return True

    scenarios.append(
        ScenarioSpec(
            id="FAIL-04",
            name="Cartesia TTS Service Failure (500 Error)",
            category=ScenarioCategory.PROVIDER_FAILURES,
            description="Cartesia TTS upstream returns 500; TTSProviderError raised cleanly without crashing.",
            custom_runner=fail_04_runner,
            is_recovery_scenario=True,
            expected_booking=False,
        )
    )

    # FAIL-05: Transient Gemini Failure & Subsequent Recovery
    def fail_05_turn2_mocks(db_path: Path, turn_idx: int) -> list[Any]:
        slot = _get_target_slot(db_path, doctor_id=1, slot_index=0)
        return make_propose_flow(slot.id, "Dr. Ada Lovelace", "Tuesday at 9:00 AM PST")

    scenarios.append(
        ScenarioSpec(
            id="FAIL-05",
            name="Transient Gemini Failure and Subsequent Turn Recovery",
            category=ScenarioCategory.PROVIDER_FAILURES,
            description="First turn hits a transient 503 error; second turn succeeds and patient confirms.",
            is_recovery_scenario=True,
            turns=[
                TurnSpec(
                    user_input="Book Dr. Lovelace.",
                    mock_responses=Exception("503 Service Unavailable"),
                    expected_pending_after=False,
                    expected_awaiting_after=False,
                    expected_reply_contains=["trouble connecting"],
                ),
                TurnSpec(
                    user_input="Book Dr. Lovelace on Tuesday.",
                    mock_responses=fail_05_turn2_mocks,
                    expected_pending_after=True,
                    expected_awaiting_after=True,
                    expected_reply_contains=["Dr. Ada Lovelace"],
                ),
                TurnSpec(
                    user_input="Yes, book it.",
                    mock_responses=[],
                    expected_pending_after=True,
                    expected_awaiting_after=False,
                    expected_reply_contains=["Appointment successfully confirmed"],
                ),
            ],
            expected_booking=True,
            expected_doctor="Dr. Ada Lovelace",
        )
    )

    # =======================================================================
    # CATEGORY 9: Session Isolation and State Recovery (5 Scenarios)
    # =======================================================================

    # SESS-01: Multi-Session State Isolation
    def sess_01_runner(spec: ScenarioSpec, db_path: Path) -> Any:
        from scheduler.agent import Agent
        slot1 = _get_target_slot(db_path, doctor_id=1, slot_index=0)
        slot2 = _get_target_slot(db_path, doctor_id=2, slot_index=0)

        # Agent A stages Dr. Lovelace
        agent_a = Agent(db_path=db_path, client=MagicMock(), patient_name="Patient A")
        agent_a._dispatch("propose_booking", {"slot_id": slot1.id})
        assert agent_a.pending_booking is not None
        assert agent_a.pending_booking.slot_id == slot1.id

        # Agent B stages Dr. Babbage
        agent_b = Agent(db_path=db_path, client=MagicMock(), patient_name="Patient B")
        agent_b._dispatch("propose_booking", {"slot_id": slot2.id})
        assert agent_b.pending_booking is not None
        assert agent_b.pending_booking.slot_id == slot2.id

        # Agent A confirms
        succ_a, msg_a = agent_a.confirm_booking()
        assert succ_a is True
        # Agent B proposal is completely unaffected
        assert agent_b.pending_booking.status == "awaiting_confirmation"
        assert agent_b.pending_booking.slot_id == slot2.id

        # Agent B confirms
        succ_b, msg_b = agent_b.confirm_booking()
        assert succ_b is True

        conn = db._connect(db_path)
        count = conn.execute("SELECT count(*) FROM bookings WHERE status = 'confirmed'").fetchone()[0]
        conn.close()
        assert count == 2
        return True

    scenarios.append(
        ScenarioSpec(
            id="SESS-01",
            name="Multi-Session State Isolation",
            category=ScenarioCategory.SESSION_ISOLATION_RECOVERY,
            description="Two concurrent sessions maintain completely isolated proposals without state leakage.",
            custom_runner=sess_01_runner,
            expected_booking=True,
            expected_booking_count=2,
        )
    )

    # SESS-02: State Recovery After Informational Question Turn
    def sess_02_turn2_mocks(db_path: Path, turn_idx: int) -> list[Any]:
        return [
            make_text_response("Appointment Scheduler is located at 100 Main St, Suite 400, San Francisco.")
        ]

    scenarios.append(
        ScenarioSpec(
            id="SESS-02",
            name="State Recovery After Informational Question Turn",
            category=ScenarioCategory.SESSION_ISOLATION_RECOVERY,
            description="Patient stages slot, asks 'Where is the clinic located?', answers question, then patient confirms.",
            is_recovery_scenario=True,
            turns=[
                TurnSpec(
                    user_input="Book an appointment with Dr. Ada Lovelace on Tuesday.",
                    mock_responses=ambig_setup,
                    expected_pending_after=True,
                    expected_awaiting_after=True,
                ),
                TurnSpec(
                    user_input="Where is the clinic located?",
                    mock_responses=sess_02_turn2_mocks,
                    expected_pending_after=False,
                    expected_awaiting_after=False,
                    expected_reply_contains=["100 Main St"],
                ),
                TurnSpec(
                    user_input="Yes, please book it.",
                    mock_responses=[],
                    expected_pending_after=True,
                    expected_awaiting_after=False,
                    expected_reply_contains=["Appointment successfully confirmed", "Dr. Ada Lovelace"],
                ),
            ],
            expected_booking=True,
            expected_doctor="Dr. Ada Lovelace",
        )
    )

    # SESS-03: Informational Interruption with New Slot Stage
    def sess_03_turn3_mocks(db_path: Path, turn_idx: int) -> list[Any]:
        slot2 = _get_target_slot(db_path, doctor_id=3, slot_index=0)
        return make_propose_flow(slot2.id, "Dr. Florence Nightingale", "Tuesday morning at 9:00 AM PST")

    scenarios.append(
        ScenarioSpec(
            id="SESS-03",
            name="Informational Interruption with New Slot Proposal",
            category=ScenarioCategory.SESSION_ISOLATION_RECOVERY,
            description="Patient stages Dr. Lovelace, asks clinic question, switches to Dr. Nightingale, confirms Nightingale.",
            is_recovery_scenario=True,
            turns=[
                TurnSpec(
                    user_input="Book Dr. Lovelace on Tuesday.",
                    mock_responses=ambig_setup,
                    expected_pending_after=True,
                    expected_awaiting_after=True,
                ),
                TurnSpec(
                    user_input="What insurance plans do you accept?",
                    mock_responses=[make_text_response("We accept most major PPO and Medicare insurance plans.")],
                    expected_pending_after=False,
                    expected_awaiting_after=False,
                ),
                TurnSpec(
                    user_input="Actually, can I book Dr. Florence Nightingale instead?",
                    mock_responses=sess_03_turn3_mocks,
                    expected_pending_after=True,
                    expected_awaiting_after=True,
                    expected_reply_contains=["Dr. Florence Nightingale"],
                ),
                TurnSpec(
                    user_input="Yes, book it.",
                    mock_responses=[],
                    expected_pending_after=True,
                    expected_awaiting_after=False,
                    expected_reply_contains=["Appointment successfully confirmed", "Dr. Florence Nightingale"],
                ),
            ],
            expected_booking=True,
            expected_doctor="Dr. Florence Nightingale",
        )
    )

    # SESS-04: Session Expiry / Unknown Session ID Handled Gracefully
    def sess_04_runner(spec: ScenarioSpec, db_path: Path) -> Any:
        from fastapi.testclient import TestClient
        from scheduler.web import app
        client = TestClient(app)
        resp = client.post("/api/sessions/non-existent-session-id/messages", json={"text": "Hello"})
        assert resp.status_code == 404
        assert "not found" in resp.json()["detail"].lower()
        return True

    scenarios.append(
        ScenarioSpec(
            id="SESS-04",
            name="Session Expiry and Unknown Session ID Handling",
            category=ScenarioCategory.SESSION_ISOLATION_RECOVERY,
            description="Web client sends message to an unknown session ID; returns HTTP 404 without crashing.",
            custom_runner=sess_04_runner,
            is_recovery_scenario=True,
            expected_booking=False,
        )
    )

    # SESS-05: Session History Rollback on Turn Failure
    def sess_05_runner(spec: ScenarioSpec, db_path: Path) -> Any:
        from scheduler.agent import Agent
        mock_client = MagicMock()
        mock_client.models.generate_content.side_effect = RuntimeError("Simulated crash mid-turn")

        agent = Agent(db_path=db_path, client=mock_client)
        assert len(agent.history) == 0

        reply = agent.turn("Book appointment")
        assert "trouble connecting" in reply
        # History must be rolled back to 0
        assert len(agent.history) == 0
        assert agent.pending_booking is None
        return True

    scenarios.append(
        ScenarioSpec(
            id="SESS-05",
            name="Session History Rollback on Turn Failure",
            category=ScenarioCategory.SESSION_ISOLATION_RECOVERY,
            description="Verifies Agent conversation history is cleanly rolled back if an exception occurs mid-turn.",
            custom_runner=sess_05_runner,
            is_recovery_scenario=True,
            expected_booking=False,
        )
    )

    return scenarios


# ---------------------------------------------------------------------------
# Public Query Functions
# ---------------------------------------------------------------------------


def get_all_scenarios() -> list[ScenarioSpec]:
    """Return all 42 reliability and evaluation scenarios."""
    return build_scenarios()


def get_scenarios_by_category(category: Union[ScenarioCategory, str]) -> list[ScenarioSpec]:
    """Filter scenarios by category enum or string name."""
    cat_str = category.value if isinstance(category, ScenarioCategory) else str(category)
    return [s for s in build_scenarios() if s.category.value == cat_str]


def get_scenario_by_id(scenario_id: str) -> Optional[ScenarioSpec]:
    """Retrieve a single scenario by its unique identifier (e.g. 'NORM-01')."""
    target = scenario_id.strip().upper()
    for s in build_scenarios():
        if s.id.upper() == target:
            return s
    return None
