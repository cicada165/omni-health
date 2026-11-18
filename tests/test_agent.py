"""
tests/test_agent.py — Unit tests for the Gemini scheduling agent (Milestones 2A & 2B).

All Gemini API calls are mocked; no network or API key required.
"""

from __future__ import annotations

import dataclasses
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from google.genai import types

from scheduler.agent import Agent, TOOLS, _dispatch, _ALLOWLIST
from scheduler.db import (
    _connect,
    book_slot,
    cancel_booking,
    get_slot,
    init_db,
    is_slot_available,
    list_available_slots,
    list_doctors,
    seed,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def db(tmp_path: Path) -> Path:
    db_path = tmp_path / "test.db"
    init_db(db_path)
    seed(db_path)
    return db_path


# ---------------------------------------------------------------------------
# Helpers — mock Gemini responses
# ---------------------------------------------------------------------------


def _make_text_response(text: str) -> MagicMock:
    """Build a mock GenerateContentResponse that has only text, no calls."""
    part = types.Part.from_text(text=text)
    content = types.Content(role="model", parts=[part])

    candidate = MagicMock()
    candidate.content = content

    response = MagicMock()
    response.candidates = [candidate]
    response.function_calls = []
    response.text = text
    return response


def _make_fn_call_response(
    calls: list[tuple[str, dict[str, Any]] | tuple[str, dict[str, Any], str]],
    text_alongside: str | None = None,
) -> MagicMock:
    """Build a mock response containing one or more function calls."""
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


# ---------------------------------------------------------------------------
# Dispatch tests (pure function validation)
# ---------------------------------------------------------------------------


class TestDispatch:
    def test_list_doctors(self, db: Path) -> None:
        result = _dispatch("list_doctors", {}, db)
        assert "doctors" in result
        assert len(result["doctors"]) == 3

    def test_search_slots_no_filter(self, db: Path) -> None:
        result = _dispatch("search_slots", {}, db)
        assert "slots" in result
        assert len(result["slots"]) > 0

    def test_search_slots_by_doctor(self, db: Path) -> None:
        doctors = list_doctors(db_path=db)
        doc_id = doctors[0].id
        result = _dispatch("search_slots", {"doctor_id": doc_id}, db)
        assert all(s["doctor_id"] == doc_id for s in result["slots"])

    def test_search_slots_bad_date(self, db: Path) -> None:
        result = _dispatch("search_slots", {"date": "not-a-date"}, db)
        assert "error" in result

    def test_propose_booking_valid(self, db: Path) -> None:
        mock_client = MagicMock()
        agent = Agent(db_path=db, client=mock_client)
        slots = list_available_slots(db_path=db)
        target = slots[0]

        result = _dispatch("propose_booking", {"slot_id": target.id}, db, agent=agent)
        assert result["status"] == "awaiting_confirmation"
        assert result["slot_id"] == target.id
        assert agent.pending_booking is not None
        assert agent.pending_booking.slot_id == target.id

        # Verify proposal did NOT write to SQLite bookings
        conn = _connect(db)
        count = conn.execute("SELECT count(*) FROM bookings").fetchone()[0]
        conn.close()
        assert count == 0

    def test_propose_booking_slot_not_found(self, db: Path) -> None:
        result = _dispatch("propose_booking", {"slot_id": 999_999}, db)
        assert "error" in result
        assert "does not exist" in result["error"]

    def test_propose_booking_missing_slot_id(self, db: Path) -> None:
        result = _dispatch("propose_booking", {}, db)
        assert "error" in result

    def test_unknown_function_raises(self, db: Path) -> None:
        with pytest.raises(ValueError, match="not allowlisted"):
            _dispatch("drop_tables", {}, db)

    def test_allowlist_safety(self) -> None:
        """Neither book_slot nor cancel_booking should be in the LLM allowlist."""
        assert "book_slot" not in _ALLOWLIST
        assert "cancel_booking" not in _ALLOWLIST
        assert "propose_booking" in _ALLOWLIST


# ---------------------------------------------------------------------------
# Agent turn — basic conversational flows
# ---------------------------------------------------------------------------


class TestAgentTextOnly:
    def test_simple_greeting(self, db: Path) -> None:
        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = _make_text_response(
            "Hello! How can I help you today?"
        )

        agent = Agent(db_path=db, client=mock_client)
        reply = agent.turn("Hi")

        assert "Hello" in reply
        assert mock_client.models.generate_content.call_count == 1

    def test_history_grows(self, db: Path) -> None:
        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = _make_text_response("Sure!")

        agent = Agent(db_path=db, client=mock_client)
        agent.turn("Hi")

        assert len(agent.history) == 2
        assert agent.history[0].role == "user"
        assert agent.history[1].role == "model"


class TestAgentSingleToolCall:
    def test_list_doctors_flow(self, db: Path) -> None:
        mock_client = MagicMock()
        mock_client.models.generate_content.side_effect = [
            _make_fn_call_response([("list_doctors", {})]),
            _make_text_response("We have 3 doctors: Ada, Charles, and Florence."),
        ]

        agent = Agent(db_path=db, client=mock_client)
        reply = agent.turn("Who are the doctors?")

        assert "3 doctors" in reply
        assert mock_client.models.generate_content.call_count == 2
        assert len(agent.history) == 4

    def test_search_slots_flow(self, db: Path) -> None:
        doctors = list_doctors(db_path=db)
        doc_id = doctors[0].id

        mock_client = MagicMock()
        mock_client.models.generate_content.side_effect = [
            _make_fn_call_response([("search_slots", {"doctor_id": doc_id})]),
            _make_text_response("Dr. Lovelace has available slots."),
        ]

        agent = Agent(db_path=db, client=mock_client)
        reply = agent.turn(f"What slots does doctor {doc_id} have?")

        assert "slots" in reply.lower()
        assert mock_client.models.generate_content.call_count == 2


class TestAgentParallelCalls:
    def test_two_calls_in_one_round(self, db: Path) -> None:
        mock_client = MagicMock()
        mock_client.models.generate_content.side_effect = [
            _make_fn_call_response([
                ("list_doctors", {}),
                ("search_slots", {}),
            ]),
            _make_text_response("We have 3 doctors and 30 slots available."),
        ]

        agent = Agent(db_path=db, client=mock_client)
        reply = agent.turn("Show me everything")

        assert mock_client.models.generate_content.call_count == 2
        tool_response_content = agent.history[2]
        assert len(tool_response_content.parts) == 2


class TestAgentThoughtPreservation:
    def test_thought_text_in_history(self, db: Path) -> None:
        mock_client = MagicMock()
        mock_client.models.generate_content.side_effect = [
            _make_fn_call_response(
                [("list_doctors", {})],
                text_alongside="Let me look that up...",
            ),
            _make_text_response("Here are the doctors."),
        ]

        agent = Agent(db_path=db, client=mock_client)
        agent.turn("Who are the doctors?")

        model_turn = agent.history[1]
        assert model_turn.role == "model"
        texts = [p.text for p in model_turn.parts if p.text]
        assert any("look that up" in t for t in texts)


# ---------------------------------------------------------------------------
# Milestone 2B Booking Confirmation Workflow Tests
# ---------------------------------------------------------------------------


class TestBookingWorkflow:
    def test_proposal_then_confirmation_flow(self, db: Path) -> None:
        """Proposal stages slot -> /confirm commits booking to SQLite."""
        slots = list_available_slots(db_path=db)
        target = slots[0]

        mock_client = MagicMock()
        mock_client.models.generate_content.side_effect = [
            _make_fn_call_response([("propose_booking", {"slot_id": target.id})]),
            _make_text_response("I have staged this slot for you. Please type /confirm to finalize."),
        ]

        agent = Agent(db_path=db, client=mock_client)
        reply = agent.turn("I'd like the 9am slot.")

        # 1. Verify confirmation banner attached to reply
        assert "Pending Confirmation" in reply
        assert str(target.id) in reply or target.doctor_name in reply

        # 2. Verify NO database booking occurred yet
        conn = _connect(db)
        count = conn.execute("SELECT count(*) FROM bookings").fetchone()[0]
        conn.close()
        assert count == 0
        assert agent.pending_booking is not None
        assert agent.pending_booking.status == "awaiting_confirmation"

        # 3. User types /confirm -> committed in SQLite
        success, confirm_msg = agent.confirm_booking()
        assert success is True
        assert "Appointment successfully confirmed" in confirm_msg
        assert "Booking ID" in confirm_msg

        # 4. Verify exactly 1 booking exists in SQLite
        conn = _connect(db)
        row = conn.execute("SELECT * FROM bookings").fetchone()
        conn.close()
        assert row is not None
        assert row["slot_id"] == target.id
        assert row["status"] == "confirmed"
        assert row["patient_name"] == agent.patient_name

    def test_no_booking_before_confirmation(self, db: Path) -> None:
        """Verifies propose_booking never writes a row to SQLite."""
        slots = list_available_slots(db_path=db)
        target = slots[0]

        mock_client = MagicMock()
        mock_client.models.generate_content.side_effect = [
            _make_fn_call_response([("propose_booking", {"slot_id": target.id})]),
            _make_text_response("Staged."),
        ]

        agent = Agent(db_path=db, client=mock_client)
        agent.turn("Book slot 1 please")

        conn = _connect(db)
        count = conn.execute("SELECT count(*) FROM bookings").fetchone()[0]
        conn.close()
        assert count == 0

    def test_confirm_with_no_pending_selection(self, db: Path) -> None:
        """/confirm fails gracefully when nothing has been proposed."""
        agent = Agent(db_path=db, client=MagicMock())
        success, msg = agent.confirm_booking()
        assert success is False
        assert "No pending appointment" in msg

    def test_cancel_discards_pending_without_db_change(self, db: Path) -> None:
        """/cancel discards the staged selection without touching SQLite."""
        slots = list_available_slots(db_path=db)
        target = slots[0]

        mock_client = MagicMock()
        mock_client.models.generate_content.side_effect = [
            _make_fn_call_response([("propose_booking", {"slot_id": target.id})]),
            _make_text_response("Staged."),
        ]

        agent = Agent(db_path=db, client=mock_client)
        agent.turn("I want that slot")
        assert agent.pending_booking is not None

        # Execute /cancel
        was_pending, cancel_msg = agent.cancel_pending()
        assert was_pending is True
        assert "Discarded pending appointment selection" in cancel_msg
        assert agent.pending_booking is None

        # Verify DB untouched
        conn = _connect(db)
        count = conn.execute("SELECT count(*) FROM bookings").fetchone()[0]
        conn.close()
        assert count == 0

    def test_cancel_with_no_pending_selection(self, db: Path) -> None:
        agent = Agent(db_path=db, client=MagicMock())
        was_pending, msg = agent.cancel_pending()
        assert was_pending is False
        assert "No pending appointment" in msg

    def test_different_slot_selection_invalidates_old_intent(self, db: Path) -> None:
        """Selecting a new slot replaces the pending booking and generates a new idempotency key."""
        slots = list_available_slots(db_path=db)
        slot1 = slots[0]
        slot2 = slots[1]

        mock_client = MagicMock()
        mock_client.models.generate_content.side_effect = [
            # Turn 1: propose slot 1
            _make_fn_call_response([("propose_booking", {"slot_id": slot1.id})]),
            _make_text_response("Staged slot 1."),
            # Turn 2: change mind, propose slot 2
            _make_fn_call_response([("propose_booking", {"slot_id": slot2.id})]),
            _make_text_response("Staged slot 2 instead."),
        ]

        agent = Agent(db_path=db, client=mock_client)
        agent.turn("I want slot 1")
        assert agent.pending_booking.slot_id == slot1.id
        key1 = agent.pending_booking.idempotency_key

        agent.turn("Actually I prefer slot 2")
        assert agent.pending_booking.slot_id == slot2.id
        key2 = agent.pending_booking.idempotency_key
        assert key1 != key2

        # Confirm commits slot2, NOT slot1
        success, _ = agent.confirm_booking()
        assert success is True

        conn = _connect(db)
        bookings = conn.execute("SELECT * FROM bookings").fetchall()
        conn.close()

        assert len(bookings) == 1
        assert bookings[0]["slot_id"] == slot2.id

        # Slot 1 remains completely available
        assert is_slot_available(slot1.id, db_path=db) is True

    def test_duplicate_confirm_does_not_duplicate_booking(self, db: Path) -> None:
        """Calling /confirm repeatedly returns the existing booking safely."""
        slots = list_available_slots(db_path=db)
        target = slots[0]

        mock_client = MagicMock()
        mock_client.models.generate_content.side_effect = [
            _make_fn_call_response([("propose_booking", {"slot_id": target.id})]),
            _make_text_response("Staged."),
        ]

        agent = Agent(db_path=db, client=mock_client)
        agent.turn("Slot 1 please")

        # Confirm 1st time
        ok1, msg1 = agent.confirm_booking()
        assert ok1 is True
        assert "successfully confirmed" in msg1

        # Confirm 2nd time
        ok2, msg2 = agent.confirm_booking()
        assert ok2 is True
        assert "already confirmed" in msg2

        conn = _connect(db)
        count = conn.execute("SELECT count(*) FROM bookings").fetchone()[0]
        conn.close()
        assert count == 1

    def test_slot_taken_before_confirmation(self, db: Path) -> None:
        """If a slot is booked by someone else between proposal and /confirm, fail safely."""
        slots = list_available_slots(db_path=db)
        target = slots[0]

        mock_client = MagicMock()
        mock_client.models.generate_content.side_effect = [
            _make_fn_call_response([("propose_booking", {"slot_id": target.id})]),
            _make_text_response("Staged."),
        ]

        agent = Agent(db_path=db, client=mock_client)
        agent.turn("Slot 1 please")
        assert agent.pending_booking is not None

        # Another patient books the slot in SQLite right before Alex confirms
        book_slot(target.id, "Competitor Patient", idempotency_key="comp-key-1", db_path=db)

        # Alex now types /confirm
        success, msg = agent.confirm_booking()
        assert success is False
        assert "no longer available" in msg
        assert agent.pending_booking is None

    def test_reject_ambiguous_simultaneous_proposals(self, db: Path) -> None:
        """If Gemini attempts parallel propose_booking calls, reject both and clear state."""
        slots = list_available_slots(db_path=db)
        mock_client = MagicMock()
        mock_client.models.generate_content.side_effect = [
            _make_fn_call_response([
                ("propose_booking", {"slot_id": slots[0].id}),
                ("propose_booking", {"slot_id": slots[1].id}),
            ]),
            _make_text_response("Which of the two slots do you prefer?"),
        ]

        agent = Agent(db_path=db, client=mock_client)
        reply = agent.turn("Book either 9am or 2pm")

        # Staged pending booking must be None (not choosing one silently)
        assert agent.pending_booking is None

        # Verify tool responses received ambiguity error
        tool_turn = agent.history[2]
        for part in tool_turn.parts:
            assert "Ambiguous proposal" in str(part.function_response.response)

    def test_call_id_preserved_in_function_response(self, db: Path) -> None:
        """Verify function-call ID from Gemini is attached to the FunctionResponse Part."""
        mock_client = MagicMock()
        mock_client.models.generate_content.side_effect = [
            _make_fn_call_response([("list_doctors", {}, "call_xyz_123")]),
            _make_text_response("Here are the doctors."),
        ]

        agent = Agent(db_path=db, client=mock_client)
        agent.turn("Who are the doctors?")

        tool_turn = agent.history[2]
        part = tool_turn.parts[0]
        assert part.function_response.id == "call_xyz_123"


# ---------------------------------------------------------------------------
# Reliability Hardening Regression Tests (FIX 1, 2, 3, 4)
# ---------------------------------------------------------------------------


class TestReliabilityHardening:
    # --- FIX 1: Gemini API failures ---

    def test_turn_api_timeout_recovers_gracefully(self, db: Path) -> None:
        """
        FIX 1: Connection timeouts must return an actionable error message,
        prevent orphaned history messages, and allow subsequent turns to succeed.
        """
        mock_client = MagicMock()
        mock_client.models.generate_content.side_effect = TimeoutError("Gemini API connection timed out")

        agent = Agent(db_path=db, client=mock_client)
        reply = agent.turn("Hello")

        # Actionable error message returned, CLI not crashed
        assert "trouble connecting" in reply or "try again" in reply
        # History rolled back: no orphaned user message left behind
        assert len(agent.history) == 0

        # Subsequent turn with working client succeeds normally
        mock_client.models.generate_content.side_effect = None
        mock_client.models.generate_content.return_value = _make_text_response("Hello! I am back online.")
        reply2 = agent.turn("Hello again")
        assert "online" in reply2
        assert len(agent.history) == 2

    def test_turn_api_error_fails_closed_on_partial_proposal(self, db: Path) -> None:
        """
        FIX 1: If an API error happens after a tool proposal has run,
        fail closed: discard the partially staged proposal and roll back history.
        """
        slots = list_available_slots(db_path=db)
        target = slots[0]

        mock_client = MagicMock()
        mock_client.models.generate_content.side_effect = [
            # Round 0: propose slot 1
            _make_fn_call_response([("propose_booking", {"slot_id": target.id})]),
            # Round 1: transient failure
            Exception("503 Service Unavailable"),
        ]

        agent = Agent(db_path=db, client=mock_client)
        reply = agent.turn("Book slot 1 please")

        assert "trouble connecting" in reply
        # Fails closed: pending booking must NOT linger
        assert agent.pending_booking is None
        # History rolled back cleanly
        assert len(agent.history) == 0

        # /confirm must fail safely
        success, msg = agent.confirm_booking()
        assert success is False
        assert "No pending appointment" in msg

    # --- FIX 2: False confirmation ---

    def test_confirm_cancel_then_repeat_confirm(self, db: Path) -> None:
        """
        FIX 2: Confirm a slot, cancel via db.cancel_booking(), then repeat /confirm.
        Must check current DB status, report failure, and NEVER report a cancelled
        booking as confirmed.
        """
        slots = list_available_slots(db_path=db)
        target = slots[0]

        mock_client = MagicMock()
        mock_client.models.generate_content.side_effect = [
            _make_fn_call_response([("propose_booking", {"slot_id": target.id})]),
            _make_text_response("Staged."),
        ]

        agent = Agent(db_path=db, client=mock_client)
        agent.turn("Book slot 1")

        # 1. Initial confirmation succeeds
        ok1, msg1 = agent.confirm_booking()
        assert ok1 is True
        assert "successfully confirmed" in msg1
        booking_id = agent.pending_booking.booking_id
        assert booking_id is not None

        # 2. Receptionist / system cancels the booking directly in database
        cancelled_booking = cancel_booking(
            booking_id=booking_id,
            patient_name=agent.patient_name,
            db_path=db,
        )
        assert cancelled_booking.status == "cancelled"

        # 3. Patient repeats /confirm
        ok2, msg2 = agent.confirm_booking()
        assert ok2 is False
        assert "cancelled" in msg2
        assert "already confirmed" not in msg2
        assert agent.pending_booking.status == "cancelled"

    # --- FIX 3: Stale pending selections ---

    def test_new_free_text_turn_invalidates_stale_proposal(self, db: Path) -> None:
        """
        FIX 3: A pending proposal awaiting confirmation must be invalidated
        if the user starts a new free-text turn without confirming.
        """
        slots = list_available_slots(db_path=db)
        target = slots[0]

        mock_client = MagicMock()
        mock_client.models.generate_content.side_effect = [
            # Turn 1: stage slot 1
            _make_fn_call_response([("propose_booking", {"slot_id": target.id})]),
            _make_text_response("Staged slot 1 for you."),
            # Turn 2: unrelated question (no booking tool called)
            _make_text_response("Dr. Ada Lovelace is in General Practice."),
        ]

        agent = Agent(db_path=db, client=mock_client)
        reply1 = agent.turn("I want slot 1")
        assert agent.pending_booking is not None
        assert agent.pending_booking.slot_id == target.id

        # Turn 2: User changes topic
        reply2 = agent.turn("Wait, what does Dr. Lovelace specialize in?")
        assert "General Practice" in reply2
        # Proposal must have been invalidated!
        assert agent.pending_booking is None

        # /confirm now must fail safely rather than confirming the stale slot
        success, msg = agent.confirm_booking()
        assert success is False
        assert "No pending appointment" in msg

    def test_direct_confirm_and_cancel_flow_preserved(self, db: Path) -> None:
        """
        FIX 3: Direct /confirm and /cancel do not pass through turn() and
        remain fully functional.
        """
        slots = list_available_slots(db_path=db)
        target = slots[0]

        mock_client = MagicMock()
        mock_client.models.generate_content.side_effect = [
            _make_fn_call_response([("propose_booking", {"slot_id": target.id})]),
            _make_text_response("Staged."),
        ]

        agent = Agent(db_path=db, client=mock_client)
        agent.turn("Stage slot 1")
        assert agent.pending_booking is not None

        # Immediate confirmation works without being invalidated
        ok, msg = agent.confirm_booking()
        assert ok is True
        assert "successfully confirmed" in msg

    # --- FIX 4: Ambiguous proposals across rounds ---

    def test_sequential_rounds_different_proposals_rejected(self, db: Path) -> None:
        """
        FIX 4: If Gemini proposes different slots across sequential tool-call rounds
        within the same turn, reject both and do not silently stage the last one.
        """
        slots = list_available_slots(db_path=db)
        slot1 = slots[0]
        slot2 = slots[1]

        mock_client = MagicMock()
        mock_client.models.generate_content.side_effect = [
            # Round 0: proposes slot 1
            _make_fn_call_response([("propose_booking", {"slot_id": slot1.id})]),
            # Round 1: model proposes slot 2 sequentially in the same user turn
            _make_fn_call_response([("propose_booking", {"slot_id": slot2.id})]),
            # Round 2: model realizes and asks user
            _make_text_response("Which slot do you prefer?"),
        ]

        agent = Agent(db_path=db, client=mock_client)
        reply = agent.turn("I'm interested in slot 1 and slot 2")

        # Crucial check: neither slot1 nor slot2 is staged!
        assert agent.pending_booking is None

        # Verify tool response for round 1 reported ambiguity
        # history: [0:user, 1:model_round0, 2:tool_round0, 3:model_round1, 4:tool_round1, 5:model_round2]
        tool_round1 = agent.history[4]
        assert "Ambiguous proposal" in str(tool_round1.parts[0].function_response.response)

        # /confirm must fail safely
        success, msg = agent.confirm_booking()
        assert success is False
        assert "No pending appointment" in msg


# ---------------------------------------------------------------------------
# Safety & Configuration Tests
# ---------------------------------------------------------------------------


class TestAgentSafetyAndConfig:
    def test_max_rounds_reached(self, db: Path) -> None:
        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = _make_fn_call_response(
            [("list_doctors", {})]
        )

        agent = Agent(db_path=db, client=mock_client)
        reply = agent.turn("infinite loop?")
        assert reply == "(max tool rounds reached)"

    def test_model_calls_unknown_function(self, db: Path) -> None:
        mock_client = MagicMock()
        mock_client.models.generate_content.side_effect = [
            _make_fn_call_response([("delete_everything", {})]),
            _make_text_response("Sorry, I can only look up doctors and slots."),
        ]

        agent = Agent(db_path=db, client=mock_client)
        reply = agent.turn("Delete the database")
        assert "sorry" in reply.lower() or "only" in reply.lower()

    def test_default_model(self, db: Path) -> None:
        mock_client = MagicMock()
        with patch("scheduler.agent.load_dotenv"), patch.dict("os.environ", {}, clear=True):
            agent = Agent(db_path=db, client=mock_client)
            assert agent.model == "gemini-3.8-flash"

    def test_env_model_override(self, db: Path) -> None:
        mock_client = MagicMock()
        with patch.dict("os.environ", {"GEMINI_MODEL": "gemini-custom-model"}):
            agent = Agent(db_path=db, client=mock_client)
            assert agent.model == "gemini-custom-model"

    def test_explicit_model_precedence(self, db: Path) -> None:
        mock_client = MagicMock()
        with patch.dict("os.environ", {"GEMINI_MODEL": "gemini-env-model"}):
            agent = Agent(db_path=db, model="gemini-explicit", client=mock_client)
            assert agent.model == "gemini-explicit"
