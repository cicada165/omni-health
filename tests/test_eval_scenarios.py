"""
tests/test_eval_scenarios.py — Automated Reliability & Evaluation Test Suite (Milestone 8).

Validates:
- Execution of all 42 conversational scheduling scenarios.
- Hard safety invariants:
    1. Unsafe Booking Rate == 0.0% (ambiguity, cancellations, and failures never commit).
    2. False Verbal Confirmation Rate == 0.0% (agent never confirms without SQLite booking).
    3. Race condition prevention (concurrent bookings on the same slot never double-book).
    4. Repeated verbal confirmation idempotency (never creates duplicate booking rows).
    5. Clean conversation history rollback on provider timeouts/503 errors.
    6. Strict multi-session state isolation.
"""

from __future__ import annotations

import concurrent.futures
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from google.genai import types

from evals.metrics import MetricsCalculator
from evals.runner import EvalRunner
from evals.scenarios import (
    ScenarioCategory,
    get_all_scenarios,
    get_scenario_by_id,
    get_scenarios_by_category,
    make_propose_flow,
)
from scheduler import db
from scheduler.agent import Agent, PendingBooking


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def eval_db(tmp_path: Path) -> Path:
    """Fixture providing a clean SQLite database seeded with clinic data."""
    db_file = tmp_path / "test_eval.db"
    db.init_db(db_file)
    db.seed(db_file)
    return db_file


# ---------------------------------------------------------------------------
# Full Scenario Suite Execution Test
# ---------------------------------------------------------------------------


class TestEvalSuite:
    """Evaluates the complete 42-scenario benchmark suite."""

    def test_all_42_scenarios_pass_deterministic(self) -> None:
        runner = EvalRunner()
        suite_result = runner.run_all()
        metrics = suite_result.metrics

        # Log any failures for developer diagnosis
        failures = [
            f"{r.scenario_id} ({r.scenario_name}): {r.error_message}"
            for r in suite_result.scenario_results
            if not r.success
        ]
        assert len(failures) == 0, f"Scenario failures encountered:\n" + "\n".join(failures)

        assert metrics.total_scenarios == 42
        assert metrics.passed_scenarios == 42
        assert metrics.failed_scenarios == 0
        assert metrics.task_completion_rate == 100.0
        assert metrics.slot_selection_accuracy == 100.0
        assert metrics.unsafe_booking_rate == 0.0
        assert metrics.false_verbal_confirmation_rate == 0.0
        assert metrics.failure_recovery_rate == 100.0

    @pytest.mark.parametrize(
        "category",
        [
            ScenarioCategory.NORMAL_BOOKING,
            ScenarioCategory.CORRECTIONS,
            ScenarioCategory.AMBIGUOUS_CONFIRMATIONS,
            ScenarioCategory.NEGATION_CANCELLATION,
            ScenarioCategory.INTERRUPTIONS,
            ScenarioCategory.DUPLICATE_COMPETING,
            ScenarioCategory.STT_ERRORS,
            ScenarioCategory.PROVIDER_FAILURES,
            ScenarioCategory.SESSION_ISOLATION_RECOVERY,
        ],
    )
    def test_category_100_percent_pass(self, category: ScenarioCategory) -> None:
        runner = EvalRunner()
        suite_result = runner.run_category(category)
        assert suite_result.metrics.pass_rate == 100.0
        assert suite_result.metrics.unsafe_booking_rate == 0.0
        assert suite_result.metrics.false_verbal_confirmation_rate == 0.0


# ---------------------------------------------------------------------------
# Dedicated Safety Invariant Tests
# ---------------------------------------------------------------------------


class TestSafetyInvariants:
    """Strict verification of core healthcare scheduling invariants."""

    def test_ambiguity_never_commits_to_sqlite(self, eval_db: Path) -> None:
        """Ambiguous or hesitant verbal responses must NEVER commit an appointment to SQLite."""
        slots = db.list_available_slots(db_path=eval_db)
        target_slot = slots[0]

        mock_client = MagicMock()
        mock_client.models.generate_content.side_effect = make_propose_flow(
            target_slot.id, "Dr. Ada Lovelace", "Tuesday at 9:00 AM PST"
        )

        agent = Agent(db_path=eval_db, client=mock_client)
        # Turn 1: Propose slot
        agent.turn("Book appointment with Dr. Ada Lovelace on Tuesday.")
        assert agent.pending_booking is not None
        assert agent.awaiting_verbal_confirmation is True

        # Test various ambiguous verbal responses
        ambiguous_utterances = [
            "Maybe, let me check my calendar.",
            "Hold on a second.",
            "Let me think about that for a minute.",
            "Not sure yet.",
        ]
        for utterance in ambiguous_utterances:
            reply = agent.turn(utterance)
            assert "Just to confirm" in reply
            assert "successfully confirmed" not in reply.lower()

            # Verify SQLite has ZERO bookings
            conn = db._connect(eval_db)
            count = conn.execute("SELECT count(*) FROM bookings").fetchone()[0]
            conn.close()
            assert count == 0, f"Ambiguous utterance '{utterance}' resulted in unsafe booking!"

    def test_cancellation_never_commits_and_clears_pending(self, eval_db: Path) -> None:
        """Explicit verbal cancellation must immediately discard proposal with zero database writes."""
        slots = db.list_available_slots(db_path=eval_db)
        target_slot = slots[0]

        mock_client = MagicMock()
        mock_client.models.generate_content.side_effect = make_propose_flow(
            target_slot.id, "Dr. Ada Lovelace", "Tuesday at 9:00 AM PST"
        )

        agent = Agent(db_path=eval_db, client=mock_client)
        agent.turn("Book Dr. Lovelace")
        assert agent.pending_booking is not None

        reply = agent.turn("Don't book that appointment.")
        assert "cancelled that booking request" in reply.lower()
        assert agent.pending_booking is None
        assert agent.awaiting_verbal_confirmation is False

        conn = db._connect(eval_db)
        count = conn.execute("SELECT count(*) FROM bookings").fetchone()[0]
        conn.close()
        assert count == 0

    def test_concurrent_competing_requests_prevent_double_booking(self, eval_db: Path) -> None:
        """Concurrent booking attempts for the same slot must guarantee exactly 1 booking."""
        slots = db.list_available_slots(db_path=eval_db)
        target_slot = slots[0]

        results = []

        def attempt_booking(worker_id: int) -> tuple[str, int]:
            try:
                b = db.book_slot(
                    target_slot.id,
                    patient_name=f"Patient {worker_id}",
                    idempotency_key=f"worker-key-{worker_id}",
                    db_path=eval_db,
                )
                return ("confirmed", b.id)
            except db.BookingError as exc:
                return ("error", exc.code)  # type: ignore[return-value]

        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(attempt_booking, i) for i in range(10)]
            for f in concurrent.futures.as_completed(futures):
                results.append(f.result())

        confirmed = [r for r in results if r[0] == "confirmed"]
        rejected = [r for r in results if r[0] == "error"]

        assert len(confirmed) == 1, f"Expected exactly 1 booking, got {len(confirmed)}"
        assert len(rejected) == 9
        assert all(code == db.SLOT_UNAVAILABLE for _, code in rejected)

        conn = db._connect(eval_db)
        total_rows = conn.execute(
            "SELECT count(*) FROM bookings WHERE slot_id = ? AND status = 'confirmed'",
            (target_slot.id,),
        ).fetchone()[0]
        conn.close()
        assert total_rows == 1

    def test_repeated_confirmation_idempotent(self, eval_db: Path) -> None:
        """Calling confirm_booking repeatedly on an already confirmed appointment is idempotent."""
        slots = db.list_available_slots(db_path=eval_db)
        target_slot = slots[0]

        mock_client = MagicMock()
        agent = Agent(db_path=eval_db, client=mock_client)
        agent.pending_booking = PendingBooking(
            slot_id=target_slot.id,
            doctor_name=target_slot.doctor_name or "Dr. Ada Lovelace",
            specialty=target_slot.specialty or "General Practice",
            start_utc=target_slot.start_utc,
            end_utc=target_slot.end_utc,
            local_time_str="Tuesday 9:00 AM",
            idempotency_key="idemp-test-key",
            patient_name="Alex Taylor",
        )

        # 1st confirm
        ok1, msg1 = agent.confirm_booking()
        assert ok1 is True
        assert "successfully confirmed" in msg1

        # 2nd confirm
        ok2, msg2 = agent.confirm_booking()
        assert ok2 is True
        assert "already confirmed" in msg2

        # 3rd confirm
        ok3, msg3 = agent.confirm_booking()
        assert ok3 is True
        assert "already confirmed" in msg3

        conn = db._connect(eval_db)
        count = conn.execute("SELECT count(*) FROM bookings").fetchone()[0]
        conn.close()
        assert count == 1, "Duplicate bookings created on repeated confirmation!"

    def test_provider_timeout_rolls_back_conversation_history(self, eval_db: Path) -> None:
        """Provider failures roll back conversation history to clean checkpoint without orphaned turns."""
        mock_client = MagicMock()
        mock_client.models.generate_content.side_effect = TimeoutError("Gemini deadline exceeded")

        agent = Agent(db_path=eval_db, client=mock_client)
        assert len(agent.history) == 0

        reply = agent.turn("I need an appointment tomorrow morning.")
        assert "trouble connecting" in reply
        assert len(agent.history) == 0
        assert agent.pending_booking is None
        assert agent.awaiting_verbal_confirmation is False

    def test_multi_session_state_isolation(self, eval_db: Path) -> None:
        """Independent Agent instances maintain strictly isolated proposals without cross-talk."""
        slots = db.list_available_slots(db_path=eval_db)
        slot_a = [s for s in slots if s.doctor_name == "Dr. Ada Lovelace"][0]
        slot_b = [s for s in slots if s.doctor_name == "Dr. Charles Babbage"][0]

        agent_a = Agent(db_path=eval_db, client=MagicMock(), patient_name="Patient A")
        agent_b = Agent(db_path=eval_db, client=MagicMock(), patient_name="Patient B")

        agent_a._dispatch("propose_booking", {"slot_id": slot_a.id})
        agent_b._dispatch("propose_booking", {"slot_id": slot_b.id})

        assert agent_a.pending_booking is not None and agent_a.pending_booking.slot_id == slot_a.id
        assert agent_b.pending_booking is not None and agent_b.pending_booking.slot_id == slot_b.id

        # Cancel A; B must remain unaffected
        agent_a.cancel_pending()
        assert agent_a.pending_booking is None
        assert agent_b.pending_booking is not None
        assert agent_b.pending_booking.slot_id == slot_b.id
