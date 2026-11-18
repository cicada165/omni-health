"""
evals/runner.py — Evaluation Execution Engine (Milestone 8).

Runs scheduling evaluation scenarios deterministically with mock providers
or against live endpoints, validating:
- Slot IDs and doctor matches
- SQLite state and partial unique index constraints
- Latency per turn and scenario duration
- Safety invariants (zero unsafe bookings, zero false verbal confirms)
"""

from __future__ import annotations

import logging
import shutil
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional
from unittest.mock import MagicMock

from evals.metrics import BenchmarkMetrics, MetricsCalculator
from evals.scenarios import (
    ScenarioCategory,
    ScenarioSpec,
    TurnSpec,
    get_all_scenarios,
    get_scenarios_by_category,
    get_scenario_by_id,
)
from scheduler import db
from scheduler.agent import Agent

logger = logging.getLogger("appointment_scheduler.evals.runner")


# ---------------------------------------------------------------------------
# Data Classes
# ---------------------------------------------------------------------------


@dataclass
class TurnResult:
    """Outcome and latency of a single evaluated conversational turn."""
    turn_index: int
    user_input: str
    agent_reply: str
    duration_ms: float
    pending_slot_id: Optional[int] = None
    awaiting_verbal: bool = False
    error: Optional[str] = None


@dataclass
class ScenarioResult:
    """Outcome and verification results for a completed scenario."""
    scenario_id: str
    scenario_name: str
    category: str
    success: bool
    task_completed: bool
    slot_accurate: bool
    unsafe_booking: bool  # Hard invariant: MUST BE FALSE
    false_verbal_confirm: bool  # Hard invariant: MUST BE FALSE
    failure_recovered: bool
    is_recovery_scenario: bool
    turn_results: list[TurnResult] = field(default_factory=list)
    total_duration_ms: float = 0.0
    sqlite_bookings_count: int = 0
    error_message: Optional[str] = None


@dataclass
class EvalSuiteResult:
    """Aggregated results and computed benchmarks for an evaluation run."""
    scenario_results: list[ScenarioResult]
    metrics: BenchmarkMetrics
    total_duration_ms: float


# ---------------------------------------------------------------------------
# Evaluation Runner
# ---------------------------------------------------------------------------


class EvalRunner:
    """
    Executes scenarios against Appointment Scheduler scheduling agent and public APIs.

    Supports:
    - Deterministic offline mock execution (default, fast, 0 network dependencies)
    - Live execution against real Gemini models (if live=True and GEMINI_API_KEY set)
    """

    def __init__(
        self,
        db_path: Optional[Path] = None,
        live: bool = False,
        model: Optional[str] = None,
    ) -> None:
        self.db_path = db_path
        self.live = live
        self.model = model

    def _setup_isolated_db(self) -> tuple[Path, Optional[str]]:
        """Create an isolated SQLite database seeded with default clinic data."""
        if self.db_path is not None:
            return self.db_path, None

        temp_dir = tempfile.mkdtemp(prefix="appointment_scheduler_eval_")
        db_file = Path(temp_dir) / "eval.db"
        db.init_db(db_file)
        db.seed(db_file)
        return db_file, temp_dir

    def _cleanup_db(self, temp_dir: Optional[str]) -> None:
        """Remove temporary directory if created."""
        if temp_dir is not None:
            try:
                shutil.rmtree(temp_dir, ignore_errors=True)
            except Exception as exc:
                logger.debug("Failed to clean up %s: %s", temp_dir, exc)

    def run_scenario(self, scenario: ScenarioSpec) -> ScenarioResult:
        """Execute a single scenario and validate all outcome and safety checks."""
        t_start = time.perf_counter()
        db_path, temp_dir = self._setup_isolated_db()

        turn_results: list[TurnResult] = []
        unsafe_booking = False
        false_verbal_confirm = False
        slot_accurate = True
        task_completed = False
        failure_recovered = True
        error_message: Optional[str] = None

        try:
            # Custom Runner Execution Path (for multi-threaded or provider service tests)
            if scenario.custom_runner is not None:
                try:
                    res = scenario.custom_runner(scenario, db_path)
                    task_completed = True if res is not False else False
                except Exception as exc:
                    error_message = f"Custom runner failed: {exc}"
                    task_completed = False
                    failure_recovered = False

                conn = db._connect(db_path)
                bookings_count = conn.execute(
                    "SELECT count(*) FROM bookings WHERE status = 'confirmed'"
                ).fetchone()[0]
                conn.close()

                if not scenario.expected_booking and bookings_count > 0:
                    unsafe_booking = True
                if scenario.expected_booking:
                    min_count = scenario.expected_booking_count or 1
                    if bookings_count < min_count:
                        task_completed = False
                        error_message = f"Expected at least {min_count} bookings, got {bookings_count}"

            # Conversational Turn Runner Execution Path
            else:
                if self.live:
                    client = None  # Agent initializes live Client with GEMINI_API_KEY
                else:
                    mock_client = MagicMock()
                    client = mock_client

                agent = Agent(
                    db_path=db_path,
                    client=client,
                    model=self.model,
                    patient_name="Alex Taylor",
                )

                for idx, turn_spec in enumerate(scenario.turns):
                    t_turn_start = time.perf_counter()

                    # Set up turn-specific mock responses
                    if not self.live and client is not None:
                        if isinstance(turn_spec.mock_responses, Exception):
                            mock_client.models.generate_content.side_effect = turn_spec.mock_responses
                        elif callable(turn_spec.mock_responses):
                            mocks = turn_spec.mock_responses(db_path, idx)
                            mock_client.models.generate_content.side_effect = mocks
                        elif isinstance(turn_spec.mock_responses, list):
                            mock_client.models.generate_content.side_effect = list(turn_spec.mock_responses)
                        else:
                            mock_client.models.generate_content.side_effect = None

                    reply = agent.turn(turn_spec.user_input)
                    turn_dur_ms = (time.perf_counter() - t_turn_start) * 1000

                    # Verification of turn expectations
                    turn_error: Optional[str] = None
                    for substr in turn_spec.expected_reply_contains:
                        if substr.lower() not in reply.lower():
                            err = f"Turn {idx + 1} reply missing expected substring '{substr}'. Reply: {reply[:120]}"
                            turn_error = err
                            error_message = err

                    for forbidden in turn_spec.reply_must_not_contain:
                        if forbidden.lower() in reply.lower():
                            err = f"Turn {idx + 1} reply contains forbidden substring '{forbidden}'. Reply: {reply[:120]}"
                            turn_error = err
                            error_message = err

                    if turn_spec.expected_pending_after is not None:
                        is_pending = (agent.pending_booking is not None)
                        if is_pending != turn_spec.expected_pending_after:
                            err = f"Turn {idx + 1} pending status mismatch: expected {turn_spec.expected_pending_after}, got {is_pending}"
                            turn_error = err
                            error_message = err

                    if turn_spec.expected_awaiting_after is not None:
                        if agent.awaiting_verbal_confirmation != turn_spec.expected_awaiting_after:
                            err = f"Turn {idx + 1} awaiting confirmation mismatch: expected {turn_spec.expected_awaiting_after}, got {agent.awaiting_verbal_confirmation}"
                            turn_error = err
                            error_message = err

                    # Safety Check: False Verbal Confirmation
                    # If reply claims confirmed, verify active booking exists in SQLite
                    if "appointment successfully confirmed" in reply.lower() or "appointment is already confirmed" in reply.lower():
                        conn = db._connect(db_path)
                        conf_count = conn.execute(
                            "SELECT count(*) FROM bookings WHERE status = 'confirmed'"
                        ).fetchone()[0]
                        conn.close()
                        if conf_count == 0:
                            false_verbal_confirm = True
                            error_message = f"Turn {idx + 1} claimed confirmation, but zero confirmed bookings in SQLite!"

                    turn_results.append(
                        TurnResult(
                            turn_index=idx + 1,
                            user_input=turn_spec.user_input,
                            agent_reply=reply,
                            duration_ms=turn_dur_ms,
                            pending_slot_id=agent.pending_booking.slot_id if agent.pending_booking else None,
                            awaiting_verbal=agent.awaiting_verbal_confirmation,
                            error=turn_error,
                        )
                    )

                # Final Database State Verification
                conn = db._connect(db_path)
                confirmed_rows = conn.execute(
                    "SELECT * FROM bookings WHERE status = 'confirmed'"
                ).fetchall()
                bookings_count = len(confirmed_rows)
                conn.close()

                # Safety Check: Unsafe Booking
                if not scenario.expected_booking and bookings_count > 0:
                    unsafe_booking = True
                    error_message = f"Scenario expected NO booking, but found {bookings_count} bookings in SQLite!"

                # Slot Accuracy & Task Completion Check
                expected_count = scenario.expected_booking_count or 1
                if scenario.expected_booking:
                    if bookings_count == expected_count:
                        b = confirmed_rows[0]
                        slot = db.get_slot(b["slot_id"], db_path=db_path)
                        if scenario.expected_doctor and (slot is None or slot.doctor_name != scenario.expected_doctor):
                            slot_accurate = False
                            error_message = f"Expected doctor '{scenario.expected_doctor}', got '{slot.doctor_name if slot else None}'"
                        if scenario.expected_specialty and (slot is None or slot.specialty != scenario.expected_specialty):
                            slot_accurate = False
                            error_message = f"Expected specialty '{scenario.expected_specialty}', got '{slot.specialty if slot else None}'"
                        task_completed = slot_accurate and (error_message is None)
                    else:
                        task_completed = False
                        slot_accurate = False
                        error_message = f"Expected {expected_count} booking(s), got {bookings_count}"
                else:
                    slot_accurate = True  # N/A for cancellation/ambiguity
                    task_completed = (bookings_count == 0) and not unsafe_booking and (error_message is None)

                if scenario.is_recovery_scenario and not task_completed:
                    failure_recovered = False

        except Exception as exc:
            logger.exception("Scenario %s encountered unhandled error: %s", scenario.id, exc)
            error_message = f"Unhandled error: {exc}"
            task_completed = False
            failure_recovered = False
        finally:
            self._cleanup_db(temp_dir)

        total_dur_ms = (time.perf_counter() - t_start) * 1000
        success = (
            task_completed
            and slot_accurate
            and not unsafe_booking
            and not false_verbal_confirm
            and (error_message is None)
        )

        return ScenarioResult(
            scenario_id=scenario.id,
            scenario_name=scenario.name,
            category=scenario.category.value,
            success=success,
            task_completed=task_completed,
            slot_accurate=slot_accurate,
            unsafe_booking=unsafe_booking,
            false_verbal_confirm=false_verbal_confirm,
            failure_recovered=failure_recovered,
            is_recovery_scenario=scenario.is_recovery_scenario,
            turn_results=turn_results,
            total_duration_ms=total_dur_ms,
            sqlite_bookings_count=bookings_count,
            error_message=error_message,
        )

    def run_scenarios(self, scenarios: list[ScenarioSpec]) -> EvalSuiteResult:
        """Run a list of scenarios sequentially and compute aggregate benchmarks."""
        t0 = time.perf_counter()
        results: list[ScenarioResult] = []
        for s in scenarios:
            res = self.run_scenario(s)
            results.append(res)
        total_dur_ms = (time.perf_counter() - t0) * 1000
        metrics = MetricsCalculator.compute(results)
        return EvalSuiteResult(
            scenario_results=results,
            metrics=metrics,
            total_duration_ms=total_dur_ms,
        )

    def run_all(self) -> EvalSuiteResult:
        """Run all 42 evaluation scenarios."""
        return self.run_scenarios(get_all_scenarios())

    def run_category(self, category: ScenarioCategory | str) -> EvalSuiteResult:
        """Run all scenarios in a specific category."""
        return self.run_scenarios(get_scenarios_by_category(category))
