"""Requested scheduling scenarios using fictional doctors and patients."""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import tempfile
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Iterator, TextIO
from unittest.mock import MagicMock

import pytest
from google.genai import types

from scheduler import db
from scheduler.agent import Agent


PROJECT_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture
def scenario_db(tmp_path: Path) -> Path:
    """Create a private database with two weeks of fictional availability."""
    database_path = tmp_path / "scenario-test.db"
    db.init_db(database_path)
    db.seed(database_path)
    return database_path


def _next_week_dates() -> tuple[date, date]:
    today = datetime.now(db.CLINIC_TZ).date()
    days_until_next_monday = 7 - today.weekday()
    next_monday = today + timedelta(days=days_until_next_monday)
    return next_monday, next_monday + timedelta(days=6)


def test_multiple_doctors_have_overlapping_afternoon_slots(
    scenario_db: Path,
) -> None:
    next_monday, _ = _next_week_dates()
    slots = db.list_available_slots(
        db_path=scenario_db,
        local_date_start=next_monday,
        local_date_end=next_monday,
        time_preference="afternoon",
    )

    doctor_names = {slot.doctor_name for slot in slots}
    assert doctor_names == {
        "Dr. Ada Lovelace",
        "Dr. Charles Babbage",
        "Dr. Florence Nightingale",
    }
    assert len(slots) == 3


def test_next_week_afternoon_request_uses_date_and_time_filters(
    scenario_db: Path,
) -> None:
    next_monday, next_sunday = _next_week_dates()
    tool_arguments = {
        "date_start": next_monday.isoformat(),
        "date_end": next_sunday.isoformat(),
        "time_preference": "afternoon",
    }
    generated_call = types.FunctionCall(
        name="search_slots",
        args=tool_arguments,
        id="next-week-afternoon-search",
    )
    first_response = MagicMock()
    first_response.function_calls = [generated_call]
    first_response.text = ""
    first_response.candidates = [
        MagicMock(
            content=types.Content(
                role="model",
                parts=[types.Part(function_call=generated_call)],
            )
        )
    ]
    final_response = MagicMock()
    final_response.function_calls = []
    final_response.text = "I found afternoon appointments next week."
    final_response.candidates = [
        MagicMock(
            content=types.Content(
                role="model",
                parts=[types.Part.from_text(text=final_response.text)],
            )
        )
    ]

    tool_events: list[dict[str, object]] = []

    def record_event(
        stage: str,
        title: str,
        details: dict[str, object],
        duration_ms: float | None = None,
        status: str = "ok",
    ) -> None:
        if stage == "tool" and details.get("function") == "search_slots":
            tool_events.append(details)

    mock_client = MagicMock()
    mock_client.models.generate_content.side_effect = [first_response, final_response]
    agent = Agent(
        db_path=scenario_db,
        client=mock_client,
        event_callback=record_event,
    )

    reply = agent.turn("Can I come next week in the afternoon?")

    assert "afternoon appointments next week" in reply
    assert len(tool_events) == 1
    assert tool_events[0]["arguments"] == {
        "doctor_id": None,
        "date": None,
        **tool_arguments,
    }
    assert tool_events[0]["result_count"] == 21

    slots = db.list_available_slots(
        db_path=scenario_db,
        local_date_start=next_monday,
        local_date_end=next_sunday,
        time_preference="afternoon",
    )
    assert len({slot.doctor_name for slot in slots}) == 3
    for slot in slots:
        local_start = datetime.fromisoformat(slot.start_utc).astimezone(db.CLINIC_TZ)
        assert next_monday <= local_start.date() <= next_sunday
        assert 12 <= local_start.hour < 17


def _free_port() -> int:
    with socket.socket() as server_socket:
        server_socket.bind(("127.0.0.1", 0))
        return int(server_socket.getsockname()[1])


def _read_server_log(log_file: TextIO) -> str:
    log_file.seek(0)
    return log_file.read()


def _wait_for_health(
    process: subprocess.Popen[bytes],
    base_url: str,
    log_file: TextIO,
) -> None:
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise AssertionError(
                f"Web process exited early:\n{_read_server_log(log_file)}"
            )
        try:
            with urllib.request.urlopen(f"{base_url}/api/health", timeout=1) as response:
                if response.status == 200:
                    return
        except OSError:
            time.sleep(0.1)
    raise AssertionError(f"Web process failed to start:\n{_read_server_log(log_file)}")


@contextmanager
def _two_web_processes(database_path: Path) -> Iterator[list[str]]:
    log_files = [tempfile.TemporaryFile(mode="w+t") for _ in range(2)]
    processes: list[subprocess.Popen[bytes]] = []
    urls: list[str] = []
    environment = os.environ.copy()
    environment.update(
        {
            "DB_PATH": str(database_path),
            "PYTHON_DOTENV_DISABLED": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
        }
    )

    try:
        for log_file in log_files:
            port = _free_port()
            base_url = f"http://127.0.0.1:{port}"
            process = subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    "uvicorn",
                    "tests.web_race_app:app",
                    "--host",
                    "127.0.0.1",
                    "--port",
                    str(port),
                    "--log-level",
                    "error",
                ],
                cwd=PROJECT_ROOT,
                env=environment,
                stdout=log_file,
                stderr=subprocess.STDOUT,
            )
            processes.append(process)
            urls.append(base_url)
            _wait_for_health(process, base_url, log_file)

        yield urls
    finally:
        for process in processes:
            process.terminate()
        for process in processes:
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
        for log_file in log_files:
            log_file.close()


def _post_json(url: str, body: dict[str, object] | None = None) -> dict[str, object]:
    request_data = json.dumps(body or {}).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=request_data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=10) as response:
        return json.loads(response.read())


def _confirm_after_barrier(
    base_url: str,
    session_id: str,
    start_barrier: threading.Barrier,
) -> dict[str, object]:
    start_barrier.wait(timeout=10)
    with urllib.request.urlopen(
        urllib.request.Request(
            f"{base_url}/api/sessions/{session_id}/confirm",
            data=b"",
            method="POST",
        ),
        timeout=10,
    ) as response:
        return json.loads(response.read())


def test_two_web_processes_racing_one_slot_have_one_winner_and_clean_recovery(
    scenario_db: Path,
) -> None:
    next_monday, next_sunday = _next_week_dates()
    afternoon_slots = db.list_available_slots(
        db_path=scenario_db,
        local_date_start=next_monday,
        local_date_end=next_sunday,
        time_preference="afternoon",
    )
    target_slot = afternoon_slots[0]

    with _two_web_processes(scenario_db) as base_urls:
        staged_sessions = [
            _post_json(f"{base_url}/__test__/stage/{target_slot.id}")
            for base_url in base_urls
        ]
        start_barrier = threading.Barrier(2)
        with ThreadPoolExecutor(max_workers=2) as executor:
            results = list(
                executor.map(
                    lambda pair: _confirm_after_barrier(
                        pair[0],
                        str(pair[1]["session_id"]),
                        start_barrier,
                    ),
                    zip(base_urls, staged_sessions),
                )
            )

    winners = [result for result in results if result["success"] is True]
    losers = [result for result in results if result["success"] is False]
    assert len(winners) == 1
    assert len(losers) == 1
    assert "no longer available" in str(losers[0]["message"]).lower()
    assert "search for another available time" in str(losers[0]["message"]).lower()

    connection = db.get_connection(scenario_db)
    try:
        bookings = connection.execute(
            "SELECT id, status FROM bookings WHERE slot_id = ?",
            (target_slot.id,),
        ).fetchall()
        slot_status = connection.execute(
            "SELECT status FROM slots WHERE id = ?",
            (target_slot.id,),
        ).fetchone()["status"]
    finally:
        connection.close()

    assert len(bookings) == 1
    assert bookings[0]["status"] == "confirmed"
    assert slot_status == "booked"
