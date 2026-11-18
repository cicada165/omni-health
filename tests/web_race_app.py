"""Test-only FastAPI wrapper for staging a proposal in each server process."""

from __future__ import annotations

import os
import uuid
from datetime import datetime
from pathlib import Path

from fastapi import HTTPException

from scheduler import db
from scheduler.agent import Agent, PendingBooking
from scheduler.web import SessionState, app, sessions


DATABASE_PATH = Path(os.environ["DB_PATH"])
app.state.db_path = DATABASE_PATH


@app.post("/__test__/stage/{slot_id}")
async def stage_test_booking(slot_id: int) -> dict[str, str]:
    """Stage a real Agent proposal without involving Gemini."""
    slot = db.get_slot(slot_id, db_path=DATABASE_PATH)
    if slot is None:
        raise HTTPException(status_code=404, detail="Test slot was not found.")

    process_id = os.getpid()
    session_id = uuid.uuid4().hex
    patient_name = f"Race Patient {process_id}"
    agent = Agent(
        db_path=DATABASE_PATH,
        client=object(),
        patient_name=patient_name,
    )
    local_start = datetime.fromisoformat(slot.start_utc).astimezone(db.CLINIC_TZ)
    agent.pending_booking = PendingBooking(
        slot_id=slot.id,
        doctor_name=slot.doctor_name or "Unknown doctor",
        specialty=slot.specialty or "Unknown specialty",
        start_utc=slot.start_utc,
        end_utc=slot.end_utc,
        local_time_str=local_start.strftime("%A, %B %d, %Y at %I:%M %p %Z"),
        idempotency_key=f"process-race-{process_id}-{uuid.uuid4().hex}",
        patient_name=patient_name,
    )
    agent.awaiting_verbal_confirmation = True
    sessions[session_id] = SessionState(agent)
    return {"session_id": session_id}
