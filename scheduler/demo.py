"""
scheduler/demo.py — interactive CLI walkthrough of the scheduling backend.

Run:
    python -m scheduler.demo
"""

from __future__ import annotations

import tempfile
from datetime import datetime, timedelta
from pathlib import Path

from scheduler.db import (
    CLINIC_TZ,
    BookingError,
    book_slot,
    cancel_booking,
    init_db,
    list_available_slots,
    seed,
)


def main() -> None:
    # Use a temp file so each demo starts clean
    db = Path(tempfile.mktemp(suffix=".db"))
    print("=== Appointment Scheduler Scheduling Demo ===\n")
    print(f"Database: {db}")
    print(f"Clinic timezone: {CLINIC_TZ}\n")

    # --- Setup -----------------------------------------------------------
    init_db(db)
    seed(db)
    print("✔ Database initialised and seeded.\n")

    # --- List available slots --------------------------------------------
    slots = list_available_slots(db_path=db)
    print(f"Available slots (all): {len(slots)}")
    for s in slots[:6]:
        local = datetime.fromisoformat(s.start_utc).astimezone(CLINIC_TZ)
        print(
            f"  Slot {s.id:>3}  {s.doctor_name} ({s.specialty})  "
            f"{local.strftime('%Y-%m-%d %I:%M %p %Z')}  (UTC: {s.start_utc})"
        )
    if len(slots) > 6:
        print(f"  … and {len(slots) - 6} more\n")
    else:
        print()

    # --- Filter by local date --------------------------------------------
    tomorrow = datetime.now(CLINIC_TZ).date() + timedelta(days=1)
    day_slots = list_available_slots(db_path=db, local_date=tomorrow)
    print(f"Slots on {tomorrow} (clinic-local): {len(day_slots)}\n")

    # --- Book a slot -----------------------------------------------------
    target = slots[0]
    booking = book_slot(
        target.id, "Alice Testpatient",
        idempotency_key="demo-key-1", db_path=db,
    )
    print(f"✔ Booked slot {target.id} → Booking #{booking.id}")
    print(f"  Patient : {booking.patient_name}")
    print(f"  Status  : {booking.status}")
    print(f"  Key     : {booking.idempotency_key}\n")

    # --- Idempotent retry ------------------------------------------------
    retry = book_slot(
        target.id, "Alice Testpatient",
        idempotency_key="demo-key-1", db_path=db,
    )
    print(f"✔ Idempotent retry returned same booking #{retry.id}\n")

    # --- Double-booking attempt ------------------------------------------
    try:
        book_slot(target.id, "Bob Duplicate", idempotency_key="demo-key-2", db_path=db)
    except BookingError as e:
        print(f"✘ Double-booking blocked: [{e.code}] {e}\n")

    # --- Unauthorised cancellation ---------------------------------------
    try:
        cancel_booking(booking.id, patient_name="Eve Intruder", db_path=db)
    except BookingError as e:
        print(f"✘ Unauthorised cancel blocked: [{e.code}] {e}\n")

    # --- Authorised cancel and rebook ------------------------------------
    cancelled = cancel_booking(
        booking.id, patient_name="Alice Testpatient", db_path=db,
    )
    print(f"✔ Cancelled booking #{cancelled.id}  (status={cancelled.status})")

    rebook = book_slot(
        target.id, "Charlie Rebooker",
        idempotency_key="demo-key-3", db_path=db,
    )
    print(f"✔ Re-booked slot {target.id} → Booking #{rebook.id}\n")

    # --- Invalid slot ----------------------------------------------------
    try:
        book_slot(999_999, "Eve Nobody", idempotency_key="demo-key-4", db_path=db)
    except BookingError as e:
        print(f"✘ Invalid slot: [{e.code}] {e}\n")

    # --- Remaining availability ------------------------------------------
    remaining = list_available_slots(db_path=db)
    print(f"Remaining available slots: {len(remaining)}")

    # Cleanup temp db
    db.unlink(missing_ok=True)
    print("\n=== Demo complete ===")


if __name__ == "__main__":
    main()
