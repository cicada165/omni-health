"""
scheduler/db.py — SQLite-backed appointment-scheduling backend.

Tables
------
doctors          – fictional physicians
slots            – time windows offered by a doctor (UTC)
bookings         – patient reservations against a slot

Concurrency
-----------
Every mutating public function opens its own connection and uses
BEGIN IMMEDIATE so that SQLite's file-lock serialises writes.

Idempotency
-----------
book_slot() accepts an ``idempotency_key``.  Retrying the same key
returns the original booking row instead of creating a duplicate.
Callers (the future voice agent) are responsible for generating stable
keys; the backend never invents one on behalf of the caller.

Error codes
-----------
The BookingError exception carries a ``code`` attribute:

* ``SLOT_NOT_FOUND``        – the requested slot_id does not exist
* ``SLOT_UNAVAILABLE``      – already booked (confirmed)
* ``SLOT_IN_PAST``          – slot start time is in the past
* ``IDEMPOTENCY_CONFLICT``  – key reused for a *different* slot/patient
* ``NOT_AUTHORISED``        – patient identity doesn't match the booking

Timezone handling
-----------------
The fictional clinic operates in America/Los_Angeles.  Seed slots are
generated at 09:00 and 14:00 *local* time (DST-aware) and stored as UTC
ISO-8601 strings.  ``list_available_slots()`` accepts an optional
``local_date`` filter that is converted to UTC boundaries for querying.
"""

from __future__ import annotations

import sqlite3
import uuid
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Optional
from zoneinfo import ZoneInfo

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_DB = Path("appointment_scheduler.db")

CLINIC_TZ = ZoneInfo("America/Los_Angeles")

# Error codes returned inside BookingError
SLOT_NOT_FOUND = "SLOT_NOT_FOUND"
SLOT_UNAVAILABLE = "SLOT_UNAVAILABLE"
SLOT_IN_PAST = "SLOT_IN_PAST"
IDEMPOTENCY_CONFLICT = "IDEMPOTENCY_CONFLICT"
NOT_AUTHORISED = "NOT_AUTHORISED"


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class BookingError(Exception):
    """Domain error raised by book_slot / cancel_booking."""

    def __init__(self, message: str, code: str) -> None:
        super().__init__(message)
        self.code = code


# ---------------------------------------------------------------------------
# Data classes (plain value objects – no ORM)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Doctor:
    id: int
    name: str
    specialty: str


@dataclass(frozen=True)
class Slot:
    id: int
    doctor_id: int
    start_utc: str  # ISO-8601
    end_utc: str
    doctor_name: str | None = None
    specialty: str | None = None


@dataclass(frozen=True)
class Booking:
    id: int
    slot_id: int
    patient_name: str
    status: str  # 'confirmed' | 'cancelled'
    idempotency_key: str
    created_utc: str


# ---------------------------------------------------------------------------
# Connection helper
# ---------------------------------------------------------------------------


def _connect(db_path: Path = DEFAULT_DB) -> sqlite3.Connection:
    conn = sqlite3.connect(str(db_path), timeout=5)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.row_factory = sqlite3.Row
    return conn


def get_connection(db_path: Path = DEFAULT_DB) -> sqlite3.Connection:
    """Return a configured sqlite3.Connection with Row factory and WAL enabled."""
    return _connect(db_path)


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

_SCHEMA_SQL = """\
CREATE TABLE IF NOT EXISTS doctors (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    name      TEXT    NOT NULL,
    specialty TEXT    NOT NULL
);

CREATE TABLE IF NOT EXISTS slots (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    doctor_id INTEGER NOT NULL REFERENCES doctors(id),
    start_utc TEXT    NOT NULL,   -- ISO-8601 UTC
    end_utc   TEXT    NOT NULL,
    status    TEXT    NOT NULL DEFAULT 'available'
                        CHECK (status IN ('available', 'booked'))
);

-- A doctor cannot have two slots starting at the same instant.
CREATE UNIQUE INDEX IF NOT EXISTS uix_doctor_start
    ON slots (doctor_id, start_utc);

CREATE TABLE IF NOT EXISTS bookings (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    slot_id         INTEGER NOT NULL REFERENCES slots(id),
    patient_name    TEXT    NOT NULL,
    status          TEXT    NOT NULL DEFAULT 'confirmed'
                        CHECK (status IN ('confirmed', 'cancelled')),
    idempotency_key TEXT    NOT NULL,
    created_utc     TEXT    NOT NULL
);

-- Partial unique index: at most ONE confirmed booking per slot.
CREATE UNIQUE INDEX IF NOT EXISTS uix_slot_confirmed
    ON bookings (slot_id)
    WHERE status = 'confirmed';

-- A given idempotency key must map to exactly one booking row.
CREATE UNIQUE INDEX IF NOT EXISTS uix_idempotency
    ON bookings (idempotency_key);
"""


def init_db(db_path: Path = DEFAULT_DB) -> None:
    """Create tables and indexes (idempotent)."""
    conn = _connect(db_path)
    try:
        conn.executescript(_SCHEMA_SQL)
        conn.execute("BEGIN IMMEDIATE")
        slot_columns = {
            row["name"] for row in conn.execute("PRAGMA table_info(slots)")
        }
        if "status" not in slot_columns:
            conn.execute(
                "ALTER TABLE slots ADD COLUMN status TEXT NOT NULL "
                "DEFAULT 'available' CHECK (status IN ('available', 'booked'))"
            )
            conn.execute(
                "UPDATE slots SET status = 'booked' "
                "WHERE EXISTS ("
                "    SELECT 1 FROM bookings b "
                "    WHERE b.slot_id = slots.id AND b.status = 'confirmed'"
                ")"
            )
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Seed data
# ---------------------------------------------------------------------------

_SEED_DOCTORS = [
    ("Dr. Ada Lovelace", "General Practice"),
    ("Dr. Charles Babbage", "Cardiology"),
    ("Dr. Florence Nightingale", "Pediatrics"),
]


def seed(db_path: Path = DEFAULT_DB, days_ahead: int = 14) -> None:
    """
    Insert fictional doctors and future appointment slots.

    * Doctors are upserted by name so the call is idempotent.
    * Slots use INSERT OR IGNORE against the (doctor_id, start_utc)
      unique index, so duplicates are silently skipped and existing
      bookings are never erased.
    * Clinic hours are 09:00 and 14:00 America/Los_Angeles (DST-aware),
      converted to UTC for storage.
    """
    conn = _connect(db_path)
    try:
        conn.execute("BEGIN IMMEDIATE")

        # Upsert doctors ------------------------------------------------
        for name, specialty in _SEED_DOCTORS:
            row = conn.execute(
                "SELECT id FROM doctors WHERE name = ?", (name,)
            ).fetchone()
            if row is None:
                conn.execute(
                    "INSERT INTO doctors (name, specialty) VALUES (?, ?)",
                    (name, specialty),
                )

        # Build slots for each doctor -----------------------------------
        today_local = datetime.now(CLINIC_TZ).date()
        doctors = conn.execute("SELECT id, name FROM doctors").fetchall()

        for doc in doctors:
            for day_offset in range(1, days_ahead + 1):
                slot_date = today_local + timedelta(days=day_offset)
                for hour in (9, 14):
                    # Build a DST-aware local datetime, then convert to UTC
                    local_dt = datetime(
                        slot_date.year, slot_date.month, slot_date.day,
                        hour, 0, tzinfo=CLINIC_TZ,
                    )
                    start_utc = local_dt.astimezone(timezone.utc)
                    end_utc = start_utc + timedelta(minutes=30)

                    # INSERT OR IGNORE: the unique index on
                    # (doctor_id, start_utc) silently skips duplicates.
                    conn.execute(
                        "INSERT OR IGNORE INTO slots "
                        "(doctor_id, start_utc, end_utc) VALUES (?, ?, ?)",
                        (
                            doc["id"],
                            start_utc.isoformat(),
                            end_utc.isoformat(),
                        ),
                    )

        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Timezone helpers
# ---------------------------------------------------------------------------


def _local_date_to_utc_range(local_date: date) -> tuple[str, str]:
    """Convert a clinic-local date to a [start, end) UTC ISO-8601 pair."""
    day_start = datetime(
        local_date.year, local_date.month, local_date.day,
        tzinfo=CLINIC_TZ,
    )
    day_end = day_start + timedelta(days=1)
    return (
        day_start.astimezone(timezone.utc).isoformat(),
        day_end.astimezone(timezone.utc).isoformat(),
    )


# ---------------------------------------------------------------------------
# Queries
# ---------------------------------------------------------------------------


def list_doctors(db_path: Path = DEFAULT_DB) -> list[Doctor]:
    """Return all doctors."""
    conn = _connect(db_path)
    try:
        rows = conn.execute(
            "SELECT id, name, specialty FROM doctors ORDER BY id"
        ).fetchall()
        return [
            Doctor(id=r["id"], name=r["name"], specialty=r["specialty"])
            for r in rows
        ]
    finally:
        conn.close()


def list_available_slots(
    db_path: Path = DEFAULT_DB,
    doctor_id: Optional[int] = None,
    local_date: Optional[date] = None,
    local_date_start: Optional[date] = None,
    local_date_end: Optional[date] = None,
    time_preference: Optional[str] = None,
) -> list[Slot]:
    """
    Return future slots that have no confirmed booking.

    Parameters
    ----------
    doctor_id : int, optional
        Filter to a single doctor.
    local_date : date, optional
        Filter to slots falling on this clinic-local date
        (America/Los_Angeles).  Boundaries are converted to UTC.
    local_date_start, local_date_end : date, optional
        Inclusive clinic-local date range boundaries. Either boundary may be omitted.
    time_preference : str, optional
        One of morning (06:00–11:59), afternoon (12:00–16:59), or
        evening (17:00–20:59), in clinic-local time.
    """
    if local_date is not None and (
        local_date_start is not None or local_date_end is not None
    ):
        raise ValueError("Use either local_date or a date range, not both.")
    if (
        local_date_start is not None
        and local_date_end is not None
        and local_date_start > local_date_end
    ):
        raise ValueError("local_date_start must be on or before local_date_end.")

    time_windows = {
        "morning": (6, 12),
        "afternoon": (12, 17),
        "evening": (17, 21),
    }
    if time_preference is not None:
        time_preference = time_preference.strip().lower()
        if time_preference not in time_windows:
            allowed = ", ".join(time_windows)
            raise ValueError(f"time_preference must be one of: {allowed}.")

    conn = _connect(db_path)
    try:
        now = datetime.now(timezone.utc).isoformat()
        sql = (
            "SELECT s.id, s.doctor_id, s.start_utc, s.end_utc, "
            "       d.name AS doctor_name, d.specialty "
            "FROM slots s "
            "JOIN doctors d ON d.id = s.doctor_id "
            "WHERE s.status = 'available' AND s.start_utc > ? "
            "  AND NOT EXISTS ("
            "      SELECT 1 FROM bookings b "
            "      WHERE b.slot_id = s.id AND b.status = 'confirmed'"
            "  ) "
        )
        params: list = [now]

        if doctor_id is not None:
            sql += "  AND s.doctor_id = ? "
            params.append(doctor_id)

        if local_date is not None:
            utc_start, utc_end = _local_date_to_utc_range(local_date)
            sql += "  AND s.start_utc >= ? AND s.start_utc < ? "
            params.extend([utc_start, utc_end])
        else:
            if local_date_start is not None:
                utc_start, _ = _local_date_to_utc_range(local_date_start)
                sql += "  AND s.start_utc >= ? "
                params.append(utc_start)
            if local_date_end is not None:
                _, utc_end = _local_date_to_utc_range(local_date_end)
                sql += "  AND s.start_utc < ? "
                params.append(utc_end)

        sql += "ORDER BY s.start_utc"

        rows = conn.execute(sql, params).fetchall()
        slots = [
            Slot(
                id=r["id"],
                doctor_id=r["doctor_id"],
                start_utc=r["start_utc"],
                end_utc=r["end_utc"],
                doctor_name=r["doctor_name"],
                specialty=r["specialty"],
            )
            for r in rows
        ]
        if time_preference is None:
            return slots

        start_hour, end_hour = time_windows[time_preference]
        return [
            slot
            for slot in slots
            if start_hour
            <= datetime.fromisoformat(slot.start_utc).astimezone(CLINIC_TZ).hour
            < end_hour
        ]
    finally:
        conn.close()


def get_slot(slot_id: int, db_path: Path = DEFAULT_DB) -> Optional[Slot]:
    """Retrieve slot details including doctor information by slot_id."""
    conn = _connect(db_path)
    try:
        row = conn.execute(
            "SELECT s.id, s.doctor_id, s.start_utc, s.end_utc, "
            "       d.name AS doctor_name, d.specialty "
            "FROM slots s "
            "JOIN doctors d ON d.id = s.doctor_id "
            "WHERE s.id = ?",
            (slot_id,),
        ).fetchone()
        if row is None:
            return None
        return Slot(
            id=row["id"],
            doctor_id=row["doctor_id"],
            start_utc=row["start_utc"],
            end_utc=row["end_utc"],
            doctor_name=row["doctor_name"],
            specialty=row["specialty"],
        )
    finally:
        conn.close()


def is_slot_available(slot_id: int, db_path: Path = DEFAULT_DB) -> bool:
    """Return True if the slot exists, is in the future, and has no confirmed booking."""
    conn = _connect(db_path)
    try:
        now = datetime.now(timezone.utc).isoformat()
        row = conn.execute(
            "SELECT 1 FROM slots s "
            "WHERE s.id = ? AND s.status = 'available' AND s.start_utc > ? "
            "  AND NOT EXISTS ("
            "      SELECT 1 FROM bookings b "
            "      WHERE b.slot_id = s.id AND b.status = 'confirmed'"
            "  )",
            (slot_id, now),
        ).fetchone()
        return row is not None
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Book
# ---------------------------------------------------------------------------


def book_slot(
    slot_id: int,
    patient_name: str,
    idempotency_key: str,
    db_path: Path = DEFAULT_DB,
) -> Booking:
    """
    Book a slot for a patient.

    Parameters
    ----------
    idempotency_key : str
        A caller-supplied key that makes retries safe.  The backend
        never generates one — the caller (voice agent) is responsible.

    Raises
    ------
    BookingError
        With code ``SLOT_NOT_FOUND``, ``SLOT_IN_PAST``,
        ``SLOT_UNAVAILABLE``, or ``IDEMPOTENCY_CONFLICT``.
    """
    conn = _connect(db_path)
    try:
        conn.execute("BEGIN IMMEDIATE")

        # 1. Idempotency check FIRST -----------------------------------
        prev = conn.execute(
            "SELECT * FROM bookings WHERE idempotency_key = ?",
            (idempotency_key,),
        ).fetchone()
        if prev is not None:
            conn.rollback()
            # Same key, same request → return existing booking with its CURRENT status
            if prev["slot_id"] == slot_id and prev["patient_name"] == patient_name:
                return Booking(
                    id=prev["id"],
                    slot_id=prev["slot_id"],
                    patient_name=prev["patient_name"],
                    status=prev["status"],
                    idempotency_key=prev["idempotency_key"],
                    created_utc=prev["created_utc"],
                )
            # Same key, different payload → conflict
            raise BookingError(
                "Idempotency key already used for a different request.",
                IDEMPOTENCY_CONFLICT,
            )

        # 2. Does the slot exist? ---------------------------------------
        slot_row = conn.execute(
            "SELECT id, start_utc FROM slots WHERE id = ?", (slot_id,)
        ).fetchone()
        if slot_row is None:
            conn.rollback()
            raise BookingError(
                f"Slot {slot_id} does not exist.", SLOT_NOT_FOUND
            )

        # 3. Is the slot in the past? -----------------------------------
        slot_start = datetime.fromisoformat(slot_row["start_utc"])
        if slot_start <= datetime.now(timezone.utc):
            conn.rollback()
            raise BookingError(
                f"Slot {slot_id} is in the past.", SLOT_IN_PAST
            )

        # 4. Claim the slot immediately before inserting the booking. ---
        claim_time = datetime.now(timezone.utc).isoformat()
        claim = conn.execute(
            "UPDATE slots SET status = 'booked' "
            "WHERE id = ? AND status = 'available' AND start_utc > ?",
            (slot_id, claim_time),
        )
        if claim.rowcount != 1:
            conn.rollback()
            raise BookingError(
                f"Slot {slot_id} is already booked.", SLOT_UNAVAILABLE
            )

        # The status claim and booking insert commit or roll back together.
        now_utc = datetime.now(timezone.utc).isoformat()
        try:
            conn.execute(
                "INSERT INTO bookings "
                "(slot_id, patient_name, status, idempotency_key, created_utc) "
                "VALUES (?, ?, 'confirmed', ?, ?)",
                (slot_id, patient_name, idempotency_key, now_utc),
            )
        except sqlite3.IntegrityError as exc:
            conn.rollback()
            msg = str(exc).lower()
            # The partial unique index on (slot_id WHERE confirmed)
            # reports as "UNIQUE constraint failed: bookings.slot_id".
            if "bookings.slot_id" in msg:
                raise BookingError(
                    f"Slot {slot_id} is already booked.", SLOT_UNAVAILABLE
                ) from exc
            # Any other integrity error (FK, etc.) is a real bug.
            raise

        booking_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
        conn.commit()

        return Booking(
            id=booking_id,
            slot_id=slot_id,
            patient_name=patient_name,
            status="confirmed",
            idempotency_key=idempotency_key,
            created_utc=now_utc,
        )
    except BookingError:
        raise  # already rolled back above
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Cancel
# ---------------------------------------------------------------------------


def cancel_booking(
    booking_id: int,
    patient_name: str,
    db_path: Path = DEFAULT_DB,
) -> Booking:
    """
    Soft-cancel a booking (sets status → 'cancelled').

    Parameters
    ----------
    patient_name : str
        The patient requesting the cancellation.  Must match the
        booking's patient_name or ``NOT_AUTHORISED`` is raised.

    Raises
    ------
    BookingError
        ``SLOT_NOT_FOUND`` if booking_id doesn't exist.
        ``NOT_AUTHORISED`` if patient_name doesn't match.
    """
    conn = _connect(db_path)
    try:
        conn.execute("BEGIN IMMEDIATE")

        row = conn.execute(
            "SELECT * FROM bookings WHERE id = ?", (booking_id,)
        ).fetchone()
        if row is None:
            conn.rollback()
            raise BookingError(
                f"Booking {booking_id} does not exist.", SLOT_NOT_FOUND
            )

        if row["patient_name"] != patient_name:
            conn.rollback()
            raise BookingError(
                f"Patient '{patient_name}' is not authorised to cancel "
                f"booking {booking_id}.",
                NOT_AUTHORISED,
            )

        if row["status"] == "confirmed":
            conn.execute(
                "UPDATE bookings SET status = 'cancelled' WHERE id = ?",
                (booking_id,),
            )
            release = conn.execute(
                "UPDATE slots SET status = 'available' "
                "WHERE id = ? AND status = 'booked'",
                (row["slot_id"],),
            )
            if release.rowcount != 1:
                raise RuntimeError(
                    f"Could not release slot {row['slot_id']} during cancellation."
                )
        conn.commit()

        return Booking(
            id=row["id"],
            slot_id=row["slot_id"],
            patient_name=row["patient_name"],
            status="cancelled",
            idempotency_key=row["idempotency_key"],
            created_utc=row["created_utc"],
        )
    except BookingError:
        raise
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()
