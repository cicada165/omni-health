"""
tests/test_db.py — pytest suite for the scheduling backend.

Covers: success path, double-booking, idempotency, cancellation,
cancellation authorisation, past-slot rejection, local-date filtering,
seed idempotence, invalid slots, and concurrent bookings.
"""

from __future__ import annotations

import sqlite3
import threading
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pytest

from scheduler.db import (
    CLINIC_TZ,
    IDEMPOTENCY_CONFLICT,
    NOT_AUTHORISED,
    SLOT_IN_PAST,
    SLOT_NOT_FOUND,
    SLOT_UNAVAILABLE,
    Booking,
    BookingError,
    _connect,
    book_slot,
    cancel_booking,
    get_slot,
    init_db,
    is_slot_available,
    list_available_slots,
    seed,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def db(tmp_path: Path) -> Path:
    """Provide a fresh, seeded database for every test."""
    db_path = tmp_path / "test.db"
    init_db(db_path)
    seed(db_path)
    return db_path


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _insert_past_slot(db: Path, minutes_ago: int = 60) -> int:
    """Insert a slot in the past and return its id."""
    conn = _connect(db)
    try:
        # Grab any doctor
        doc = conn.execute("SELECT id FROM doctors LIMIT 1").fetchone()
        past = datetime.now(timezone.utc) - timedelta(minutes=minutes_ago)
        end = past + timedelta(minutes=30)
        conn.execute(
            "INSERT INTO slots (doctor_id, start_utc, end_utc) VALUES (?, ?, ?)",
            (doc["id"], past.isoformat(), end.isoformat()),
        )
        slot_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
        conn.commit()
        return slot_id
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Schema & seed
# ---------------------------------------------------------------------------


class TestSeedIdempotence:
    def test_seed_twice_no_duplicate_doctors(self, db: Path) -> None:
        """Calling seed() again must not duplicate doctors."""
        seed(db)  # second call
        slots_before = list_available_slots(db_path=db)
        seed(db)  # third call
        slots_after = list_available_slots(db_path=db)
        assert len(slots_before) == len(slots_after)

    def test_seed_preserves_bookings(self, db: Path) -> None:
        """Seeding after a booking must not erase the booking."""
        slots = list_available_slots(db_path=db)
        key = "seed-preserve-key"
        booking = book_slot(slots[0].id, "Keep Me", idempotency_key=key, db_path=db)
        seed(db)  # re-seed
        # Booking should still exist — cancelling it should work
        result = cancel_booking(booking.id, patient_name="Keep Me", db_path=db)
        assert result.status == "cancelled"

    def test_seed_allows_extending_schedule(self, db: Path) -> None:
        """Seeding with more days_ahead must add new slots, not fail."""
        count_before = len(list_available_slots(db_path=db))
        seed(db, days_ahead=20)  # extend schedule further out
        count_after = len(list_available_slots(db_path=db))
        assert count_after > count_before

    def test_slot_uniqueness_constraint(self, db: Path) -> None:
        """The (doctor_id, start_utc) unique index must exist."""
        conn = _connect(db)
        try:
            doc = conn.execute("SELECT id FROM doctors LIMIT 1").fetchone()
            ts = "2099-01-01T17:00:00+00:00"
            conn.execute(
                "INSERT INTO slots (doctor_id, start_utc, end_utc) VALUES (?, ?, ?)",
                (doc["id"], ts, "2099-01-01T17:30:00+00:00"),
            )
            conn.commit()
            # Duplicate must be rejected
            with pytest.raises(sqlite3.IntegrityError):
                conn.execute(
                    "INSERT INTO slots (doctor_id, start_utc, end_utc) VALUES (?, ?, ?)",
                    (doc["id"], ts, "2099-01-01T17:30:00+00:00"),
                )
        finally:
            conn.close()


class TestSchemaMigration:
    def test_legacy_confirmed_bookings_mark_slots_booked(self, tmp_path: Path) -> None:
        """Adding slot status preserves active bookings in existing databases."""
        db_path = tmp_path / "legacy.db"
        start = datetime.now(timezone.utc) + timedelta(days=10)
        end = start + timedelta(minutes=30)
        created = datetime.now(timezone.utc).isoformat()

        with sqlite3.connect(db_path) as conn:
            conn.executescript(
                """
                CREATE TABLE doctors (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    specialty TEXT NOT NULL
                );
                CREATE TABLE slots (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    doctor_id INTEGER NOT NULL REFERENCES doctors(id),
                    start_utc TEXT NOT NULL,
                    end_utc TEXT NOT NULL
                );
                CREATE TABLE bookings (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    slot_id INTEGER NOT NULL REFERENCES slots(id),
                    patient_name TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'confirmed',
                    idempotency_key TEXT NOT NULL,
                    created_utc TEXT NOT NULL
                );
                INSERT INTO doctors (id, name, specialty)
                VALUES (1, 'Dr. Ada Lovelace', 'General Practice');
                """
            )
            conn.execute(
                "INSERT INTO slots (id, doctor_id, start_utc, end_utc) "
                "VALUES (1, 1, ?, ?)",
                (start.isoformat(), end.isoformat()),
            )
            conn.execute(
                "INSERT INTO slots (id, doctor_id, start_utc, end_utc) "
                "VALUES (2, 1, ?, ?)",
                (
                    (start + timedelta(hours=1)).isoformat(),
                    (end + timedelta(hours=1)).isoformat(),
                ),
            )
            conn.execute(
                "INSERT INTO bookings "
                "(slot_id, patient_name, status, idempotency_key, created_utc) "
                "VALUES (1, 'Alex Taylor', 'confirmed', 'legacy-confirmed', ?)",
                (created,),
            )

        init_db(db_path)

        conn = _connect(db_path)
        try:
            statuses = conn.execute(
                "SELECT id, status FROM slots ORDER BY id"
            ).fetchall()
        finally:
            conn.close()

        assert [(row["id"], row["status"]) for row in statuses] == [
            (1, "booked"),
            (2, "available"),
        ]


# ---------------------------------------------------------------------------
# Listing
# ---------------------------------------------------------------------------


class TestListSlots:
    def test_returns_future_slots(self, db: Path) -> None:
        slots = list_available_slots(db_path=db)
        assert len(slots) > 0

    def test_filter_by_doctor(self, db: Path) -> None:
        all_slots = list_available_slots(db_path=db)
        doc_id = all_slots[0].doctor_id
        filtered = list_available_slots(db_path=db, doctor_id=doc_id)
        assert all(s.doctor_id == doc_id for s in filtered)
        assert len(filtered) <= len(all_slots)

    def test_filter_by_local_date(self, db: Path) -> None:
        """Filtering by a clinic-local date returns only that day's slots."""
        tomorrow = datetime.now(CLINIC_TZ).date() + timedelta(days=1)
        slots = list_available_slots(db_path=db, local_date=tomorrow)
        assert len(slots) > 0
        # Verify every returned slot falls on the requested local date
        for s in slots:
            dt = datetime.fromisoformat(s.start_utc).astimezone(CLINIC_TZ)
            assert dt.date() == tomorrow

    def test_filter_by_doctor_and_date(self, db: Path) -> None:
        """Combined doctor + date filter works."""
        all_slots = list_available_slots(db_path=db)
        doc_id = all_slots[0].doctor_id
        tomorrow = datetime.now(CLINIC_TZ).date() + timedelta(days=1)
        filtered = list_available_slots(
            db_path=db, doctor_id=doc_id, local_date=tomorrow
        )
        assert all(s.doctor_id == doc_id for s in filtered)
        for s in filtered:
            dt = datetime.fromisoformat(s.start_utc).astimezone(CLINIC_TZ)
            assert dt.date() == tomorrow


class TestSlotHelpers:
    def test_get_slot_found(self, db: Path) -> None:
        slots = list_available_slots(db_path=db)
        slot = get_slot(slots[0].id, db_path=db)
        assert slot is not None
        assert slot.id == slots[0].id
        assert slot.doctor_name is not None

    def test_get_slot_not_found(self, db: Path) -> None:
        assert get_slot(999_999, db_path=db) is None

    def test_is_slot_available_lifecycle(self, db: Path) -> None:
        slots = list_available_slots(db_path=db)
        target_id = slots[0].id
        assert is_slot_available(target_id, db_path=db) is True

        # Book it
        b = book_slot(target_id, "Patient", idempotency_key="life-1", db_path=db)
        assert is_slot_available(target_id, db_path=db) is False

        # Cancel it
        cancel_booking(b.id, patient_name="Patient", db_path=db)
        assert is_slot_available(target_id, db_path=db) is True



# ---------------------------------------------------------------------------
# Booking — happy path
# ---------------------------------------------------------------------------


class TestBookSlot:
    def test_book_returns_confirmed(self, db: Path) -> None:
        slots = list_available_slots(db_path=db)
        booking = book_slot(
            slots[0].id, "Test Patient",
            idempotency_key="book-happy-1", db_path=db,
        )
        assert isinstance(booking, Booking)
        assert booking.status == "confirmed"
        assert booking.patient_name == "Test Patient"

    def test_booked_slot_no_longer_available(self, db: Path) -> None:
        slots = list_available_slots(db_path=db)
        target_id = slots[0].id
        book_slot(target_id, "Occupier", idempotency_key="book-happy-2", db_path=db)
        remaining_ids = {s.id for s in list_available_slots(db_path=db)}
        assert target_id not in remaining_ids


# ---------------------------------------------------------------------------
# Past-slot rejection
# ---------------------------------------------------------------------------


class TestSlotInPast:
    def test_book_past_slot_raises(self, db: Path) -> None:
        past_id = _insert_past_slot(db)
        with pytest.raises(BookingError) as exc:
            book_slot(past_id, "TimeTraveler", idempotency_key="past-1", db_path=db)
        assert exc.value.code == SLOT_IN_PAST


# ---------------------------------------------------------------------------
# Double-booking
# ---------------------------------------------------------------------------


class TestDoubleBooking:
    def test_second_booking_raises_unavailable(self, db: Path) -> None:
        slots = list_available_slots(db_path=db)
        book_slot(slots[0].id, "First", idempotency_key="dbl-1", db_path=db)
        with pytest.raises(BookingError) as exc:
            book_slot(slots[0].id, "Second", idempotency_key="dbl-2", db_path=db)
        assert exc.value.code == SLOT_UNAVAILABLE

    def test_failed_booking_insert_rolls_back_slot_claim(self, db: Path) -> None:
        """A failed booking insert must release the conditional slot claim."""
        slot = list_available_slots(db_path=db)[0]
        conn = _connect(db)
        try:
            conn.execute(
                "CREATE TRIGGER reject_test_booking BEFORE INSERT ON bookings "
                "BEGIN SELECT RAISE(ABORT, 'simulated insert failure'); END"
            )
            conn.commit()
        finally:
            conn.close()

        with pytest.raises(sqlite3.IntegrityError, match="simulated insert failure"):
            book_slot(
                slot.id,
                "Rollback Patient",
                idempotency_key="rollback-claim",
                db_path=db,
            )

        assert is_slot_available(slot.id, db_path=db)
        conn = _connect(db)
        try:
            status = conn.execute(
                "SELECT status FROM slots WHERE id = ?", (slot.id,)
            ).fetchone()["status"]
        finally:
            conn.close()
        assert status == "available"


# ---------------------------------------------------------------------------
# Idempotency
# ---------------------------------------------------------------------------


class TestIdempotency:
    def test_same_key_same_request_returns_original(self, db: Path) -> None:
        slots = list_available_slots(db_path=db)
        key = "idem-abc"
        b1 = book_slot(slots[0].id, "Pat", idempotency_key=key, db_path=db)
        b2 = book_slot(slots[0].id, "Pat", idempotency_key=key, db_path=db)
        assert b1.id == b2.id
        assert b2.status == "confirmed"

    def test_same_key_different_request_raises_conflict(self, db: Path) -> None:
        slots = list_available_slots(db_path=db)
        key = "idem-xyz"
        book_slot(slots[0].id, "Pat", idempotency_key=key, db_path=db)
        with pytest.raises(BookingError) as exc:
            book_slot(slots[1].id, "Pat", idempotency_key=key, db_path=db)
        assert exc.value.code == IDEMPOTENCY_CONFLICT

    def test_idempotent_retry_after_slot_passes(self, db: Path) -> None:
        """
        Regression test: An existing booking can be safely retried even after
        the appointment slot's start time has passed into history.
        """
        slots = list_available_slots(db_path=db)
        slot_id = slots[0].id
        key = "idem-past-safe"
        b1 = book_slot(slot_id, "Alice", idempotency_key=key, db_path=db)

        # Manually backdate the slot in SQLite to simulate time passing
        conn = _connect(db)
        try:
            past_time = (datetime.now(timezone.utc) - timedelta(hours=3)).isoformat()
            conn.execute("UPDATE slots SET start_utc = ? WHERE id = ?", (past_time, slot_id))
            conn.commit()
        finally:
            conn.close()

        # Retrying with the same idempotency key must succeed and NOT raise SLOT_IN_PAST
        b2 = book_slot(slot_id, "Alice", idempotency_key=key, db_path=db)
        assert b2.id == b1.id
        assert b2.status == "confirmed"

    def test_cancelled_booking_retry_returns_cancelled_status(self, db: Path) -> None:
        """
        Regression test: Retrying an idempotency key for a cancelled booking
        must return the booking with status='cancelled', never describing it as confirmed.
        """
        slots = list_available_slots(db_path=db)
        slot_id = slots[0].id
        key = "idem-cancelled"
        b1 = book_slot(slot_id, "Bob", idempotency_key=key, db_path=db)
        cancel_booking(b1.id, patient_name="Bob", db_path=db)

        # Retry with the same key
        b2 = book_slot(slot_id, "Bob", idempotency_key=key, db_path=db)
        assert b2.id == b1.id
        assert b2.status == "cancelled"


# ---------------------------------------------------------------------------
# Cancellation
# ---------------------------------------------------------------------------


class TestCancellation:
    def test_cancel_sets_status(self, db: Path) -> None:
        slots = list_available_slots(db_path=db)
        booking = book_slot(
            slots[0].id, "Soon Cancelled",
            idempotency_key="cancel-1", db_path=db,
        )
        result = cancel_booking(booking.id, patient_name="Soon Cancelled", db_path=db)
        assert result.status == "cancelled"

    def test_cancel_reopens_slot(self, db: Path) -> None:
        slots = list_available_slots(db_path=db)
        target_id = slots[0].id
        booking = book_slot(
            target_id, "Temp", idempotency_key="cancel-2", db_path=db
        )
        cancel_booking(booking.id, patient_name="Temp", db_path=db)
        available_ids = {s.id for s in list_available_slots(db_path=db)}
        assert target_id in available_ids
        conn = _connect(db)
        try:
            status = conn.execute(
                "SELECT status FROM slots WHERE id = ?", (target_id,)
            ).fetchone()["status"]
        finally:
            conn.close()
        assert status == "available"

    def test_rebook_after_cancel(self, db: Path) -> None:
        slots = list_available_slots(db_path=db)
        target_id = slots[0].id
        b1 = book_slot(target_id, "First", idempotency_key="cancel-3a", db_path=db)
        cancel_booking(b1.id, patient_name="First", db_path=db)
        b2 = book_slot(target_id, "Second", idempotency_key="cancel-3b", db_path=db)
        assert b2.status == "confirmed"
        assert b2.id != b1.id

    def test_cancel_nonexistent_raises(self, db: Path) -> None:
        with pytest.raises(BookingError) as exc:
            cancel_booking(999_999, patient_name="Nobody", db_path=db)
        assert exc.value.code == SLOT_NOT_FOUND


# ---------------------------------------------------------------------------
# Cancellation authorisation
# ---------------------------------------------------------------------------


class TestCancelAuthorisation:
    def test_wrong_patient_raises_not_authorised(self, db: Path) -> None:
        slots = list_available_slots(db_path=db)
        booking = book_slot(
            slots[0].id, "Alice",
            idempotency_key="auth-1", db_path=db,
        )
        with pytest.raises(BookingError) as exc:
            cancel_booking(booking.id, patient_name="Eve", db_path=db)
        assert exc.value.code == NOT_AUTHORISED

    def test_correct_patient_can_cancel(self, db: Path) -> None:
        slots = list_available_slots(db_path=db)
        booking = book_slot(
            slots[0].id, "Alice",
            idempotency_key="auth-2", db_path=db,
        )
        result = cancel_booking(booking.id, patient_name="Alice", db_path=db)
        assert result.status == "cancelled"


# ---------------------------------------------------------------------------
# Invalid slot
# ---------------------------------------------------------------------------


class TestInvalidSlot:
    def test_book_nonexistent_slot(self, db: Path) -> None:
        with pytest.raises(BookingError) as exc:
            book_slot(999_999, "Ghost", idempotency_key="inv-1", db_path=db)
        assert exc.value.code == SLOT_NOT_FOUND


# ---------------------------------------------------------------------------
# Concurrent bookings
# ---------------------------------------------------------------------------


class TestConcurrency:
    def test_concurrent_bookings_only_one_wins(self, db: Path) -> None:
        """
        Race two threads booking the *same* slot.
        Exactly one must succeed; the other must get SLOT_UNAVAILABLE.
        """
        slots = list_available_slots(db_path=db)
        target_id = slots[0].id
        results: list[Booking | BookingError] = [None, None]  # type: ignore[list-item]

        def attempt(idx: int) -> None:
            try:
                results[idx] = book_slot(
                    target_id, f"Racer-{idx}",
                    idempotency_key=f"race-{idx}",
                    db_path=db,
                )
            except BookingError as e:
                results[idx] = e

        t0 = threading.Thread(target=attempt, args=(0,))
        t1 = threading.Thread(target=attempt, args=(1,))
        t0.start()
        t1.start()
        t0.join()
        t1.join()

        wins = [r for r in results if isinstance(r, Booking)]
        fails = [r for r in results if isinstance(r, BookingError)]
        assert len(wins) == 1, f"Expected exactly 1 winner, got {len(wins)}"
        assert len(fails) == 1
        assert fails[0].code == SLOT_UNAVAILABLE

        conn = _connect(db)
        try:
            status = conn.execute(
                "SELECT status FROM slots WHERE id = ?", (target_id,)
            ).fetchone()["status"]
        finally:
            conn.close()
        assert status == "booked"
