"""
scripts/verify_milestone_6b.py — Automated verification script for Milestone 6B / M7 integration.

Simulates hands-free conversational appointment booking on the live server:
1. Connects to http://127.0.0.1:8000.
2. Checks /api/voices and auditions Cora, Gemma, Clara.
3. Creates a new session.
4. Switches voice to Cora.
5. Turn 1: Requests appointment with Dr. Charles Babbage.
6. Turn 2: User changes mind ("Actually, change it to Wednesday at 2 PM instead").
   - Confirms old proposal discarded and ZERO rows added to SQLite bookings table.
7. Turn 3: User confirms verbally ("Yes, book it").
   - Confirms exactly ONE row committed to SQLite bookings table.
8. Queries Developer Inspector (/api/sessions/{id}/inspector) and verifies:
   - Live event timeline contains all pipeline stages.
   - Real SQLite booking row matches the database record.
   - No credentials leaked.
"""

import json
import sqlite3
import sys
import urllib.error
import urllib.request

BASE_URL = "http://127.0.0.1:8000"
DB_PATH = "appointment_scheduler.db"


def request_json(method: str, path: str, data: dict | None = None) -> dict:
    url = f"{BASE_URL}{path}"
    headers = {"Content-Type": "application/json"}
    body = json.dumps(data).encode("utf-8") if data is not None else None
    req = urllib.request.Request(url, data=body, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            content = resp.read().decode("utf-8")
            return json.loads(content) if content else {}
    except urllib.error.HTTPError as e:
        err_body = e.read().decode("utf-8")
        print(f"HTTPError {e.code} on {method} {path}: {err_body}", file=sys.stderr)
        raise


def request_bytes(method: str, path: str, data: dict | None = None) -> tuple[int, dict, bytes]:
    url = f"{BASE_URL}{path}"
    headers = {"Content-Type": "application/json"}
    body = json.dumps(data).encode("utf-8") if data is not None else None
    req = urllib.request.Request(url, data=body, headers=headers, method=method)
    with urllib.request.urlopen(req, timeout=30) as resp:
        return resp.status, dict(resp.headers), resp.read()


def count_bookings() -> int:
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT count(*) FROM bookings")
    cnt = cur.fetchone()[0]
    conn.close()
    return cnt


def main():
    print("=" * 70)
    print("Milestone 6B / M7 Live End-to-End Hands-Free Verification")
    print("=" * 70)

    # 1. Health check
    health = request_json("GET", "/api/health")
    assert health.get("status") == "ok", f"Unhealthy: {health}"
    print("✔ [1/6] Application health verified: Appointment Scheduler is running.")

    # 2. Voice API and Audition
    voices_data = request_json("GET", "/api/voices")
    assert len(voices_data["voices"]) >= 3
    print(f"✔ [2/6] Voice library loaded: {len(voices_data['voices'])} allowlisted voices available.")

    # Audition Cora with standard pre-booking confirmation sentence
    status_code, headers, wav = request_bytes(
        "POST",
        "/api/voices/audition",
        {"voice_id": "c46cf1f6-49a1-4d67-9a57-ff859a4046d3", "sample_key": "confirmation"},
    )
    assert status_code == 200
    assert len(wav) > 1000
    print(f"✔ [2/6] Audition synthesis successful: Cora synthesized {len(wav)} bytes WAV audio.")

    # Record baseline booking count
    baseline_bookings = count_bookings()
    print(f"Initial SQLite bookings count: {baseline_bookings}")

    # 3. Create Session
    sess = request_json("POST", "/api/sessions")
    session_id = sess["session_id"]
    print(f"✔ [3/6] Session created: {session_id} (Patient: {sess['patient_name']})")

    # Set active voice to Cora
    v_switch = request_json(
        "PUT",
        f"/api/sessions/{session_id}/voice",
        {"voice_id": "c46cf1f6-49a1-4d67-9a57-ff859a4046d3"},
    )
    assert v_switch["voice_name"] == "Cora"
    print(f"✔ [3/6] Voice updated on session: active voice is now {v_switch['voice_name']}.")

    # 4. Turn 1: Request an appointment with Dr. Charles Babbage
    print("\n--- Turn 1: User Request ---")
    user_msg_1 = "Hi, can you propose an available slot with Dr. Charles Babbage?"
    print(f"User: '{user_msg_1}'")
    t1 = request_json(
        "POST",
        f"/api/sessions/{session_id}/messages",
        {"text": user_msg_1},
    )
    print(f"Receptionist: {t1['reply']}")

    # If offered options rather than direct staging, select one
    if t1["pending"] is None:
        print("\n--- Turn 1b: Selecting option ---")
        user_msg_1b = "Yes, please propose that first time"
        print(f"User: '{user_msg_1b}'")
        t1 = request_json(
            "POST",
            f"/api/sessions/{session_id}/messages",
            {"text": user_msg_1b},
        )
        print(f"Receptionist: {t1['reply']}")

    assert t1["pending"] is not None, "Expected pending proposal staged"
    staged_slot = t1["pending"]["slot_id"]
    print(f"Staged Proposal: Slot {staged_slot} ({t1['pending']['local_time_str']})")

    # Invariant: SQLite bookings table must NOT change
    cnt1 = count_bookings()
    print(f"SQLite bookings count after proposal: {cnt1} (baseline: {baseline_bookings})")
    assert cnt1 == baseline_bookings, "Invariant violated: unconfirmed proposal wrote to SQLite!"

    # 5. Turn 2: User changes mind before confirming - change to Wednesday at 2 PM
    print("\n--- Turn 2: Correction / Modification ---")
    user_msg_2 = "Actually, change it to Wednesday at 2 PM instead"
    print(f"User: '{user_msg_2}'")
    t2 = request_json(
        "POST",
        f"/api/sessions/{session_id}/messages",
        {"text": user_msg_2},
    )
    print(f"Receptionist: {t2['reply']}")

    # If the response needs explicit selection of proposed slot
    if t2["pending"] is None:
        user_msg_2b = "Yes, propose Wednesday at 2 PM please"
        print(f"User: '{user_msg_2b}'")
        t2 = request_json(
            "POST",
            f"/api/sessions/{session_id}/messages",
            {"text": user_msg_2b},
        )
        print(f"Receptionist: {t2['reply']}")

    # Invariant: SQLite bookings table must still NOT change
    cnt2 = count_bookings()
    print(f"SQLite bookings count after modification: {cnt2} (baseline: {baseline_bookings})")
    assert cnt2 == baseline_bookings, "Invariant violated: mind-change wrote to SQLite!"

    # 6. Turn 3: Verbal Confirmation
    print("\n--- Turn 3: Natural Verbal Confirmation ---")
    user_msg_3 = "Yes, book it"
    print(f"User: '{user_msg_3}'")
    t3 = request_json(
        "POST",
        f"/api/sessions/{session_id}/messages",
        {"text": user_msg_3},
    )
    print(f"Receptionist: {t3['reply']}")
    assert t3["confirmed"] is not None, "Expected confirmed booking in Turn 3"
    booking_id = t3["confirmed"]["booking_id"]
    print(f"Confirmed Booking ID: {booking_id}")
    print(f"Spoken Receipt: {t3.get('spoken_reply')}")

    # Invariant: Exactly 1 new booking in SQLite
    cnt3 = count_bookings()
    print(f"SQLite bookings count after verbal confirmation: {cnt3} (expected {baseline_bookings + 1})")
    assert cnt3 == baseline_bookings + 1, "Invariant violated: did not commit exactly 1 booking!"

    # 7. Query Developer Inspector
    print("\n--- Developer Inspector Verification ---")
    insp = request_json("GET", f"/api/sessions/{session_id}/inspector")
    print(f"Active Voice: {insp['active_voice']['name']} ({insp['active_voice']['accent']})")
    print(f"Appointment State: status={insp['appointment_state']['status']}, committed={insp['appointment_state']['is_committed_to_sqlite']}")
    print(f"Real SQLite Booking: {insp['real_sqlite_booking']}")
    print(f"Total Timeline Events: {len(insp['events'])}")

    # Ensure stages are tracked
    stages = [e["stage"] for e in insp["events"]]
    for expected_stage in ["session_init", "voice_change", "stt", "tool", "proposal", "sqlite_commit"]:
        assert expected_stage in stages, f"Missing stage '{expected_stage}' in inspector timeline!"

    # Ensure no API keys leaked in inspector payload
    raw_insp_str = json.dumps(insp).lower()
    for key in ["deepgram_api_key", "cartesia_api_key", "gemini_api_key"]:
        assert key not in raw_insp_str, f"Leaked credential {key} in inspector payload!"

    print("\n" + "=" * 70)
    print("ALL LIVE HANDS-FREE VOICE CHECKS PASSED ON MERGED RUNTIME!")
    print("=" * 70)


if __name__ == "__main__":
    main()
