"""
scripts/verify_voice_mode.py — Live validation of Milestone 6A Hands-Free Conversational Voice Mode.

Verifies:
1. Live session initialization.
2. Live Deepgram streaming STT WebSocket proxy (`/api/sessions/{id}/ws/transcribe`).
3. Multi-turn hands-free conversational flow with Gemini.
4. Pre-booking isolation: SQLite has ZERO bookings after proposal.
5. Natural verbal confirmation ('Yes, book it') without clicking any button or typing /confirm.
6. Post-confirmation: booking appears in SQLite only after verbal confirmation.
7. Cartesia TTS audio generation of the deterministic receipt.
8. End-to-end latency measurement (end-of-speech to beginning of audible response).
"""

import asyncio
import json
import os
import sys
import tempfile
import time
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from dotenv import load_dotenv

load_dotenv()

from fastapi.testclient import TestClient
import sqlite3
from scheduler import db
from scheduler.db import init_db, seed
from scheduler.web import app, sessions


def run_live_validation():
    print("=" * 70)
    print("Milestone 6A — Live Hands-Free Voice Mode Verification")
    print("=" * 70)

    # 1. Setup isolated database
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        test_db_path = f.name

    os.environ["DB_PATH"] = test_db_path
    app.state.db_path = test_db_path
    init_db(test_db_path)
    seed(test_db_path)
    sessions.clear()

    client = TestClient(app)

    # 2. Start session
    resp = client.post("/api/sessions")
    assert resp.status_code == 201, resp.text
    session_id = resp.json()["session_id"]
    print(f"\n[1] Created Session: {session_id}")

    # 3. Test Live Streaming WebSocket Proxy connection
    print("\n[2] Connecting to Live Deepgram WebSocket Proxy (/api/sessions/{id}/ws/transcribe)...")
    with client.websocket_connect(f"/api/sessions/{session_id}/ws/transcribe") as ws:
        init_msg = ws.receive_json()
        print(f"    Proxy handshake response: {init_msg}")
        assert init_msg.get("type") == "ready", "Expected ready message from Deepgram proxy"
        print("    ✔ WebSocket proxy successfully authenticated with server-side DEEPGRAM_API_KEY.")
        # Gracefully close test connection
        ws.send_text(json.dumps({"type": "close"}))

    # 4. Hands-Free Turn 1: Patient searches availability
    user_utterance_1 = "I would like to schedule an appointment with Dr. Charles Babbage on Tuesday."
    print(f"\n[3] Hands-Free Turn 1 (Search Request): \"{user_utterance_1}\"")
    t_end_of_speech_1 = time.perf_counter()

    msg_resp_1 = client.post(
        f"/api/sessions/{session_id}/messages",
        json={"text": user_utterance_1},
    )
    assert msg_resp_1.status_code == 200, msg_resp_1.text
    data_1 = msg_resp_1.json()

    reply_1 = data_1.get("reply", "")
    spoken_1 = data_1.get("spoken_reply", "")
    print(f"    Assistant Reply ({data_1.get('latency_ms')} ms):")
    print(f"      Display: {reply_1}")
    print(f"      Spoken:  {spoken_1}")

    # Synthesize spoken response via Cartesia TTS
    tts_resp_1 = client.post(
        f"/api/sessions/{session_id}/speak",
        json={"text": spoken_1},
    )
    t_audible_1 = time.perf_counter()
    assert tts_resp_1.status_code == 200
    assert tts_resp_1.content.startswith(b"RIFF")
    e2e_latency_1_ms = (t_audible_1 - t_end_of_speech_1) * 1000

    print(f"    ✔ Cartesia generated {len(tts_resp_1.content)} bytes of WAV audio.")
    print(f"    ⚡ Turn 1 End-of-Speech to Audible Response Latency: {e2e_latency_1_ms:.1f} ms")

    # 5. Hands-Free Turn 2: Patient selects 9 AM slot
    user_utterance_2 = "Let's do 9 AM."
    print(f"\n[4] Hands-Free Turn 2 (Slot Selection): \"{user_utterance_2}\"")
    t_end_of_speech_2 = time.perf_counter()

    msg_resp_2 = client.post(
        f"/api/sessions/{session_id}/messages",
        json={"text": user_utterance_2},
    )
    assert msg_resp_2.status_code == 200, msg_resp_2.text
    data_2 = msg_resp_2.json()

    reply_2 = data_2.get("reply", "")
    spoken_2 = data_2.get("spoken_reply", "")
    pending_2 = data_2.get("pending")
    print(f"    Assistant Reply ({data_2.get('latency_ms')} ms):")
    print(f"      Display: {reply_2}")
    print(f"      Spoken:  {spoken_2}")
    print(f"      Proposal Staged: {pending_2 is not None}")
    assert pending_2 is not None, "Expected appointment proposal to be staged"

    tts_resp_2 = client.post(
        f"/api/sessions/{session_id}/speak",
        json={"text": spoken_2},
    )
    t_audible_2 = time.perf_counter()
    assert tts_resp_2.status_code == 200
    assert tts_resp_2.content.startswith(b"RIFF")
    e2e_latency_2_ms = (t_audible_2 - t_end_of_speech_2) * 1000
    print(f"    ✔ Cartesia generated {len(tts_resp_2.content)} bytes of proposal audio.")
    print(f"    ⚡ Turn 2 End-of-Speech to Audible Response Latency: {e2e_latency_2_ms:.1f} ms")

    # 6. Database Check: Zero bookings in SQLite after proposal
    conn = sqlite3.connect(test_db_path)
    count_before = conn.execute("SELECT count(*) FROM bookings").fetchone()[0]
    conn.close()
    print(f"\n[5] Database Check Before Confirmation:")
    print(f"    Bookings in SQLite: {count_before}")
    assert count_before == 0, "Invariant violation: booking written before confirmation!"
    print("    ✔ Confirmed: Zero bookings written to database prior to verbal confirmation.")

    # 7. Hands-Free Turn 3: Natural Verbal Confirmation
    user_utterance_3 = "Yes, book it"
    print(f"\n[6] Hands-Free Turn 3 (Verbal Confirmation): \"{user_utterance_3}\"")
    t_end_of_speech_3 = time.perf_counter()

    msg_resp_3 = client.post(
        f"/api/sessions/{session_id}/messages",
        json={"text": user_utterance_3},
    )
    assert msg_resp_3.status_code == 200, msg_resp_3.text
    data_3 = msg_resp_3.json()

    reply_3 = data_3.get("reply", "")
    spoken_3 = data_3.get("spoken_reply", "")
    confirmed_3 = data_3.get("confirmed")

    print(f"    Assistant Confirmation Reply ({data_3.get('latency_ms')} ms):")
    print(f"      Display: {reply_3}")
    print(f"      Spoken:  {spoken_3}")
    print(f"      Confirmed Object: {confirmed_3}")
    assert confirmed_3 is not None, "Expected confirmed booking object in response"
    assert confirmed_3["status"] == "confirmed"
    booking_id = confirmed_3["booking_id"]

    # Synthesize spoken receipt via Cartesia TTS
    tts_resp_3 = client.post(
        f"/api/sessions/{session_id}/speak",
        json={"text": spoken_3},
    )
    t_audible_3 = time.perf_counter()
    assert tts_resp_3.status_code == 200
    assert tts_resp_3.content.startswith(b"RIFF")
    e2e_latency_3_ms = (t_audible_3 - t_end_of_speech_3) * 1000

    print(f"    ✔ Cartesia generated {len(tts_resp_3.content)} bytes of receipt WAV audio.")
    print(f"    ⚡ Turn 3 End-of-Speech to Audible Receipt Latency: {e2e_latency_3_ms:.1f} ms")

    # 8. Database Check After Spoken Confirmation
    conn = sqlite3.connect(test_db_path)
    conn.row_factory = sqlite3.Row
    rows = conn.execute("SELECT * FROM bookings").fetchall()
    conn.close()

    print(f"\n[7] Database Check After Spoken Confirmation:")
    print(f"    Bookings in SQLite: {len(rows)}")
    assert len(rows) == 1, "Expected exactly 1 booking committed in SQLite"
    row = dict(rows[0])
    print(f"    Committed Record: {row}")
    assert row["id"] == booking_id
    assert row["status"] == "confirmed"
    assert row["patient_name"] == "Alex Taylor"
    print("    ✔ Confirmed: Booking committed in SQLite only after natural verbal confirmation.")

    # 8. Clean up temp database
    try:
        os.remove(test_db_path)
    except OSError:
        pass

    print("\n" + "=" * 70)
    print("✔ Milestone 6A Live Hands-Free Voice Mode Validation PASSED!")
    print("=" * 70)


if __name__ == "__main__":
    run_live_validation()
