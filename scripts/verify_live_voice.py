import os
import sys
import time
import tempfile
from dotenv import load_dotenv

# Ensure scheduler module can be imported
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

load_dotenv()

from fastapi.testclient import TestClient
from scheduler.web import app, sessions
from scheduler.db import init_db, seed

# Create a clean temp database
with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
    db_path = f.name

os.environ["DB_PATH"] = db_path
app.state.db_path = db_path
init_db(db_path)
seed(db_path)
sessions.clear()

client = TestClient(app)

print("--- 1. Testing Live Session Creation ---")
resp = client.post("/api/sessions")
assert resp.status_code == 201, resp.text
session_id = resp.json()["session_id"]
print(f"Created session: {session_id}")

print("\n--- 2. Testing Live Cartesia TTS (/speak) ---")
speak_text = "Hello! I can help you schedule an appointment with Dr. Rivera or Dr. Chen."
t0 = time.perf_counter()
resp = client.post(f"/api/sessions/{session_id}/speak", json={"text": speak_text})
tts_ms = (time.perf_counter() - t0) * 1000
assert resp.status_code == 200, resp.text
audio_bytes = resp.content
assert audio_bytes.startswith(b"RIFF"), "Missing RIFF header"
assert b"WAVE" in audio_bytes[:16], "Missing WAVE format"
print(f"Cartesia TTS returned {len(audio_bytes)} bytes of valid RIFF/WAVE audio in {tts_ms:.1f}ms")

print("\n--- 3. Testing Live Deepgram STT (/transcribe) ---")
t0 = time.perf_counter()
files = {"file": ("sample.wav", audio_bytes, "audio/wav")}
resp = client.post(f"/api/sessions/{session_id}/transcribe", files=files)
stt_ms = (time.perf_counter() - t0) * 1000
assert resp.status_code == 200, resp.text
tr_data = resp.json()
transcript = tr_data.get("transcript", "")
confidence = tr_data.get("confidence", 0.0)
print(f"Deepgram transcribed in {stt_ms:.1f}ms:")
print(f"  Transcript: \"{transcript}\" (confidence: {confidence})")

print("\n--- 4. Testing Live Conversation Loop with Gemini ---")
# User turn 1: Search doctors / slots
user_msg_1 = "I need an appointment with Dr. Ada Lovelace next week."
t0 = time.perf_counter()
resp = client.post(f"/api/sessions/{session_id}/messages", json={"text": user_msg_1})
llm_ms_1 = (time.perf_counter() - t0) * 1000
assert resp.status_code == 200, resp.text
msg_data_1 = resp.json()
reply_1 = msg_data_1.get("reply", "")
spoken_1 = msg_data_1.get("spoken_reply", "")
pending_1 = msg_data_1.get("pending")
print(f"Turn 1 ({llm_ms_1:.1f}ms):")
print(f"  Assistant reply: {reply_1}")
print(f"  Spoken reply: {spoken_1}")
print(f"  Pending: {pending_1 is not None}")
assert "slot_id" not in spoken_1.lower(), "Slot ID leaked in spoken text!"

# User turn 2: Change appointment / choose a different doctor
user_msg_2 = "Actually what does Dr. Charles Babbage have on Tuesday?"
t0 = time.perf_counter()
resp = client.post(f"/api/sessions/{session_id}/messages", json={"text": user_msg_2})
llm_ms_2 = (time.perf_counter() - t0) * 1000
assert resp.status_code == 200, resp.text
msg_data_2 = resp.json()
reply_2 = msg_data_2.get("reply", "")
spoken_2 = msg_data_2.get("spoken_reply", "")
pending_2 = msg_data_2.get("pending")
print(f"\nTurn 2 ({llm_ms_2:.1f}ms):")
print(f"  Assistant reply: {reply_2}")
print(f"  Spoken reply: {spoken_2}")
print(f"  Pending: {pending_2 is not None}")
assert "slot_id" not in spoken_2.lower(), "Slot ID leaked in spoken text!"

# User turn 3: Ask to book the morning slot
user_msg_3 = "Let us book the 9 AM slot with Dr. Charles Babbage on Tuesday, November 17."
t0 = time.perf_counter()
resp = client.post(f"/api/sessions/{session_id}/messages", json={"text": user_msg_3})
llm_ms_3 = (time.perf_counter() - t0) * 1000
assert resp.status_code == 200, resp.text
msg_data_3 = resp.json()
reply_3 = msg_data_3.get("reply", "")
spoken_3 = msg_data_3.get("spoken_reply", "")
pending_3 = msg_data_3.get("pending")
print(f"\nTurn 3 ({llm_ms_3:.1f}ms):")
print(f"  Assistant reply: {reply_3}")
print(f"  Spoken reply: {spoken_3}")
print(f"  Pending: {pending_3 is not None}")
assert pending_3 is not None, "Expected pending proposal to be staged"
assert "[Pending Confirmation]" not in spoken_3, "Pending confirmation banner leaked into spoken text!"
assert "/confirm" not in spoken_3, "CLI /confirm instruction leaked into spoken text!"

# Step 5: Explicit confirmation via backend
print("\n--- 5. Testing Explicit Confirmation ---")
resp = client.post(f"/api/sessions/{session_id}/confirm")
assert resp.status_code == 200, resp.text
conf_data = resp.json()
success_val = conf_data.get("success")
status_val = conf_data.get("status")
booking_id_val = conf_data.get("booking_id")
spoken_receipt = conf_data.get("spoken_message")
print("Confirmation response:")
print(f"  Success: {success_val}")
print(f"  Status: {status_val}")
print(f"  Booking ID: {booking_id_val}")
print(f"  Spoken message: \"{spoken_receipt}\"")
assert success_val is True
assert status_val == "confirmed"
assert spoken_receipt is not None

# Step 6: Spoken receipt audio generation via Cartesia
print("\n--- 6. Testing Spoken Receipt Audio Synthesis ---")
t0 = time.perf_counter()
resp = client.post(f"/api/sessions/{session_id}/speak", json={"text": spoken_receipt})
receipt_tts_ms = (time.perf_counter() - t0) * 1000
assert resp.status_code == 200, resp.text
receipt_audio = resp.content
assert receipt_audio.startswith(b"RIFF"), "Missing RIFF header on receipt audio"
print(f"Spoken receipt generated {len(receipt_audio)} bytes of RIFF/WAVE in {receipt_tts_ms:.1f}ms")

print("\n=== Live End-to-End Voice Workflow Passed Successfully! ===")

# Clean up temp db
try:
    os.remove(db_path)
except OSError:
    pass
