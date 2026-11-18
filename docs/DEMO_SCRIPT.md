# Appointment Scheduler — 5-Minute Live Interview Demo Script

This document provides a turnkey, turn-by-turn spoken dialogue script and demonstration guide for showcasing Appointment Scheduler during a technical interview. It is designed to be executed in under 5 minutes while highlighting the system's core technical differentiators: **hands-free voice streaming**, **the "Hard Boundary" staging architecture**, **deterministic verbal confirmation**, **acoustic barge-in handling**, and the **Developer Inspector**.

---

## 1. Demo Overview & Persona

- **Time Required**: 4 to 5 minutes.
- **Presenter Role**: Candidate / Systems Engineer presenting architecture and live behavior.
- **Demonstrated Patient Persona**: **Alex Taylor** (fictional demo identity; unauthenticated test user).
- **Primary Goal**: Walk through a natural, multi-turn clinical scheduling consultation where the patient searches for a doctor, stages an appointment, changes their mind mid-flow, and confirms verbally—all while proving via the Developer Inspector that SQLite is never touched until explicit verbal confirmation occurs.

---

## 2. Prerequisites & Setup Checklist

Before opening the call or screen share:

1. **Verify Environment Variables (`.env`)**:
   ```bash
   GEMINI_API_KEY=your_gemini_api_key
   DEEPGRAM_API_KEY=your_deepgram_api_key
   CARTESIA_API_KEY=your_cartesia_api_key
   ```
2. **Start Backend Web Server**:
   ```bash
   python -m uvicorn scheduler.web:app --host 127.0.0.1 --port 8000 --reload
   ```
3. **Open Browser**:
   - Navigate to `http://127.0.0.1:8000` in Google Chrome or Arc.
   - Grant microphone permissions when prompted.
4. **Prepare Browser Windows**:
   - Set browser zoom to 100% or 110%.
   - Click the **"🔍 Inspector"** button in the top right navigation bar to open the Developer Inspector side drawer.
   - Point out the initial state: `Appointment State: IDLE`, `Committed to SQLite: None`, `Events: 0`.
   - Point out the active Cartesia voice selector (default: *Madame Forestier* / British or *Dr. Sarah Brooks* / American).

---

## 3. Step-by-Step Live Demo Script

```
                +-------------------------------------------------------+
                |                    DEMO TIMELINE                      |
                |                                                       |
 [0:00 - 0:45]  | Turn 1: Initial Greeting & Specialty Search           |
 [0:45 - 1:45]  | Turn 2: Slot Selection & In-Memory Staging            |
 [1:45 - 2:45]  | Turn 3: Correction & Stale State Discard              |
 [2:45 - 3:45]  | Turn 4: Deterministic Verbal Confirmation & Commit    |
 [3:45 - 4:45]  | Inspector Deep-Dive, Acoustic Barge-In & Failure Modes|
                +-------------------------------------------------------+
```

---

### Turn 1: Greeting & Specialty Search

**Interviewer Context**: Demonstrates initial browser-to-FastAPI WebSocket initialization, streaming audio capture via Deepgram Nova-3, read-only tool invocation (`list_doctors` and `search_slots`), natural conversational response shaping (max 2 choices, 15–30 words), and Cartesia voice playback.

1. **Presenter Action**: Click **"🎙️ Start Voice Mode"**.
   - *Visual cue*: Status changes to `● Streaming Active`.
   - The microphone opens; WebSocket connects to `/api/sessions/{session_id}/ws/transcribe`.
2. **Spoken User Dialogue**:
   > *"Hi there, I'm looking to book an appointment with a cardiologist sometime this week."*
3. **Expected Assistant Spoken Response**:
   > *"Hello! Dr. Ada Lovelace in Cardiology has openings this Tuesday, November 17th at 9:00 AM and 2:00 PM. Would either of those times work for you?"*
4. **Technical Narration for Interviewer**:
   > *"Notice how concise that turn was—Gemini is instructed to present at most two options and ask one focused question. If you look at the Inspector on the right, you can see the event cascade: STT transcribed the audio, Gemini called `search_slots()`, and Cartesia synthesized the audio response. Most importantly: look at the database state in the inspector—it is completely empty. No booking has been staged, and no database transaction has executed."*

---

### Turn 2: Slot Selection & In-Memory Staging (The "Hard Boundary")

**Interviewer Context**: Demonstrates Gemini executing `propose_booking(slot_id)` to stage the slot into session memory without writing to SQLite. Shows clear read-back and confirmation prompt.

1. **Spoken User Dialogue**:
   > *"Let's do Tuesday morning at 9:00 AM."*
2. **Expected Assistant Spoken Response**:
   > *"I have staged Dr. Ada Lovelace for Tuesday, November 17th at 9:00 AM. Would you like me to go ahead and book this appointment for you?"*
3. **Technical Narration for Interviewer**:
   > *"Here is the most critical architectural boundary in our system. Gemini called `propose_booking(slot_id=1)`, but notice what happened in the Developer Inspector:
   > - The badge now reads **AWAITING_CONFIRMATION** in amber.
   > - The slot is staged in Python memory with a server-generated idempotency key `book-1-xxxx`.
   > - But look at the 'Real SQLite Database Row' card: **No appointment committed in SQLite yet**.
   >
   > Gemini does not have write access to SQLite. It literally lacks the tool to commit a booking. It can only stage intent. If the patient drops the call right now, nothing is booked in the clinic database."*

---

### Turn 3: Correction & Interruption (Changing Mind Mid-Flow)

**Interviewer Context**: Tests intent modification, stale state invalidation, and acoustic echo protection. Shows that when a patient changes their mind, Python's deterministic classifier intercepts the utterance, discards the staged proposal, and instructs Gemini to stage the new selection.

1. **Spoken User Dialogue**:
   > *"Actually, wait—something came up Tuesday morning. Can we switch to the afternoon at 2:00 PM instead?"*
2. **Expected Assistant Spoken Response**:
   > *"No problem. I've updated that to Dr. Ada Lovelace on Tuesday, November 17th at 2:00 PM. Would you like me to book this afternoon appointment for you?"*
3. **Technical Narration for Interviewer**:
   > *"Notice how the system handled that change of mind:
   > 1. Python's regex classifier evaluated `is_verbal_modification()`. It matched the words 'actually', 'switch', and '2:00 PM'.
   > 2. Because a modification was detected, Python immediately discarded the 9:00 AM proposal from session memory.
   > 3. The request was forwarded to Gemini, which called `propose_booking(slot_id=2)` for the 2:00 PM slot.
   > 4. In the Inspector, the staged slot updated from Slot #1 to Slot #2. The database remains completely untouched."*

---

### Turn 4: Deterministic Verbal Confirmation & Atomic SQLite Commit

**Interviewer Context**: Demonstrates the climax of the booking flow: natural verbal confirmation ("Yes, please book it") intercepted by Python, executing `confirm_booking()` directly via an ACID transaction (`BEGIN IMMEDIATE`), and returning a deterministic spoken receipt with the committed Booking ID.

1. **Spoken User Dialogue**:
   > *"Yes, please book that appointment."*
2. **Expected Assistant Spoken Response**:
   > *"Your appointment with Dr. Ada Lovelace on Tuesday, November 17 at 2:00 PM PST is confirmed. Your confirmation number is Booking #1. We look forward to seeing you."*
3. **Technical Narration for Interviewer**:
   > *"Now look at the Developer Inspector:
   > - The status badge flipped to green: **CONFIRMED**.
   > - Look at the 'Real SQLite Database Row' card: we just queried the actual SQLite database on disk. Booking ID #1 is stored with `slot_id=2`, `patient_name='Alex Taylor'`, `status='confirmed'`, and the exact UTC timestamp.
   > - Look at the timeline event for this turn: `sqlite_commit` took **3.2 milliseconds**.
   > - Crucially, **Gemini was never invoked on this turn**. Python's verbal confirmation classifier detected the affirmative phrase, bypassed the LLM completely to eliminate hallucination risk, executed the atomic SQL transaction, and formatted the spoken receipt deterministically."*

---

## 4. Demonstrating Key Edge Cases & Advanced Features

### Feature A: The Developer Inspector Deep Dive
- Click on individual events in the Inspector timeline.
- Expand the `proposal` event: show the parameters (`slot_id`, `doctor`, `time`, `idempotency_key`).
- Expand the `sqlite_commit` event: show `committed_to_sqlite: true` and the SQL transaction latency.
- Explain the security boundary:
  > *"The inspector endpoint is guarded by `_is_local_request()`. If called from any non-loopback IP, it returns HTTP 403 Forbidden. Furthermore, API keys and internal tokens are sanitized before rendering."*

---

### Feature B: Acoustic Barge-In & Interruption Handling
Explain and demonstrate how Appointment Scheduler handles patient interruption:
1. **The Problem**: If the assistant is speaking and the patient interrupts, speech from the computer speakers can feed back into the microphone, creating an acoustic echo loop or false cancellation.
2. **The Defense**:
   - When assistant audio starts playing, the frontend marks `isAssistantSpeaking = true` and suppresses outgoing audio chunks from triggering premature turn submissions.
   - A **300ms post-playback guard window** absorbs acoustic reverberation before re-arming the microphone.
   - If the user clicks **"Mute"** or starts speaking with high energy, the browser instantly halts the WebAudio playback buffer and signals the backend to flush state.

---

### Feature C: Error Fallback & Guardrails
If the interviewer asks to see failure recovery, describe or demonstrate these three scenarios:

1. **Negation Guard ("Do NOT book it")**:
   - If a proposal is pending and the user says *"Actually, do not book that"*, the system does **not** trigger on the word "book". Python's `negative_words` filter (`no, not, dont, never, cancel, stop, discard`) catches the negation and routes to `cancel_pending()`.
2. **Ambiguity Guard ("Wait, let me check my calendar")**:
   - If the user hesitates, `is_verbal_ambiguity()` catches phrases like *"Hold on"*, *"Maybe"*, or *"Wait"*.
   - Instead of canceling or booking, the agent patiently asks: *"Just to confirm, would you like me to book Dr. Lovelace on Tuesday at 2:00 PM? Say yes to confirm or let me know if you want to change it."*
3. **Double-Booking Race Condition Defense**:
   - If another patient books the slot while this patient is deliberating, `book_slot()` catches SQLite's `UNIQUE constraint failed: bookings.slot_id` error on the partial index `uix_slot_confirmed`.
   - The transaction cleanly rolls back, and the agent gracefully tells the patient: *"Cannot confirm: this appointment is unavailable in the database. Please search for another available slot."*

---

## 5. Quick Reference Card for Presenter

| Turn | Spoken User Input | Target Backend Action | Visual Verification in Inspector |
|---|---|---|---|
| **Turn 1** | *"Hi, I need a cardiologist this week."* | STT -> Gemini `search_slots()` -> Cartesia | DB is empty (`status: idle`). |
| **Turn 2** | *"Let's do Tuesday at 9:00 AM."* | STT -> Gemini `propose_booking(1)` -> Cartesia | Amber `AWAITING_CONFIRMATION` banner. DB still empty! |
| **Turn 3** | *"Actually, can we switch to 2:00 PM?"* | Python regex -> Discard Slot 1 -> Gemini `propose_booking(2)` | Staged slot updates to Slot #2. DB still empty. |
| **Turn 4** | *"Yes, please book that appointment."* | Python regex -> Deterministic `confirm_booking()` -> SQLite | Green `CONFIRMED` badge. Real SQLite row rendered. |
