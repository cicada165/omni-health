# Appointment Scheduler — Verbal Confirmation Sequence & Staging Flow

This document provides a deep architectural trace of the **verbal confirmation sequence** in Appointment Scheduler. It illustrates how patient speech transitions from raw audio into an uncommitted proposal, and how verbal affirmation is deterministically intercepted and committed to SQLite without delegating booking authority to the LLM.

---

## 1. Sequence Diagram: Turn-by-Turn Trace

The following Mermaid sequence diagram traces two critical conversational turns:
1. **Turn N (Staging)**: The patient selects a slot; Gemini invokes `propose_booking(slot_id)` to stage the appointment in memory.
2. **Turn N+1 (Confirmation)**: The patient affirms verbally (*"Yes, book it"*); Python intercepts the utterance, executes an atomic ACID transaction against SQLite, formats the receipt deterministically, and invokes Cartesia TTS.

```mermaid
sequenceDiagram
    autonumber
    actor Patient as Patient (User)
    participant Browser as Browser Client (WebAudio / JS)
    participant WS as FastAPI WS STT Proxy
    participant Deepgram as Deepgram Nova-3 (STT)
    participant Web as FastAPI Message Handler
    participant Verbal as Python Verbal Classifier
    participant Agent as Gemini Agent (google-genai)
    participant DB as SQLite WAL (appointment_scheduler.db)
    participant Cartesia as Cartesia TTS (sonic-3.6)

    %% TURN N: STAGING
    Note over Patient, Cartesia: TURN N: Patient Selects Slot -> In-Memory Staging
    Patient->>Browser: Speaks: "Tuesday at 9 AM sounds good."
    Browser->>WS: Stream Binary PCM Audio (WebSocket)
    WS->>Deepgram: Forward Binary Audio (Upstream WSS)
    Deepgram-->>WS: speech_final=true: "Tuesday at 9:00 AM sounds good."
    WS-->>Browser: JSON Final Transcript
    Browser->>Web: POST /api/sessions/{id}/messages {"text": "Tuesday at 9:00 AM..."}
    Web->>Web: Acquire session.lock
    Web->>Verbal: Check awaiting_verbal_confirmation (False)
    Verbal->>Agent: Forward utterance to Gemini turn()
    Agent->>Agent: LLM Reason: Match slot for Tuesday 9:00 AM
    Agent->>DB: Tool: search_slots(local_date="2026-11-17")
    DB-->>Agent: Slot #1: Dr. Lovelace, 09:00 PST (Available)
    Agent->>Agent: Tool: propose_booking(slot_id=1)
    Note over Agent: Validates slot in SQLite<br/>Stages PendingBooking in RAM<br/>Generates book-1-uuid<br/>awaiting_verbal_confirmation = True<br/>NO DATABASE INSERT!
    Agent-->>Web: Staged confirmation prompt: "I've staged Dr. Lovelace... Would you like me to book it?"
    Web->>Cartesia: Synthesize response audio
    Cartesia-->>Browser: Stream 44.1kHz WAV Audio
    Browser->>Patient: Plays: "I've staged Dr. Lovelace for Tuesday at 9:00 AM. Would you like me to book it?"
    Web->>Web: Release session.lock

    %% TURN N+1: VERBAL CONFIRMATION
    Note over Patient, Cartesia: TURN N+1: Verbal Confirmation -> Deterministic SQLite Commit
    Patient->>Browser: Speaks: "Yes, please book it."
    Browser->>WS: Stream Binary PCM Audio (WebSocket)
    WS->>Deepgram: Forward Binary Audio
    Deepgram-->>WS: speech_final=true: "Yes, please book it."
    WS-->>Browser: JSON Final Transcript
    Browser->>Web: POST /api/sessions/{id}/messages {"text": "Yes, please book it."}
    Web->>Web: Acquire session.lock
    Web->>Verbal: Evaluate is_verbal_confirmation("Yes, please book it.")
    Note over Verbal: awaits_conf = True<br/>Negation Guard: Passed (no negations)<br/>Affirmation Match: "yes", "book"<br/>INTERCEPTION TRIGGERED!
    
    %% BYPASS GEMINI ENTIRELY
    Verbal->>Agent: Call Agent.confirm_booking() (Bypasses Gemini LLM)
    Agent->>DB: BEGIN IMMEDIATE
    Agent->>DB: Check idempotency key & slot availability
    Agent->>DB: INSERT INTO bookings (slot_id=1, status='confirmed', ...)
    Note over DB: Partial Index Guard:<br/>uix_slot_confirmed<br/>Transaction Committed!
    DB-->>Agent: Returns Booking #1 (Committed)
    Note over Agent: pending_booking.status = 'confirmed'<br/>pending_booking.booking_id = 1<br/>awaiting_verbal_confirmation = False
    Agent->>Agent: format_spoken_receipt(Dr. Lovelace, 9:00 AM, Booking #1)
    Agent-->>Web: Return deterministic receipt text
    Web->>Cartesia: Synthesize receipt audio
    Cartesia-->>Browser: Stream 44.1kHz WAV Audio
    Browser->>Patient: Plays: "Your appointment with Dr. Ada Lovelace on Tuesday, November 17 at 9:00 AM PST is confirmed. Booking #1."
    Web->>Web: Release session.lock
```

---

## 2. Step-by-Step Architectural Walkthrough

### Phase 1: Ingestion & Transcription
1. **Audio Capture**: The patient speaks into their microphone. WebAudio captures 16-bit PCM frames.
2. **Streaming WebSocket**: Audio is streamed over `/api/sessions/{id}/ws/transcribe` to the FastAPI backend, which proxies frames to Deepgram Nova-3.
3. **Endpointing**: Deepgram's Voice Activity Detection (VAD) detects a 300ms pause and marks `speech_final=true`.
4. **Utterance Dispatch**: The browser receives the JSON transcript and immediately dispatches a `POST` request to `/api/sessions/{id}/messages`.

---

### Phase 2: In-Memory Staging (`propose_booking`)
1. **Session Serialization**: FastAPI acquires `session.lock` to serialize turn execution.
2. **Gemini Tool Invocation**: Gemini parses the user's intent (*"Tuesday morning at 9:00 AM"*), queries slot availability via `search_slots()`, and decides to call `propose_booking(slot_id=1)`.
3. **The Staging Invariant**:
   - `propose_booking()` checks that the slot exists and is unbooked.
   - It instantiates a `PendingBooking` object in the agent's memory:
     ```python
     PendingBooking(
         slot_id=1,
         doctor_name="Dr. Ada Lovelace",
         specialty="Cardiology",
         local_time_str="Tuesday, Nov 17, 2026 at 09:00 AM PST",
         patient_name="Alex Taylor",
         idempotency_key="book-1-e4f8b2a19c3d",
         status="awaiting_confirmation",
     )
     ```
   - It sets `agent.awaiting_verbal_confirmation = True`.
   - **Zero SQL Mutating Queries**: No `INSERT` or `UPDATE` statements are sent to SQLite. The database remains in its pristine initial state.
4. **Spoken Confirmation Read-Back**: Gemini generates a concise spoken prompt reading back the physician, day, date, and local time, ending with an explicit confirmation question:
   > *"I have staged Dr. Ada Lovelace for Tuesday, November 17th at 9:00 AM. Would you like me to go ahead and book this appointment for you?"*

---

### Phase 3: Verbal Confirmation Interception
1. **User Affirmation**: The patient responds naturally: *"Yes, please book it."*
2. **Deterministic Interception**: In `scheduler/agent.py:handle_turn()`, the code checks if `self.awaiting_verbal_confirmation` is `True`.
3. **Regex Classification Pipeline**:
   - **Modification Check (`is_verbal_modification`)**: Checks if the patient is changing their mind (e.g. *"Actually, what about 2 PM?"*). If true, discards the proposal and invokes Gemini.
   - **Cancellation Check (`is_verbal_cancellation`)**: Checks if the patient is canceling (e.g. *"No, don't book that"*). If true, calls `cancel_pending()` and returns immediately.
   - **Ambiguity Check (`is_verbal_ambiguity`)**: Checks if the user is hesitant (e.g. *"Wait, let me check"*). If true, repeats the confirmation prompt without mutating state.
   - **Affirmation Match (`is_verbal_confirmation`)**:
     - Validates that no negation words exist (`no`, `not`, `dont`, `never`, `cancel`, `stop`, `discard`).
     - Matches affirmative tokens (`yes`, `book`, `confirm`, `please`, `sounds good`, `proceed`).
4. **Total LLM Bypass**: Once affirmative confirmation is verified, **Gemini is not invoked**. The flow routes directly to Python's `agent.confirm_booking()`.

---

### Phase 4: Atomic Commit & Spoken Receipt
1. **ACID Transaction**:
   - `scheduler/db.py:book_slot()` opens a connection and executes `BEGIN IMMEDIATE`, instantly obtaining SQLite's reserved file lock.
   - Validates that the slot is still in the future and not booked.
   - Attempts insertion into the `bookings` table with status `'confirmed'` and the pre-generated `idempotency_key`.
   - SQLite enforces the partial unique index `uix_slot_confirmed`. If a race condition occurred and another session booked the slot, SQLite raises `sqlite3.IntegrityError`, which rolls back the transaction and raises `BookingError(SLOT_UNAVAILABLE)`.
2. **State Finalization**:
   - On commit, the row ID is retrieved (`last_insert_rowid()`).
   - `pending_booking.status` transitions from `'awaiting_confirmation'` to `'confirmed'`.
   - `pending_booking.booking_id` is populated with the generated ID (e.g. `#1`).
   - `awaiting_verbal_confirmation` resets to `False`.
3. **Spoken Receipt Generation**:
   - `format_spoken_receipt()` deterministically constructs the audio transcript:
     > *"Your appointment with Dr. Ada Lovelace on Tuesday, November 17 at 9:00 AM PST is confirmed. Your confirmation number is Booking #1. We look forward to seeing you."*
   - Cartesia synthesizes the audio in ~300ms, and the browser plays the confirmation receipt.

---

## 3. Architectural Contrast: LLM Staging vs. Deterministic Backend Commit

The table below contrasts the technical characteristics of LLM staging against the deterministic backend commit:

| Feature / Attribute | LLM Staging (`propose_booking`) | Deterministic Commit (`confirm_booking`) |
|---|---|---|
| **Execution Authority** | LLM (Google Gemini Flash via tool calling) | Python Backend (`scheduler/db.py`) |
| **Storage Medium** | Ephemeral Session Memory (`Agent.pending_booking`) | Disk-backed SQLite Database (WAL Mode) |
| **Persistence Across Reboots** | Lost on server crash or restart | Fully ACID persistent on disk (`appointment_scheduler.db`) |
| **Concurrency Guarantees** | None (advisory proposal only) | Strict mutual exclusion via `BEGIN IMMEDIATE` |
| **Double-Booking Defense** | None (cannot prevent concurrent claims) | Absolute mathematical guard via `uix_slot_confirmed` |
| **Hallucination Risk** | High if allowed to book; Zero because write is prohibited | **Zero**: Deterministic regex + hardcoded SQL query |
| **Failure Semantics** | Fails closed: invalidates proposal | Full transaction rollback (`ROLLBACK` on exception) |
| **Idempotency** | Stable key generated (`book-{slot_id}-{uuid}`) | Key enforced by `uix_idempotency` index |
| **Turn Latency** | ~1200ms – 2500ms (LLM inference + tool round) | **3.2ms** (Direct SQL insert; skips LLM entirely) |
| **Spoken Output** | Conversational prompt shaped by system instructions | Deterministic receipt via `format_spoken_receipt()` |

---

## 4. Failure Mode Matrix & State Invariants

| Failure Scenario | Where Detected | System Behavior | Resulting State |
|---|---|---|---|
| **Patient says "Actually, cancel that"** | `is_verbal_cancellation` in Python | `cancel_pending()` called; DB untouched | Pending proposal discarded; `idle` |
| **Patient says "Can I do 2 PM instead?"** | `is_verbal_modification` in Python | Pending Slot #1 discarded; forwarded to Gemini | Gemini stages Slot #2 in memory |
| **Patient says "Wait a second"** | `is_verbal_ambiguity` in Python | Proposal preserved; agent prompts for clarification | Pending proposal retained; awaits clarification |
| **Patient drops call / disconnects** | Browser / WebSocket disconnect | In-memory session discarded on timeout or restart | Zero orphaned bookings in database |
| **Concurrent session claims slot first** | SQLite `book_slot()` insert | Catches `IntegrityError: bookings.slot_id` | Transaction rolled back; patient notified |
| **Patient repeats "Yes, book it" twice** | SQLite `book_slot()` idempotency check | Identifies reused `idempotency_key` | Returns existing booking without duplicate row |
