# Appointment Scheduler — Milestone 8 Reliability & Evaluation Report

## 1. Executive Summary

- **Total Scenarios Evaluated**: 42
- **Scenarios Passed**: 42 (100.0%)
- **Scenarios Failed**: 0
- **Total Benchmark Runtime**: 503.0 ms
- **Unsafe Booking Rate**: **0.0%** (Target: 0.0% — PASSED)
- **False Verbal Confirmation Rate**: **0.0%** (Target: 0.0% — PASSED)

## 2. Quantitative Evaluation Benchmarks

| Evaluation Metric | Measured Value | Target SLA / Safety Invariant | Compliance Status |
|:-------------------|:--------------:|:-----------------------------:|:-----------------:|
| **Task Completion Rate** | 100.0% | ≥ 95.0% | PASS |
| **Slot-Selection Accuracy** | 100.0% | 100.0% | PASS |
| **Unsafe Booking Rate** | 0.0% | **0.0%** (Hard Invariant) | PASS |
| **False Verbal Confirmation Rate** | 0.0% | **0.0%** (Hard Invariant) | PASS |
| **Failure Recovery Rate** | 100.0% | 100.0% | PASS |
| **Turn Latency (p50)** | 2.0 ms | < 100.0 ms (mocked) | PASS |
| **Turn Latency (p90)** | 3.0 ms | < 250.0 ms (mocked) | PASS |
| **Turn Latency (p99)** | 12.6 ms | < 500.0 ms (mocked) | PASS |

## 3. Performance by Scenario Category

| Category # | Category Name | Scenarios | Passed | Pass Rate | Unsafe Bookings | False Confirms | Mean Latency |
|:----------:|:--------------|:---------:|:------:|:---------:|:---------------:|:--------------:|:------------:|
| 1 | Normal Booking | 5 | 5 | 100.0% | 0 | 0 | 12.0 ms |
| 2 | Doctor and Time Corrections | 5 | 5 | 100.0% | 0 | 0 | 12.6 ms |
| 3 | Ambiguous Verbal Confirmations | 4 | 4 | 100.0% | 0 | 0 | 7.2 ms |
| 4 | Negation and Cancellation | 4 | 4 | 100.0% | 0 | 0 | 7.2 ms |
| 5 | Interruption During Confirmation | 4 | 4 | 100.0% | 0 | 0 | 12.5 ms |
| 6 | Duplicate and Competing Requests | 5 | 5 | 100.0% | 0 | 0 | 8.3 ms |
| 7 | STT Transcription Errors | 5 | 5 | 100.0% | 0 | 0 | 7.4 ms |
| 8 | Provider Timeouts and Failures | 5 | 5 | 100.0% | 0 | 0 | 15.1 ms |
| 9 | Session Isolation and State Recovery | 5 | 5 | 100.0% | 0 | 0 | 23.3 ms |

## 4. Scenario Execution Ledger

| Scenario ID | Name | Category | Status | Turns | Bookings | Latency |
|:-----------:|:-----|:---------|:------:|:-----:|:--------:|:-------:|
| `NORM-01` | Single Doctor Specific Date Booking | Normal Booking | **PASS** | 2 | 1 | 18.6 ms |
| `NORM-02` | Multiple Doctors Comparison and Selection | Normal Booking | **PASS** | 3 | 1 | 11.6 ms |
| `NORM-03` | Next Available Slot Booking | Normal Booking | **PASS** | 2 | 1 | 9.5 ms |
| `NORM-04` | Specific Afternoon Slot with Pediatrician | Normal Booking | **PASS** | 2 | 1 | 10.2 ms |
| `NORM-05` | Direct Specialty Search (Cardiology) | Normal Booking | **PASS** | 2 | 1 | 10.3 ms |
| `CORR-01` | Day Correction Pre-Confirmation | Doctor and Time Corrections | **PASS** | 3 | 1 | 12.7 ms |
| `CORR-02` | Time of Day Correction (Morning to Afternoon) | Doctor and Time Corrections | **PASS** | 3 | 1 | 11.9 ms |
| `CORR-03` | Doctor Switch Mid-Conversation | Doctor and Time Corrections | **PASS** | 3 | 1 | 12.4 ms |
| `CORR-04` | Multiple Sequential Corrections (Day, Time, Doctor) | Doctor and Time Corrections | **PASS** | 4 | 1 | 14.1 ms |
| `CORR-05` | Correction Directly at Confirmation Prompt | Doctor and Time Corrections | **PASS** | 3 | 1 | 12.0 ms |
| `AMBIG-01` | Hesitant Confirmation ('Maybe') | Ambiguous Verbal Confirmations | **PASS** | 2 | 0 | 8.0 ms |
| `AMBIG-02` | Pause and Delay ('Hold on a second') | Ambiguous Verbal Confirmations | **PASS** | 2 | 0 | 6.7 ms |
| `AMBIG-03` | Hesitation ('Let me think') | Ambiguous Verbal Confirmations | **PASS** | 2 | 0 | 6.8 ms |
| `AMBIG-04` | Uncertainty ('Not sure yet') | Ambiguous Verbal Confirmations | **PASS** | 2 | 0 | 7.4 ms |
| `CANC-01` | Direct Verbal Negation ('No') | Negation and Cancellation | **PASS** | 2 | 0 | 8.0 ms |
| `CANC-02` | Explicit Verbal Cancellation ('Don't book that') | Negation and Cancellation | **PASS** | 2 | 0 | 7.0 ms |
| `CANC-03` | Conversational Dismissal ('Never mind') | Negation and Cancellation | **PASS** | 2 | 0 | 7.9 ms |
| `CANC-04` | Deterministic Cancel Method (cancel_pending) | Negation and Cancellation | **PASS** | 0 | 0 | 5.7 ms |
| `INTR-01` | Interruption ('Wait stop, do not book') | Interruption During Confirmation | **PASS** | 2 | 0 | 24.3 ms |
| `INTR-02` | Interruption with Slot Change ('Actually change to 2 PM') | Interruption During Confirmation | **PASS** | 3 | 1 | 11.3 ms |
| `INTR-03` | Informational Interruption ('Wait, what is her specialty?') | Interruption During Confirmation | **PASS** | 2 | 0 | 7.7 ms |
| `INTR-04` | Barge-in Noise During Confirmation | Interruption During Confirmation | **PASS** | 2 | 0 | 6.6 ms |
| `COMP-01` | Concurrent Two-Patient Race for Same Slot | Duplicate and Competing Requests | **PASS** | 0 | 1 | 8.0 ms |
| `COMP-02` | Repeated Verbal Confirmation Idempotency | Duplicate and Competing Requests | **PASS** | 0 | 1 | 7.8 ms |
| `COMP-03` | Same Slot Re-booking After Prior Cancellation | Duplicate and Competing Requests | **PASS** | 0 | 1 | 8.0 ms |
| `COMP-04` | Idempotency Key Conflict Protection | Duplicate and Competing Requests | **PASS** | 0 | 1 | 5.8 ms |
| `COMP-05` | Multi-Threaded 10-Worker Race Condition Safety | Duplicate and Competing Requests | **PASS** | 0 | 1 | 11.9 ms |
| `STT-01` | Fuzzy Doctor Name ('Dr. Cabbage') | STT Transcription Errors | **PASS** | 2 | 1 | 9.9 ms |
| `STT-02` | Misheard Date Homophone ('to stay' -> Tuesday) | STT Transcription Errors | **PASS** | 2 | 1 | 9.0 ms |
| `STT-03` | Phonetic Doctor Name ('Dr. Loveless') | STT Transcription Errors | **PASS** | 2 | 1 | 8.6 ms |
| `STT-04` | Garbled Background Noise Utterance | STT Transcription Errors | **PASS** | 1 | 0 | 4.8 ms |
| `STT-05` | Unsupported Audio MIME Type Error Handling | STT Transcription Errors | **PASS** | 0 | 0 | 4.8 ms |
| `FAIL-01` | Gemini API 503 Service Unavailable | Provider Timeouts and Failures | **PASS** | 1 | 0 | 4.2 ms |
| `FAIL-02` | Gemini API Request Timeout | Provider Timeouts and Failures | **PASS** | 1 | 0 | 3.8 ms |
| `FAIL-03` | Deepgram Speech STT Timeout | Provider Timeouts and Failures | **PASS** | 0 | 0 | 4.9 ms |
| `FAIL-04` | Cartesia TTS Service Failure (500 Error) | Provider Timeouts and Failures | **PASS** | 0 | 0 | 54.1 ms |
| `FAIL-05` | Transient Gemini Failure and Subsequent Turn Recovery | Provider Timeouts and Failures | **PASS** | 3 | 1 | 8.6 ms |
| `SESS-01` | Multi-Session State Isolation | Session Isolation and State Recovery | **PASS** | 0 | 2 | 9.6 ms |
| `SESS-02` | State Recovery After Informational Question Turn | Session Isolation and State Recovery | **PASS** | 3 | 1 | 9.1 ms |
| `SESS-03` | Informational Interruption with New Slot Proposal | Session Isolation and State Recovery | **PASS** | 4 | 1 | 10.4 ms |
| `SESS-04` | Session Expiry and Unknown Session ID Handling | Session Isolation and State Recovery | **PASS** | 0 | 0 | 82.9 ms |
| `SESS-05` | Session History Rollback on Turn Failure | Session Isolation and State Recovery | **PASS** | 0 | 0 | 4.4 ms |

## 5. Invariant Safety Guarantees

- **Zero Unsafe Bookings**: No uncommitted, ambiguous, cancelled, or interrupted proposals were written to SQLite.
- **Zero False Confirmations**: The receptionist never confirmed an appointment without an active, verified SQLite booking row.
- **Atomic Race Condition Safety**: SQLite partial unique indexes prevent double-booking under concurrent load.
- **Deterministic Rollback**: Conversation history is cleanly reverted on provider timeouts without orphaned turns.
