# Appointment Scheduler — Healthcare Privacy, Security & HIPAA Compliance

This document provides a candid, rigorous architectural evaluation of the privacy and security characteristics of **Appointment Scheduler**. It details the explicit boundaries of the current technical prototype, identifies HIPAA/PHI compliance gaps, and defines the mandatory security controls required for clinical production deployment.

---

## 1. Prototype Privacy & Security Boundaries

Appointment Scheduler was developed as an advanced proof-of-concept for real-time voice scheduling and deterministic LLM guardrails. To maintain engineering focus on latency and concurrency, the prototype operates within strict, transparent boundaries:

1. **Unauthenticated Persona Boundary**:
   - The system operates with a hardcoded, fictional demo patient identity: **Alex Taylor** (`DEFAULT_PATIENT_NAME`).
   - Callers are not authenticated via Multi-Factor Authentication (MFA), password, or insurance card verification.
2. **Localhost Isolation**:
   - The web and WebSocket interfaces bind strictly to `127.0.0.1` / `localhost`.
   - The Developer Inspector (`/api/sessions/{id}/inspector`) enforces `_is_local_request()`, rejecting any non-loopback network packets with `HTTP 403 Forbidden`.
3. **Non-Persistent Audio Buffers & Ephemeral Memory**:
   - Audio chunks streamed via the primary WebSocket proxy (`/api/sessions/{id}/ws/transcribe`) and synthesized WAV buffers from Cartesia are held strictly in ephemeral RAM and forwarded directly to client AudioContexts.
   - The system does not maintain an application database or S3 bucket for audio recordings.
   - *Technical Spooling Caveat*: The REST audio endpoint (`POST /transcribe`) uses Starlette's `UploadFile`, which spools uploads >1 MB to OS temporary buffers (`/tmp`). For clinical production, all audio ingestion must be restricted exclusively to streaming WebSockets or configured with strictly in-memory `BytesIO` buffers.
4. **Credential Isolation**:
   - Cloud provider credentials (`GEMINI_API_KEY`, `DEEPGRAM_API_KEY`, `CARTESIA_API_KEY`) reside exclusively in server-side environment variables (`.env`).
   - The browser client never receives an API key, preventing token extraction from client memory.

---

## 2. HIPAA & PHI Compliance Gap Analysis

In the United States, deployment in a healthcare setting subject to the **Health Insurance Portability and Accountability Act (HIPAA)** mandates compliance with the **Privacy Rule**, **Security Rule**, and **Breach Notification Rule**. 

The current prototype has several critical compliance gaps that must be bridged prior to clinical deployment:

```
+----------------------------------------------------------------------------------------------------+
|                               HIPAA COMPLIANCE STATUS MATRIX                                       |
+------------------------------+---------------------------+-----------------------------------------+
| Requirement                  | Prototype Status          | Production Remediation Required         |
+------------------------------+---------------------------+-----------------------------------------+
| Business Associate Agrmt     | MISSING                   | Execute BAAs with Google, Deepgram, and |
| (BAA) with Cloud Vendors     | (Standard Developer APIs) | Cartesia under enterprise tiers.        |
+------------------------------+---------------------------+-----------------------------------------+
| PHI Encryption in Transit    | PARTIAL                   | Enforce strict TLS 1.3 / WSS and mTLS   |
|                              | (Localhost HTTP / WSS)    | across all microservices.               |
+------------------------------+---------------------------+-----------------------------------------+
| PHI Encryption at Rest       | MISSING                   | AES-256 disk encryption (LUKS / EBS)    |
|                              | (Plain SQLite file)       | + SQLCipher / Column-level KMS keys.    |
+------------------------------+---------------------------+-----------------------------------------+
| Zero Data Retention Policies | MISSING                   | Enforce opt-out of model training & log |
|                              | (Default Cloud Tiers)     | retention via enterprise agreements.    |
+------------------------------+---------------------------+-----------------------------------------+
| Patient Authentication &     | MISSING                   | SMART on FHIR OAuth2 / Telephony OTP /  |
| Identity Verification        | (Hardcoded Alex Taylor)   | Master Patient Index (MPI) matching.    |
+------------------------------+---------------------------+-----------------------------------------+
| Immutable Audit Logging      | PARTIAL                   | Append-only, tamper-evident audit logs  |
|                              | (In-memory Inspector)     | shipped to AWS CloudWatch / Datadog.    |
+------------------------------+---------------------------+-----------------------------------------+
```

---

## 3. Deep Dive: The Four Pillars of Production Healthcare Security

### Pillar 1: Business Associate Agreements (BAAs) & Zero Data Retention
Under HIPAA § 164.502(e), any third-party service provider that processes, transmits, or stores Protected Health Information (PHI)—including patient voice, names, or appointment dates—is legally classified as a **Business Associate (BA)**.

- **Deepgram**:
  - *Risk*: Standard consumer tiers may log audio streams for model improvement.
  - *Remediation*: Deploy Deepgram on an Enterprise plan with a signed BAA and `no-log` headers enabled, or deploy on-premise Deepgram containerized models within the clinic's private VPC.
- **Google Cloud (Gemini)**:
  - *Risk*: Consumer Gemini APIs may retain user prompts for system tuning.
  - *Remediation*: Utilize **Vertex AI with Gemini 2.5 Flash** under a Google Cloud BAA. Under the Vertex AI BAA, customer data and prompts are never used to train foundational models and are encrypted at rest with Customer-Managed Encryption Keys (CMEK).
- **Cartesia**:
  - *Risk*: Synthesis text containing patient names or medical specialties transmitted to external endpoints.
  - *Remediation*: Execute an enterprise BAA with Cartesia ensuring zero data retention on synthesized text payloads.

---

### Pillar 2: Cryptographic Controls (Transit & Rest)

1. **Data in Transit**:
   - All browser, telephony, and API communication must enforce **TLS 1.3** with forward secrecy cipher suites (`ECDHE-RSA-AES128-GCM-SHA256` or `TLS_AES_256_GCM_SHA384`).
   - Internal microservice calls (FastAPI to EHR or Redis) must enforce **mutual TLS (mTLS)** using SPIFFE/SPIRE or service mesh certificates.
2. **Data at Rest**:
   - **Database Encryption**: Upgrade from plaintext SQLite to **PostgreSQL with Transparent Data Encryption (TDE)** or **SQLCipher (AES-256-CBC)** for embedded databases.
   - **Column-Level PHI Envelope Encryption**: Sensitive attributes (`patient_name`, `phone_number`, `medical_record_number`) must be encrypted before database insertion using a key managed by AWS KMS or HashiCorp Vault.

---

### Pillar 3: Immutable Audit Trails (HIPAA Security Rule § 164.312(b))
Healthcare regulations mandate that every read, write, modification, or cancellation of an appointment must generate a tamper-evident audit event:

```sql
CREATE TABLE audit_logs (
    id              UUID PRIMARY KEY,
    timestamp_utc   TIMESTAMP WITH TIME ZONE NOT NULL,
    actor_id        TEXT NOT NULL,          -- e.g. "patient-phone:+14155552671" or "receptionist:104"
    action          TEXT NOT NULL,          -- "SEARCH_SLOTS", "PROPOSE_BOOKING", "CONFIRM_BOOKING"
    resource_id     TEXT NOT NULL,          -- "booking:104" or "slot:1"
    ip_address      TEXT NOT NULL,
    session_id      TEXT NOT NULL,
    outcome         TEXT NOT NULL,          -- "SUCCESS", "FAILURE", "INTEGRITY_VIOLATION"
    prev_hash       TEXT NOT NULL,          -- Cryptographic hash of previous row (hash chain)
    signature       TEXT NOT NULL           -- HMAC signature using HSM key
);
```
- In-memory event lists (such as the current Inspector log) must be replaced by asynchronous, non-blocking pipeline delivery to a dedicated, append-only SIEM (e.g. AWS CloudWatch, Datadog Security Monitoring, or Splunk).

---

### Pillar 4: Patient Identity Verification & Minimum Necessary Disclosure
- **Telephony Caller ID (ANI) Verification**:
  - Caller ID is inherently spoofable. In production, a voice assistant must send a One-Time Passcode (OTP) via SMS to the verified phone number on file or require the patient to confirm two independent identifiers (e.g. Date of Birth + Last 4 digits of SSN) before revealing existing medical appointments.
- **Minimum Necessary Standard (§ 164.502(b))**:
  - The voice assistant must never vocalize sensitive clinical diagnoses over voice.
  - Notice our system prompt instruction: appointments are referenced solely by **physician name, date, and local time** (e.g. *"Dr. Ada Lovelace on Tuesday at 9:00 AM"*), deliberately avoiding phrases like *"your cardiology heart failure consultation"*.

---

## 4. Guardrails Against PII & Credential Leakage in Code

The Appointment Scheduler codebase implements several defensive patterns to guard against data leakage:

1. **Developer Inspector Sanitization**:
   - In `scheduler/web.py:get_session_inspector()`, internal credentials, idempotency tokens, and raw binary audio chunks are excluded from the returned payload.
2. **Spoken Text Cleaning**:
   - `scheduler/tts.py:format_spoken_text()` strips internal metadata, markdown, slot primary keys, and bracketed tags (`[System Event: ...]`) so that internal orchestrations are never spoken to the caller.
3. **Failsafe Rollback on Timeout**:
   - In `scheduler/agent.py:turn()`, if Gemini times out or raises an API exception, the turn history is rolled back to prevent dirty or orphaned patient dialogue fragments from leaking into future conversation context.
