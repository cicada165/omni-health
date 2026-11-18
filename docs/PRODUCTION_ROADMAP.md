# Appointment Scheduler — Production Engineering Roadmap

This document outlines a phased engineering roadmap to evolve **Appointment Scheduler** from an advanced localhost proof-of-concept into a scalable, fault-tolerant, HIPAA-compliant enterprise clinical scheduling platform.

---

## 1. Roadmap Architecture Overview

```
 [PHASE 1: Telephony & SIP]    --> [PHASE 2: Distributed State]    --> [PHASE 3: EHR / FHIR Integration]
 - Twilio/Vonage SIP Trunking      - Redis Session State               - HL7 FHIR R4 API Adapters
 - LiveKit / WebRTC Audio Media    - Multi-Instance FastAPI Clusters   - Epic & Cerner Bi-directional Sync
 - Carrier Acoustic Echo Cancel    - Postgres Aurora WAL / Pooling     - Real-Time Provider Calendars
              |                                     |                                     |
              v                                     v                                     v
 [PHASE 4: Enterprise Security]---------------------------------------> [PHASE 5: Clinical Triage & Transfer]
 - SOC 2 Type II & HIPAA Compliance                                    - Emergency Keyword Detection (911)
 - Enterprise Cloud BAAs (Google, Deepgram)                            - SIP REFER Warm Transfer to Human
 - Immutable Cryptographic SIEM Audit Logs                             - Real-Time Agent Handoff Summary
```

---

## 2. Phase 1: WebRTC, LiveKit & Telephony Integration (PSTN / SIP)

**Objective**: Transition from browser microphone inputs to live public telephone network (PSTN) calls, enabling patients to call the clinic's landline number from any mobile or landline device.

### Architectural Deliverables:
1. **SIP Trunking & Telephony Gateway**:
   - Establish SIP trunks via **Twilio Voice**, **Vonage**, or **Telnyx**.
   - Negotiate standard telecom codecs: **G.711 µ-law / A-law (PCMU/PCMA)** at 8 kHz sample rate, with dynamic transcoding to 16 kHz PCM for Deepgram.
2. **Media Server Layer (LiveKit WebRTC)**:
   - Deploy **LiveKit Server** to manage bidirectional RTP/RTCP media streams.
   - Leverage WebRTC media servers for kernel-level jitter buffering, packet loss concealment (PLC), and carrier-grade **Acoustic Echo Cancellation (AEC)**.
3. **DTMF Tone Fallback**:
   - Enable dual-tone multi-frequency (DTMF) detection (e.g. *"Press 1 to confirm, 2 to reschedule"*) alongside natural speech recognition for noisy environments.

---

## 3. Phase 2: Distributed State & Scalable Microservices

**Objective**: Eliminate single-process in-memory session bottlenecks and migrate to an horizontally scalable, stateless microservice architecture capable of handling 10,000+ concurrent calls.

### Architectural Deliverables:
1. **Distributed Session Storage (Redis Enterprise / Valkey)**:
   - Migrate `SessionState` and `Agent.pending_booking` from process memory to Redis with automatic TTL expiration (e.g. 15 minutes of inactivity).
   - Use **Redlock (Distributed Lock Manager)** or Redis Lua scripts to replace Python's local `asyncio.Lock`, ensuring strict turn serialization across horizontally autoscaled FastAPI worker pods.
2. **Database Migration to PostgreSQL Aurora**:
   - Migrate from SQLite to **Amazon Aurora PostgreSQL** (Multi-AZ with read replicas).
   - Maintain the identical partial unique index strategy:
     ```sql
     CREATE UNIQUE INDEX uix_slot_confirmed
     ON bookings (slot_id)
     WHERE status = 'confirmed';
     ```
   - Deploy **PgBouncer** connection pooling to manage thousands of concurrent microservice connections with sub-millisecond connection acquisition.
3. **Stateless API Deployment**:
   - Containerize FastAPI backend into Kubernetes (EKS/GKE) worker pods behind an AWS Application Load Balancer (ALB) with sticky sessions or Redis-backed state routing.

---

## 4. Phase 3: EHR / EMR Integration via HL7 FHIR APIs

**Objective**: Replace the local database with real-time bidirectional synchronization with hospital Electronic Health Record (EHR) systems like **Epic Systems** and **Cerner (Oracle Health)**.

### Architectural Deliverables:
1. **HL7 FHIR R4 Standards Compliance**:
   - Implement native FHIR REST resource mapping:
     - `doctors` ➔ `Practitioner` & `PractitionerRole`
     - `slots` ➔ `Schedule` & `Slot` (`status = 'free'`)
     - `bookings` ➔ `Appointment` (`status = 'booked' | 'cancelled'`)
     - `patient` ➔ `Patient`
2. **SMART on FHIR Backend Services**:
   - Authenticate via asymmetric JWT signed tokens (OAuth 2.0 Backend Services Authorization) to query Epic's Open and Restricted APIs.
3. **Two-Way Realtime Webhooks**:
   - Subscribe to FHIR `Subscription` events to invalidate cached availability in real time when a human receptionist books a slot directly inside the EHR desktop client.

---

## 5. Phase 4: Production HIPAA Compliance & Enterprise Security

**Objective**: Implement bank-grade and healthcare-grade governance, passing third-party SOC 2 Type II and HIPAA regulatory audits.

### Architectural Deliverables:
1. **Enterprise Business Associate Agreements (BAAs)**:
   - Transition Google Gemini API to **Google Cloud Vertex AI** under enterprise BAA.
   - Deploy Deepgram and Cartesia on enterprise BAA agreements with zero-data-retention guarantees.
2. **Envelope Encryption & Key Management**:
   - Implement **Envelope Encryption** for patient identifiers using AWS KMS or Google Cloud KMS (AES-256).
   - Encrypt PostgreSQL storage volumes with Customer Managed Encryption Keys (CMEK).
3. **Tamper-Evident SIEM Audit Logs**:
   - Stream structured JSON audit events asynchronously via Kafka or AWS Kinesis to an immutable append-only datastore (Datadog Security, AWS CloudWatch, Splunk).
   - Enforce cryptographic hash chaining on audit records.

---

## 6. Phase 5: Automated Clinical Triage & Human Warm Handoff

**Objective**: Guarantee patient safety by detecting life-threatening medical emergencies and orchestrating seamless warm transfers to human triage nurses.

### Architectural Deliverables:
1. **Clinical Red-Flag Emergency Guardrail**:
   - Run a sub-50ms parallel regex and embedding classifier on every incoming user utterance to detect critical emergency symptoms (e.g. *"chest pain"*, *"difficulty breathing"*, *"severe bleeding"*, *"stroke symptoms"*, *"suicidal ideation"*).
   - **Immediate Clinical Protocol**:
     - Interrupt any scheduling dialogue immediately.
     - Spoken instruction: *"If you are experiencing a life-threatening medical emergency, please hang up and dial 911 immediately."*
     - Simultaneously initiate priority transfer to the on-call emergency clinical queue.
2. **SIP REFER Warm Receptionist Transfer**:
   - When a patient expresses confusion, requests a human, or exhibits high frustration, the system executes an RFC 5589 **SIP REFER** transfer to the clinic's front-desk phone queue.
   - **Contextual Handoff Screen Pop**:
     - The telephony system delivers a structured clinical summary to the human receptionist's desktop before the call connects:
       ```json
       {
         "patient_name": "Alex Taylor",
         "caller_id": "+1-415-555-0192",
         "target_specialty": "Cardiology",
         "preferred_time": "Tuesday morning",
         "attempted_doctor": "Dr. Ada Lovelace",
         "reason_for_transfer": "Patient requested human confirmation",
         "call_duration_seconds": 128
       }
       ```
     - The human agent greets the caller by name and immediately picks up where the AI left off without forcing the patient to repeat their information.
