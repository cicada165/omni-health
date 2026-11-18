"""
scheduler/chat.py — Terminal chat interface for the scheduling agent (Milestone 2B).

Run:
    python -m scheduler.chat
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from google import genai

from scheduler.agent import Agent, DEFAULT_MODEL, DEFAULT_PATIENT_NAME
from scheduler.db import init_db, seed


def main() -> None:
    load_dotenv()

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("Error: set GEMINI_API_KEY in .env or environment.", file=sys.stderr)
        sys.exit(1)

    model = os.getenv("GEMINI_MODEL", DEFAULT_MODEL)
    db_path = Path(os.getenv("DB_PATH", "appointment_scheduler.db"))

    # Ensure DB is ready
    init_db(db_path)
    seed(db_path)

    client = genai.Client(api_key=api_key)
    agent = Agent(
        db_path=db_path,
        model=model,
        client=client,
        patient_name=DEFAULT_PATIENT_NAME,
    )

    print("=" * 60)
    print("  Appointment Scheduler — Receptionist Assistant (Milestone 2B)")
    print("=" * 60)
    print(f"Model   : {model}")
    print(f"Patient : {agent.patient_name} (fictional demo identity; unauthenticated)")
    print("Commands:")
    print("  /confirm  -> Confirm the currently proposed appointment")
    print("  /cancel   -> Discard the currently proposed appointment")
    print("  quit      -> Exit the chat")
    print("-" * 60 + "\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not user_input:
            continue

        if user_input.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break

        # Explicit confirmation / cancellation intercepted BEFORE calling Gemini
        if user_input.lower() == "/confirm":
            success, msg = agent.confirm_booking()
            print(f"\n{msg}\n")
            continue

        if user_input.lower() == "/cancel":
            was_pending, msg = agent.cancel_pending()
            print(f"\n{msg}\n")
            continue

        reply = agent.turn(user_input)
        print(f"\nAssistant: {reply}\n")


if __name__ == "__main__":
    main()
