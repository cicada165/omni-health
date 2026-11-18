# Omni Health

A personal prototype for scheduling appointments at a fictional clinic. It combines a Gemini-powered conversational agent with deterministic Python booking logic and SQLite. A local web interface supports text and optional voice interaction.

## Features

- Search fictional doctors and future appointment slots in the clinic time zone.
- Stage an appointment proposal before explicit confirmation.
- Commit bookings using a SQLite transaction and conditional slot claim to handle competing requests.
- Inspect conversation and tool execution traces in the web interface.
- Use optional speech-to-text and text-to-speech services.

## Run locally

Install the dependencies in a virtual environment and supply credentials through your environment. The web server seeds a fictional clinic database on startup.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export GEMINI_API_KEY=your_key
python -m uvicorn scheduler.web:app --host 127.0.0.1 --port 8000
```

Open http://127.0.0.1:8000. For voice features, configure `DEEPGRAM_API_KEY` and `CARTESIA_API_KEY`. `DB_PATH` selects the SQLite file. Keep credentials out of Git.

Run the tests with `python -m pytest -q`. The tests mock external model and audio services.

## Scope

This is a demonstration using fictional clinicians and patients. Sessions are held in memory, so each browser session must continue with the same server process. The project is not intended for clinical use or real patient data. See `docs/` for architecture and concurrency details.
