# Alive or Dead POC

This is a Proof of Concept for the "Alive or Dead" game, featuring an AI-driven game loop and dynamic UI generation. Gemini returns the round JSON and HTML fragments directly, plus a portrait search query that the backend resolves into a verified Wikimedia image URL.

## Project Structure

- `main.py`: The FastAPI server containing the game logic and the AI System Prompt.
- `requirements.txt`: Python dependencies.

## How to Run

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the Server**:
   ```bash
   python main.py
   ```

3. **Access the Game**:
   Open your browser and navigate to `http://localhost:8000`.

## Apache + systemd deployment

The repository includes deployment templates for serving the app behind Apache at `/alive-or-dead`:

- `deploy/alive-or-dead.service`: `systemd` unit running Uvicorn on `127.0.0.1:8000`
- `deploy/alive-or-dead-apache.conf`: Apache proxy snippet for `/alive-or-dead/`

The service expects:

- project path: `/home/ubuntu/alive-or-dead`
- environment file: `/home/ubuntu/alive-or-dead/.env`
- app base path: `APP_BASE_PATH=/alive-or-dead`

## Gemini Audit Log

- Gemini prompt/response audit events are written to `gemini_audit.jsonl` in the project root by default.
- Each JSONL entry records the exact prompt text, model, SDK, raw `response_text`, portrait-resolution events, and whether the round was accepted or rejected.
- Set `GEMINI_AUDIT_LOG_ENABLED=false` to disable it.
- Set `GEMINI_AUDIT_LOG_FILE=/custom/path/gemini_audit.jsonl` to change the log location.

## Game Loop

1. **Start**: The user clicks "START POC".
2. **Guessing State**: Gemini selects a subject (e.g., Keanu Reeves), returns themed Tailwind UI plus a `portrait_search_query`, and the backend resolves a live Wikimedia portrait for the user to guess [ALIVE] or [DEAD].
3. **Dramatic Reveal**: Upon selection, the UI transforms into a themed reveal (celebratory for alive, respectful for dead).
4. **Session Tracking**: The backend tracks the score over 10 rounds.
5. **Finale**: After 10 rounds, a final score screen is displayed.

## AI System Prompt

The core logic for the AI subject selection and UI generation is defined in the `SYSTEM_PROMPT` variable within `main.py`. It instructs the AI to:
- Autonomously pick a famous person.
- Fact-check their current status.
- Generate a clean, themed UI without revealing the answer.
- Return a portrait search query while keeping the HTML on a fixed portrait placeholder that the backend replaces with a verified Wikimedia image URL.
