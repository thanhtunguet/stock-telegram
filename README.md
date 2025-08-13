## Telegram Stock Signal Watcher

A small async utility that watches a Telegram group for potential stock trading messages (Vietnamese/English), extracts actionable buy/sell signals using an LLM (Ollama or OpenAI-compatible endpoints like LM Studio), and forwards a concise summary to a Telegram bot/channel.

### How it works
- **Prefilter**: Reads new messages from a source `GROUP` using Telethon and filters by keywords and ticker patterns to avoid noise.
- **Batch**: Chunks candidate messages into groups of up to `MAX_MESSAGES_PER_CALL` to respect context/token limits.
- **LLM extraction**: Sends each chunk to the configured LLM with a strict prompt to return only valid JSON of signals.
- **Aggregate & notify**: Combines extracted signals and sends a Markdown summary to a destination via a Telegram bot (`BOT_TOKEN` + `BOT_CHAT_ID`, optional `BOT_THREAD_ID`).
- **State & schedule**: Stores `last_id` and `last_run` in a state file so only new messages are processed, and repeats every `POLL_INTERVAL_SEC` seconds.
- **Debugging**: Saves each LLM input batch into timestamped files in `DEBUG_INPUT_DIR` for inspection.

### Requirements
- **Python**: 3.10+ recommended
- **Dependencies**: `telethon`, `httpx`, `python-dotenv`
- **An LLM backend**:
  - Ollama chat API (default): `http://127.0.0.1:11434/api/chat`
  - OpenAI API: `https://api.openai.com/v1/chat/completions` (needs `OPENAI_API_KEY`)
  - LM Studio or other OpenAI-compatible server: `http://localhost:1234/v1/chat/completions`
- **Telegram API credentials**: `API_ID`, `API_HASH`, `PHONE`
- **Telegram bot**: `BOT_TOKEN`, `BOT_CHAT_ID` (and optional `BOT_THREAD_ID` for forum topics)

### Setup
1. **Install dependencies**
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
pip install telethon httpx python-dotenv
```

2. **Copy and edit environment config**
```bash
cp .env.example .env
# Open .env and fill in your values
```

3. **Start your LLM backend**
- **Ollama**: ensure it is running and the model exists (e.g., `llama3.1:8b`).
- **OpenAI API**: set `OLLAMA_URL=https://api.openai.com/v1/chat/completions` and `OPENAI_API_KEY`.
- **LM Studio** (or similar): set `OLLAMA_URL=http://localhost:1234/v1/chat/completions` and `LM_STUDIO_API_KEY`.

4. **Run the watcher**
```bash
python tg_stock_watcher.py
```
- On first run, Telethon will send a code to `PHONE`. Enter it to sign in. A local session file `tg_stock_session.session` will be created.

### Environment variables
Use `.env` for local configuration (keep it private; do not commit). See `.env.example` for reference. Key variables:
- **API_ID**: Telegram API ID from `my.telegram.org`.
- **API_HASH**: Telegram API hash from `my.telegram.org`.
- **PHONE**: Your Telegram phone number, e.g., `+84xxxxxxxxx`.
- **GROUP**: Source group identifier. Accepts `@username`, full `t.me` link, or numeric ID.
- **OLLAMA_URL**: LLM chat endpoint. Defaults to Ollama chat API `http://127.0.0.1:11434/api/chat`.
- **MODEL**: Model name (e.g., `llama3.1:8b`). For Qwen, suffix `/no_think` is auto-appended when possible.
- **LLM_MAX_TOKENS**: Max tokens for responses (OpenAI: `max_tokens`; Ollama: `options.num_predict`).
- **LLM_TEMPERATURE**: Sampling temperature.
- **LM_STUDIO_API_KEY**: Bearer used for OpenAI-compatible local servers (e.g., LM Studio).
- **OPENAI_API_KEY**: Required if using `api.openai.com`.
- **MAX_MESSAGES_PER_CALL**: Messages per LLM request (limits context), default `10`.
- **BOT_TOKEN**: Telegram Bot token (from `@BotFather`).
- **BOT_CHAT_ID**: Target chat ID to receive summaries (negative for supergroups, e.g., `-1001234567890`).
- **BOT_THREAD_ID**: Topic/thread ID for forum-enabled supergroups (use `0` to disable).
- **POLL_INTERVAL_SEC**: Interval between polling cycles, default `300`.
- **STATE_FILE**: Path to JSON state file, default `./tg_stock_state.json` in code; `.env.example` shows `./state.json` as an example.
- **DEBUG_INPUT_DIR**: Directory to save LLM input batches, default `./debug_inputs`.
- **LOG_LEVEL**: `DEBUG`, `INFO`, `WARNING`, `ERROR` (default `INFO`).

Example `.env`:
```dotenv
API_ID=123456
API_HASH=xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
PHONE=+84xxxxxxxxx
GROUP=@ten_nhom_hoac_link_hoac_idso

# LLM (choose one backend)
# OpenAI
# OLLAMA_URL=https://api.openai.com/v1/chat/completions
# OPENAI_API_KEY=sk-...

# LM Studio / OpenAI-compatible local
# OLLAMA_URL=http://localhost:1234/v1/chat/completions
# LM_STUDIO_API_KEY=lm-studio

# Ollama (default works without setting OLLAMA_URL)
MODEL=llama3.1:8b
LLM_MAX_TOKENS=4096
LLM_TEMPERATURE=0.2
MAX_MESSAGES_PER_CALL=10

# Bot notify
BOT_TOKEN=123456:abc_def...
BOT_CHAT_ID=-1001234567890
BOT_THREAD_ID=0

# Schedule & state
POLL_INTERVAL_SEC=300
STATE_FILE=./tg_stock_state.json

# Debug
DEBUG_INPUT_DIR=./debug_inputs

# Logging
LOG_LEVEL=INFO
```

### Telegram-specific notes
- **Getting API_ID/API_HASH**: Log in at `https://my.telegram.org`, go to API Development Tools, create an app, and copy `API_ID` and `API_HASH`.
- **Getting BOT_TOKEN**: Use `@BotFather` to create a bot and copy the token.
- **Adding the bot to a chat**: Add it to the destination supergroup/channel and grant permission to post.
- **Finding BOT_CHAT_ID**: Send a message in the target chat, then query updates:
```bash
curl -s "https://api.telegram.org/bot<YOUR_BOT_TOKEN>/getUpdates" | jq
```
Look for `message.chat.id` (supergroups usually start with `-100`). Alternatively, use helper bots like `@getidsbot`.
- **Forum topics**: For supergroup topics, set `BOT_THREAD_ID` to the topic id; otherwise use `0`.

### Logs, state and debug artifacts
- **Logs**: Controlled via `LOG_LEVEL` and printed to stdout.
- **State**: JSON file that tracks `last_id` to prevent duplicate processing (`STATE_FILE`).
- **Debug inputs**: Each LLM batch sent is saved to `DEBUG_INPUT_DIR` with a timestamped filename for inspection.

### Development notes
- The LLM prompt and regex rules are tailored to Vietnamese stock chat rooms and aim to extract clear buy/sell intents. False positives are reduced by a simple prefilter before the LLM stage.
- The code automatically strips common fenced blocks and think tags that some models output, and tries to recover a valid JSON array/object.

### Disclaimer
This tool is for research/monitoring convenience. It does not constitute financial advice. Verify all signals independently before making trading decisions.
