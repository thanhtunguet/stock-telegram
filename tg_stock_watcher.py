import os, re, json, signal, asyncio, logging, time
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Any, Optional

import httpx
from dotenv import load_dotenv
from telethon import TelegramClient
from telethon.errors import FloodWaitError, RPCError
from telethon.tl.types import PeerChannel

# ========= Load env =========
load_dotenv()

API_ID = int(os.environ["API_ID"])
API_HASH = os.environ["API_HASH"]
PHONE = os.environ["PHONE"]
GROUP = os.environ["GROUP"]

# Endpoint for LLM calls. Can be Ollama chat endpoint (e.g. http://127.0.0.1:11434/api/chat)
# or an OpenAI-compatible endpoint like LM Studio (e.g. http://localhost:1234/v1/chat/completions)
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434/api/chat")
MODEL = os.environ.get("MODEL", "llama3.1:8b")

# In case of OpenAI-compatible endpoints (LM Studio/OpenRouter/etc.), these help avoid 400 errors
LLM_MAX_TOKENS = int(os.environ.get("LLM_MAX_TOKENS", "256"))
LLM_TEMPERATURE = float(os.environ.get("LLM_TEMPERATURE", "0.2"))
LM_STUDIO_API_KEY = os.environ.get("LM_STUDIO_API_KEY", "lm-studio")

# Limit how many messages are sent to the LLM per request to avoid context/token overflows
# Default: 10 messages per request
MAX_MESSAGES_PER_CALL = int(os.environ.get("MAX_MESSAGES_PER_CALL", "10"))

BOT_TOKEN = os.environ["BOT_TOKEN"]
BOT_CHAT_ID = os.environ["BOT_CHAT_ID"]
BOT_THREAD_ID = int(os.environ.get("BOT_THREAD_ID", "0"))

POLL_INTERVAL_SEC = int(os.environ.get("POLL_INTERVAL_SEC", "300"))
STATE_FILE = os.environ.get("STATE_FILE", "./tg_stock_state.json")
LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO").upper()
DEBUG_INPUT_DIR = os.environ.get("DEBUG_INPUT_DIR", "./debug_inputs")

# ========= Logging =========
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger("tg-stock")

# ========= Regex sơ bộ =========
KEYWORDS = [
    r"\b(mua|buy|long|gom)\b",
    r"\b(bán|ban|sell|short|xả|xa)\b",
    r"\b(chốt lời|chot loi|tp|target)\b",
    r"\b(stop ?loss|sl|cắt lỗ|cat lo)\b",
    r"\b(điểm vào|entry|vào|vao|ra)\b",
    r"\b(phím|kèo|keo|tín hiệu|tin hieu)\b",
]
TICKER = r"\b[A-Z]{2,5}(?:\.[A-Z]{2,3})?\b"

KEYWORD_RE = re.compile("|".join(KEYWORDS), flags=re.IGNORECASE)
TICKER_RE = re.compile(TICKER)
TZ = timezone(timedelta(hours=7))

# ========= State =========
def load_state() -> Dict[str, Any]:
    try:
        if os.path.exists(STATE_FILE):
            with open(STATE_FILE, "r") as f:
                return json.load(f)
    except Exception as e:
        log.warning(f"Cannot read STATE_FILE: {e}")
    return {"last_id": 0, "last_run": None}

def save_state(state: Dict[str, Any]) -> None:
    try:
        os.makedirs(os.path.dirname(STATE_FILE), exist_ok=True)
        with open(STATE_FILE, "w") as f:
            json.dump(state, f)
    except Exception as e:
        log.error(f"Cannot write STATE_FILE: {e}")

# ========= HTTP helpers with retry =========
async def http_post_json(
    url: str,
    json_body: dict,
    timeout: int = 30,
    max_retries: int = 3,
    headers: Optional[Dict[str, str]] = None,
) -> dict:
    delay = 1
    last_err = None
    async with httpx.AsyncClient(timeout=timeout) as client:
        for attempt in range(1, max_retries + 1):
            try:
                resp = await client.post(url, json=json_body, headers=headers)
                resp.raise_for_status()
                return resp.json()
            except Exception as e:
                last_err = e
                log.warning(f"POST {url} failed (attempt {attempt}/{max_retries}): {e}")
                await asyncio.sleep(delay)
                delay = min(delay * 2, 10)
    raise RuntimeError(f"POST {url} failed after {max_retries} attempts: {last_err}")

async def http_get_json(url: str, timeout=30, max_retries=3) -> dict:
    delay = 1
    last_err = None
    async with httpx.AsyncClient(timeout=timeout) as client:
        for attempt in range(1, max_retries + 1):
            try:
                resp = await client.get(url)
                resp.raise_for_status()
                return resp.json()
            except Exception as e:
                last_err = e
                log.warning(f"GET {url} failed (attempt {attempt}/{max_retries}): {e}")
                await asyncio.sleep(delay)
                delay = min(delay * 2, 10)
    raise RuntimeError(f"GET {url} failed after {max_retries} attempts: {last_err}")

# ========= Bot notify =========
async def send_bot_message(text: str) -> None:
    base = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": BOT_CHAT_ID,
        "text": text,
        "parse_mode": "Markdown",
        "disable_web_page_preview": True,
    }
    if BOT_THREAD_ID:
        payload["message_thread_id"] = BOT_THREAD_ID
    await http_post_json(base, payload, timeout=30, max_retries=3)

# ========= LLM call (Ollama-compatible chat) =========
def build_llm_prompt(messages: List[str]) -> str:
    prompt = (
        "Bạn là trợ lý phân tích tín hiệu từ phòng chat chứng khoán Việt Nam.\n"
        "- Nhiệm vụ: từ các tin nhắn bên dưới, chỉ trích xuất những gợi ý MUA/BÁN thực sự.\n"
        "- Với mỗi tín hiệu, hãy chuẩn hóa JSON:\n"
        "  {ticker, action[mua|bán], entry(optional), range(optional), sl(optional), tp(optional), note(optional)}\n"
        "- Bỏ qua meme/đùa/hỏi đáp chung.\n"
        "- Nếu không có tín hiệu rõ, trả về []\n\n"
        "Tin nhắn:\n"
    )
    for i, m in enumerate(messages, 1):
        prompt += f"{i}. {m}\n"
    prompt += "\nChỉ trả lời **một JSON hợp lệ**."
    return prompt

def save_input_messages_to_txt(messages: List[str]) -> Optional[str]:
    """Save the list of input messages (sent to the LLM) into a timestamped .txt file.

    Returns the file path on success, or None if writing fails.
    """
    try:
        os.makedirs(DEBUG_INPUT_DIR, exist_ok=True)
        timestamp_string = datetime.now(TZ).strftime("%Y%m%d_%H%M%S")
        filename = f"inputs_{timestamp_string}.txt"
        filepath = os.path.join(DEBUG_INPUT_DIR, filename)
        with open(filepath, "w", encoding="utf-8") as file_handle:
            for index, line in enumerate(messages, 1):
                file_handle.write(f"{index}. {line}\n")
        log.info(f"Saved LLM input messages to {filepath}")
        return filepath
    except Exception as e:
        log.warning(f"Cannot write debug inputs: {e}")
        return None

async def call_llm(messages: List[str]) -> Optional[List[Dict[str, Any]]]:
    if not messages:
        return None
    # Base payload
    # Auto-adjust model for Qwen to suppress chain-of-thought if supported
    model_name = MODEL
    try:
        if re.search(r"qwen", MODEL, flags=re.IGNORECASE) and "/no_think" not in MODEL:
            model_name = f"{MODEL}/no_think"
    except Exception:
        model_name = MODEL

    payload: Dict[str, Any] = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": "Bạn trả lời NGẮN GỌN và chỉ bằng JSON hợp lệ."},
            {"role": "user", "content": build_llm_prompt(messages)},
        ],
        "stream": False,
    }
    # Endpoint-specific shaping
    if "/v1/" in OLLAMA_URL:
        # LM Studio / OpenAI-compatible
        payload.update({
            "max_tokens": LLM_MAX_TOKENS,
            "temperature": LLM_TEMPERATURE,
        })
    else:
        # Ollama chat API
        payload.update({
            "options": {
                "num_predict": LLM_MAX_TOKENS,
                "temperature": LLM_TEMPERATURE,
            }
        })
    try:
        # Attach Authorization header if calling OpenAI-compatible APIs
        headers: Optional[Dict[str, str]] = None
        if "api.openai.com" in OLLAMA_URL:
            api_key = os.environ.get("OPENAI_API_KEY")
            if not api_key:
                log.error("OPENAI_API_KEY is required when using OpenAI API endpoint")
                return None
            headers = {"Authorization": f"Bearer {api_key}"}
        elif "/v1/" in OLLAMA_URL:
            # Likely LM Studio or another OpenAI-compatible server
            headers = {"Authorization": f"Bearer {LM_STUDIO_API_KEY}"}

        data = await http_post_json(OLLAMA_URL, payload, timeout=60, max_retries=3, headers=headers)
        # Ollama returns {"message":{"content": "..."} ...} or {"choices":[{"message":{"content": ...}}]}
        content = None
        if "message" in data and isinstance(data["message"], dict):
            content = data["message"].get("content")
        elif "choices" in data and data["choices"]:
            content = data["choices"][0]["message"]["content"]
        if not content:
            log.warning("LLM returned empty content")
            return None
        # Try parse JSON
        content = content.strip()
        # Strip any Qwen-style think tags before parsing
        content = re.sub(r"<think>[\s\S]*?</think>", "", content, flags=re.IGNORECASE)
        # Remove trivial wrappers that sometimes appear
        content = re.sub(r"^<answer>\s*", "", content, flags=re.IGNORECASE)
        content = re.sub(r"\s*</answer>$", "", content, flags=re.IGNORECASE)

        if content.startswith("```"):
            # remove fences if model wrapped in code block
            # strip enclosing backticks and optional leading language tag
            content = content.strip("`")
            content = re.sub(r"^json\n", "", content, flags=re.IGNORECASE)
        try:
            result = json.loads(content)
        except Exception:
            # Fallback: extract first JSON object/array substring
            match = re.search(r"(\{[\s\S]*\}|\[[\s\S]*\])", content)
            if not match:
                raise
            result = json.loads(match.group(1))
        if isinstance(result, dict):
            # normalize to list if single object
            result = [result]
        if not isinstance(result, list):
            log.warning("LLM JSON is not a list; ignoring")
            return None
        return result
    except Exception as e:
        # Log a short preview of the unparsed content to aid debugging
        try:
            preview = (content or "")[:200].replace("\n", " ")
            log.error(f"LLM error: {e}. Preview: {preview}")
        except Exception:
            log.error(f"LLM error: {e}")
        return None

# ========= Prefilter =========
def is_potential_signal(text: str) -> bool:
    return bool(KEYWORD_RE.search(text)) or bool(TICKER_RE.search(text))

# ========= Utils =========
def chunk_list(sequence: List[Any], chunk_size: int) -> List[List[Any]]:
    if chunk_size <= 0:
        return [sequence]
    return [sequence[i : i + chunk_size] for i in range(0, len(sequence), chunk_size)]

# ========= Main loop =========
class Runner:
    def __init__(self):
        self.client = TelegramClient("tg_stock_session", API_ID, API_HASH)
        self.shutdown = asyncio.Event()
        self.state = load_state()

    async def init(self):
        await self.client.connect()
        if not await self.client.is_user_authorized():
            log.info("Signing in...")
            await self.client.send_code_request(PHONE)
            code = input("Enter the code you received: ")
            await self.client.sign_in(PHONE, code)
        log.info("Connected & authorized.")

    async def resolve_group(self):
        # Accept @username, link, or numeric id
        try:
            if isinstance(GROUP, str) and (GROUP.startswith("@") or "t.me" in GROUP):
                entity = await self.client.get_entity(GROUP)
            else:
                # try numeric id
                entity = await self.client.get_entity(int(GROUP))
            if not isinstance(entity, (PeerChannel,)) and not getattr(entity, "megagroup", True):
                log.info("Resolved group entity.")
            return entity
        except Exception as e:
            log.error(f"Cannot resolve GROUP {GROUP}: {e}")
            raise

    async def fetch_new_messages(self, entity) -> List[dict]:
        last_id = int(self.state.get("last_id", 0))
        msgs = []
        async for m in self.client.iter_messages(entity, min_id=last_id, reverse=True):
            if not m.message:
                continue
            text = m.message.strip()
            if not text:
                continue
            if is_potential_signal(text):
                msgs.append({
                    "id": m.id,
                    "date": (m.date.astimezone(TZ)).isoformat(),
                    "from_id": getattr(m.from_id, "user_id", None) if m.from_id else None,
                    "text": text,
                })
            # Track last seen id even if filtered (to avoid re-reading forever)
            last_id = max(last_id, m.id)
        if last_id > int(self.state.get("last_id", 0)):
            self.state["last_id"] = last_id
            self.state["last_run"] = datetime.now(TZ).isoformat()
            save_state(self.state)
        return msgs

    async def cycle_once(self, entity):
        try:
            raw_msgs = await self.fetch_new_messages(entity)
            if not raw_msgs:
                log.info("No new candidate messages this cycle.")
                return

            batch_texts = [f"[{r['date']}] {r['text']}" for r in raw_msgs]

            # Process in batches to respect context limits
            chunks = chunk_list(batch_texts, MAX_MESSAGES_PER_CALL)
            if len(chunks) > 1:
                log.info(f"Splitting {len(batch_texts)} messages into {len(chunks)} chunks of up to {MAX_MESSAGES_PER_CALL} each")

            aggregated_outputs: List[Dict[str, Any]] = []
            for index, chunk in enumerate(chunks, start=1):
                save_input_messages_to_txt(chunk)
                outputs = await call_llm(chunk)
                if not outputs:
                    log.info(f"Chunk {index}/{len(chunks)} produced no signals.")
                    continue
                aggregated_outputs.extend(outputs)

            if not aggregated_outputs:
                log.info("LLM produced no signals across all chunks.")
                return

            # Format summary for bot
            lines = ["*Tín hiệu mới từ nhóm nguồn:*"]
            for item in aggregated_outputs:
                ticker = item.get("ticker", "").upper()
                action = item.get("action", "").lower()
                entry = item.get("entry")
                rng = item.get("range")
                sl = item.get("sl")
                tp = item.get("tp")
                note = item.get("note")
                parts = [f"`{ticker}`", f"*{action}*"]
                if entry: parts.append(f"entry: `{entry}`")
                if rng: parts.append(f"range: `{rng}`")
                if sl: parts.append(f"SL: `{sl}`")
                if tp: parts.append(f"TP: `{tp}`")
                if note: parts.append(f"_({note})_")
                lines.append(" • " + " | ".join(parts))
            text = "\n".join(lines)
            await send_bot_message(text)
            log.info(f"Sent {len(aggregated_outputs)} signals to bot.")
        except FloodWaitError as fw:
            log.warning(f"FloodWait: sleeping {fw.seconds}s")
            await asyncio.sleep(fw.seconds)
        except RPCError as r:
            log.error(f"Telegram RPC error: {r}")
        except Exception as e:
            log.exception(f"Cycle error: {e}")

    async def run(self):
        await self.init()
        entity = await self.resolve_group()
        log.info(f"Start polling every {POLL_INTERVAL_SEC}s")
        while not self.shutdown.is_set():
            start = time.time()
            await self.cycle_once(entity)
            spent = time.time() - start
            sleep_for = max(0, POLL_INTERVAL_SEC - spent)
            try:
                await asyncio.wait_for(self.shutdown.wait(), timeout=sleep_for)
            except asyncio.TimeoutError:
                pass
        await self.client.disconnect()
        log.info("Shutdown complete.")

async def main():
    runner = Runner()

    def _handle_signal(*_):
        log.info("Signal received, shutting down...")
        runner.shutdown.set()

    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, _handle_signal)
    await runner.run()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass

