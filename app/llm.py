"""LLM + embedding access.
LangChain's ChatOpenAI is the model client (any OpenAI-compatible endpoint: Groq, Gemini, Ollama, ...).
We keep our own retry loop on top of it so rate-limit waits are visible, honest and excluded from latency."""
import hashlib, json, re, time
from functools import lru_cache
import numpy as np
from langchain_core.messages import convert_to_messages
from langchain_openai import ChatOpenAI
from openai import BadRequestError, RateLimitError
from . import config

WAITED = [0.0]  # seconds spent sleeping on provider rate limits (excluded from latency)
USAGE = {"calls": 0, "tokens": 0}  # uncached LLM calls and total tokens used since process start
_ROLE = {"human": "user", "ai": "assistant", "system": "system"}


class RateLimitExhausted(RuntimeError):
    pass


def _canonical(messages) -> list[dict]:
    """Accept OpenAI-style dicts or LangChain messages; return plain {'role','content'} dicts (stable cache keys)."""
    return [{"role": _ROLE.get(m.type, m.type), "content": m.content} for m in convert_to_messages(messages)]


@lru_cache(maxsize=16)
def _model(model: str, json_mode: bool, max_tokens: int):
    extra = {"reasoning_effort": "low"} if "gpt-oss" in model else None  # reasoning models: keep thinking short
    m = ChatOpenAI(model=model, base_url=config.LLM_BASE_URL, api_key=config.LLM_API_KEY or "none",
                   temperature=0, max_tokens=max_tokens, max_retries=0, timeout=120, extra_body=extra)
    return m.bind(response_format={"type": "json_object"}) if json_mode else m


def _wait_seconds(e: Exception, attempt: int) -> float:
    """How long the provider asks us to wait: retry-after header, else 'try again in 1m23s', else backoff."""
    try:
        h = e.response.headers.get("retry-after")
        if h:
            return float(h) + 1
    except Exception:
        pass
    m = re.search(r"try again in (?:(\d+)h)?(?:(\d+)m(?!s))?(?:([\d.]+)s)?", str(e))
    if m and any(m.groups()):
        h, mi, s = (float(x) if x else 0.0 for x in m.groups())
        return h * 3600 + mi * 60 + s + 1
    return min(2 ** attempt * 2, 60)


def _text(content) -> str:
    if isinstance(content, str):
        return content
    return "".join(p.get("text", "") if isinstance(p, dict) else str(p) for p in content)


def chat(messages, model=None, json_mode=False, max_tokens=1500, use_cache=False) -> str:
    """Deterministic (temperature 0) chat call. use_cache=True memoizes on disk, keyed by the full request hash."""
    model = model or config.LLM_MODEL
    canon = _canonical(messages)
    key = hashlib.sha256(json.dumps([model, canon, json_mode], sort_keys=True).encode()).hexdigest()
    path = config.CACHE_DIR / f"{key}.json"
    if use_cache and path.exists():
        return json.loads(path.read_text())["out"]
    budget = max(max_tokens, 2500)  # reasoning models count "thinking" tokens against max_tokens
    active_json = json_mode
    for attempt in range(12):
        try:
            resp = _model(model, active_json, budget).invoke(convert_to_messages(canon))
            out = _text(resp.content)
            USAGE["calls"] += 1
            USAGE["tokens"] += (getattr(resp, "usage_metadata", None) or {}).get("total_tokens", 0)
            break
        except RateLimitError as e:
            wait = _wait_seconds(e, attempt)
            if attempt == 0:
                print("  rate limited:", str(e)[:600])  # shows whether it is a per-minute or per-day limit
            if wait > 900:
                raise RateLimitExhausted(
                    f"Provider asks us to wait {wait / 60:.0f} min (probably a daily limit). "
                    "Progress is saved: re-run the same command later.")
            print(f"  waiting {wait:.0f}s for the rate limit ...")
            WAITED[0] += wait
            time.sleep(wait)
        except BadRequestError:
            if not active_json:
                raise
            active_json = False  # model rejected JSON mode: the prompts already demand JSON
    else:
        raise RateLimitExhausted("Rate limit retries exhausted. Progress is saved: re-run the same command.")
    if use_cache:
        path.write_text(json.dumps({"out": out}))
    return out


def parse_json(text: str) -> dict:
    text = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.M).strip()
    m = re.search(r"\{.*\}", text, flags=re.S)
    if not m:
        raise ValueError("no JSON object found")
    return json.loads(m.group(0))


def chat_json(messages, **kw) -> dict:
    return parse_json(chat(messages, json_mode=True, **kw))


@lru_cache(maxsize=1)
def _embedder():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(config.EMBED_MODEL)


def embed(texts: list[str]) -> np.ndarray:
    """Local embeddings, L2-normalized so dot product == cosine similarity."""
    return _embedder().encode(texts, normalize_embeddings=True)
