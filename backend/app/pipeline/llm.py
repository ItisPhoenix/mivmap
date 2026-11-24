"""
Layer 4 — LLM integration (Groq first, open-source fallbacks second).

Two kinds of calls:

* `complete()` / `stream()` — ONE attempt, streamed, with a hard deadline. Used by the
  map labeller (one call per map) and the node chatbot (one call per message).
  A timeout, 429 or any error returns what arrived so far (possibly nothing) —
  callers always have a heuristic fallback, so there are no sleeping retries.
* `chat_json()` — user-triggered actions (AI refine, study set). Provider chain:
  Groq main model → Groq fallback model → Hugging Face router → local transformers.

One module-level keep-alive `httpx.Client` is shared by every call, and every call
is counted (calls / prompt tokens / completion tokens) so usage is visible in logs.
"""
from __future__ import annotations

import json
import logging
import re
import threading
import time
from dataclasses import dataclass
from urllib.parse import urlsplit
from functools import lru_cache
from typing import Any, Callable, Iterator

import httpx

from ..config import has_module, settings
from .text_utils import extract_json

log = logging.getLogger("mivmap.llm")

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODELS_URL = "https://api.groq.com/openai/v1/models"
HF_URL = "https://router.huggingface.co/v1/chat/completions"

# Fastest reliable first (id prefixes; the newest live version wins). Non-reasoning models are
# preferred; reasoning ones run with reasoning switched off / low (see `reasoning_params`).
FAST_MODEL_PREFERENCE = ["llama-3.1-8b-instant", "qwen/qwen3", "llama-3.3-70b", "openai/gpt-oss-20b", "openai/gpt-oss-120b"]

_semaphore = threading.BoundedSemaphore(4)
_client: httpx.Client | None = None
_client_lock = threading.Lock()
_CHAT_JSON_STATE = threading.local()
_stats_lock = threading.Lock()
STATS = {
    "calls": 0,               # successful model responses (backwards compatible)
    "prompt_tokens": 0,
    "completion_tokens": 0,
    "errors": 0,
    "logical_calls": 0,       # one per chat_json()/complete() logical request
    "http_requests": 0,       # every outbound POST, including ones that fail
}
_groq_models: list[str] | None = None
_cooldown_until = 0.0  # after a 429 the provider is skipped (0 calls) until this monotonic time
DEFAULT_COOLDOWN = 20.0
LAST_STATUS: int | None = None
LAST_RETRY_AFTER: float | None = None

# (endpoint, model) pairs observed rejecting `response_format` are remembered for
# this process so later JSON calls skip the wasted json_object attempt (HTTP 400).
# The endpoint is normalised (scheme/host lower-cased, userinfo removed) and the
# model is kept verbatim: two models sharing one gateway must never inherit each
# other's capability verdict. Neither credentials nor keys are ever stored here.
_NO_JSON_MODE: set[tuple[str, str]] = set()


def _normalize_endpoint(url: str) -> str:
    """Comparable endpoint identity without credentials, query or fragment."""
    raw = str(url or "").strip()
    if not raw:
        return ""
    try:
        parts = urlsplit(raw if "://" in raw else "http://" + raw)
        host = (parts.hostname or "").lower().rstrip(".")
        port = f":{parts.port}" if parts.port else ""
        path = (parts.path or "").rstrip("/")
        return f"{parts.scheme.lower()}://{host}{port}{path}"
    except ValueError:
        return raw.rstrip("/")


def json_capability_key(url: str, model: str) -> tuple[str, str]:
    """Cache key for a response-format capability verdict."""
    return (_normalize_endpoint(url), str(model or ""))


def json_mode_supported(url: str, model: str) -> bool:
    """True unless this exact endpoint/model pair rejected JSON mode."""
    return json_capability_key(url, model) not in _NO_JSON_MODE


def mark_json_mode_unsupported(url: str, model: str) -> None:
    """Remember a confirmed response-format rejection for this endpoint/model."""
    _NO_JSON_MODE.add(json_capability_key(url, model))


# Only exact fields the outgoing request contract can actually carry. `json_mode`
# is not a payload field (json_mode sets `response_format`), so it is deliberately
# not an alias here.
_PARAMETER_NAMES = (
    "temperature",
    "reasoning_effort",
    "include_reasoning",
    "reasoning_format",
    "response_format",
)


def _unsupported_parameter(error: str) -> bool:
    """True when a 400 body says the endpoint rejects the submitted parameter."""
    lowered = error.lower()
    return any(marker in lowered for marker in (
        "unsupported value", "unsupported parameter", "unknown parameter",
        "unexpected parameter", "invalid value", "must be", "not allowed",
    ))


# A message-only rejection must tie the field to a rejection phrase. A quoted
# field mentioned somewhere else in an unrelated error is deliberately NOT enough.
_REJECTION_PREFIX = re.compile(
    r"(?:unsupported|unknown|unexpected|invalid|unrecognized|unrecognised)\s+"
    r"(?:value|parameter|field|argument)\s*(?::|for)?\s*"
)
_REJECTION_SUFFIX = re.compile(
    r"(?:is\s+)?(?:not\s+supported|unsupported|not\s+allowed|not\s+permitted|"
    r"disallowed|must\s+be|is\s+invalid|invalid|unknown|rejected)"
)


def _message_rejects_field(message: str, name: str) -> bool:
    """True only when the message explicitly names `name` as rejected."""
    lowered = message.lower()
    quoted = r"['\"]" + re.escape(name) + r"['\"]"
    # "Unsupported value: 'temperature' must be 1" / "Unknown parameter: 'reasoning_effort'."
    if re.search(_REJECTION_PREFIX.pattern + quoted, lowered):
        return True
    # "'response_format.type' must be 'json_schema' or 'text'" / "'response_format' is not allowed"
    if re.search(quoted + r"\s+" + _REJECTION_SUFFIX.pattern, lowered):
        return True
    # Unquoted is accepted only with an explicit "parameter/field <name> ..." lead-in.
    if re.search(
        rf"(?:parameter|field|argument)\s+{re.escape(name)}\s+" + _REJECTION_SUFFIX.pattern,
        lowered,
    ):
        return True
    return False


def _rejected_parameter(error: str) -> str | None:
    """Return the exact request field a 400 rejects, if any.

    Structured `error.param` wins. Unknown structured params fail closed. Without
    one, the message must explicitly identify the rejected field next to a
    rejection phrase; merely mentioning it (even in quotes) is insufficient.
    """
    structured = None
    try:
        structured = json.loads(error)
    except (TypeError, ValueError):
        pass

    param = ""
    if isinstance(structured, dict):
        detail = structured.get("error")
        if isinstance(detail, dict):
            param = detail.get("param") or ""
        if not param:
            param = structured.get("param") or ""
    name = str(param).strip().strip("'\"").lower()
    if name:
        # Never guess from message text when the server already named a field.
        return name if name in _PARAMETER_NAMES else None

    # Check the sub-field first so 'response_format.type' is not mistaken for
    # the parent parameter. Both map to `response_format` on the request.
    if _message_rejects_field(error, "response_format.type"):
        return "response_format"
    for candidate in _PARAMETER_NAMES:
        if _message_rejects_field(error, candidate):
            return candidate
    return None


def _confirmed_json_mode_rejection(error: str, payload: dict[str, Any]) -> bool:
    """Only a confirmed rejection of a response_format we actually sent."""
    return "response_format" in payload and _rejected_parameter(error) == "response_format"


def _drop_rejected_parameter(payload: dict[str, Any], name: str) -> bool:
    """Remove exactly `name`. Return False if absent; preserve all other fields."""
    if name not in payload:
        return False
    del payload[name]
    return True


class LLMError(RuntimeError):
    pass


@dataclass
class Completion:
    text: str
    model: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    ms: float = 0.0
    error: str | None = None
    partial: bool = False  # deadline hit mid-stream: `text` holds what arrived
    sent: bool = False  # a request actually went out (False: no LLM configured / cooling down)

    @property
    def ok(self) -> bool:
        return self.error is None and not self.partial


def client() -> httpx.Client:
    global _client
    with _client_lock:
        if _client is None:
            _client = httpx.Client(
                timeout=httpx.Timeout(30.0, connect=5.0),
                limits=httpx.Limits(max_connections=16, max_keepalive_connections=8, keepalive_expiry=120),
            )
        return _client


# ---------------------------------------------------------------------------
# Provider / model selection
# ---------------------------------------------------------------------------
def llm_chat_url() -> str:
    """Build the chat-completions URL for the configured OpenAI-compatible provider."""
    base = (settings.llm_base_url or "").strip().rstrip("/")
    if not base:
        return ""
    if base.endswith("/chat/completions"):
        return base
    # Some gateways already expose /v1 or /api/gateway; append the standard path.
    return base + "/chat/completions"


def provider_name() -> str:
    if settings.llm_enabled and llm_chat_url():
        return f"openai-compatible:{settings.llm_model}"
    if settings.groq_enabled:
        return f"groq:{settings.groq_model}"
    if settings.hf_token:
        return f"huggingface:{settings.hf_model}"
    if settings.hf_local_model and has_module("transformers"):
        return f"local:{settings.hf_local_model}"
    return "heuristic"


def available() -> bool:
    return provider_name() != "heuristic"


def fast_model() -> str:
    """Model used for map labels and the node chatbot (latency matters more than depth)."""
    if settings.llm_enabled and settings.llm_model:
        return settings.llm_model
    if settings.groq_label_model:
        return settings.groq_label_model
    if _groq_models is None and settings.groq_enabled:
        warm_up()  # learn which models are live (normally already done at server start)
    live = _groq_models or []
    if not live:
        return FAST_MODEL_PREFERENCE[0]
    for prefix in FAST_MODEL_PREFERENCE:
        matches = sorted((m for m in live if m.startswith(prefix)), reverse=True)
        if matches:
            return matches[0]
    return settings.groq_model


def fast_provider() -> tuple[str, str, str] | None:
    """(url, key, model) for single-shot calls, or None when no remote LLM is configured."""
    if settings.llm_enabled and llm_chat_url():
        return llm_chat_url(), settings.llm_api_key, fast_model()
    if settings.groq_enabled:
        return GROQ_URL, settings.groq_api_key, fast_model()
    if settings.hf_token:
        return HF_URL, settings.hf_token, settings.hf_model
    return None


def fast_available() -> bool:
    """An HTTP single-shot provider (remote or unauthenticated loopback) is configured."""
    return fast_provider() is not None


def local_available() -> bool:
    """A local transformers model is configured and importable.

    This is deliberately separate from `fast_provider()`: local inference runs
    in-process via `chat_json()`/_local_generate(), not through the HTTP
    streaming path used by `complete()`/`stream()`.
    """
    return bool(settings.hf_local_model) and has_module("transformers")


def generation_available() -> bool:
    """True when any provider can serve JSON lesson generation.

    Fresh explanations go through `chat_json()`, which supports the HTTP
    providers *and* an in-process local transformers model. Gate generation on
    this, not on `fast_available()` alone, so HF_LOCAL_MODEL is not silently
    excluded.
    """
    return fast_available() or local_available()


def fast_name() -> str:
    p = fast_provider()
    if not p:
        return "heuristic"
    url = p[0]
    if settings.llm_enabled and url == llm_chat_url():
        return f"openai-compatible:{p[2]}"
    if url == GROQ_URL:
        return f"groq:{p[2]}"
    if url == HF_URL:
        return f"huggingface:{p[2]}"
    return f"remote:{p[2]}"


def warm_up() -> None:
    """Open the keep-alive connection (TLS handshake) and learn which Groq models are live."""
    global _groq_models
    if settings.llm_enabled and llm_chat_url():
        base = settings.llm_base_url.rstrip("/")
        try:
            client().get(base + "/models", headers=_headers(settings.llm_api_key), timeout=5)
        except Exception as exc:  # noqa: BLE001
            log.info("LLM warm-up skipped: %s", exc)
        log.info("LLM warm: openai-compatible model %s", settings.llm_model)
        return
    if not settings.groq_enabled:
        return
    try:
        resp = client().get(GROQ_MODELS_URL, headers=_headers(settings.groq_api_key), timeout=5)
        _groq_models = sorted(m["id"] for m in resp.json().get("data", [])) if resp.status_code == 200 else []
    except Exception as exc:  # noqa: BLE001 - warm-up is best effort
        _groq_models = []
        log.info("LLM warm-up skipped: %s", exc)
    log.info("LLM warm: fast model %s", fast_model())


def reasoning_params(model: str) -> dict:
    """Keep hidden 'thinking' tokens (latency + cost) to a minimum on reasoning models."""
    m = model.lower()
    if "gpt-oss" in m:
        return {"reasoning_effort": "low", "include_reasoning": False}
    if "qwen3" in m:
        return {"reasoning_effort": "none"}
    if "gemma-4" in m or "gemma4" in m:
        # Gemma 4 reasoning models otherwise spend most of the token budget on
        # reasoning_content, leaving content empty or truncated. Local gateways
        # accept reasoning_effort="none" to disable it (already an allowlisted
        # parameter; the HTTP-400 retry drops it if an endpoint disagrees).
        return {"reasoning_effort": "none"}
    if "deepseek-r1" in m:
        return {"reasoning_format": "hidden"}
    return {}


def _headers(key: str) -> dict:
    """Build request headers. An empty key means an unauthenticated local server.

    Local inference endpoints (LM Studio, Ollama, vLLM) typically ignore or
    reject an Authorization header, so it is omitted entirely when no key is
    configured. Remote providers always supply a real key (see
    `Settings.llm_enabled`), so they still receive the Bearer header.
    """
    headers: dict[str, str] = {"Content-Type": "application/json"}
    if key:
        headers["Authorization"] = f"Bearer {key}"
    return headers


def _count(c: Completion) -> None:
    with _stats_lock:
        STATS["calls"] += 1
        STATS["prompt_tokens"] += c.prompt_tokens
        STATS["completion_tokens"] += c.completion_tokens
        STATS["errors"] += 1 if c.error else 0


def estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4)


# ---------------------------------------------------------------------------
# Single-shot streamed calls (labels, chat)
# ---------------------------------------------------------------------------
def stream(
    messages: list[dict],
    *,
    max_tokens: int,
    temperature: float = 0.2,
    timeout: float = 3.0,
    result: Completion | None = None,
) -> Iterator[str]:
    """
    Yield text deltas from a streamed request. Never raises: errors, 429s and the
    deadline end the stream early and are recorded in `result` (pass one in to read
    usage/errors afterwards). No sleeping; at most one compatibility retry when the
    endpoint explicitly rejects a submitted request parameter with HTTP 400.
    """
    out = result if result is not None else Completion("")
    provider = fast_provider()
    if provider is None:
        out.error = "no LLM configured"
        return
    url, key, model = provider
    out.model = model
    if time.monotonic() < _cooldown_until:
        out.error = "rate limited (cooling down)"
        return
    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": True,
        **reasoning_params(model),
    }
    started = time.monotonic()
    deadline = started + timeout
    parts: list[str] = []
    usage: dict = {}
    out.sent = True
    compat_retried = False
    attempts = 0
    try:
        with _semaphore:
            while attempts < 2:
                req_timeout = httpx.Timeout(max(0.5, deadline - time.monotonic()), connect=min(2.0, timeout))
                with _stats_lock:
                    STATS["http_requests"] += 1
                with client().stream("POST", url, headers=_headers(key), json=payload, timeout=req_timeout) as resp:
                    if resp.status_code == 400:
                        resp.read()
                        body = resp.text[:300]
                        rejected = _rejected_parameter(body)
                        if rejected and not compat_retried and _drop_rejected_parameter(payload, rejected):
                            # ONE compatibility retry per request: drop only a parameter
                            # that is actually present, then resend once. A 400 spends no
                            # tokens; an unchanged payload is never resent.
                            compat_retried = True
                            continue
                        out.error = f"HTTP 400: {body}"
                        break
                    attempts += 1
                    if resp.status_code != 200:
                        resp.read()
                        out.error = f"HTTP {resp.status_code}: {resp.text[:160]}"
                        if resp.status_code == 429:
                            _cool_down(resp)
                        break
                    for line in resp.iter_lines():
                        if time.monotonic() > deadline:
                            out.partial = True
                            out.error = "deadline"
                            break
                        if not line.startswith("data:"):
                            continue
                        data = line[5:].strip()
                        if data == "[DONE]":
                            break
                        try:
                            chunk = json.loads(data)
                        except json.JSONDecodeError:
                            continue
                        usage = chunk.get("usage") or (chunk.get("x_groq") or {}).get("usage") or usage
                        for choice in chunk.get("choices") or []:
                            delta = (choice.get("delta") or {}).get("content")
                            if delta:
                                parts.append(delta)
                                yield delta
                    break
    except httpx.TimeoutException:
        out.partial, out.error = bool(parts), "timeout"
    except Exception as exc:  # noqa: BLE001 - never break the caller
        out.error = f"{type(exc).__name__}: {exc}"
    finally:
        out.text = "".join(parts)
        out.ms = (time.monotonic() - started) * 1000
        prompt_chars = sum(len(m.get("content", "")) for m in messages)
        out.prompt_tokens = int(usage.get("prompt_tokens") or max(1, prompt_chars // 4))
        out.completion_tokens = int(usage.get("completion_tokens") or (estimate_tokens(out.text) if out.text else 0))
        _count(out)
        if out.error:
            log.warning("LLM %s: %s after %.0f ms (%d chars kept)", model, out.error, out.ms, len(out.text))


def complete(system: str, user: str, *, max_tokens: int, temperature: float = 0.2, timeout: float = 3.0,
             on_text: Callable[[str], None] | None = None) -> Completion:
    """One streamed call collected into a `Completion` (partial text is kept on timeout)."""
    result = Completion("")
    messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    with _stats_lock:
        STATS["logical_calls"] += 1
    for delta in stream(messages, max_tokens=max_tokens, temperature=temperature, timeout=timeout, result=result):
        if on_text:
            on_text(delta)
    return result


# ---------------------------------------------------------------------------
# JSON calls for user-triggered features (refine, study)
# ---------------------------------------------------------------------------
def chat_json_diagnostics() -> dict[str, Any]:
    """Return diagnostics for the CURRENT thread's most recent chat_json call."""
    return dict(getattr(_CHAT_JSON_STATE, "diagnostics", {}) or {})


def _reset_call_diagnostics() -> None:
    """Clear per-call state so a previous (or another thread's) call cannot leak."""
    _CHAT_JSON_STATE.status = None
    _CHAT_JSON_STATE.retry_after = None
    _CHAT_JSON_STATE.diagnostics = {}


def _record_response_status(status: int | None, retry_after: float | None) -> None:
    """Record the response status on the calling thread, never on a global."""
    _CHAT_JSON_STATE.status = status
    _CHAT_JSON_STATE.retry_after = retry_after


def _redact(text: Any) -> str | None:
    """Strip URL userinfo (credentials) from text stored in diagnostics."""
    if text is None:
        return None
    return re.sub(r"://[^/\s:@]+:[^/\s@]+@", "://***:***@", str(text))[:1200]


def _record_chat_json_diagnostics(
    provider: str,
    error: str | None = None,
    error_kind: str | None = None,
) -> None:
    _CHAT_JSON_STATE.diagnostics = {
        "provider": provider,
        "status": getattr(_CHAT_JSON_STATE, "status", None),
        "retry_after": getattr(_CHAT_JSON_STATE, "retry_after", None),
        "error": _redact(error),
        "error_kind": error_kind,
    }


def chat_json(system: str, user: str, max_tokens: int = 1800, temperature: float = 0.3) -> Any | None:
    """Return parsed JSON and retain per-thread diagnostics when every provider fails."""
    global LAST_STATUS, LAST_RETRY_AFTER
    LAST_STATUS = None
    LAST_RETRY_AFTER = None
    _reset_call_diagnostics()
    with _stats_lock:
        STATS["logical_calls"] += 1
    _record_chat_json_diagnostics("none")

    messages = [
        {"role": "system", "content": system + "\nRespond with valid JSON only. No markdown fences."},
        {"role": "user", "content": user},
    ]
    errors = []
    attempted_provider = "none"
    last_error_kind = None
    with _semaphore:
        if settings.llm_enabled and llm_chat_url():
            attempted_provider = f"openai-compatible:{settings.llm_model}"
            # Reasoning models sometimes answer in <reasoning> only, or emit truncated JSON.
            # Retry once without json_mode and with a larger budget before giving up.
            last_error: Exception | None = None
            attempts_modes = ((True, max_tokens), (False, min(max_tokens * 2, 3000)))
            if not json_mode_supported(llm_chat_url(), settings.llm_model):
                # This exact endpoint/model pair already proved it unsupported.
                attempts_modes = ((False, min(max_tokens * 2, 3000)),)
            for json_mode, budget in attempts_modes:
                try:
                    text = _openai_compatible(
                        llm_chat_url(), settings.llm_api_key, settings.llm_model, messages,
                        budget, temperature, json_mode=json_mode, attempts=1,
                    )
                    if text and text.strip():
                        parsed = extract_json(text)
                        _record_chat_json_diagnostics(attempted_provider)
                        return parsed
                    last_error = RuntimeError("empty LLM response")
                    last_error_kind = "empty_response"
                except Exception as exc:  # noqa: BLE001
                    last_error = exc
                    last_error_kind = (
                        "response_parse" if isinstance(exc, (ValueError, json.JSONDecodeError))
                        else "provider_error"
                    )
            if last_error:
                errors.append(f"{attempted_provider}: {type(last_error).__name__}: {last_error}")
                log.warning("OpenAI-compatible model %s failed: %s", settings.llm_model, last_error)

        if settings.groq_enabled:
            for model in dict.fromkeys([settings.groq_model, settings.groq_fallback_model]):
                attempted_provider = f"groq:{model}"
                try:
                    parsed = extract_json(_openai_compatible(
                        GROQ_URL, settings.groq_api_key, model, messages, max_tokens,
                        temperature, json_mode=json_mode_supported(GROQ_URL, model),
                    ))
                    _record_chat_json_diagnostics(attempted_provider)
                    return parsed
                except Exception as exc:  # noqa: BLE001
                    errors.append(f"{attempted_provider}: {type(exc).__name__}: {exc}")
                    last_error_kind = (
                        "response_parse" if isinstance(exc, (ValueError, json.JSONDecodeError))
                        else "provider_error"
                    )
                    log.warning("Groq %s failed: %s", model, exc)

        if settings.hf_token:
            attempted_provider = f"huggingface:{settings.hf_model}"
            try:
                # The HF router never receives response_format, so it has no
                # JSON-mode capability verdict to consult or record.
                parsed = extract_json(_openai_compatible(
                    HF_URL, settings.hf_token, settings.hf_model, messages,
                    max_tokens, temperature, json_mode=False,
                ))
                _record_chat_json_diagnostics(attempted_provider)
                return parsed
            except Exception as exc:  # noqa: BLE001
                errors.append(f"{attempted_provider}: {type(exc).__name__}: {exc}")
                last_error_kind = (
                    "response_parse" if isinstance(exc, (ValueError, json.JSONDecodeError))
                    else "provider_error"
                )
                log.warning("Hugging Face router failed: %s", exc)

        if settings.hf_local_model and has_module("transformers"):
            attempted_provider = f"local:{settings.hf_local_model}"
            try:
                parsed = extract_json(_local_generate(messages, max_tokens, temperature))
                _record_chat_json_diagnostics(attempted_provider)
                return parsed
            except Exception as exc:  # noqa: BLE001
                errors.append(f"{attempted_provider}: {type(exc).__name__}: {exc}")
                last_error_kind = (
                    "response_parse" if isinstance(exc, (ValueError, json.JSONDecodeError))
                    else "provider_error"
                )
                log.warning("Local model failed: %s", exc)

    if errors:
        error = " | ".join(errors)
        error_kind = last_error_kind or "provider_error"
        log.error("All LLM providers failed: %s", error)
    else:
        error = "no LLM provider configured"
        error_kind = "not_configured"
    _record_chat_json_diagnostics(attempted_provider, error, error_kind)
    return None


def _openai_compatible(
    url: str,
    key: str,
    model: str,
    messages: list[dict],
    max_tokens: int,
    temperature: float,
    json_mode: bool,
    attempts: int = 2,
    max_wait: float = 3.0,
) -> str:
    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        **reasoning_params(model),
    }
    if json_mode:
        payload["response_format"] = {"type": "json_object"}

    started = time.monotonic()
    attempt = 0
    compat_retried = False
    while attempt < attempts:
        global LAST_STATUS, LAST_RETRY_AFTER
        with _stats_lock:
            STATS["http_requests"] += 1
        resp = client().post(url, headers=_headers(key), json=payload, timeout=60)
        LAST_STATUS = resp.status_code
        LAST_RETRY_AFTER = None if resp.status_code != 429 else _retry_after(resp)
        _record_response_status(resp.status_code, LAST_RETRY_AFTER)
        if resp.status_code == 200:
            data = resp.json()
            usage = data.get("usage") or {}
            _count(Completion("", model, int(usage.get("prompt_tokens") or 0), int(usage.get("completion_tokens") or 0), (time.monotonic() - started) * 1000))
            return data["choices"][0]["message"]["content"]
        body = resp.text[:300]
        if resp.status_code == 400:
            rejected = _rejected_parameter(body)
            if _confirmed_json_mode_rejection(body, payload):
                mark_json_mode_unsupported(url, model)
            if rejected and not compat_retried and _drop_rejected_parameter(payload, rejected):
                # ONE compatibility retry per request: drop only a parameter that is
                # actually present. No tokens were spent, so the caller's attempt
                # budget for 429/5xx stays intact. An unchanged payload is never resent.
                compat_retried = True
                continue
            raise LLMError(f"HTTP {resp.status_code}: {body}")
        if resp.status_code == 429 or 500 <= resp.status_code < 600:
            wait = _retry_after(resp) or 1.0
            if wait > max_wait or attempt == attempts - 1:  # user is waiting: fail fast
                raise LLMError(f"HTTP {resp.status_code} (rate limited/unavailable): {body}")
            time.sleep(wait)
            attempt += 1
            continue
        # 401/403/404 and every other non-retryable status: never resend a
        # permanent authentication or client error.
        raise LLMError(f"HTTP {resp.status_code}: {body}")
    raise LLMError(f"HTTP {resp.status_code}: {body}")


def _cool_down(resp: httpx.Response) -> None:
    global _cooldown_until
    wait = min(_retry_after(resp) or DEFAULT_COOLDOWN, 120.0)
    _cooldown_until = time.monotonic() + wait
    log.info("LLM rate limited: single-shot calls paused for %.0f s", wait)


def _retry_after(resp: httpx.Response) -> float | None:
    value = resp.headers.get("retry-after")
    try:
        return float(value) if value else None
    except ValueError:
        return None


@lru_cache(maxsize=1)
def _local_pipeline():
    from transformers import pipeline

    log.info("Loading local model %s", settings.hf_local_model)
    return pipeline("text-generation", model=settings.hf_local_model, device_map="auto")


def _local_generate(messages: list[dict], max_tokens: int, temperature: float) -> str:
    pipe = _local_pipeline()
    out = pipe(messages, max_new_tokens=max_tokens, do_sample=temperature > 0, temperature=max(temperature, 0.01))
    generated = out[0]["generated_text"]
    if isinstance(generated, list):  # chat format returns the whole conversation
        return generated[-1]["content"]
    return str(generated)


def compact(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))
