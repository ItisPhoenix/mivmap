"""Generic HTTP 400 parameter-compatibility retries for LLM requests."""
from __future__ import annotations

import json

import httpx
import pytest

from app.pipeline import llm

TEMPERATURE_400 = (
    '{"error":{"message":"Unsupported value: \'temperature\' must be 1.",'
    '"type":"invalid_request_error","code":"unsupported_value","param":"temperature"}}'
)
JSON_400 = (
    '{"error":{"message":"Unsupported value: \'response_format\' json_object is not supported.",'
    '"type":"invalid_request_error","code":"unsupported_value","param":"response_format"}}'
)
REASONING_400 = (
    '{"error":{"message":"Unknown parameter: \'reasoning_effort\'.",'
    '"type":"invalid_request_error","code":"unknown_parameter","param":"reasoning_effort"}}'
)
REASONING_PARAMS = {
    "reasoning_effort": "low",
    "include_reasoning": False,
    "reasoning_format": "hidden",
}
CHAT_URL = "https://endpoint.invalid/v1/chat/completions"


def _param_400(name: str) -> str:
    """Structured 400 naming exactly one request field."""
    return json.dumps({
        "error": {
            "message": f"Unsupported value: '{name}' is not supported.",
            "type": "invalid_request_error",
            "code": "unsupported_value",
            "param": name,
        }
    })


JSON_MODE_400 = _param_400("json_mode")


def _json_ok(content: str = "{}") -> httpx.Response:
    return httpx.Response(200, json={"choices": [{"message": {"content": content}}], "usage": {}})


def _sse_ok(text: str = "accepted") -> httpx.Response:
    return httpx.Response(200, text=_sse(text), headers={"content-type": "text/event-stream"})


def _reasoning_model(monkeypatch):
    """Simulate a reasoning model without hardcoding a provider or model family."""
    monkeypatch.setattr(llm, "reasoning_params", lambda model: dict(REASONING_PARAMS))


class _ClientFactory:
    """Return a fresh MockTransport client and remember every submitted payload."""

    def __init__(self, handler):
        self.payloads: list[dict] = []
        self.clients: list[httpx.Client] = []

        def wrapped(request: httpx.Request) -> httpx.Response:
            self.payloads.append(json.loads(request.content))
            return handler(request, self.payloads[-1])

        self.handler = wrapped

    def __call__(self) -> httpx.Client:
        client = httpx.Client(transport=httpx.MockTransport(self.handler))
        self.clients.append(client)
        return client

    def close_all(self) -> None:
        for client in self.clients:
            client.close()
        self.clients.clear()


def _sse(text: str) -> str:
    chunk = {"choices": [{"delta": {"content": text}}]}
    return f"data: {json.dumps(chunk)}\n\ndata: [DONE]\n\n"


def test_unsupported_parameter_detection():
    assert llm._unsupported_parameter(TEMPERATURE_400.lower()) is True
    assert llm._unsupported_parameter("invalid value for top_p: must be positive") is True
    assert llm._unsupported_parameter("HTTP 500 internal error") is False


def test_openai_compatible_retries_without_temperature(monkeypatch):
    factory = _ClientFactory(
        lambda request, payload: (
            httpx.Response(400, text=TEMPERATURE_400) if "temperature" in payload
            else httpx.Response(200, json={"choices": [{"message": {"content": "{\"ok\": true}"}}], "usage": {}})
        )
    )
    monkeypatch.setattr(llm, "client", factory)
    try:
        text = llm._openai_compatible(
            "https://endpoint.invalid/v1/chat/completions", "k", "test-model",
            [{"role": "user", "content": "hello"}], 100, 0.25, json_mode=True, attempts=1,
        )
    finally:
        factory.close_all()
    assert "ok" in text
    assert len(factory.payloads) == 2
    assert "temperature" in factory.payloads[0]
    assert "temperature" not in factory.payloads[1]


def test_openai_compatible_keeps_reporting_persistent_errors(monkeypatch):
    factory = _ClientFactory(lambda request, payload: httpx.Response(400, text=TEMPERATURE_400))
    monkeypatch.setattr(llm, "client", factory)
    try:
        with pytest.raises(llm.LLMError, match="HTTP 400"):
            llm._openai_compatible(
                "https://endpoint.invalid/v1/chat/completions", "k", "test-model",
                [{"role": "user", "content": "hello"}], 100, 0.25, json_mode=False, attempts=1,
            )
    finally:
        factory.close_all()
    assert len(factory.payloads) == 2, "one compatibility retry, then the real error surfaces"


def test_stream_retries_without_temperature(monkeypatch):
    factory = _ClientFactory(
        lambda request, payload: (
            httpx.Response(400, text=TEMPERATURE_400) if "temperature" in payload
            else httpx.Response(200, text=_sse("accepted"), headers={"content-type": "text/event-stream"})
        )
    )
    monkeypatch.setattr(llm, "client", factory)
    monkeypatch.setattr(llm, "fast_provider", lambda: ("https://endpoint.invalid/v1/chat/completions", "k", "test-model"))
    result = llm.Completion("")
    try:
        deltas = list(llm.stream([{"role": "user", "content": "hello"}], max_tokens=20,
                                 temperature=0.2, timeout=5.0, result=result))
    finally:
        factory.close_all()
    assert "".join(deltas) == "accepted"
    assert result.error is None
    assert len(factory.payloads) == 2
    assert "temperature" not in factory.payloads[1]


# ---------------------------------------------------------------------------
# Review corrections: body-driven parameter selection + one shared retry budget
# ---------------------------------------------------------------------------
def test_temperature_rejection_preserves_reasoning_parameters(monkeypatch):
    _reasoning_model(monkeypatch)
    factory = _ClientFactory(
        lambda request, payload: httpx.Response(400, text=TEMPERATURE_400) if "temperature" in payload else _sse_ok()
    )
    monkeypatch.setattr(llm, "client", factory)
    monkeypatch.setattr(llm, "fast_provider", lambda: (CHAT_URL, "k", "test-model"))
    result = llm.Completion("")
    try:
        deltas = list(llm.stream([{"role": "user", "content": "hello"}], max_tokens=20,
                                 temperature=0.25, timeout=5.0, result=result))
    finally:
        factory.close_all()
    assert "".join(deltas) == "accepted"
    assert result.error is None
    assert len(factory.payloads) == 2
    first, second = factory.payloads
    assert "temperature" in first and "temperature" not in second
    assert all(key in second for key in REASONING_PARAMS), "a temperature rejection must keep reasoning params"
    assert second["model"] == first["model"]
    assert second["messages"] == first["messages"]
    assert second["stream"] is True and second["max_tokens"] == first["max_tokens"]


def test_openai_compatible_temperature_retry_uses_exactly_two_requests(monkeypatch):
    _reasoning_model(monkeypatch)
    factory = _ClientFactory(
        lambda request, payload: httpx.Response(400, text=TEMPERATURE_400) if "temperature" in payload else _json_ok('{"ok": true}')
    )
    monkeypatch.setattr(llm, "client", factory)
    try:
        text = llm._openai_compatible(CHAT_URL, "k", "test-model",
                                     [{"role": "user", "content": "hello"}], 100, 0.25,
                                     json_mode=True, attempts=3)
    finally:
        factory.close_all()
    assert "ok" in text
    assert len(factory.payloads) == 2, "exactly one compatibility retry regardless of attempt budget"
    first, second = factory.payloads
    assert "temperature" in first and "temperature" not in second
    assert all(key in second for key in REASONING_PARAMS)
    assert second["response_format"] == first["response_format"]
    assert second["model"] == first["model"] and second["messages"] == first["messages"]


def test_second_incompatible_parameter_rejection_does_not_retry(monkeypatch):
    _reasoning_model(monkeypatch)

    def handler(request, payload):
        if "temperature" in payload:
            return httpx.Response(400, text=TEMPERATURE_400)
        if any(key in payload for key in REASONING_PARAMS):
            return httpx.Response(400, text=REASONING_400)
        return _json_ok('{"unreachable": true}')

    factory = _ClientFactory(handler)
    monkeypatch.setattr(llm, "client", factory)
    try:
        with pytest.raises(llm.LLMError, match="HTTP 400"):
            llm._openai_compatible(CHAT_URL, "k", "test-model",
                                   [{"role": "user", "content": "hello"}], 100, 0.25,
                                   json_mode=True, attempts=3)
    finally:
        factory.close_all()
    assert len(factory.payloads) == 2, "a second 400 must surface, not start another compatibility retry"
    assert "temperature" not in factory.payloads[1]
    assert any(key in factory.payloads[1] for key in REASONING_PARAMS)


def test_persistent_streaming_error_populates_completion_error(monkeypatch):
    factory = _ClientFactory(lambda request, payload: httpx.Response(400, text=TEMPERATURE_400))
    monkeypatch.setattr(llm, "client", factory)
    monkeypatch.setattr(llm, "fast_provider", lambda: (CHAT_URL, "k", "test-model"))
    result = llm.Completion("")
    try:
        deltas = list(llm.stream([{"role": "user", "content": "hello"}], max_tokens=20,
                                 temperature=0.2, timeout=5.0, result=result))
    finally:
        factory.close_all()
    assert deltas == []
    assert result.error and "HTTP 400" in result.error
    assert len(factory.payloads) == 2, "one compat retry, then the error surfaces"


def test_stream_second_incompatible_parameter_rejection_surfaces_error(monkeypatch):
    _reasoning_model(monkeypatch)

    def handler(request, payload):
        if "temperature" in payload:
            return httpx.Response(400, text=TEMPERATURE_400)
        if any(key in payload for key in REASONING_PARAMS):
            return httpx.Response(400, text=REASONING_400)
        return _sse_ok("unreachable")

    factory = _ClientFactory(handler)
    monkeypatch.setattr(llm, "client", factory)
    monkeypatch.setattr(llm, "fast_provider", lambda: (CHAT_URL, "k", "test-model"))
    result = llm.Completion("")
    try:
        deltas = list(llm.stream([{"role": "user", "content": "hello"}], max_tokens=20,
                                 temperature=0.2, timeout=5.0, result=result))
    finally:
        factory.close_all()
    assert deltas == []
    assert result.error and "HTTP 400" in result.error
    assert len(factory.payloads) == 2
    assert any(key in factory.payloads[1] for key in REASONING_PARAMS)


def test_json_mode_compat_drops_only_response_format(monkeypatch):
    _reasoning_model(monkeypatch)
    factory = _ClientFactory(
        lambda request, payload: httpx.Response(400, text=JSON_400) if "response_format" in payload else _json_ok()
    )
    monkeypatch.setattr(llm, "client", factory)
    try:
        llm._openai_compatible(CHAT_URL, "k", "test-model",
                               [{"role": "user", "content": "hello"}], 100, 0.25,
                               json_mode=True, attempts=2)
    finally:
        factory.close_all()
    assert len(factory.payloads) == 2
    first, second = factory.payloads
    assert "response_format" in first and "response_format" not in second
    assert second["temperature"] == first["temperature"]
    assert all(key in second for key in REASONING_PARAMS)


@pytest.mark.parametrize("rejected", ["reasoning_effort", "include_reasoning", "reasoning_format"])
def test_openai_compatible_removes_only_the_rejected_reasoning_field(monkeypatch, rejected):
    _reasoning_model(monkeypatch)
    factory = _ClientFactory(
        lambda request, payload: httpx.Response(400, text=_param_400(rejected)) if rejected in payload else _json_ok()
    )
    monkeypatch.setattr(llm, "client", factory)
    try:
        llm._openai_compatible(CHAT_URL, "k", "test-model",
                               [{"role": "user", "content": "hello"}], 100, 0.25,
                               json_mode=True, attempts=2)
    finally:
        factory.close_all()
    assert len(factory.payloads) == 2
    first, second = factory.payloads
    assert rejected in first and rejected not in second
    for other in REASONING_PARAMS:
        if other != rejected:
            assert second[other] == first[other], f"{other} must survive a {rejected} rejection"
    assert second["temperature"] == first["temperature"]
    assert second["response_format"] == first["response_format"]


@pytest.mark.parametrize("rejected", ["reasoning_effort", "include_reasoning", "reasoning_format"])
def test_stream_removes_only_the_rejected_reasoning_field(monkeypatch, rejected):
    _reasoning_model(monkeypatch)
    factory = _ClientFactory(
        lambda request, payload: httpx.Response(400, text=_param_400(rejected)) if rejected in payload else _sse_ok()
    )
    monkeypatch.setattr(llm, "client", factory)
    monkeypatch.setattr(llm, "fast_provider", lambda: (CHAT_URL, "k", "test-model"))
    result = llm.Completion("")
    try:
        deltas = list(llm.stream([{"role": "user", "content": "hello"}], max_tokens=20,
                                 temperature=0.2, timeout=5.0, result=result))
    finally:
        factory.close_all()
    assert "".join(deltas) == "accepted"
    assert result.error is None
    assert len(factory.payloads) == 2
    first, second = factory.payloads
    assert rejected in first and rejected not in second
    for other in REASONING_PARAMS:
        if other != rejected:
            assert second[other] == first[other], f"{other} must survive a {rejected} rejection"
    assert second["temperature"] == first["temperature"]


@pytest.mark.parametrize("name", ["temperature", "reasoning_effort", "include_reasoning", "reasoning_format", "response_format"])
def test_drop_absent_parameter_returns_false_and_preserves_payload(name):
    payload = {"model": "m", "messages": [], "max_tokens": 10}
    before = dict(payload)
    assert llm._drop_rejected_parameter(payload, name) is False
    assert payload == before


def test_compatibility_rejection_identifies_only_the_named_parameter():
    assert llm._rejected_parameter(TEMPERATURE_400) == "temperature"
    assert llm._rejected_parameter(JSON_400) == "response_format"
    assert llm._rejected_parameter(REASONING_400) == "reasoning_effort"
    for name in ("reasoning_effort", "include_reasoning", "reasoning_format"):
        assert llm._rejected_parameter(_param_400(name)) == name
    assert llm._rejected_parameter("generic bad request") is None


def test_unknown_structured_param_does_not_fallback_to_message():
    body = json.dumps({
        "error": {
            "message": "Unsupported value: 'temperature' is not supported.",
            "type": "invalid_request_error",
            "code": "unsupported_value",
            "param": "top_p",
        }
    })
    assert llm._rejected_parameter(body) is None


def test_structured_response_format_takes_precedence_over_temperature_message():
    body = json.dumps({
        "error": {
            "message": "Unsupported value: 'temperature' must be 1.",
            "type": "invalid_request_error",
            "code": "unsupported_value",
            "param": "response_format",
        }
    })
    assert llm._rejected_parameter(body) == "response_format"


def test_message_only_temperature_rejection_still_detected():
    body = '{"error":{"message":"Unsupported value: \'temperature\' must be 1.","type":"invalid_request_error","code":"unsupported_value"}}'
    assert llm._rejected_parameter(body) == "temperature"


def test_json_mode_rejection_behavior_unchanged():
    assert llm._rejected_parameter(JSON_MODE_400) is None


def test_429_retry_budget_is_preserved_after_compatibility_retry(monkeypatch):
    responses = iter([
        httpx.Response(400, text=TEMPERATURE_400),
        httpx.Response(429, text="rate limited", headers={"retry-after": "0"}),
        _json_ok('{"ok": true}'),
    ])
    factory = _ClientFactory(lambda request, payload: next(responses))
    monkeypatch.setattr(llm, "client", factory)
    monkeypatch.setattr(llm.time, "sleep", lambda seconds: None)
    try:
        text = llm._openai_compatible(CHAT_URL, "k", "test-model",
                                      [{"role": "user", "content": "hello"}], 100, 0.25,
                                      json_mode=False, attempts=2)
    finally:
        factory.close_all()
    assert "ok" in text
    assert len(factory.payloads) == 3, "compat retry (400) + ordinary 429 retry + success"
    assert "temperature" in factory.payloads[0]
    assert "temperature" not in factory.payloads[1]
    assert "temperature" not in factory.payloads[2]


# ---------------------------------------------------------------------------
# Final review: fail-fast on permanent statuses + precise parameter detection
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("status", [401, 403, 404])
def test_non_retryable_status_produces_exactly_one_request(monkeypatch, status):
    factory = _ClientFactory(lambda request, payload: httpx.Response(status, text="permanent failure"))
    monkeypatch.setattr(llm, "client", factory)
    try:
        with pytest.raises(llm.LLMError, match=f"HTTP {status}"):
            llm._openai_compatible(CHAT_URL, "k", "test-model",
                                   [{"role": "user", "content": "hello"}], 100, 0.25,
                                   json_mode=False, attempts=3)
    finally:
        factory.close_all()
    assert len(factory.payloads) == 1, "auth/client errors must never be resent"


@pytest.mark.parametrize("status", [429, 500])
def test_retryable_status_retains_bounded_retry(monkeypatch, status):
    responses = iter([
        httpx.Response(status, text="retry", headers={"retry-after": "0"}),
        httpx.Response(status, text="retry", headers={"retry-after": "0"}),
        _json_ok('{"ok": true}'),
    ])
    factory = _ClientFactory(lambda request, payload: next(responses))
    monkeypatch.setattr(llm, "client", factory)
    monkeypatch.setattr(llm.time, "sleep", lambda seconds: None)
    try:
        text = llm._openai_compatible(CHAT_URL, "k", "test-model",
                                      [{"role": "user", "content": "hello"}], 100, 0.25,
                                      json_mode=False, attempts=3)
    finally:
        factory.close_all()
    assert "ok" in text
    assert len(factory.payloads) == 3, f"HTTP {status} keeps the ordinary attempt budget"


@pytest.mark.parametrize("body, word", [
    ('{"error":{"message":"reasoning tokens were exhausted","type":"server_error"}}', "reasoning"),
    ('{"error":{"message":"json response was malformed","type":"server_error"}}', "json"),
])
def test_generic_mention_does_not_trigger_retry(monkeypatch, body, word):
    factory = _ClientFactory(lambda request, payload: httpx.Response(400, text=body))
    monkeypatch.setattr(llm, "client", factory)
    try:
        with pytest.raises(llm.LLMError, match="HTTP 400"):
            llm._openai_compatible(CHAT_URL, "k", "test-model",
                                   [{"role": "user", "content": "hello"}], 100, 0.25,
                                   json_mode=True, attempts=3)
    finally:
        factory.close_all()
    assert len(factory.payloads) == 1, f"a generic '{word}' mention is not a parameter rejection"
    assert llm._rejected_parameter(body) is None


def test_rejected_parameter_absent_from_payload_does_not_retry(monkeypatch):
    # JSON_400 names response_format, but json_mode=False means it was never sent.
    factory = _ClientFactory(lambda request, payload: httpx.Response(400, text=JSON_400))
    monkeypatch.setattr(llm, "client", factory)
    try:
        with pytest.raises(llm.LLMError, match="HTTP 400"):
            llm._openai_compatible(CHAT_URL, "k", "test-model",
                                   [{"role": "user", "content": "hello"}], 100, 0.25,
                                   json_mode=False, attempts=3)
    finally:
        factory.close_all()
    assert len(factory.payloads) == 1, "no matching parameter in payload means no retry"


def test_absent_reasoning_parameter_does_not_retry(monkeypatch):
    monkeypatch.setattr(llm, "reasoning_params", lambda model: {})
    factory = _ClientFactory(lambda request, payload: httpx.Response(400, text=REASONING_400))
    monkeypatch.setattr(llm, "client", factory)
    try:
        with pytest.raises(llm.LLMError, match="HTTP 400"):
            llm._openai_compatible(CHAT_URL, "k", "test-model",
                                   [{"role": "user", "content": "hello"}], 100, 0.25,
                                   json_mode=False, attempts=3)
    finally:
        factory.close_all()
    assert len(factory.payloads) == 1


def test_structured_param_field_is_preferred_over_message_text():
    body = '{"error":{"message":"bad request","param":"response_format"}}'
    assert llm._rejected_parameter(body) == "response_format"
    quoted_only = '{"error":{"message":"Unsupported value: \'response_format\' is not allowed."}}'
    assert llm._rejected_parameter(quoted_only) == "response_format"
    assert llm._rejected_parameter(JSON_MODE_400) is None


def test_json_mode_error_does_not_remove_response_format(monkeypatch):
    _reasoning_model(monkeypatch)
    factory = _ClientFactory(lambda request, payload: httpx.Response(400, text=JSON_MODE_400))
    monkeypatch.setattr(llm, "client", factory)
    try:
        with pytest.raises(llm.LLMError, match="HTTP 400"):
            llm._openai_compatible(CHAT_URL, "k", "test-model",
                                   [{"role": "user", "content": "hello"}], 100, 0.25,
                                   json_mode=True, attempts=3)
    finally:
        factory.close_all()
    assert len(factory.payloads) == 1, "json_mode is not a supported payload alias; never retry"
    assert "response_format" in factory.payloads[0]
