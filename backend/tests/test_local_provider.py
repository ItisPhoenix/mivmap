"""Local OpenAI-compatible provider regressions — no key, no live quota.

Covers the real-world case of LM Studio / Ollama / vLLM on loopback (no auth),
remote endpoints that must still require a key, the in-process local model
generation path, and honest failure classification. All LLM calls are mocked.
"""
from __future__ import annotations

from dataclasses import replace

import pytest

from app import config, main
from app.config import is_loopback_url
from app.features import explanations
from app.pipeline import labels, lesson_cache, llm
from test_explanation_recovery import VALID_LESSON, _map_with_concept


@pytest.fixture(autouse=True)
def isolated_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(lesson_cache, "cache_dir", lambda: tmp_path)
    return tmp_path


@pytest.fixture
def cfg(monkeypatch):
    """Swap the module-level settings with a controlled copy."""
    def apply(**overrides):
        fake = replace(config.settings, **overrides)
        monkeypatch.setattr(config, "settings", fake)
        monkeypatch.setattr(llm, "settings", fake)
        monkeypatch.setattr(explanations, "settings", fake)
        monkeypatch.setattr(main, "settings", fake)
        return fake
    return apply


LOCAL = dict(
    groq_api_key="", hf_token="", hf_local_model="",
    llm_api_key="", llm_base_url="http://127.0.0.1:1234/v1", llm_model="lfm3-1.7b",
)
REMOTE = dict(
    groq_api_key="", hf_token="", hf_local_model="",
    llm_api_key="", llm_base_url="https://api.example.com/v1", llm_model="gpt-example",
)


@pytest.mark.parametrize("url,expected", [
    ("http://127.0.0.1:1234/v1", True),
    ("http://127.0.0.2:8080", True),
    ("http://localhost:11434/v1", True),
    ("http://[::1]:8080/v1", True),
    ("https://api.example.com/v1", False),
    ("http://192.168.1.50:1234/v1", False),
    ("", False),
])
def test_loopback_detection(url, expected):
    assert is_loopback_url(url) is expected


def test_local_endpoint_without_key_is_enabled(cfg):
    cfg(**LOCAL)
    assert config.settings.llm_is_local is True
    assert config.settings.llm_enabled is True
    assert llm.fast_provider() == (llm.llm_chat_url(), "", "lfm3-1.7b")
    assert llm.fast_available() is True
    assert llm.generation_available() is True
    assert llm.provider_name() == "openai-compatible:lfm3-1.7b"
    assert llm.fast_model() == "lfm3-1.7b"


def test_local_endpoint_with_key_sends_auth(cfg):
    cfg(**{**LOCAL, "llm_api_key": "local-only-secret"})
    assert config.settings.llm_enabled is True
    assert llm.fast_provider()[1] == "local-only-secret"
    assert llm._headers(llm.fast_provider()[1])["Authorization"] == "Bearer local-only-secret"


def test_no_auth_header_omitted_for_unauthenticated_endpoint():
    headers = llm._headers("")
    assert "Content-Type" in headers
    assert "Authorization" not in headers


def test_gemma4_reasoning_is_disabled_so_content_is_not_truncated():
    assert llm.reasoning_params("google/gemma-4-e2b") == {"reasoning_effort": "none"}
    assert llm.reasoning_params("google/gemma-4-e4b") == {"reasoning_effort": "none"}
    assert llm.reasoning_params("lfm3-1.7b") == {}


def test_remote_endpoint_without_key_is_rejected(cfg):
    cfg(**REMOTE)
    assert config.settings.llm_is_local is False
    assert config.settings.llm_enabled is False
    assert llm.fast_provider() is None
    assert llm.fast_available() is False
    assert llm.provider_name() == "heuristic"


def test_placeholder_key_never_enables_a_remote_endpoint(cfg):
    cfg(**{**REMOTE, "llm_api_key": "your_api_key_here"})
    assert config.settings.llm_enabled is False
    assert llm.fast_available() is False


def test_hf_local_model_is_generation_capable_but_not_http_fast(cfg):
    cfg(groq_api_key="", hf_token="", llm_api_key="", llm_base_url="",
        hf_local_model="Qwen/Qwen2.5-7B-Instruct")
    assert llm.provider_name() == "local:Qwen/Qwen2.5-7B-Instruct"
    assert llm.local_available() is True
    assert llm.generation_available() is True
    assert llm.fast_available() is False  # no HTTP provider for streamed calls
    assert llm.fast_provider() is None


def test_hf_local_model_can_generate_a_validated_lesson(cfg, isolated_cache, monkeypatch):
    cfg(groq_api_key="", hf_token="", llm_api_key="", llm_base_url="",
        hf_local_model="Qwen/Qwen2.5-7B-Instruct")
    mm = _map_with_concept()
    monkeypatch.setattr(labels.llm, "chat_json", lambda *a, **kw: {"c1": dict(VALID_LESSON)})
    monkeypatch.setattr(llm.time, "sleep", lambda *_: None)
    report = explanations.generate_explanations(mm, ["c1"])
    assert report["generated"] == ["c1"], report
    assert explanations.has_explanation(mm["root"]["children"][0]["children"][0])


@pytest.mark.parametrize("rejection,fragment", [
    ({"providerErrorKind": "not_configured"}, "no llm provider"),
    ({"providerStatus": 404, "providerError": "model not found"}, "not loaded"),
    ({"providerStatus": 401, "providerError": "bad key"}, "authentication"),
    ({"providerStatus": 403, "providerError": "forbidden"}, "authentication"),
    ({"providerStatus": 400, "providerError": "unsupported parameter"}, "invalid configuration"),
    ({"providerStatus": 503, "providerError": "busy"}, "503"),
    ({"providerError": "ConnectError: connection refused"}, "unreachable"),
    ({"failureKind": "validation", "errors": ["explain too short"]}, "quality validation"),
    ({"failureKind": "lesson_parse"}, "not valid lesson json"),
])
def test_failure_classification_is_specific(rejection, fragment):
    assert fragment in explanations.describe_generation_failure(rejection).lower()


def test_health_reports_local_generation_capability(cfg):
    cfg(**LOCAL)
    from fastapi.testclient import TestClient
    data = TestClient(main.app).get("/api/health").json()
    assert data["llm"] == "openai-compatible"
    assert data["openai_compatible"] is True
    assert data["llm_local"] is True
    assert data["can_generate"] is True
    assert data["local"] is False


def test_generation_gate_explains_remote_key_requirement(cfg):
    cfg(**REMOTE)
    mm = _map_with_concept()
    report = explanations.generate_explanations(mm, ["c1"])
    assert report["generated"] == []
    assert "without an api key" in report["failed"][0]["reason"].lower()
